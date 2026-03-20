#!/usr/bin/env python3
"""Visualize mask correctness for every edit spec in the pipeline.

Traces the full chain: VLM labels → edit spec → voxel mask, producing
a per-spec diagnostic image with:
  Row 1: rendered views of the object (edit parts highlighted in red)
  Row 2: 3-axis voxel projections (edit / preserved / final mask / SLAT overlay)
  Row 3: text info (edit_id, type, prompt, part IDs, mask stats)

Usage:
    # All specs for a config
    python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml

    # Single edit
    python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml --edit-id del_000000

    # Limit number of specs
    python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml --limit 10

    # Specify tag (default: v1)
    python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml --tag v1

    # Use a specific VD outputs path (for SLAT / mesh.ply)
    python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml \
        --vd-path /Node11_nvme/wjw/3D_Editing/Vinedresser3D-main
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── project root on sys.path ──
_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from partcraft.io.partcraft_loader import PartCraftDataset, ObjectRecord


# ── Colors ──
PART_COLORS = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
    (153, 153, 153), (0, 206, 209), (127, 255, 0), (255, 20, 147),
    (100, 149, 237), (255, 215, 0), (0, 128, 128), (220, 20, 60),
    (75, 0, 130), (0, 255, 127), (255, 69, 0), (148, 103, 189),
]


def _load_specs(spec_path: str) -> list[dict]:
    specs = []
    with open(spec_path) as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))
    return specs


def _load_semantic_labels(label_path: str) -> dict[str, dict]:
    """Load Phase 0 semantic labels, keyed by obj_id."""
    labels = {}
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                labels[d["obj_id"]] = d
    return labels


def _get_edit_part_ids(spec: dict) -> list[int]:
    """Derive edit_part_ids from spec (same logic as run_pipeline.py)."""
    edit_type = spec["edit_type"].capitalize()
    if edit_type == "Deletion":
        return spec.get("remove_part_ids", [])
    elif edit_type == "Modification":
        if spec.get("remove_part_ids"):
            return spec["remove_part_ids"]
        else:
            pid = spec.get("old_part_id", -1)
            return [pid] if pid >= 0 else []
    elif edit_type == "Addition":
        return spec.get("add_part_ids", [])
    elif edit_type == "Global":
        return []
    return []


def _render_part_highlight(
    obj_record: ObjectRecord,
    edit_part_ids: list[int],
    view_idx: int,
    resolution: int = 256,
) -> Image.Image:
    """Render a single view with edit parts highlighted in red overlay."""
    # Base rendered image
    try:
        img = obj_record.get_image_pil(view_idx).convert("RGB")
    except (KeyError, Exception):
        return Image.new("RGB", (resolution, resolution), (128, 128, 128))
    img = img.resize((resolution, resolution), Image.LANCZOS)

    if not edit_part_ids:
        return img

    # Overlay mask for edit parts
    try:
        mask_full = obj_record.get_mask(view_idx)
        if mask_full is not None:
            overlay = np.array(img).copy()
            for pid in edit_part_ids:
                part_mask = (mask_full == pid)
                # Resize part_mask to match resolution
                from PIL import Image as _Img
                pm_img = _Img.fromarray(part_mask.astype(np.uint8) * 255, "L")
                pm_img = pm_img.resize((resolution, resolution), Image.NEAREST)
                pm = np.array(pm_img) > 128
                # Red tint
                overlay[pm, 0] = np.clip(overlay[pm, 0].astype(int) + 100, 0, 255).astype(np.uint8)
                overlay[pm, 1] = (overlay[pm, 1] * 0.5).astype(np.uint8)
                overlay[pm, 2] = (overlay[pm, 2] * 0.5).astype(np.uint8)
            return Image.fromarray(overlay)
    except Exception:
        pass

    return img


def _voxelize_parts(
    obj_record: ObjectRecord,
    edit_part_ids: list[int],
    vd_mesh_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Voxelize edit and preserved parts into 64³ grids.

    Returns: (edit_grid, preserved_grid, stats_dict)
    Same logic as TrellisRefiner.build_part_mask but without SLAT or torch.
    """
    import open3d as o3d
    import trimesh

    stats = {}

    # Load VD mesh for reference frame
    vd_mesh = o3d.io.read_triangle_mesh(vd_mesh_path)
    vd_verts = np.asarray(vd_mesh.vertices)
    vd_center = (vd_verts.max(0) + vd_verts.min(0)) / 2
    vd_extent = (vd_verts.max(0) - vd_verts.min(0)).max()

    # Load HY3D-Part full mesh
    hy3d_full = obj_record.get_full_mesh(colored=False)
    hy3d_verts = np.array(hy3d_full.vertices)
    hy3d_center = (hy3d_verts.max(0) + hy3d_verts.min(0)) / 2
    hy3d_extent = (hy3d_verts.max(0) - hy3d_verts.min(0)).max()

    scale_factor = vd_extent / hy3d_extent if hy3d_extent > 0 else 1.0
    stats["scale_factor"] = float(scale_factor)
    stats["hy3d_extent"] = float(hy3d_extent)
    stats["vd_extent"] = float(vd_extent)

    all_part_ids = [p.part_id for p in obj_record.parts]
    edit_set = set(edit_part_ids)
    edit_meshes, preserved_meshes = [], []

    for pid in all_part_ids:
        try:
            pm = obj_record.get_part_mesh(pid, colored=False)
        except (KeyError, Exception):
            continue
        if pid in edit_set:
            edit_meshes.append(pm)
        else:
            preserved_meshes.append(pm)

    def _voxelize_combined(meshes: list) -> np.ndarray:
        grid = np.zeros((64, 64, 64), dtype=bool)
        if not meshes:
            return grid
        combined = trimesh.util.concatenate(meshes)
        verts = np.array(combined.vertices)
        verts_vd = (verts - hy3d_center) * scale_factor + vd_center
        # No axis reorder — SLAT coords are in VD space directly
        verts_vd = np.clip(verts_vd, -0.5 + 1e-6, 0.5 - 1e-6)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_vd)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(combined.faces))
        try:
            vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                o3d_mesh, voxel_size=1/64,
                min_bound=(-0.5, -0.5, -0.5),
                max_bound=(0.5, 0.5, 0.5))
            voxels = np.array([v.grid_index for v in vg.get_voxels()])
        except Exception:
            vg_grid = ((verts_vd + 0.5) * 64).astype(np.int32)
            voxels = np.unique(np.clip(vg_grid, 0, 63), axis=0)
        if len(voxels) > 0:
            grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
        return grid

    edit_grid = _voxelize_combined(edit_meshes)
    preserved_grid = _voxelize_combined(preserved_meshes)

    stats["edit_voxels"] = int(edit_grid.sum())
    stats["preserved_voxels"] = int(preserved_grid.sum())
    stats["n_edit_parts"] = len(edit_meshes)
    stats["n_preserved_parts"] = len(preserved_meshes)

    return edit_grid, preserved_grid, stats


def _build_mask_numpy(
    edit_grid: np.ndarray,
    preserved_grid: np.ndarray,
    edit_type: str,
) -> np.ndarray:
    """Build final mask (same logic as build_part_mask but pure numpy)."""
    from scipy import ndimage

    if edit_type == "Global":
        return np.ones((64, 64, 64), dtype=bool)

    if edit_type == "Deletion":
        mask = edit_grid.copy()
        # Dilate by 1 into preserved
        struct = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.binary_dilation(mask, structure=struct, iterations=1)
        transition = dilated & ~mask & preserved_grid
        mask = mask | transition
        return mask

    if edit_type in ("Modification", "Addition"):
        # Simplified _compute_editing_region: expand edit into nearby empty space
        # Use bbox of occupied region + pad
        occupied = edit_grid | preserved_grid
        occ_coords = np.argwhere(occupied)
        if len(occ_coords) == 0:
            return edit_grid.copy()

        pad = 3
        omin = occ_coords.min(axis=0)
        omax = occ_coords.max(axis=0)
        vicinity = np.zeros((64, 64, 64), dtype=bool)
        vicinity[
            max(omin[0] - pad, 0):min(omax[0] + pad + 1, 64),
            max(omin[1] - pad, 0):min(omax[1] + pad + 1, 64),
            max(omin[2] - pad, 0):min(omax[2] + pad + 1, 64),
        ] = True

        # Empty voxels in vicinity
        empty = ~(preserved_grid | edit_grid) & vicinity
        empty_coords = np.argwhere(empty)

        mask = np.zeros((64, 64, 64), dtype=bool)

        if len(empty_coords) > 0:
            # Assign empty voxels: check if nearest occupied voxel is edit
            from sklearn.neighbors import NearestNeighbors
            occ_is_edit = edit_grid[occ_coords[:, 0], occ_coords[:, 1], occ_coords[:, 2]]
            nbrs = NearestNeighbors(n_neighbors=min(20, len(occ_coords)),
                                     algorithm='ball_tree').fit(occ_coords)
            distances, indices = nbrs.kneighbors(empty_coords)
            neighbor_edit = occ_is_edit[indices].mean(axis=1)
            expand = neighbor_edit > 0.5
            mask[empty_coords[expand, 0], empty_coords[expand, 1],
                 empty_coords[expand, 2]] = True

        mask = mask | edit_grid
        mask = mask & ~preserved_grid
        return mask

    return edit_grid.copy()


def _proj_to_rgb(
    edit_grid: np.ndarray,
    preserved_grid: np.ndarray,
    mask: np.ndarray,
    axis: int,
    scale: int = 4,
) -> Image.Image:
    """Create RGB projection along axis.

    Red = mask, Green = preserved, Blue = edit (not in mask), Yellow = mask ∩ edit.
    """
    e_proj = edit_grid.any(axis=axis)
    p_proj = preserved_grid.any(axis=axis)
    m_proj = mask.any(axis=axis)

    h, w = e_proj.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Background
    rgb[:] = [30, 30, 30]
    # Preserved only: green
    rgb[p_proj & ~m_proj] = [0, 180, 0]
    # Edit only (not in mask): blue
    rgb[e_proj & ~m_proj] = [0, 100, 255]
    # Mask only (not edit, not preserved): dim red
    rgb[m_proj & ~e_proj & ~p_proj] = [180, 50, 50]
    # Mask ∩ edit: yellow
    rgb[m_proj & e_proj] = [255, 230, 0]
    # Mask ∩ preserved (transition zone): orange
    rgb[m_proj & p_proj] = [255, 140, 0]

    img = Image.fromarray(rgb)
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    return img


def _text_block(lines: list[str], width: int, height: int,
                font_size: int = 14) -> Image.Image:
    """Render text lines onto an image."""
    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                   font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    y = 4
    for line in lines:
        draw.text((8, y), line, fill=(220, 220, 220), font=font)
        y += font_size + 4
        if y > height - font_size:
            break
    return img


def visualize_one_spec(
    spec: dict,
    obj_record: ObjectRecord,
    semantic_label: dict | None,
    vd_outputs_path: str,
    save_path: str,
    resolution: int = 256,
):
    """Generate full diagnostic image for one edit spec."""
    obj_id = spec["obj_id"]
    edit_id = spec["edit_id"]
    edit_type = spec["edit_type"].capitalize()
    edit_part_ids = _get_edit_part_ids(spec)

    vd_mesh_path = os.path.join(vd_outputs_path, f"outputs/img_Enc/{obj_id}/mesh.ply")
    has_vd_mesh = os.path.exists(vd_mesh_path)

    # ── Row 1: Rendered views with part highlight ──
    # Pick up to 4 views: best_view + 3 orthogonal
    views = []
    best_view = spec.get("best_view", -1)
    if best_view >= 0:
        views.append(best_view)
    # Add some evenly spaced views
    n_views = obj_record.num_views
    if n_views > 0:
        for v in [0, n_views // 4, n_views // 2, 3 * n_views // 4]:
            if v not in views and len(views) < 4:
                views.append(v)
    while len(views) < 4:
        views.append(0)

    view_imgs = []
    for v in views:
        img = _render_part_highlight(obj_record, edit_part_ids, v, resolution)
        view_imgs.append(img)

    row1_w = resolution * 4 + 30
    row1 = Image.new("RGB", (row1_w, resolution), (20, 20, 20))
    for i, img in enumerate(view_imgs):
        row1.paste(img, (i * (resolution + 10), 0))

    # ── Row 2: Voxel projections ──
    vox_size = resolution  # each projection is resolution x resolution
    if has_vd_mesh and edit_type != "Global" and edit_part_ids:
        try:
            edit_grid, preserved_grid, vox_stats = _voxelize_parts(
                obj_record, edit_part_ids, vd_mesh_path)
            mask = _build_mask_numpy(edit_grid, preserved_grid, edit_type)

            # 3 projections: XY (axis=2), XZ (axis=1), YZ (axis=0)
            proj_labels = ["XY (top)", "XZ (front)", "YZ (side)"]
            proj_imgs = []
            scale = max(1, vox_size // 64)
            for ax in [2, 1, 0]:
                proj_imgs.append(_proj_to_rgb(edit_grid, preserved_grid, mask, ax, scale))

            row2_w = row1_w
            proj_h = proj_imgs[0].size[1]
            row2 = Image.new("RGB", (row2_w, proj_h + 20), (20, 20, 20))
            draw2 = ImageDraw.Draw(row2)
            try:
                small_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            except (OSError, IOError):
                small_font = ImageFont.load_default()

            x_offset = 0
            for i, (pimg, label) in enumerate(zip(proj_imgs, proj_labels)):
                row2.paste(pimg, (x_offset, 18))
                draw2.text((x_offset + 4, 2), label, fill=(200, 200, 200),
                           font=small_font)
                x_offset += pimg.size[0] + 10

            # Legend
            legend_x = x_offset + 10
            legend_items = [
                ((255, 230, 0), "mask∩edit"),
                ((255, 140, 0), "mask∩preserved (transition)"),
                ((0, 180, 0), "preserved"),
                ((0, 100, 255), "edit (not in mask)"),
                ((180, 50, 50), "mask (empty expansion)"),
            ]
            for j, (color, text) in enumerate(legend_items):
                y = 2 + j * 16
                draw2.rectangle([legend_x, y, legend_x + 10, y + 10], fill=color)
                draw2.text((legend_x + 14, y - 1), text,
                           fill=(200, 200, 200), font=small_font)

        except Exception as e:
            vox_stats = {"error": str(e)}
            row2 = _text_block([f"Voxelization error: {e}"], row1_w, 80)
            mask = None
    elif edit_type == "Global":
        vox_stats = {"mask": "full 64³"}
        row2 = _text_block(["Global edit: full 64³ mask (262144 voxels)"], row1_w, 40)
        mask = np.ones((64, 64, 64), dtype=bool)
    else:
        vox_stats = {"skip": "no VD mesh or no edit parts"}
        row2 = _text_block(
            [f"No VD mesh at {vd_mesh_path}" if not has_vd_mesh
             else "No edit parts"], row1_w, 40)
        mask = None

    # ── Row 3: Text info ──
    # Part labels from semantic
    part_labels = {}
    if semantic_label:
        for p in semantic_label.get("parts", []):
            part_labels[p["part_id"]] = f"{p['label']} ({p.get('group', '?')})"

    edit_labels = [part_labels.get(pid, f"part_{pid}") for pid in edit_part_ids]

    info_lines = [
        f"edit_id: {edit_id}    type: {edit_type}    obj: {obj_id}",
        f"prompt: {spec.get('edit_prompt', '?')}",
        f"edit_parts: {edit_part_ids} → {edit_labels}",
        f"after_desc: {spec.get('after_desc', '?')[:100]}",
    ]

    if mask is not None:
        mask_vox = int(mask.sum())
        info_lines.append(
            f"mask: {mask_vox} voxels ({mask_vox/64**3*100:.1f}% of 64³)  "
            f"| edit_vox={vox_stats.get('edit_voxels', '?')}  "
            f"preserved_vox={vox_stats.get('preserved_voxels', '?')}"
        )
    else:
        info_lines.append(f"voxel stats: {vox_stats}")

    if spec.get("before_part_desc"):
        info_lines.append(f"before_part: {spec['before_part_desc']}")
    if spec.get("after_part_desc"):
        info_lines.append(f"after_part: {spec['after_part_desc']}")

    # Show VLM labels for all parts
    if semantic_label:
        all_labels = [f"  {p['part_id']}:{p['label']}" for p in semantic_label.get("parts", [])]
        info_lines.append(f"all parts: {' '.join(all_labels)}")

    row3 = _text_block(info_lines, row1_w, 16 * len(info_lines) + 16, font_size=13)

    # ── Compose ──
    total_h = row1.size[1] + row2.size[1] + row3.size[1] + 20
    canvas = Image.new("RGB", (row1_w, total_h), (20, 20, 20))
    y = 0
    canvas.paste(row1, (0, y)); y += row1.size[1] + 5
    canvas.paste(row2, (0, y)); y += row2.size[1] + 5
    canvas.paste(row3, (0, y))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize masks for all edit specs in the pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Pipeline config YAML")
    parser.add_argument("--tag", type=str, default="v1",
                        help="Pipeline run tag (default: v1)")
    parser.add_argument("--edit-id", type=str, default=None,
                        help="Only visualize this edit_id")
    parser.add_argument("--obj-id", type=str, default=None,
                        help="Only visualize specs for this object")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of specs to visualize")
    parser.add_argument("--vd-path", type=str, default=None,
                        help="Vinedresser3D root path (default: from config)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory (default: outputs/.../vis_masks)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="View render resolution (default: 256)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Paths — apply same cache_dir normalization as run_pipeline.py
    output_dir = cfg["data"].get("output_dir", "outputs")
    for _phase_key in ("phase0", "phase1", "phase2", "phase2_5", "phase3", "phase4"):
        _pcfg = cfg.get(_phase_key, {})
        _cd = _pcfg.get("cache_dir", "")
        if _cd and not os.path.isabs(_cd) and not _cd.startswith(output_dir):
            _pcfg["cache_dir"] = os.path.join(output_dir, _cd)
    phase0_cache = cfg.get("phase0", {}).get("cache_dir",
                    os.path.join(output_dir, "cache/phase0"))
    phase1_cache = cfg.get("phase1", {}).get("cache_dir",
                    os.path.join(output_dir, "cache/phase1"))

    tag = args.tag
    spec_path = os.path.join(phase1_cache, f"edit_specs_{tag}.jsonl")
    label_path = os.path.join(phase0_cache, f"semantic_labels_{tag}.jsonl")


    vd_path = args.vd_path or cfg.get("phase2_5", {}).get(
        "vinedresser_path", "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main")

    save_dir = args.save_dir or os.path.join(output_dir, "vis_masks", tag)
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print(f"Loading specs from {spec_path}")
    if not os.path.exists(spec_path):
        print(f"ERROR: spec file not found: {spec_path}")
        sys.exit(1)
    specs = _load_specs(spec_path)
    print(f"  → {len(specs)} edit specs")

    semantic_labels = {}
    if os.path.exists(label_path):
        print(f"Loading VLM labels from {label_path}")
        semantic_labels = _load_semantic_labels(label_path)
        print(f"  → {len(semantic_labels)} objects")
    else:
        print(f"WARNING: no semantic labels at {label_path}")

    # Load dataset
    dataset = PartCraftDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )

    # Filter specs
    if args.edit_id:
        specs = [s for s in specs if s["edit_id"] == args.edit_id]
    if args.obj_id:
        specs = [s for s in specs if s["obj_id"] == args.obj_id]
    if args.limit:
        specs = specs[:args.limit]

    if not specs:
        print("No specs to visualize after filtering.")
        sys.exit(0)

    print(f"Visualizing {len(specs)} specs → {save_dir}/")
    print()

    # Cache loaded objects
    obj_cache: dict[str, ObjectRecord] = {}

    for i, spec in enumerate(specs):
        obj_id = spec["obj_id"]
        shard = spec["shard"]
        edit_id = spec["edit_id"]

        print(f"[{i+1}/{len(specs)}] {edit_id} ({spec['edit_type']}) "
              f"obj={obj_id[:12]}... ", end="", flush=True)

        # Load object record
        cache_key = f"{shard}/{obj_id}"
        if cache_key not in obj_cache:
            try:
                obj_record = dataset.load_object(shard, obj_id)
                obj_cache[cache_key] = obj_record
            except Exception as e:
                print(f"SKIP (load error: {e})")
                continue
        obj_record = obj_cache[cache_key]

        save_path = os.path.join(save_dir, f"{edit_id}_{obj_id[:8]}.png")

        try:
            visualize_one_spec(
                spec=spec,
                obj_record=obj_record,
                semantic_label=semantic_labels.get(obj_id),
                vd_outputs_path=vd_path,
                save_path=save_path,
                resolution=args.resolution,
            )
            print(f"OK → {os.path.basename(save_path)}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Results in {save_dir}/")

    # Generate index HTML
    _generate_index_html(save_dir, specs)


def _generate_index_html(save_dir: str, specs: list[dict]):
    """Generate an HTML index for easy browsing."""
    html_path = os.path.join(save_dir, "index.html")

    rows = []
    for spec in specs:
        edit_id = spec["edit_id"]
        obj_id = spec["obj_id"]
        img_name = f"{edit_id}_{obj_id[:8]}.png"
        img_path = os.path.join(save_dir, img_name)
        if not os.path.exists(img_path):
            continue

        edit_parts = _get_edit_part_ids(spec)
        rows.append(f"""
        <div class="card" id="{edit_id}">
            <h3>{edit_id} — {spec['edit_type']} — {obj_id[:12]}...</h3>
            <p><b>Prompt:</b> {spec.get('edit_prompt', '?')}</p>
            <p><b>Edit parts:</b> {edit_parts}</p>
            <img src="{img_name}" loading="lazy" />
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Mask Visualization — {os.path.basename(save_dir)}</title>
<style>
body {{ background: #1a1a1a; color: #ddd; font-family: monospace; padding: 20px; }}
.card {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; }}
.card h3 {{ color: #ffcc00; margin: 0 0 8px 0; }}
.card img {{ max-width: 100%; border: 1px solid #444; }}
.card p {{ margin: 4px 0; }}
.nav {{ position: sticky; top: 0; background: #111; padding: 10px; z-index: 10; }}
.nav a {{ color: #6cf; margin-right: 10px; text-decoration: none; }}
.nav a:hover {{ text-decoration: underline; }}
</style>
</head><body>
<h1>Mask Visualization ({len(rows)} specs)</h1>
<div class="nav">
{''.join(f'<a href="#{s["edit_id"]}">{s["edit_id"]}</a>' for s in specs if os.path.exists(os.path.join(save_dir, f"{s['edit_id']}_{s['obj_id'][:8]}.png")))}
</div>
{''.join(rows)}
</body></html>"""

    with open(html_path, "w") as f:
        f.write(html)
    print(f"Index: {html_path}")


if __name__ == "__main__":
    main()
