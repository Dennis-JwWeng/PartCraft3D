#!/usr/bin/env python3
"""Visualize PartObjaverse test data: Objaverse full mesh vs PartObjaverse parts.

For each object, generates a multi-panel figure showing:
  - Row 0: Full mesh (with baked vertex colors) from multiple views
  - Row 1: All parts assembled with distinct per-part colors, same views
  - Row 2: Individual parts side by side (vertex colors from PLY)
  - Right panel: object metadata and part list

Usage:
    # Visualize all 3 test objects
    python scripts/visualize_partobjaverse.py

    # Single object by ID
    python scripts/visualize_partobjaverse.py --obj-id 002e462c8bfa4267a9c9f038c7966f3b

    # Single object by index
    python scripts/visualize_partobjaverse.py --index 0

    # Custom output
    python scripts/visualize_partobjaverse.py --save-dir outputs/partobjaverse_vis

    # Export parts as individual PLY files for external viewers
    python scripts/visualize_partobjaverse.py --export-ply --index 0
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from partcraft.io.hy3d_loader import HY3DPartDataset, ObjectRecord

# 20 distinct colors for part visualization
PART_PALETTE = np.array([
    [0.894, 0.102, 0.110],  # red
    [0.216, 0.494, 0.722],  # blue
    [0.302, 0.686, 0.290],  # green
    [0.596, 0.306, 0.639],  # purple
    [1.000, 0.498, 0.000],  # orange
    [1.000, 1.000, 0.200],  # yellow
    [0.651, 0.337, 0.157],  # brown
    [0.969, 0.506, 0.749],  # pink
    [0.600, 0.600, 0.600],  # gray
    [0.400, 0.761, 0.647],  # teal
    [0.553, 0.827, 0.780],
    [0.745, 0.729, 0.855],
    [0.984, 0.502, 0.447],
    [0.502, 0.694, 0.827],
    [0.992, 0.706, 0.384],
    [0.702, 0.871, 0.412],
    [0.988, 0.804, 0.898],
    [0.749, 0.357, 0.090],
    [0.400, 0.400, 0.400],
    [0.200, 0.200, 0.600],
], dtype=np.float64)


def extract_face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract per-face RGB colors (float64, 0-1) from a Trimesh."""
    n_faces = len(mesh.faces)
    # Vertex colors
    if mesh.visual.kind == "vertex":
        try:
            vc = mesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
            return vc[mesh.faces].mean(axis=1)
        except Exception:
            pass
    # Face colors
    try:
        fc = mesh.visual.face_colors
        if fc is not None and len(fc) == n_faces:
            return fc[:, :3].astype(np.float64) / 255.0
    except Exception:
        pass
    return np.full((n_faces, 3), 0.6)


def render_mesh_view(vertices, faces, face_colors, elev, azim, resolution=400):
    """Render one view of a colored mesh using matplotlib 3D projection."""
    dpi = 100
    figsize = resolution / dpi
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    poly3d = vertices[faces]
    collection = Poly3DCollection(poly3d, linewidths=0.0, edgecolors="none")
    collection.set_facecolor(face_colors)
    ax.add_collection3d(collection)

    mx, mn = vertices.max(axis=0), vertices.min(axis=0)
    center = (mx + mn) / 2
    extent = (mx - mn).max() / 2 * 1.15
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)

    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_box_aspect([1, 1, 1])
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize((resolution, resolution), Image.LANCZOS)


def normalize_vertices(vertices):
    """Center and scale vertices to [-1, 1]."""
    centroid = vertices.mean(axis=0)
    v = vertices - centroid
    scale = np.abs(v).max()
    if scale > 0:
        v = v / scale
    return v


def build_colored_parts_mesh(obj: ObjectRecord):
    """Load all parts and assign distinct palette colors per part.

    Returns (vertices, faces, face_colors, part_info_list)
    """
    obj._ensure_mesh_npz()
    all_verts, all_faces, all_fcolors = [], [], []
    part_info = []
    offset = 0

    for p in obj.parts:
        key = f"part_{p.part_id}.ply"
        if key not in obj._mesh_npz:
            continue
        ply_bytes = obj._mesh_npz[key].tobytes()
        mesh = trimesh.load(io.BytesIO(ply_bytes), file_type="ply")

        color = PART_PALETTE[p.part_id % len(PART_PALETTE)]
        fc = np.tile(color, (len(mesh.faces), 1))

        all_verts.append(mesh.vertices)
        all_faces.append(mesh.faces + offset)
        all_fcolors.append(fc)
        part_info.append({
            "part_id": p.part_id,
            "name": p.cluster_name,
            "nodes": p.mesh_node_names,
            "faces": p.cluster_size,
            "verts": len(mesh.vertices),
        })
        offset += len(mesh.vertices)

    vertices = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)
    face_colors = np.concatenate(all_fcolors)
    return vertices, faces, face_colors, part_info


def load_part_with_vertex_colors(obj: ObjectRecord, part_id: int):
    """Load a single part mesh with its original vertex colors."""
    obj._ensure_mesh_npz()
    key = f"part_{part_id}.ply"
    ply_bytes = obj._mesh_npz[key].tobytes()
    mesh = trimesh.load(io.BytesIO(ply_bytes), file_type="ply")
    fc = extract_face_colors(mesh)
    return mesh.vertices.copy(), mesh.faces.copy(), fc


def visualize_object(obj: ObjectRecord, save_path: str, resolution: int = 400,
                     max_faces: int = 60000):
    """Create comprehensive visualization for one object."""
    print(f"  Loading full mesh with baked vertex colors...")
    full_mesh = obj.get_full_mesh(colored=True)
    full_verts = normalize_vertices(full_mesh.vertices.copy())
    full_faces = full_mesh.faces.copy()
    full_fc = extract_face_colors(full_mesh)

    print(f"  Loading parts with palette colors...")
    parts_verts, parts_faces, parts_fc, part_info = build_colored_parts_mesh(obj)
    parts_verts = normalize_vertices(parts_verts)

    n_parts = len(part_info)
    azimuths = [0, 90, 180, 270]
    elev = 25
    n_views = len(azimuths)

    # Downsample for rendering if needed
    def maybe_downsample(faces, fc, max_f):
        if len(faces) > max_f:
            step = max(1, len(faces) // max_f)
            idx = np.arange(0, len(faces), step)
            return faces[idx], fc[idx]
        return faces, fc

    full_faces_r, full_fc_r = maybe_downsample(full_faces, full_fc, max_faces)
    parts_faces_r, parts_fc_r = maybe_downsample(parts_faces, parts_fc, max_faces)

    # --- Render full mesh views ---
    print(f"  Rendering full mesh ({len(full_faces):,} faces)...")
    full_views = []
    for az in azimuths:
        img = render_mesh_view(full_verts, full_faces_r, full_fc_r, elev, az, resolution)
        full_views.append(img)

    # --- Render parts (palette colored) views ---
    print(f"  Rendering assembled parts ({len(parts_faces):,} faces)...")
    parts_views = []
    for az in azimuths:
        img = render_mesh_view(parts_verts, parts_faces_r, parts_fc_r, elev, az, resolution)
        parts_views.append(img)

    # --- Render individual parts with vertex colors ---
    max_individual = min(n_parts, 10)
    print(f"  Rendering {max_individual} individual parts...")
    individual_views = []
    for pi in part_info[:max_individual]:
        pid = pi["part_id"]
        v, f, fc = load_part_with_vertex_colors(obj, pid)
        v = normalize_vertices(v)
        f, fc = maybe_downsample(f, fc, max_faces // 2)
        img = render_mesh_view(v, f, fc, elev, 135, resolution)
        individual_views.append((img, pi))

    # --- Build figure ---
    ind_cols = min(5, max_individual)
    ind_rows = (max_individual + ind_cols - 1) // ind_cols
    total_rows = 2 + ind_rows
    total_cols = max(n_views, ind_cols)

    fig = plt.figure(figsize=(3.5 * total_cols + 4, 3.2 * total_rows + 1))
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              width_ratios=[total_cols * 3.5, 4], wspace=0.02)
    gs = gridspec.GridSpecFromSubplotSpec(total_rows, total_cols,
                                          subplot_spec=outer[0],
                                          hspace=0.2, wspace=0.05)
    gs_info = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    # Row 0: Full mesh (Objaverse original with baked colors)
    for i, img in enumerate(full_views):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        label = f"az={azimuths[i]}" if i > 0 else "Objaverse Full Mesh (baked colors)"
        ax.set_title(label, fontsize=8, color="white", fontweight="bold")
        ax.axis("off")

    # Row 1: Parts assembled (palette colors)
    for i, img in enumerate(parts_views):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img)
        label = f"az={azimuths[i]}" if i > 0 else f"PartObjaverse Parts ({n_parts} parts)"
        ax.set_title(label, fontsize=8, color="white", fontweight="bold")
        ax.axis("off")

    # Rows 2+: Individual parts
    for idx, (img, pi) in enumerate(individual_views):
        row = 2 + idx // ind_cols
        col = idx % ind_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        color = PART_PALETTE[pi["part_id"] % len(PART_PALETTE)]
        ax.set_title(f"part_{pi['part_id']}: {pi['nodes'][0] if pi['nodes'] else '?'}",
                     fontsize=7, color=color, fontweight="bold")
        ax.axis("off")

    # Info panel
    ax_info = fig.add_subplot(gs_info[0, 0])
    ax_info.axis("off")

    lines = [
        f"Object ID:",
        f"  {obj.obj_id}",
        f"Shard: {obj.shard}",
        f"Views: {obj.num_views}",
        f"Full mesh: {len(full_mesh.vertices):,} verts, {len(full_mesh.faces):,} faces",
        f"Parts: {n_parts}",
        "",
        "--- Part Details ---",
    ]
    for pi in part_info:
        nodes_str = ", ".join(pi["nodes"][:2])
        if len(pi["nodes"]) > 2:
            nodes_str += f" +{len(pi['nodes'])-2}"
        lines.append(f"  part_{pi['part_id']}: {pi['faces']} faces, {pi['verts']} verts")
        lines.append(f"    {nodes_str}")

    y_pos = 0.97
    for line in lines:
        color = "white"
        if line.strip().startswith("part_"):
            try:
                pid = int(line.strip().split("part_")[1].split(":")[0])
                color = PART_PALETTE[pid % len(PART_PALETTE)]
            except (ValueError, IndexError):
                pass
        ax_info.text(0.02, y_pos, line, transform=ax_info.transAxes,
                     fontsize=7.5, fontfamily="monospace", color=color,
                     verticalalignment="top")
        y_pos -= 0.032

    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"PartObjaverse Viewer: {obj.obj_id[:20]}...",
        fontsize=12, color="white", fontweight="bold", y=0.99,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def export_ply_files(obj: ObjectRecord, export_dir: str):
    """Export full mesh and parts as individual PLY files."""
    out = Path(export_dir) / obj.obj_id
    out.mkdir(parents=True, exist_ok=True)

    # Full mesh with baked colors
    print(f"  Exporting full mesh with baked vertex colors...")
    full = obj.get_full_mesh(colored=True)
    full.export(str(out / "full_colored.ply"))
    print(f"    -> {out / 'full_colored.ply'}")

    # Individual parts with vertex colors
    obj._ensure_mesh_npz()
    for p in obj.parts:
        key = f"part_{p.part_id}.ply"
        if key not in obj._mesh_npz:
            continue
        ply_bytes = obj._mesh_npz[key].tobytes()
        mesh = trimesh.load(io.BytesIO(ply_bytes), file_type="ply")
        out_path = out / f"part_{p.part_id}.ply"
        mesh.export(str(out_path))
        print(f"    -> {out_path} ({len(mesh.vertices)} verts)")

    print(f"  Exported to {out}/")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PartObjaverse: Objaverse full mesh vs segmented parts")
    parser.add_argument("--config", type=str, default="configs/partobjaverse_test.yaml")
    parser.add_argument("--obj-id", type=str, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="outputs/partobjaverse_vis")
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--export-ply", action="store_true",
                        help="Export meshes as individual PLY files")
    args = parser.parse_args()

    # Load config
    from partcraft.utils.config import load_config
    cfg = load_config(args.config)

    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )

    print(f"Dataset: {len(dataset)} objects")

    # Select objects
    if args.obj_id:
        objects = []
        for shard in cfg["data"]["shards"]:
            try:
                objects.append(dataset.load_object(shard, args.obj_id))
                break
            except Exception:
                continue
        if not objects:
            print(f"Object {args.obj_id} not found")
            sys.exit(1)
    elif args.index is not None:
        objects = [list(dataset)[args.index]]
    else:
        objects = list(dataset)

    for obj in objects:
        print(f"\nObject: {obj.obj_id} (shard={obj.shard}, {len(obj.parts)} parts)")

        if args.export_ply:
            export_ply_files(obj, args.save_dir)

        save_path = str(Path(args.save_dir) / f"{obj.obj_id}.png")
        visualize_object(obj, save_path, resolution=args.resolution)
        obj.close()

    print(f"\nDone! Results in {args.save_dir}/")


if __name__ == "__main__":
    main()
