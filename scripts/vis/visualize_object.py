#!/usr/bin/env python3
"""Visualize a single HY3D-Part object: rendered views, segmentation masks,
per-part highlights, and mesh statistics.

Usage:
    # Visualize by object ID
    python scripts/visualize_object.py --obj-id 0000c32fde7f45efb8d14e8ba737d50c

    # Visualize Nth object in the dataset
    python scripts/visualize_object.py --index 0

    # Specify shard
    python scripts/visualize_object.py --index 5 --shard 00

    # Save to file instead of showing
    python scripts/visualize_object.py --index 0 --save vis_output.png
"""

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from partcraft.utils.config import load_config
from partcraft.io.hy3d_loader import HY3DPartDataset, ObjectRecord


# ---------- color palette ----------

# 20 distinct colors for parts (RGBA 0-1)
PART_COLORS = np.array([
    [0.894, 0.102, 0.110, 1.0],  # red
    [0.216, 0.494, 0.722, 1.0],  # blue
    [0.302, 0.686, 0.290, 1.0],  # green
    [0.596, 0.306, 0.639, 1.0],  # purple
    [1.000, 0.498, 0.000, 1.0],  # orange
    [1.000, 1.000, 0.200, 1.0],  # yellow
    [0.651, 0.337, 0.157, 1.0],  # brown
    [0.969, 0.506, 0.749, 1.0],  # pink
    [0.600, 0.600, 0.600, 1.0],  # gray
    [0.400, 0.761, 0.647, 1.0],  # teal
    [0.553, 0.827, 0.780, 1.0],
    [0.745, 0.729, 0.855, 1.0],
    [0.984, 0.502, 0.447, 1.0],
    [0.502, 0.694, 0.827, 1.0],
    [0.992, 0.706, 0.384, 1.0],
    [0.702, 0.871, 0.412, 1.0],
    [0.988, 0.804, 0.898, 1.0],
    [0.749, 0.357, 0.090, 1.0],
    [0.400, 0.400, 0.400, 1.0],
    [0.200, 0.200, 0.600, 1.0],
], dtype=np.float32)

BG_COLOR = np.array([0.15, 0.15, 0.15, 1.0])  # dark gray background


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert integer mask (H, W) to RGBA image. -1 = background."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:] = BG_COLOR
    for pid in np.unique(mask):
        if pid < 0:
            continue
        color = PART_COLORS[pid % len(PART_COLORS)]
        rgba[mask == pid] = color
    return rgba


def highlight_part(img_arr: np.ndarray, mask: np.ndarray, part_id: int) -> np.ndarray:
    """Highlight a specific part: dim others, add red border."""
    out = img_arr.astype(np.float32).copy()
    part_mask = mask == part_id

    # Dim non-part pixels
    out[~part_mask] = out[~part_mask] * 0.25

    # Red border
    border = binary_dilation(part_mask, iterations=3) & ~part_mask
    out[border] = [255, 50, 50] if out.shape[2] == 3 else [255, 50, 50, 255]

    return out.astype(np.uint8)


def load_image(obj: ObjectRecord, view_idx: int,
               bg_color: tuple[int, ...] = (255, 255, 255)) -> np.ndarray:
    """Load a rendered view as numpy RGB array, compositing onto bg_color.

    The raw views are RGBA with transparent background. Direct .convert("RGB")
    discards alpha and produces dark halos (ghosting) around edges.  Instead we
    alpha-composite onto a solid background first.
    """
    img_bytes = obj.get_image_bytes(view_idx)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (*bg_color, 255))
    composited = Image.alpha_composite(bg, img)
    return np.array(composited.convert("RGB"))


def get_part_stats(obj: ObjectRecord) -> list[dict]:
    """Compute per-part statistics."""
    stats = []
    for part in obj.parts:
        # Find best view and pixel coverage
        best_view = obj.get_best_view_for_part(part.part_id)
        pixel_counts = obj.get_mask_pixel_counts(best_view)
        total_fg = sum(v for k, v in pixel_counts.items() if k >= 0)
        part_pixels = pixel_counts.get(part.part_id, 0)
        coverage = part_pixels / max(total_fg, 1)

        stats.append({
            "part_id": part.part_id,
            "name": part.cluster_name,
            "mesh_nodes": part.mesh_node_names,
            "faces": part.cluster_size,
            "best_view": best_view,
            "coverage": coverage,
        })
    return stats


def visualize_object(obj: ObjectRecord, save_path: str | None = None):
    """Create a comprehensive visualization of a single object."""
    n_parts = len(obj.parts)
    part_ids = sorted(p.part_id for p in obj.parts)

    # Pick 6 evenly spaced views
    n_views = min(6, obj.num_views)
    view_step = max(1, obj.num_views // n_views)
    view_ids = list(range(0, obj.num_views, view_step))[:n_views]

    # ---------- build figure ----------
    # Layout:
    #   Row 0: rendered views (6)
    #   Row 1: segmentation masks (6)
    #   Row 2: per-part highlight (up to 10, 2 rows of 5)
    #   Right side: text info panel

    n_highlight_cols = min(5, n_parts)
    n_highlight_rows = (n_parts + n_highlight_cols - 1) // n_highlight_cols
    n_highlight_rows = min(n_highlight_rows, 3)  # cap at 3 rows
    max_parts_shown = n_highlight_cols * n_highlight_rows

    total_rows = 2 + n_highlight_rows
    total_cols = max(n_views, n_highlight_cols)

    fig = plt.figure(figsize=(3.2 * total_cols + 3.5, 3.0 * total_rows + 0.8))
    # Main grid: images on left, info panel on right
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[total_cols * 3.2, 3.5],
                              wspace=0.02)
    gs_img = gridspec.GridSpecFromSubplotSpec(
        total_rows, total_cols, subplot_spec=outer[0], hspace=0.15, wspace=0.05)
    gs_info = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    # ---- Row 0: Rendered views ----
    for i, vid in enumerate(view_ids):
        ax = fig.add_subplot(gs_img[0, i])
        img = load_image(obj, vid)
        ax.imshow(img)
        ax.set_title(f"view {vid}", fontsize=8, color="white")
        ax.axis("off")

    # ---- Row 1: Masks ----
    for i, vid in enumerate(view_ids):
        ax = fig.add_subplot(gs_img[1, i])
        mask = obj.get_mask(vid)
        ax.imshow(mask_to_rgb(mask))
        ax.set_title(f"mask {vid}", fontsize=8, color="white")
        ax.axis("off")

    # ---- Rows 2+: Per-part highlights ----
    part_stats = get_part_stats(obj)
    for idx, ps in enumerate(part_stats[:max_parts_shown]):
        row = 2 + idx // n_highlight_cols
        col = idx % n_highlight_cols
        ax = fig.add_subplot(gs_img[row, col])

        pid = ps["part_id"]
        bv = ps["best_view"]
        img = load_image(obj, bv)
        mask = obj.get_mask(bv)
        highlighted = highlight_part(img, mask, pid)
        ax.imshow(highlighted)

        color = PART_COLORS[pid % len(PART_COLORS)][:3]
        ax.set_title(
            f"part_{pid} ({ps['faces']} faces)",
            fontsize=7, color=color, fontweight="bold",
        )
        ax.axis("off")

    # ---- Info panel ----
    ax_info = fig.add_subplot(gs_info[0, 0])
    ax_info.axis("off")

    info_lines = [
        f"Object ID:",
        f"  {obj.obj_id}",
        f"Shard: {obj.shard}",
        f"Views: {obj.num_views}",
        f"Parts: {n_parts}",
        "",
        "─── Part Details ───",
    ]
    for ps in part_stats:
        color_hex = "#{:02x}{:02x}{:02x}".format(
            *[int(c * 255) for c in PART_COLORS[ps["part_id"] % len(PART_COLORS)][:3]]
        )
        info_lines.append(
            f"  part_{ps['part_id']}: {ps['faces']:,} faces, "
            f"cov={ps['coverage']:.1%}"
        )
        if ps["mesh_nodes"]:
            node_str = ", ".join(ps["mesh_nodes"][:3])
            if len(ps["mesh_nodes"]) > 3:
                node_str += f" (+{len(ps['mesh_nodes'])-3})"
            info_lines.append(f"    nodes: {node_str}")

    # Render text with part colors
    y_pos = 0.97
    line_height = 0.035
    for line in info_lines:
        # Color-code part lines
        color = "white"
        if line.strip().startswith("part_"):
            try:
                pid = int(line.strip().split("part_")[1].split(":")[0])
                color = PART_COLORS[pid % len(PART_COLORS)][:3]
            except (ValueError, IndexError):
                pass
        ax_info.text(
            0.02, y_pos, line, transform=ax_info.transAxes,
            fontsize=7.5, fontfamily="monospace", color=color,
            verticalalignment="top",
        )
        y_pos -= line_height

    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"PartCraft3D Object Viewer — {obj.obj_id[:16]}...",
        fontsize=11, color="white", fontweight="bold", y=0.99,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize a single HY3D-Part object")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--obj-id", type=str, default=None, help="Object ID to visualize")
    parser.add_argument("--index", type=int, default=None, help="Object index in dataset")
    parser.add_argument("--shard", type=str, default=None, help="Override shard")
    parser.add_argument("--save", type=str, default="./outputs/sample", help="Save path (PNG). If not set, plt.show()")
    args = parser.parse_args()

    cfg = load_config(args.config)
    shards = [args.shard] if args.shard else cfg["data"]["shards"]

    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        shards,
    )

    if args.obj_id:
        # Find object by ID
        obj = None
        for shard in shards:
            try:
                obj = dataset.load_object(shard, args.obj_id)
                break
            except Exception:
                continue
        if obj is None:
            print(f"Object {args.obj_id} not found in shards {shards}")
            sys.exit(1)
    elif args.index is not None:
        if args.index >= len(dataset):
            print(f"Index {args.index} out of range (dataset has {len(dataset)} objects)")
            sys.exit(1)
        obj = list(dataset)[args.index]  # TODO: direct indexing
    else:
        # Default: first object
        obj = next(iter(dataset))

    print(f"Visualizing: {obj.obj_id} (shard={obj.shard}, {obj.num_views} views, {len(obj.parts)} parts)")
    visualize_object(obj, save_path=args.save)
    obj.close()


if __name__ == "__main__":
    main()
