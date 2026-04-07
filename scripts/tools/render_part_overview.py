#!/usr/bin/env python3
"""Render a 4-view part overview pairing original photos with colored renders.

Reads ``data/partverse/{images,mesh}/{shard}/{obj_id}.npz``:

  * From ``images``: pulls 4 fixed-index original PNGs + their NeRF-style
    camera matrices (``transforms.json``).
  * From ``mesh``: extracts each ``part_<id>.ply`` to a tmp dir and runs
    ``scripts/blender_render_parts.py`` with the SAME 4 camera matrices to
    produce part-colored versions of those exact views.

Stitches the result as 2 rows × 5 columns: top = clean originals, bottom =
colored part renders. This image is the input the VLM sees during phase 1.

The 5 fixed view indices are 4 overhead cardinal-ish views (idx 89/90/91/100,
camera above origin tilted ~30° downward, covering 4 yaw octants) plus 1
steep upward front view (idx 8, pitch +52°) for inspecting undersides.
Reusable for any edit type (deletion / modification / scale / material / global).

Usage:
    python scripts/tools/render_part_overview.py \
        --obj-id 112204b2a12c4e25bbbdcc0d196b1ad5 --shard 01 \
        --out outputs/_debug/part_overview_vanity.png
"""
from __future__ import annotations
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BLENDER_SCRIPT = _PROJECT_ROOT / "scripts" / "blender_render_parts.py"

# Fixed 5 view indices from the saved 16-view set:
#   89  back-left  overhead   (yaw -143°, pitch -27°)
#   90  back-right overhead   (yaw +127°, pitch -28°)
#   91  front-left overhead   (yaw  -53°, pitch -28°)
#   100 front-right overhead  (yaw  +53°, pitch -34°)
#   8   front upward          (yaw  +22°, pitch +52°)
# The 4 overhead views are the only above-origin views the dataset stores;
# they give clean coverage of top faces (lids, seats, table tops) and overall
# topology. The 5th (idx 8) is the most extreme upward view available, used
# to expose underside features (chair bottoms, table undersides, base plates).
VIEW_INDICES = [89, 90, 91, 100, 8]

# 16 named colors VLMs reliably distinguish & name. part_<i> gets _PALETTE[i % 16].
_PALETTE_NAMES = [
    "red", "orange", "yellow", "lime", "green", "teal", "cyan", "blue",
    "navy", "purple", "magenta", "pink", "brown", "tan", "black", "gray",
]
_PALETTE = [
    [220,  30,  30],  # red
    [255, 140,   0],  # orange
    [255, 220,   0],  # yellow
    [130, 220,  30],  # lime
    [ 30, 160,  50],  # green
    [  0, 150, 150],  # teal
    [ 60, 220, 230],  # cyan
    [ 30,  90, 240],  # blue
    [ 20,  30, 130],  # navy
    [140,  40, 200],  # purple
    [230,  40, 200],  # magenta
    [255, 150, 200],  # pink
    [130,  70,  30],  # brown
    [220, 180, 130],  # tan
    [ 30,  30,  30],  # black
    [130, 130, 130],  # gray
]


def extract_parts(npz_path: Path, out_dir: Path) -> list[int]:
    """Extract part_*.ply from a mesh NPZ. Returns sorted part_ids."""
    z = np.load(npz_path, allow_pickle=True)
    pids = []
    for k in z.files:
        if k.startswith("part_") and k.endswith(".ply"):
            pid = int(k.replace("part_", "").replace(".ply", ""))
            (out_dir / f"part_{pid}.ply").write_bytes(bytes(z[k]))
            pids.append(pid)
    pids.sort()
    return pids


def load_views_from_npz(images_npz: Path, view_indices: list[int]):
    """Return (list of BGR np.ndarray, list of frame dicts) for the chosen views.

    Each frame dict has at least 'transform_matrix' and 'camera_angle_x'.
    """
    z = np.load(images_npz, allow_pickle=True)
    if "transforms.json" not in z.files:
        raise RuntimeError(f"transforms.json missing in {images_npz}")
    tf = json.loads(bytes(z["transforms.json"]).decode())
    frames_by_name = {f["file_path"]: f for f in tf["frames"]}

    imgs, frames = [], []
    for idx in view_indices:
        png_key = f"{idx:03d}.png"
        if png_key not in z.files:
            raise RuntimeError(f"view {png_key} not present in {images_npz}")
        if png_key not in frames_by_name:
            raise RuntimeError(f"frame for {png_key} missing in transforms.json")
        img = cv2.imdecode(np.frombuffer(bytes(z[png_key]), np.uint8),
                           cv2.IMREAD_UNCHANGED)
        if img.ndim == 3 and img.shape[2] == 4:
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            bg = np.full_like(rgb, 255)
            img = (rgb * a + bg * (1 - a)).astype(np.uint8)
        imgs.append(img)  # BGR
        frames.append(frames_by_name[png_key])
    return imgs, frames


def run_blender(
    parts_dir: Path,
    blender: str,
    resolution: int,
    pid_palette: list[list[int]],
    frames: list[dict],
) -> list[np.ndarray]:
    """Render parts from the supplied camera frames. Returns list of BGR images."""
    with tempfile.TemporaryDirectory() as out:
        cmd = [
            blender, "-b", "-P", str(_BLENDER_SCRIPT), "--",
            "--parts_dir", str(parts_dir),
            "--palette", json.dumps(pid_palette),
            "--output_folder", out,
            "--frames", json.dumps(frames),
            "--resolution", str(resolution),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print("[blender stdout]\n" + r.stdout[-3000:])
            print("[blender stderr]\n" + r.stderr[-2000:])
            raise RuntimeError(f"blender failed exit={r.returncode}")
        imgs = []
        for i in range(len(frames)):
            p = os.path.join(out, f"{i:03d}.png")
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"missing render {p}")
            if img.shape[2] == 4:
                a = img[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                bg = np.full_like(rgb, 255)
                img = (rgb * a + bg * (1 - a)).astype(np.uint8)
            imgs.append(img)  # BGR
        return imgs


def stitch_two_rows(top: list[np.ndarray], bot: list[np.ndarray]) -> np.ndarray:
    """Top row = original views, bottom row = colored renders."""
    assert len(top) == len(bot)
    # Resize bottom to top resolution if needed
    H, W = top[0].shape[:2]
    bot = [cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)
           if b.shape[:2] != (H, W) else b for b in bot]

    sep_w = 4
    col_sep = np.full((H, sep_w, 3), 200, dtype=np.uint8)

    def make_row(imgs):
        row = imgs[0]
        for im in imgs[1:]:
            row = np.hstack([row, col_sep, im])
        return row

    row_top = make_row(top)
    row_bot = make_row(bot)
    row_sep_h = 6
    sep_row = np.full((row_sep_h, row_top.shape[1], 3), 180, dtype=np.uint8)
    return np.vstack([row_top, sep_row, row_bot])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-id", required=True)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--mesh-root", default="data/partverse/mesh")
    ap.add_argument("--images-root", default="data/partverse/images")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--blender",
        default="/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender",
    )
    args = ap.parse_args()

    mesh_npz = Path(args.mesh_root) / args.shard / f"{args.obj_id}.npz"
    img_npz = Path(args.images_root) / args.shard / f"{args.obj_id}.npz"
    for p in (mesh_npz, img_npz):
        if not p.is_file():
            print(f"[ERR] missing: {p}", file=sys.stderr)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load original views + camera frames
    top_imgs, frames = load_views_from_npz(img_npz, VIEW_INDICES)
    print(f"[INFO] loaded {len(top_imgs)} original views from {img_npz.name}")
    H = top_imgs[0].shape[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        part_ids = extract_parts(mesh_npz, tmp)
        print(f"[INFO] extracted {len(part_ids)} parts")
        if len(part_ids) > len(_PALETTE):
            print(f"[WARN] {len(part_ids)} parts > {len(_PALETTE)} palette colors; "
                  "this is a long-tail object — colors will repeat")

        max_pid = max(part_ids) + 1
        pid_palette = [[200, 200, 200]] * max_pid
        for pid in part_ids:
            pid_palette[pid] = _PALETTE[pid % len(_PALETTE)]

        bot_imgs = run_blender(tmp, args.blender, H, pid_palette, frames)

    final = stitch_two_rows(top_imgs, bot_imgs)
    cv2.imwrite(str(args.out), final)
    print(f"[OK] wrote {args.out} ({final.shape[1]}x{final.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
