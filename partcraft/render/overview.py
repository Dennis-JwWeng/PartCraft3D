"""Part overview rendering utilities.

Provides the 5x2 grid image that the VLM sees during Phase 1 (top row =
original photos, bottom row = palette-colored part renders).

Previously lived in ``scripts/tools/render_part_overview.py`` as a
standalone script.  Moved here so pipeline_v2 modules can import cleanly
without ``sys.path`` manipulation.

``scripts/tools/render_part_overview.py`` is kept as a thin shim + CLI
entry point that re-exports everything from this module.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BLENDER_SCRIPT = _PROJECT_ROOT / "scripts" / "blender_render_parts.py"

# Fixed 5 view indices from the saved 150-view set:
#   89  back-left  overhead   (yaw -143 deg, pitch -27 deg)
#   90  back-right overhead   (yaw +127 deg, pitch -28 deg)
#   91  front-left overhead   (yaw  -53 deg, pitch -28 deg)
#   100 front-right overhead  (yaw  +53 deg, pitch -34 deg)
#   8   front upward          (yaw  +22 deg, pitch +52 deg)
# Changing this list requires updating scripts/tools/render_part_overview.py in lockstep.
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
    """Extract per-part mesh files from a mesh NPZ to out_dir.

    Supports both GLB-format (part_N.glb) and PLY-format (part_N.ply) NPZs.
    Returns sorted list of part IDs extracted.
    """
    npz = np.load(npz_path, allow_pickle=False)

    # Detect format from available keys
    glb_keys = [k for k in npz.files if re.match(r'^part_\d+\.glb$', k)]
    ply_keys = [k for k in npz.files if re.match(r'^part_\d+\.ply$', k)]

    if glb_keys:
        part_keys = glb_keys
        ext = ".glb"
    else:
        part_keys = ply_keys
        ext = ".ply"

    pids = []
    for key in part_keys:
        pid = int(re.search(r'\d+', key).group())
        out_path = out_dir / f"part_{pid}{ext}"
        out_path.write_bytes(bytes(npz[key]))
        pids.append(pid)

    return sorted(pids)


def load_views_from_npz(images_npz: Path, view_indices: list[int]):
    """Return (list of BGR np.ndarray, list of frame dicts) for the chosen views.

    Each frame dict has at least ``transform_matrix`` and ``camera_angle_x``.
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
    *,
    use_vertex_colors: bool = False,
    samples: int | None = None,
) -> list[np.ndarray]:
    """Render parts from the supplied camera frames. Returns list of BGR images.

    When ``use_vertex_colors=True`` the Blender script reads the 'Col' face-corner
    color attribute (set by PLY import) instead of painting a solid palette color.
    The color_attributes must NOT be stripped in that mode.

    ``samples`` overrides the Cycles sample count (default: 32 in vertex-color
    mode, 4 in solid-palette mode).  Pass a lower value (e.g. 8) for fast
    preview renders where denoising is not needed.
    """
    with tempfile.TemporaryDirectory() as out:
        cmd = [
            blender, "-b", "-P", str(_BLENDER_SCRIPT), "--",
            "--parts_dir", str(parts_dir),
            "--palette", json.dumps(pid_palette),
            "--output_folder", out,
            "--frames", json.dumps(frames),
            "--resolution", str(resolution),
        ]
        if use_vertex_colors:
            cmd.append("--use_vertex_colors")
        if samples is not None:
            cmd.extend(["--samples", str(samples)])
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
    sep_row = np.full((6, row_top.shape[1], 3), 180, dtype=np.uint8)
    return np.vstack([row_top, sep_row, row_bot])


__all__ = [
    "VIEW_INDICES",
    "_PALETTE", "_PALETTE_NAMES",
    "extract_parts", "load_views_from_npz",
    "run_blender", "stitch_two_rows",
]
