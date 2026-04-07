#!/usr/bin/env python3
"""Render a single-view "highlight" image for one edit's selected parts.

For phase 1.5 VLM scoring. Given an obj_id, a view_index (0..4 into the
phase1 5-view set), and a list of selected_part_ids, this script:

  1. Loads the original photo for that view from images/{shard}/{obj_id}.npz
  2. Re-renders the SAME camera with Blender Workbench, painting:
       - selected parts        → their normal phase1 palette color
       - non-selected parts    → light gray (210,210,210)
       - background            → white
  3. Stitches a 1x2 image (original | highlight) with an optional header bar
     showing edit_type and prompt, and writes it to --out.

Used as a preview tool / building block for the phase 1.5 scoring runner.

Usage:
    python scripts/tools/render_part_highlight.py \
        --obj-id 112204b2a12c4e25bbbdcc0d196b1ad5 --shard 01 \
        --view-index 3 --part-ids 0 \
        --edit-type deletion \
        --prompt "Remove the top drawer from the cabinet." \
        --out outputs/_debug/highlight_demo.png
"""
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_part_overview import (  # noqa: E402
    VIEW_INDICES, extract_parts, load_views_from_npz, run_blender,
)

# Highlight color used for ALL selected parts (regardless of part_id).
# Single bright color → unambiguous "this is the edit target", works for both
# single- and multi-part edits without VLM having to disambiguate.
HIGHLIGHT = [230, 40, 200]   # magenta — high contrast on white/wood/gray
GRAY = [210, 210, 210]       # non-selected parts


def render_highlight(mesh_npz: Path, img_npz: Path, view_index: int,
                     selected_part_ids: list[int], blender: str
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (original_bgr, highlight_bgr) for the chosen view."""
    if not (0 <= view_index < len(VIEW_INDICES)):
        raise ValueError(f"view_index must be in [0,{len(VIEW_INDICES)})")
    top_imgs, frames = load_views_from_npz(img_npz, VIEW_INDICES)
    orig = top_imgs[view_index]
    frame = frames[view_index]
    H = orig.shape[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        part_ids = extract_parts(mesh_npz, tmp)
        if not part_ids:
            raise RuntimeError("no parts in mesh npz")
        max_pid = max(part_ids) + 1
        pid_palette = [list(GRAY)] * max_pid
        sel_set = set(selected_part_ids)
        for pid in part_ids:
            if pid in sel_set:
                pid_palette[pid] = list(HIGHLIGHT)
        # render only this single frame
        rendered = run_blender(tmp, blender, H, pid_palette, [frame])
    return orig, rendered[0]


def make_header(width: int, edit_type: str, prompt: str,
                height: int = 56) -> np.ndarray:
    bar = np.full((height, width, 3), 245, dtype=np.uint8)
    line1 = f"[{edit_type}]"
    cv2.putText(bar, line1, (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (30, 30, 30), 2, cv2.LINE_AA)
    # truncate prompt if too long for the bar
    max_chars = max(20, width // 11)
    p = prompt if len(prompt) <= max_chars else prompt[:max_chars - 1] + "…"
    cv2.putText(bar, p, (12, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return bar


def stitch_pair(orig: np.ndarray, hl: np.ndarray,
                edit_type: str = "", prompt: str = "") -> np.ndarray:
    H, W = orig.shape[:2]
    if hl.shape[:2] != (H, W):
        hl = cv2.resize(hl, (W, H), interpolation=cv2.INTER_AREA)
    sep = np.full((H, 6, 3), 180, dtype=np.uint8)
    # column labels
    def label_strip(text):
        s = np.full((28, W, 3), 230, dtype=np.uint8)
        cv2.putText(s, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 30, 30), 1, cv2.LINE_AA)
        return s
    orig_lab = np.vstack([label_strip("ORIGINAL"), orig])
    hl_lab = np.vstack([label_strip("HIGHLIGHTED PARTS"), hl])
    sep2 = np.full((orig_lab.shape[0], 6, 3), 180, dtype=np.uint8)
    body = np.hstack([orig_lab, sep2, hl_lab])
    if edit_type or prompt:
        header = make_header(body.shape[1], edit_type, prompt)
        body = np.vstack([header, body])
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-id", required=True)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--mesh-root", default="data/partverse/mesh")
    ap.add_argument("--images-root", default="data/partverse/images")
    ap.add_argument("--view-index", type=int, required=True)
    ap.add_argument("--part-ids", type=int, nargs="*", default=[],
                    help="Selected part_ids to highlight (may be empty for global)")
    ap.add_argument("--edit-type", default="")
    ap.add_argument("--prompt", default="")
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
    orig, hl = render_highlight(mesh_npz, img_npz, args.view_index,
                                args.part_ids, args.blender)
    out_img = stitch_pair(orig, hl, args.edit_type, args.prompt)
    cv2.imwrite(str(args.out), out_img)
    print(f"[OK] wrote {args.out} ({out_img.shape[1]}x{out_img.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
