#!/usr/bin/env python3
"""Part overview rendering — CLI entry point.

The library code has moved to ``partcraft.render.overview``.
This file re-exports everything for backward compatibility and provides
the standalone CLI (``python scripts/tools/render_part_overview.py --obj-id ...``).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from partcraft.render.overview import (  # noqa: F401 — re-export for compat
    VIEW_INDICES, _PALETTE, _PALETTE_NAMES,
    extract_parts, load_views_from_npz, run_blender, stitch_two_rows,
)

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BLENDER_SCRIPT = _PROJECT_ROOT / "scripts" / "blender_render_parts.py"


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
