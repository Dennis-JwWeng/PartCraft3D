#!/usr/bin/env python3
"""Part highlight rendering — CLI entry point.

The library code has moved to ``partcraft.render.highlight``.
This file re-exports everything for backward compatibility and provides
the standalone CLI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from partcraft.render.highlight import (  # noqa: F401 — re-export for compat
    HIGHLIGHT, GRAY, render_highlight, make_header, stitch_pair,
)
from partcraft.render.overview import VIEW_INDICES  # noqa: F401

import cv2

_HERE = Path(__file__).resolve().parent


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
