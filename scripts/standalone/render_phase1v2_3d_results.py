#!/usr/bin/env python3
"""Render phase1_v2 step4 3D edit results at the original phase1 view.

For each edit in <in-dir>:
  1. Load mesh_pairs_<tag>/{edit_id}/after.npz (SLAT)
  2. Look up the edit's view_index in the matching parsed.json
  3. Pull the corresponding camera frame from images/<shard>/<obj>.npz
  4. Decode SLAT → Gaussian, render at that single camera
  5. Save PNG to <in-dir>/_3d_renders/{edit_id}_after.png
     (and _before.png from before.npz)

Usage:
    python scripts/standalone/render_phase1v2_3d_results.py \
        --in-dir outputs/_debug/phase1_v2_mirror5 \
        --shard 01 --tag mirror5
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "third_party"))
sys.path.insert(0, str(_ROOT / "scripts" / "tools"))

from render_part_overview import VIEW_INDICES, load_views_from_npz  # noqa: E402

PREFIX = {"modification": "mod", "scale": "scl",
          "material": "mat", "global": "glb", "deletion": "del"}


def frame_to_extrinsic_intrinsic(frame: dict, device="cuda"):
    """NeRF c2w → TRELLIS extrinsic (w2c) + fov intrinsic."""
    from trellis.utils.render_utils import (
        yaw_pitch_r_fov_to_extrinsics_intrinsics,  # noqa
    )
    import utils3d  # type: ignore
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    extr = torch.inverse(c2w).to(device)
    fov = torch.tensor(frame["camera_angle_x"], dtype=torch.float32)
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov).to(device)
    return extr, intr


def render_one_view(pipeline, slat, frame, resolution=518) -> np.ndarray:
    from trellis.utils.render_utils import render_frames
    outputs = pipeline.decode_slat(slat, ["gaussian"])
    gaussian = outputs["gaussian"][0]
    extr, intr = frame_to_extrinsic_intrinsic(frame)
    res = render_frames(
        gaussian, [extr], [intr],
        {"resolution": resolution, "bg_color": (1, 1, 1)},
        verbose=False,
    )
    img = res["color"][0]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img  # RGB


def load_slat(path: Path, device="cuda"):
    from trellis.modules import sparse as sp
    d = np.load(str(path))
    feats = torch.from_numpy(d["slat_feats"]).to(device)
    coords = torch.from_numpy(d["slat_coords"]).to(device)
    return sp.SparseTensor(feats=feats, coords=coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--tag", default="mirror5")
    ap.add_argument("--images-root", default="data/partverse/images", type=Path)
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pairs_dir = args.in_dir / f"mesh_pairs_{args.tag}"
    out_dir = args.in_dir / "_3d_renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    text_ckpt = str(Path(args.ckpt_root) / "TRELLIS-text-xlarge")
    print(f"[trellis] loading {text_ckpt}")
    from trellis.pipelines import TrellisTextTo3DPipeline
    pipeline = TrellisTextTo3DPipeline.from_pretrained(text_ckpt)
    pipeline.cuda()
    print("[trellis] ready")

    parsed_files = sorted(args.in_dir.glob("*.parsed.json"))
    n_ok = n_skip = n_fail = 0
    frame_cache: dict[str, list] = {}

    for pf in parsed_files:
        j = json.loads(pf.read_text())
        obj_id = j["obj_id"]
        edits = (j.get("parsed") or {}).get("edits") or []
        if obj_id not in frame_cache:
            img_npz = args.images_root / args.shard / f"{obj_id}.npz"
            if not img_npz.is_file():
                print(f"  [skip obj] {obj_id}: missing images npz")
                continue
            _, frames = load_views_from_npz(img_npz, VIEW_INDICES)
            frame_cache[obj_id] = frames
        frames = frame_cache[obj_id]

        seq_per_obj = 0
        for e in edits:
            et = e.get("edit_type")
            if et not in PREFIX:
                continue
            if et == "deletion":
                continue  # no SLAT npz
            edit_id = f"{PREFIX[et]}_{obj_id}_{seq_per_obj:03d}"
            seq_per_obj += 1
            vi = int(e.get("view_index", 0))
            if not (0 <= vi < len(frames)):
                continue
            frame = frames[vi]

            pair_dir = pairs_dir / edit_id
            for which in ("before", "after"):
                npz = pair_dir / f"{which}.npz"
                out_png = out_dir / f"{edit_id}_{which}.png"
                if out_png.is_file() and not args.force:
                    n_skip += 1
                    continue
                if not npz.is_file():
                    n_fail += 1
                    continue
                try:
                    slat = load_slat(npz)
                    rgb = render_one_view(pipeline, slat, frame,
                                          args.resolution)
                    Image.fromarray(rgb).save(str(out_png))
                    n_ok += 1
                    print(f"  [ok] {out_png.name}")
                except Exception as ex:
                    print(f"  [FAIL] {edit_id}/{which}: {ex}")
                    n_fail += 1

    # Re-derive flux-style numbering from parsed_to_edit_specs.py and rename
    # the files to match (shared seq across mod/scl/mat/glb).
    print(f"\n[done] ok={n_ok} skip={n_skip} fail={n_fail} → {out_dir}")


if __name__ == "__main__":
    main()
