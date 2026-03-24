#!/usr/bin/env python3
"""Verify PartVerse SLAT encoding by decoding back to 3D and rendering images.

Loads encoded SLAT (feats.pt + coords.pt) for a few objects, decodes them to
Gaussian Splatting via TRELLIS, renders multiview images, and saves them to an
output directory for visual inspection.

Usage:
    cd /DATA_EDS2/shenlc2403/zsn/3dedit/PartCraft3D
    CUDA_VISIBLE_DEVICES=7 python scripts/datasets/partverse/verify_decode.py
    CUDA_VISIBLE_DEVICES=7 python scripts/datasets/partverse/verify_decode.py --n 5 --out outputs/verify_decode
    CUDA_VISIBLE_DEVICES=7 python scripts/datasets/partverse/verify_decode.py --obj-ids <id1> <id2>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT  = Path(__file__).resolve().parents[3]
_SLAT_DIR      = _PROJECT_ROOT / "data" / "partverse" / "slat"
_THIRD_PARTY   = _PROJECT_ROOT / "third_party"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIRD_PARTY))

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_decode")


def load_decoder():
    import trellis.models as models
    logger.info("Loading SLAT Gaussian decoder (JeffreyXiang/TRELLIS-image-large)...")
    decoder = models.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16"
    ).eval().cuda()
    return decoder


def load_slat(obj_id: str, slat_dir: Path):
    from trellis.modules import sparse as sp
    # Search recursively (supports both flat slat/ and shard-based slat/{shard}/)
    candidates = list(slat_dir.rglob(f"{obj_id}_feats.pt"))
    if not candidates:
        raise FileNotFoundError(f"SLAT not found for {obj_id} under {slat_dir}")
    feats_path  = candidates[0]
    coords_path = feats_path.parent / f"{obj_id}_coords.pt"
    if not coords_path.exists():
        raise FileNotFoundError(f"coords.pt not found for {obj_id}")
    feats  = torch.load(feats_path,  weights_only=True).float().cuda()
    coords = torch.load(coords_path, weights_only=True).cuda()
    return sp.SparseTensor(feats=feats, coords=coords)


def decode_and_render(decoder, slat, obj_id: str, out_dir: Path, nviews: int = 8):
    from trellis.utils.render_utils import render_multiview
    import numpy as np
    from PIL import Image

    logger.info(f"  Decoding {obj_id} (feats {tuple(slat.feats.shape)})...")
    with torch.no_grad():
        gaussians = decoder(slat)
    gaussian = gaussians[0]

    logger.info(f"  Rendering {nviews} views...")
    frames, _, _ = render_multiview(gaussian, resolution=512, nviews=nviews)

    obj_out = out_dir / obj_id
    obj_out.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(obj_out / f"view_{i:03d}.png")

    # Save a 2xN grid montage for quick inspection
    n_cols = min(nviews, 4)
    n_rows = (nviews + n_cols - 1) // n_cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, n_cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
    Image.fromarray(grid).save(obj_out / "grid.png")
    logger.info(f"  Saved {nviews} views + grid → {obj_out}")


def main():
    parser = argparse.ArgumentParser(description="Verify SLAT decode for PartVerse")
    parser.add_argument("--obj-ids", nargs="+", default=None,
                        help="Specific object IDs to decode (default: pick --n at random)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of random objects to verify (default: 3)")
    parser.add_argument("--nviews", type=int, default=8,
                        help="Number of render views per object (default: 8)")
    parser.add_argument("--out", type=str, default="outputs/verify_decode_partverse",
                        help="Output directory for rendered images")
    parser.add_argument("--slat-dir", type=str, default=None,
                        help="Override SLAT directory (default: data/partverse/slat)")
    args = parser.parse_args()

    slat_dir = Path(args.slat_dir) if args.slat_dir else _SLAT_DIR
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect available encoded objects (search recursively for shard subdirs)
    feats_files = sorted(slat_dir.rglob("*_feats.pt"))
    available   = [f.stem.removesuffix("_feats") for f in feats_files
                   if (f.parent / f"{f.stem.removesuffix('_feats')}_coords.pt").exists()]

    if not available:
        logger.error(f"No encoded objects found in {slat_dir}")
        sys.exit(1)

    logger.info(f"Found {len(available)} encoded objects in {slat_dir}")

    if args.obj_ids:
        obj_ids = args.obj_ids
    else:
        import random
        random.seed(42)
        obj_ids = random.sample(available, min(args.n, len(available)))

    logger.info(f"Verifying {len(obj_ids)} objects: {obj_ids}")

    decoder = load_decoder()

    ok, failed = 0, []
    for obj_id in obj_ids:
        logger.info(f"[{ok+len(failed)+1}/{len(obj_ids)}] {obj_id}")
        try:
            slat = load_slat(obj_id, slat_dir)
            decode_and_render(decoder, slat, obj_id, out_dir, nviews=args.nviews)
            ok += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            failed.append(obj_id)

    logger.info(f"Done: {ok}/{len(obj_ids)} succeeded, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")
    logger.info(f"Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
