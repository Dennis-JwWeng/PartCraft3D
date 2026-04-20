#!/usr/bin/env python3
"""Decode H3D_v1 SLAT NPZs back to Gaussians and render multiview grids.

Walks `<root>/edits/<NN>/<obj_id>/<edit_id>/{before,after}.npz` (smoke layout)
or `<root>/<edit_type>/<NN>/<obj_id>/<edit_id>/...` (canonical) and writes
side-by-side comparison images vs the ground-truth per-edit
`before.png` / `after.png` (flat schema-v3 layout).

Outputs go under `<edit_dir>/decoded/`:
    decoded/{before,after}_decoded_grid.png   ← decoded SLAT renderings
    decoded/{before,after}_compare.png        ← top: GT 5 views, bottom: decoded grid

Usage::

    PYTHONPATH=.:third_party \
    PARTCRAFT_CKPT_ROOT=$PWD/checkpoints \
    /mnt/zsn/miniconda3/envs/vinedresser3d/bin/python \
      -m scripts.datasets.h3d_v1.decode_inspect \
      --root data/H3D_v1_smoke5 --device cuda:0 --nviews 8

Decoder caches by inode so duplicated `before.npz` (hardlinked from
`_assets/<obj>/object.npz`) is only decoded once per run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("h3d_v1.decode_inspect")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
for p in (_PROJECT_ROOT, _THIRD_PARTY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")


def _normalize_device(device: str) -> str:
    if device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return "cuda"
    return device


def _load_decoder(ckpt_root: Path, device: str):
    import trellis.models as models
    ckpt_path = ckpt_root / "TRELLIS-image-large" / "ckpts" / "slat_dec_gs_swin8_B_64l8gs32_fp16"
    if ckpt_path.exists():
        src = str(ckpt_path)
        logger.info(f"Loading SLAT GS decoder from local: {src}")
    else:
        src = "JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16"
        logger.info(f"Loading SLAT GS decoder from HF hub: {src}")
    decoder = models.from_pretrained(src).eval()
    if device.startswith("cuda"):
        decoder = decoder.cuda()
    return decoder


def _load_slat_npz(path: Path, device: str):
    from trellis.modules import sparse as sp
    d = np.load(path)
    feats = torch.from_numpy(d["slat_feats"]).float()
    coords = torch.from_numpy(d["slat_coords"]).int()
    if device.startswith("cuda"):
        feats = feats.cuda()
        coords = coords.cuda()
    return sp.SparseTensor(feats=feats, coords=coords)


@torch.no_grad()
def _decode_render(decoder, slat, *, nviews: int, resolution: int):
    from trellis.utils.render_utils import render_multiview
    gaussians = decoder(slat)
    gaussian = gaussians[0]
    frames, _, _ = render_multiview(gaussian, resolution=resolution, nviews=nviews)
    return frames


def _grid(frames, n_cols: int = 4) -> Image.Image:
    n = len(frames)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, n_cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return Image.fromarray(grid)


def _gt_strip(gt_image: Path, *, target_h: int):
    """Load the single per-edit ground-truth png and resize to ``target_h``.

    Schema v3 ships one ``before.png`` / ``after.png`` per edit at the
    best_view_index (see ``partcraft.cleaning.h3d_v1.promoter``).  The
    "strip" is a 1-image strip now; name kept for call-site compat.
    """
    if not gt_image.is_file():
        return None
    im = Image.open(gt_image).convert("RGB")
    scale = target_h / im.height
    new_w = int(round(im.width * scale))
    return im.resize((new_w, target_h), Image.LANCZOS)


def _stack_compare(gt_strip, decoded_grid, label: str) -> Image.Image:
    pad = 8
    target_w = max(gt_strip.width if gt_strip is not None else 0, decoded_grid.width)

    def _pad_to(im):
        if im.width == target_w:
            return im
        out = Image.new("RGB", (target_w, im.height), (255, 255, 255))
        out.paste(im, ((target_w - im.width) // 2, 0))
        return out

    parts = []
    if gt_strip is not None:
        parts.append(_pad_to(gt_strip))
    parts.append(_pad_to(decoded_grid))
    total_h = sum(p.height for p in parts) + pad * (len(parts) - 1)
    out = Image.new("RGB", (target_w, total_h), (255, 255, 255))
    y = 0
    for i, p in enumerate(parts):
        out.paste(p, (0, y))
        y += p.height + (pad if i < len(parts) - 1 else 0)
    return out


def _iter_edit_dirs(root: Path):
    bases = []
    if (root / "edits").is_dir():
        bases.append(root / "edits")
    for sub in ("deletion", "addition", "modification", "scale", "material", "color", "global"):
        if (root / sub).is_dir():
            bases.append(root / sub)
    if not bases:
        bases.append(root)
    for base in bases:
        for edit_dir in sorted(base.glob("*/*/*")):
            if not edit_dir.is_dir():
                continue
            if not (edit_dir / "before.npz").exists() and not (edit_dir / "after.npz").exists():
                continue
            yield edit_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode H3D_v1 SLAT NPZs and render multiview grids")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ckpt-root", type=Path,
                        default=Path(os.environ.get("PARTCRAFT_CKPT_ROOT", _PROJECT_ROOT / "checkpoints")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nviews", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    edits = list(_iter_edit_dirs(args.root))
    if args.limit > 0:
        edits = edits[:args.limit]
    logger.info(f"Found {len(edits)} edit dirs under {args.root}")
    if not edits:
        logger.error("No edits with .npz found")
        return 1

    device = _normalize_device(args.device)
    decoder = _load_decoder(args.ckpt_root, device)

    cache_grid = {}
    ok = 0
    fail = 0

    for i, ed in enumerate(edits, 1):
        logger.info(f"[{i}/{len(edits)}] {ed.relative_to(args.root)}")
        out = ed / "decoded"
        if args.skip_existing and (out / "before_compare.png").exists() and (out / "after_compare.png").exists():
            logger.info("  skip (existing)")
            ok += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        try:
            for tag in ("before", "after"):
                npz = ed / f"{tag}.npz"
                if not npz.exists():
                    logger.warning(f"  missing {tag}.npz")
                    continue
                ino = npz.stat().st_ino
                if ino in cache_grid:
                    grid = cache_grid[ino]
                    logger.info(f"  {tag}: cached (inode={ino})")
                else:
                    slat = _load_slat_npz(npz, device)
                    frames = _decode_render(decoder, slat, nviews=args.nviews, resolution=args.resolution)
                    grid = _grid(frames, n_cols=4)
                    cache_grid[ino] = grid
                    logger.info(f"  {tag}: decoded ({len(frames)} views, slat N={slat.feats.shape[0]})")
                grid.save(out / f"{tag}_decoded_grid.png")

                target_h = grid.height // 2 if args.nviews >= 8 else grid.height
                gt_strip = _gt_strip(ed / f"{tag}.png", target_h=target_h)
                compare = _stack_compare(gt_strip, grid, label=tag)
                compare.save(out / f"{tag}_compare.png")
            ok += 1
        except Exception as e:
            fail += 1
            logger.exception(f"  FAILED: {e}")

    logger.info(f"Done: ok={ok} fail={fail} -> results under each edit's decoded/ subdir")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
