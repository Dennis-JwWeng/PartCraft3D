#!/usr/bin/env python3
"""Slim down images/ NPZ files by keeping only a subset of views.

The full 150-view NPZ is ~26MB per object. The pipeline only reads 4 views
([0,10,20,30]) for VLM labeling. This script repacks each NPZ keeping:
  - A configurable set of PNG views (default: 16 evenly spaced)
  - transforms.json (all 150 camera params — needed for mask rendering)
  - split_mesh.json (part names)

Savings: ~90% reduction (31G → ~3G per shard for 16 views).

Usage:
    # Slim shard 00, keep 16 views (default)
    python scripts/datasets/partverse/repack_images_slim.py --shard 00

    # Keep only the 4 VLM views
    python scripts/datasets/partverse/repack_images_slim.py --shard 00 --views 0 10 20 30

    # Dry run — report sizes without writing
    python scripts/datasets/partverse/repack_images_slim.py --shard 00 --dry-run
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import os

_PROJECT_ROOT  = Path(__file__).resolve().parents[3]
_PARTVERSE_DIR = Path(os.environ.get(
    "PARTVERSE_DATA_ROOT", str(_PROJECT_ROOT / "data" / "partverse")))
_IMAGES_DIR    = _PARTVERSE_DIR / "images"

sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.datasets.partverse.pack_npz import PACK_VIEWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("repack_slim")


def _default_views(total: int = 150, n: int = 16) -> list[int]:
    """Return n evenly spaced view indices from [0, total)."""
    step = total / n
    return sorted(set(int(i * step) for i in range(n)))


# Imported from pack_npz.py — single source of truth
SEMANTIC_VIEWS = PACK_VIEWS


def repack_one(npz_path: Path, keep_views: list[int],
               dry_run: bool) -> tuple[str, int, int]:
    """Repack a single NPZ in-place. Returns (obj_id, old_bytes, new_bytes)."""
    obj_id = npz_path.stem
    old_bytes = npz_path.stat().st_size

    npz = np.load(str(npz_path), allow_pickle=False)
    all_keys = set(npz.files)

    # Determine which PNG keys to keep
    keep_png = set()
    for vid in keep_views:
        for ext in (".png", ".webp"):
            k = f"{vid:03d}{ext}"
            if k in all_keys:
                keep_png.add(k)

    # Non-image keys (transforms.json, split_mesh.json, etc.)
    non_image = {k for k in all_keys if not (k.endswith(".png") or k.endswith(".webp"))}
    keep_keys = keep_png | non_image

    dropped_keys = all_keys - keep_keys
    if not dropped_keys:
        npz.close()
        return obj_id, old_bytes, old_bytes  # nothing to do

    if dry_run:
        # Estimate new size proportionally
        est_new = int(old_bytes * len(keep_keys) / len(all_keys))
        npz.close()
        return obj_id, old_bytes, est_new

    data = {k: npz[k] for k in keep_keys}
    npz.close()

    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_path), **data)
    tmp_path.rename(npz_path)

    new_bytes = npz_path.stat().st_size
    return obj_id, old_bytes, new_bytes


def main():
    parser = argparse.ArgumentParser(description="Slim down images/ NPZ files")
    parser.add_argument("--data-root", type=str, default=None,
                        help="PartVerse data root (overrides PARTVERSE_DATA_ROOT env var)")
    parser.add_argument("--shard", type=str, required=True,
                        help="Shard to process, e.g. '00'")
    parser.add_argument("--views", type=int, nargs="+", default=None,
                        help="View indices to keep (default: 16 evenly spaced)")
    parser.add_argument("--n-views", type=int, default=0,
                        help="Keep N evenly spaced views (ignored if --views is set; "
                             "default: use semantic preset of 13 views)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate savings without writing anything")
    args = parser.parse_args()

    global _IMAGES_DIR
    if args.data_root:
        _IMAGES_DIR = Path(args.data_root) / "images"

    if args.views:
        keep_views = sorted(set(args.views))
    elif args.n_views:
        keep_views = _default_views(150, args.n_views)
    else:
        keep_views = SEMANTIC_VIEWS
    logger.info(f"Keeping {len(keep_views)} views: {keep_views}")

    shard_dir = _IMAGES_DIR / args.shard
    if not shard_dir.exists():
        logger.error(f"Shard dir not found: {shard_dir}")
        sys.exit(1)

    npz_files = sorted(shard_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"No NPZ files in {shard_dir}")
        sys.exit(1)

    logger.info(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(npz_files)} NPZ files...")

    total_old = total_new = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(repack_one, f, keep_views, args.dry_run): f
            for f in npz_files
        }
        for fut in as_completed(futures):
            obj_id, old, new = fut.result()
            total_old += old
            total_new += new
            done += 1
            if done % 100 == 0 or done == len(npz_files):
                logger.info(f"  {done}/{len(npz_files)} done")

    saved = total_old - total_new
    pct = 100 * saved / total_old if total_old else 0
    logger.info(
        f"{'[DRY RUN] ' if args.dry_run else ''}Done: "
        f"{total_old/1e9:.1f}G → {total_new/1e9:.1f}G "
        f"(saved {saved/1e9:.1f}G, {pct:.0f}%)"
    )


if __name__ == "__main__":
    main()
