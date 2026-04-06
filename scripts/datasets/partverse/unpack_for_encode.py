#!/usr/bin/env python3
"""Restore transforms.json + PNGs from packed NPZ back to img_Enc/ for encoding.

After --render-only packs everything into images/{shard}/*.npz and deletes the
raw files, --encode-only fails because encode_into_SLAT needs the originals.
This script extracts them back from the NPZ so encode can proceed.

Usage:
    # Restore all shards
    python scripts/datasets/partverse/unpack_for_encode.py

    # Restore specific shard
    python scripts/datasets/partverse/unpack_for_encode.py --shard 00 --num-shards 10

    # Dry run (just report what would be restored)
    python scripts/datasets/partverse/unpack_for_encode.py --dry-run
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
_IMG_ENC_DIR   = _PARTVERSE_DIR / "img_Enc"
_IMAGES_DIR    = _PARTVERSE_DIR / "images"
_SLAT_DIR      = _PARTVERSE_DIR / "slat"

sys.path.insert(0, str(_PROJECT_ROOT))


def unpack_one(obj_id: str, npz_path: Path, img_enc_dir: Path,
               dry_run: bool, logger: logging.Logger) -> tuple[str, str]:
    """Extract transforms.json + PNGs from NPZ into img_Enc/{obj_id}/."""
    if (img_enc_dir / "transforms.json").exists():
        return obj_id, "skip_exists"

    # Check both flat slat/ and shard subdirs slat/*/{obj_id}_feats.pt
    if (any(_SLAT_DIR.rglob(f"{obj_id}_feats.pt")) and
            any(_SLAT_DIR.rglob(f"{obj_id}_coords.pt"))):
        return obj_id, "skip_encoded"

    if not (img_enc_dir / "voxels.ply").exists():
        return obj_id, "skip_no_voxels"

    if dry_run:
        return obj_id, "would_restore"

    try:
        npz = np.load(str(npz_path), allow_pickle=False)
    except Exception as e:
        logger.error(f"Failed to load {npz_path}: {e}")
        return obj_id, "error"

    img_enc_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    for key in npz.files:
        if key == "transforms.json" or key.endswith(".png"):
            (img_enc_dir / key).write_bytes(npz[key].tobytes())
            restored += 1

    return obj_id, f"restored_{restored}"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("unpack_for_encode")

    parser = argparse.ArgumentParser(description="Restore img_Enc files from NPZ for encoding")
    parser.add_argument("--data-root", type=str, default=None,
                        help="PartVerse data root (overrides PARTVERSE_DATA_ROOT env var)")
    parser.add_argument("--shard", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel threads for I/O (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be restored without writing files")
    args = parser.parse_args()

    global _PARTVERSE_DIR, _IMG_ENC_DIR, _IMAGES_DIR, _SLAT_DIR
    if args.data_root:
        _PARTVERSE_DIR = Path(args.data_root)
        _IMG_ENC_DIR   = _PARTVERSE_DIR / "img_Enc"
        _IMAGES_DIR    = _PARTVERSE_DIR / "images"
        _SLAT_DIR      = _PARTVERSE_DIR / "slat"

    if args.dry_run:
        logger.info("DRY RUN — no files will be written")

    # Collect NPZ files to process
    if args.shard is not None:
        shard_dirs = [_IMAGES_DIR / args.shard]
    else:
        shard_dirs = sorted(_IMAGES_DIR.iterdir()) if _IMAGES_DIR.exists() else []

    npz_files: list[tuple[str, Path]] = []
    for shard_dir in shard_dirs:
        if not shard_dir.is_dir():
            continue
        for npz_path in sorted(shard_dir.glob("*.npz")):
            npz_files.append((npz_path.stem, npz_path))

    if not npz_files:
        logger.warning(f"No NPZ files found in {_IMAGES_DIR}")
        return

    logger.info(f"Found {len(npz_files)} NPZ files to check, {args.workers} workers")

    counts: dict[str, int] = {}
    done = 0
    total = len(npz_files)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(unpack_one, obj_id, npz_path,
                        _IMG_ENC_DIR / obj_id, args.dry_run, logger): obj_id
            for obj_id, npz_path in npz_files
        }
        for fut in as_completed(futures):
            obj_id, result = fut.result()
            done += 1
            counts[result] = counts.get(result, 0) + 1
            if result.startswith("restored") or result == "would_restore":
                logger.info(f"[{done}/{total}] {obj_id}: {result}")
            elif result == "error":
                logger.error(f"[{done}/{total}] {obj_id}: {result}")

    logger.info("Summary: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
