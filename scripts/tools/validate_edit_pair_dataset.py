#!/usr/bin/env python3
"""Smoke-test EditPairDataset against the new pipeline_v2 layout.

Usage:
    python scripts/tools/validate_edit_pair_dataset.py \
        --root outputs/partverse/pipeline_v2_mirror5
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.io.edit_pair_dataset import EditPairDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--shards", nargs="*", default=None)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--collate", action="store_true",
                    help="also try collate_fn (requires trellis import)")
    args = ap.parse_args()

    ds = EditPairDataset(
        args.root,
        shards=args.shards,
        edit_types=set(args.types) if args.types else None,
    )
    print(ds)

    if len(ds) == 0:
        print("[fail] empty dataset"); sys.exit(1)

    # Single-item check
    item = ds[0]
    print("\n[item 0]")
    for k, v in item.items():
        if hasattr(v, "shape"):
            print(f"  {k:14s} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:14s} {v!r}")

    # Iterate everything (cheap, just verifies no IO errors)
    type_counts = Counter()
    n_voxels = []
    for i in range(len(ds)):
        it = ds[i]
        type_counts[it["edit_type"]] += 1
        n_voxels.append(it["before_coords"].shape[0])
    print(f"\n[scan] {len(ds)} items ok")
    print(f"  types:        {dict(type_counts)}")
    print(f"  voxels  min/median/max: "
          f"{min(n_voxels)} / {sorted(n_voxels)[len(n_voxels)//2]} / {max(n_voxels)}")

    # Optional batched collate
    if args.collate:
        try:
            batch = [ds[i] for i in range(min(args.batch_size, len(ds)))]
            pack = EditPairDataset.collate_fn(batch)
            print("\n[collate]")
            for k in ("before_slat", "after_slat", "before_ss", "after_ss"):
                v = pack[k]
                shape = (tuple(v.shape) if hasattr(v, "shape")
                         else (v.feats.shape, v.coords.shape))
                print(f"  {k:14s} {shape}")
            print(f"  prompts: {pack['prompt']}")
        except ImportError as e:
            print(f"\n[collate skipped] {e}")

    print("\n[OK]")


if __name__ == "__main__":
    main()
