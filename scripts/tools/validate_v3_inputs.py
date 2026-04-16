#!/usr/bin/env python3
"""Validate pipeline_v3 input data for a set of objects.

Delegates all checking logic to :class:`partcraft.io.partverse_dataset.PartVerseDataset`.

Checks (per object)
-------------------
images NPZ  — split_mesh.json, transforms.json, view images (png/webp)
mesh NPZ    — full.glb, part_{i}.glb, anno_info.json, part_captions.json,
              vd_scale, vd_offset
SLAT        — <slat_dir>/<shard>/<obj_id>_{coords,feats}.pt
              (skipped if slat_dir is absent in config or torch unavailable)

Exit codes
----------
0  all objects pass
1  one or more objects have blocking errors

Usage
-----
    # validate objects listed in a txt file
    python scripts/tools/validate_v3_inputs.py \\
        --config configs/pipeline_v3_shard08_test20.yaml \\
        --shard 08 \\
        --obj-ids-file configs/shard08_test20_obj_ids.txt

    # validate all objects in mesh_root/<shard>/
    python scripts/tools/validate_v3_inputs.py \\
        --config configs/pipeline_v3_shard08_test20.yaml \\
        --shard 08 --all

    # quick check for a single object
    python scripts/tools/validate_v3_inputs.py \\
        --config configs/pipeline_v3_shard08_test20.yaml \\
        --shard 08 \\
        --obj-ids c3d88711e2f34164b1eb8803a3e2448a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.io.partverse_dataset import PartVerseDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="pipeline_v3 YAML config")
    ap.add_argument("--shard",  default="08")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",          action="store_true",
                       help="validate all objects in mesh_root/<shard>/")
    group.add_argument("--obj-ids-file", metavar="TXT",
                       help="newline-separated obj_id list")
    group.add_argument("--obj-ids",      nargs="+", metavar="OBJ_ID",
                       help="one or more obj_ids on the command line")
    ap.add_argument("--verbose", "-v",   action="store_true",
                    help="show warnings even for passing objects")
    args = ap.parse_args()

    cfg  = yaml.safe_load(Path(args.config).read_text())
    data = cfg.get("data", {})

    images_root = Path(data.get("images_root", ""))
    mesh_root   = Path(data.get("mesh_root",   ""))
    slat_dir    = data.get("slat_dir")          # may be absent — OK
    shard       = args.shard.zfill(2)

    dataset = PartVerseDataset(images_root, mesh_root, shards=[shard], slat_dir=slat_dir)

    if args.all:
        shard_dir = mesh_root / shard
        if not shard_dir.is_dir():
            sys.exit(f"mesh_root/{shard} not found: {shard_dir}")
        obj_ids = sorted(p.stem for p in shard_dir.glob("*.npz"))
    elif args.obj_ids_file:
        obj_ids = [
            l.strip()
            for l in Path(args.obj_ids_file).read_text().splitlines()
            if l.strip()
        ]
    else:
        obj_ids = args.obj_ids

    if not obj_ids:
        sys.exit("No object IDs resolved — check --all / --obj-ids-file / --obj-ids")

    print(f"Validating {len(obj_ids)} objects (shard={shard})")
    print(f"  images_root: {images_root}")
    print(f"  mesh_root:   {mesh_root}")
    print(f"  slat_dir:    {slat_dir or '(skipped)'}")
    print()

    reports = dataset.validate_batch(
        obj_ids, shard,
        verbose=args.verbose,
    )

    n_ok  = sum(1 for r in reports if r.ok)
    n_err = len(reports) - n_ok
    n_warn = sum(1 for r in reports if r.warnings)

    print(f"\n{'=' * 50}")
    print(f"  PASS {n_ok:3d}  FAIL {n_err:3d}  WARN {n_warn:3d}  (of {len(reports)} objects)")

    if n_err:
        print("\nFailed objects:")
        for r in reports:
            if not r.ok:
                print(f"  {r.obj_id}")
                for e in r.errors:
                    print(f"    {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
