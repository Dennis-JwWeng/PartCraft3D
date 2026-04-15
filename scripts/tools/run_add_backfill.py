#!/usr/bin/env python3
"""One-shot backfill: create add_*/meta.json for all existing del_* PLY pairs.

Usage:
    python scripts/tools/run_add_backfill.py --config configs/pipeline_v2_shard02.yaml
    python scripts/tools/run_add_backfill.py --config configs/pipeline_v2_shard02.yaml --dry-run

The script is CPU-only and idempotent: already-existing meta.json files are
skipped. Re-running is safe.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.pipeline_v2.paths import PipelineRoot, ObjectContext
from partcraft.pipeline_v2.specs import iter_deletion_specs
from partcraft.pipeline_v2.qc_io import is_gate_a_failed
from partcraft.pipeline_v2.s5b_deletion import _backfill_add


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", default="02")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be created without writing")
    ap.add_argument("--force-update", action="store_true",
                    help="Re-write existing meta.json (e.g. to add new fields)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("add_backfill")

    cfg = json.loads(Path(args.config).read_text()) if args.config.endswith(".json") \
        else __import__("yaml").safe_load(Path(args.config).read_text())
    output_dir = Path(cfg["data"]["output_dir"])
    root = PipelineRoot(output_dir)

    shard = args.shard
    objects_root = root.objects_root / shard
    if not objects_root.is_dir():
        log.error("No objects dir: %s", objects_root)
        sys.exit(1)

    obj_dirs = sorted(objects_root.iterdir())
    log.info("Scanning %d object dirs for shard %s", len(obj_dirs), shard)

    n_obj = n_created = n_skipped = n_no_ply = n_gate_fail = 0
    for obj_dir in obj_dirs:
        if not obj_dir.is_dir():
            continue
        ctx = root.context(shard, obj_dir.name)
        specs = list(iter_deletion_specs(ctx))
        if not specs:
            continue
        n_obj += 1

        add_seq = 0
        for spec in specs:
            if is_gate_a_failed(ctx, spec.edit_id):
                n_gate_fail += 1
                add_seq += 1
                continue
            pair_dir = ctx.edit_3d_dir(spec.edit_id)
            a_ply = pair_dir / "after.ply"
            if not a_ply.is_file():
                n_no_ply += 1
                add_seq += 1
                continue
            # Check if meta.json already exists
            add_id = ctx.edit_id("addition", add_seq)
            add_dir = ctx.edit_3d_dir(add_id)
            meta_path = add_dir / "meta.json"
            if meta_path.is_file() and not args.force_update:
                n_skipped += 1
            else:
                if args.dry_run:
                    log.debug("Would create/update: %s", meta_path)
                    n_created += 1
                else:
                    created = _backfill_add(ctx, spec, add_seq,
                                            force=args.force_update, logger=log)
                    if created:
                        n_created += 1
                    else:
                        n_skipped += 1
            add_seq += 1

    log.info(
        "Done. objects=%d  created=%d  skipped(exists)=%d  "
        "no_ply=%d  gate_a_fail=%d",
        n_obj, n_created, n_skipped, n_no_ply, n_gate_fail,
    )


if __name__ == "__main__":
    main()
