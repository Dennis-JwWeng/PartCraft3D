#!/usr/bin/env python3
"""Promote addition edits into ``H3D_v1``.

Usage::

    python -m scripts.cleaning.h3d_v1.pull_addition \
        --pipeline-cfg configs/pipeline_v3_shard08.yaml \
        --shard 08 \
        --dataset-root data/H3D_v1 \
        --workers 8

Pure IO. Each ``add_<obj>_NNN`` requires its paired ``del_<obj>_NNN``
to already be in the dataset (from ``pull_deletion``). Run **after**
``pull_deletion`` for the same shard. Edits whose paired deletion is
missing are skipped silently with a counted reason; rerun once more
deletions land if needed.

Skipped: there is no per-edit gate filter for addition — eligibility
is purely a function of dataset state. The paired-deletion's gates
were already validated when it was promoted.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from partcraft.cleaning.h3d_v1.layout import H3DLayout
from partcraft.cleaning.h3d_v1.pipeline_io import iter_edits, resolve_paths
from partcraft.cleaning.h3d_v1.promoter import promote_addition
from scripts.cleaning.h3d_v1._common import (
    PullStats,
    add_common_args,
    build_promote_context,
    print_summary,
    resolve_obj_filter,
    run_promote_pool,
    setup_logging,
)

LOG = logging.getLogger("h3d_v1.pull_addition")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    layout = H3DLayout(root=args.dataset_root)
    obj_filter = resolve_obj_filter(args)

    LOG.info("scanning shard=%s for addition edits", args.shard)
    paths = resolve_paths(args.pipeline_cfg)
    edits = list(iter_edits(args.pipeline_cfg, args.shard,
                             types=("addition",),
                             obj_id_allowlist=obj_filter))
    stats = PullStats(accepted=len(edits))
    if args.limit:
        edits = edits[: args.limit]
    LOG.info("found %d addition edits", len(edits))

    if args.dry_run:
        n_paired = sum(
            1 for e in edits
            if (layout.after_npz("deletion", e.shard, e.obj_id, "del_" + e.edit_id[4:])).is_file()
        )
        LOG.info("dry-run: %d/%d additions have paired deletion in dataset", n_paired, len(edits))
        print_summary(stats, "pull_addition (dry-run)")
        return 0

    ctx = build_promote_context(paths)
    run_promote_pool(
        edits, layout, ctx,
        promote_fn=lambda e, l: promote_addition(e, l, ctx=ctx),
        workers=args.workers, stats=stats,
    )
    print_summary(stats, "pull_addition")
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
