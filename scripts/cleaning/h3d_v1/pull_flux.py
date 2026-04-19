#!/usr/bin/env python3
"""Promote flux edits (modification | scale | material | color | global) into ``H3D_v1``.

Usage::

    python -m scripts.cleaning.h3d_v1.pull_flux \
        --pipeline-cfg configs/pipeline_v3_shard08.yaml \
        --shard 08 \
        --dataset-root data/H3D_v1 \
        --workers 8 \
        [--types modification scale material color global]

Pure IO. ``before.npz`` and ``after.npz`` are produced by pipeline_v3
(s5 + s6) and live under the per-edit ``edits_3d/<edit_id>/`` dir;
``promote_flux`` hardlinks them into the dataset.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from partcraft.cleaning.h3d_v1.filter import accept_flux
from partcraft.cleaning.h3d_v1.layout import EDIT_TYPES_FLUX, H3DLayout
from partcraft.cleaning.h3d_v1.pipeline_io import load_edit_status
from partcraft.cleaning.h3d_v1.promoter import promote_flux
from scripts.cleaning.h3d_v1._common import (
    add_common_args,
    build_promote_context,
    collect_edits,
    print_summary,
    resolve_obj_filter,
    run_promote_pool,
    setup_logging,
)

LOG = logging.getLogger("h3d_v1.pull_flux")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--types", nargs="+", default=list(EDIT_TYPES_FLUX),
                    choices=list(EDIT_TYPES_FLUX),
                    help="Restrict to these flux edit_types (default: all 5).")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    layout = H3DLayout(root=args.dataset_root)
    obj_filter = resolve_obj_filter(args)

    LOG.info("scanning shard=%s for flux types=%s", args.shard, args.types)
    edits, stats, paths = collect_edits(
        args.pipeline_cfg, args.shard, tuple(args.types), obj_filter,
        accept_fn=accept_flux, load_status=load_edit_status,
    )
    if args.limit:
        edits = edits[: args.limit]
    LOG.info("after filter+limit: %d flux edits to promote", len(edits))

    if args.dry_run:
        from collections import Counter
        by_type = Counter(e.edit_type for e in edits)
        LOG.info("dry-run breakdown: %s", dict(by_type))
        print_summary(stats, "pull_flux (dry-run)")
        return 0

    ctx = build_promote_context(paths, pipeline_cfg_path=args.pipeline_cfg)
    run_promote_pool(
        edits, layout, ctx,
        promote_fn=lambda e, l: promote_flux(e, l, ctx=ctx),
        workers=args.workers, stats=stats,
    )
    print_summary(stats, "pull_flux")
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
