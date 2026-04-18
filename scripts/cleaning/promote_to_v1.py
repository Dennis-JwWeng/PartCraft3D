#!/usr/bin/env python3
"""Promote cleaned edits from one or more pipeline runs into ``data/partverse_edit_v1/``.

Usage::

    python -m scripts.cleaning.promote_to_v1 \
        --source-runs outputs/partverse/pipeline_v2_shard05 \
                       outputs/partverse/shard08/mode_e_text_align \
        --rules configs/cleaning/promote_v1.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from partcraft.cleaning.v1.layout import V1Layout
from partcraft.cleaning.v1.linker import LinkMode
from partcraft.cleaning.v1.promoter import (
    PromoterConfig, promote_records, PromotionSummary,
)
from partcraft.cleaning.v1.pending import DelLatentPending
from partcraft.cleaning.v1.source_v2 import iter_records_from_v2_run
from partcraft.cleaning.v1.source_v3 import iter_records_from_v3_run

LOG = logging.getLogger("promote_to_v1")


def _detect_pipeline_version(run_root: Path) -> str:
    if (run_root / "objects").is_dir():
        return "v2"
    for child in run_root.iterdir():
        if child.is_dir() and child.name.startswith("mode_") and (child / "objects").is_dir():
            return "v3"
    return "v2"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-runs", nargs="+", type=Path, required=True)
    ap.add_argument("--rules", type=Path,
                    default=Path("configs/cleaning/promote_v1.yaml"))
    ap.add_argument("--v1-root", type=Path, default=None)
    ap.add_argument("--link-mode", choices=[m.value for m in LinkMode], default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-pipeline", choices=["v2", "v3"], default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    rules = yaml.safe_load(args.rules.read_text())
    v1_root = args.v1_root or Path(rules["v1_root"])
    link_mode = LinkMode(args.link_mode or rules.get("link_mode", "hardlink"))
    layout = V1Layout(root=v1_root)
    cfg = PromoterConfig(
        rule=rules["promote_rules"],
        link_mode=link_mode,
        img_enc_root=Path(rules["before_assets"]["img_enc_root"]),
        slat_root=Path(rules["before_assets"]["slat_root"]),
        view_indices=list(rules["before_assets"]["view_indices"]),
        force=args.force,
    )
    pending = DelLatentPending(layout.pending_del_latent_file())

    overall = PromotionSummary()
    for run_root in args.source_runs:
        run_root = run_root.resolve()
        version = args.force_pipeline or _detect_pipeline_version(run_root)
        LOG.info("processing %s as pipeline %s", run_root, version)
        if version == "v2":
            recs = iter_records_from_v2_run(run_root, run_tag=run_root.name)
        else:
            recs = iter_records_from_v3_run(run_root, run_tag=run_root.name)
        s = promote_records(recs, layout=layout, cfg=cfg, pending=pending)
        LOG.info("  promoted=%d skipped=%d deferred=%d failed=%d fallback=%d",
                 s.promoted, s.skipped_existing, s.deferred, s.failed, s.fallback_count)
        for n in s.notes[:20]:
            LOG.info("  note: %s", n)
        overall.promoted += s.promoted
        overall.skipped_existing += s.skipped_existing
        overall.deferred += s.deferred
        overall.failed += s.failed
        overall.fallback_count += s.fallback_count

    LOG.info("TOTAL promoted=%d skipped=%d deferred=%d failed=%d fallback=%d",
             overall.promoted, overall.skipped_existing,
             overall.deferred, overall.failed, overall.fallback_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
