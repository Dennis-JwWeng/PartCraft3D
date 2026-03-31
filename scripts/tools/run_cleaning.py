#!/usr/bin/env python3
"""CLI entry point for data cleaning on object-centric training data.

Runs Layer 1 (NPZ sanity) + Layer 2 (edit-type-specific pair checks) on
repacked ``partverse_pairs/shard_XX/{obj_id}/`` directories.

Usage
-----
Clean all shards::

    python scripts/tools/run_cleaning.py --input-dir partverse_pairs

Clean specific shards with parallel workers::

    python scripts/tools/run_cleaning.py \\
        --input-dir partverse_pairs \\
        --shards 00 01 \\
        --workers 16

Clean only specific edit types::

    python scripts/tools/run_cleaning.py \\
        --input-dir partverse_pairs \\
        --edit-types deletion modification scale

Dry run (report only, no quality.json written)::

    python scripts/tools/run_cleaning.py \\
        --input-dir partverse_pairs --dry-run

With pipeline config for custom thresholds::

    python scripts/tools/run_cleaning.py \\
        --input-dir partverse_pairs \\
        --config configs/partverse_node39_shard01.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from partcraft.cleaning.cleaner import run_cleaning  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Default cleaning config ─────────────────────────────────────────────

DEFAULT_CLEANING_CFG: dict = {
    # Layer 1: NPZ sanity
    "min_voxels": 100,
    "max_voxels": 40000,
    "max_feat_abs": 50.0,
    "min_feat_std": 0.01,
    "max_ss_abs": 100.0,
    "min_ss_std": 0.001,

    # Layer 2: per-type thresholds
    "deletion": {
        "min_voxel_ratio": 0.05,
        "max_voxel_ratio": 0.95,
        "min_delete_ratio": 0.02,
        "max_delete_ratio": 0.80,
        "min_bbox_iou": 0.15,
        "min_ss_change": 0.01,
        "max_ss_change": 0.90,
        "max_components": 3,
    },
    "addition": {
        "min_voxel_ratio": 1.05,
        "max_voxel_ratio": 20.0,
        "min_add_ratio": 0.02,
        "max_add_ratio": 0.80,
        "min_bbox_iou": 0.15,
    },
    "modification": {
        "min_voxel_ratio": 0.3,
        "max_voxel_ratio": 3.0,
        "min_ss_cosine": 0.3,
        "min_edit_locality": 0.02,
        "max_edit_locality": 0.70,
        "max_components": 5,
        "max_center_drift": 0.3,
        "max_feat_kl": 5.0,
    },
    "scale": {
        "min_voxel_ratio": 0.5,
        "max_voxel_ratio": 2.0,
        "min_ss_cosine": 0.5,
        "min_edit_locality": 0.01,
        "max_edit_locality": 0.50,
        "max_center_drift": 0.2,
        "min_bbox_axis_ratio": 0.7,
        "max_bbox_axis_ratio": 1.8,
    },
    "material": {
        "require_coords_match": True,
        "require_ss_match": True,
        "ss_match_tol": 1e-4,
        "min_feat_change": 0.01,
        "max_feat_change": 2.0,
    },
    "global": {
        "require_coords_match": True,
        "require_ss_match": True,
        "ss_match_tol": 1e-4,
        "min_feat_change": 0.02,
        "max_feat_change": 2.0,
        "min_change_coverage": 0.3,
    },

    # Tier thresholds
    "tier_thresholds": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
    },
}


def _merge_cleaning_cfg(base: dict, override: dict) -> dict:
    """Deep merge override into base cleaning config."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge_cleaning_cfg(result[k], v)
        else:
            result[k] = v
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Clean object-centric training data (Layer 1 + Layer 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Root of repacked data (contains shard_XX dirs)")
    parser.add_argument("--config", default=None,
                        help="Pipeline YAML config (uses 'cleaning' section)")
    parser.add_argument("--shards", nargs="*", default=None,
                        help="Shard IDs to process (default: all)")
    parser.add_argument("--edit-types", nargs="*", default=None,
                        help="Edit types to process (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers per shard")
    parser.add_argument("--min-tier", default="medium",
                        choices=["high", "medium", "low", "negative"],
                        help="Minimum tier for manifest_clean.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, don't write quality.json")
    args = parser.parse_args()

    # Build config
    cfg: dict = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Merge cleaning config: defaults ← yaml overrides
    yaml_cleaning = cfg.get("cleaning", {})
    cfg["cleaning"] = _merge_cleaning_cfg(DEFAULT_CLEANING_CFG, yaml_cleaning)

    edit_types = set(args.edit_types) if args.edit_types else None

    summary_path = run_cleaning(
        input_dir=args.input_dir,
        cfg=cfg,
        shards=args.shards,
        edit_types=edit_types,
        workers=args.workers,
        min_tier=args.min_tier,
        dry_run=args.dry_run,
    )

    # Print summary
    with open(summary_path) as f:
        summary = json.load(f)
    total = summary.get("total_edits", 0)
    clean = summary.get("clean_edits", 0)
    pct = clean / max(total, 1) * 100
    print(f"\n{'='*60}")
    print(f"Cleaning complete: {clean}/{total} edits passed ({pct:.1f}%)")
    print(f"Summary: {summary_path}")
    print(f"Clean manifest: {Path(args.input_dir) / 'manifest_clean.jsonl'}")

    # Per-type breakdown
    g = summary.get("global", {})
    by_type_tier = g.get("by_type_tier", {})
    if by_type_tier:
        print(f"\n{'Type':<15} {'Total':>6} {'High':>6} {'Medium':>6} {'Low':>6} {'Neg':>6} {'Rej':>6}")
        print("-" * 61)
        for etype in sorted(by_type_tier.keys()):
            tiers = by_type_tier[etype]
            total_t = sum(tiers.values())
            print(f"{etype:<15} {total_t:>6} "
                  f"{tiers.get('high', 0):>6} {tiers.get('medium', 0):>6} "
                  f"{tiers.get('low', 0):>6} {tiers.get('negative', 0):>6} "
                  f"{tiers.get('rejected', 0):>6}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
