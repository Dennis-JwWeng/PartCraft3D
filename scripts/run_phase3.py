#!/usr/bin/env python3
"""Run Phase 3: VLM-based quality filter for Phase 2.5 edit results.

Renders before/after 3D models, sends comparison images to VLM,
and scores each edit on execution, localization, preservation, and quality.

Usage:
    # Filter all edits from Phase 2.5
    ATTN_BACKEND=xformers python scripts/run_phase3.py \
        --config configs/partobjaverse.yaml --tag action_v2

    # Limit to first 10
    ATTN_BACKEND=xformers python scripts/run_phase3.py --tag action_v2 --limit 10

    # Custom results file
    ATTN_BACKEND=xformers python scripts/run_phase3.py \
        --results outputs/.../edit_results_action_v2.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.phase3_filter.vlm_filter import run_vlm_filter


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: VLM Quality Filter for Phase 2.5")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag (matches Phase 2.5 --tag)")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to edit_results.jsonl (overrides --tag)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max edits to evaluate")
    parser.add_argument("--num-views", type=int, default=4,
                        help="Views to render per model (default: 4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "phase3")
    logger.info("Starting Phase 3: VLM Quality Filter")

    p25_cfg = cfg.get("phase2_5", {})
    cache_dir = Path(p25_cfg["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    tag_suffix = f"_{args.tag}" if args.tag else ""

    # Resolve paths
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = cache_dir / f"edit_results{tag_suffix}.jsonl"

    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # Phase 3 output
    p3_output = Path(cfg.get("phase3", {}).get(
        "cache_dir", "cache/phase3"))
    if args.tag:
        p3_output = p3_output.parent / f"{p3_output.name}{tag_suffix}"

    logger.info(f"Results: {results_path}")
    logger.info(f"Mesh pairs: {mesh_pairs_dir}")
    logger.info(f"Output: {p3_output}")

    all_scored = run_vlm_filter(
        cfg=cfg,
        results_path=results_path,
        mesh_pairs_dir=mesh_pairs_dir,
        output_dir=p3_output,
        limit=args.limit,
        num_views=args.num_views,
    )
    n_positive = sum(1 for e in all_scored
                     if e.get("quality_tier") in ("high", "medium"))
    n_negative = sum(1 for e in all_scored
                     if e.get("quality_tier") == "negative")
    logger.info(f"Phase 3: {len(all_scored)} scored, "
                f"{n_positive} positive, {n_negative} negative samples")


if __name__ == "__main__":
    main()
