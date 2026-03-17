#!/usr/bin/env python3
"""Run Phase 1: Edit planning from Part Catalog.

Usage:
    # Default (uses semantic_labels.jsonl)
    python scripts/run_phase1.py --config configs/partobjaverse.yaml

    # Action-style prompts (uses semantic_labels_action.jsonl)
    python scripts/run_phase1.py --config configs/partobjaverse.yaml \
        --labels data/partobjaverse_tiny/cache/phase0/semantic_labels_action.jsonl \
        --suffix _action
"""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.phase0_semantic.catalog import PartCatalog
from partcraft.phase1_planning.planner import run_phase1


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Edit Planning")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--catalog", type=str, default=None,
                        help="Path to pre-built catalog JSON")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to semantic_labels JSONL "
                             "(default: {phase0.cache_dir}/semantic_labels.jsonl)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Output suffix, e.g. '_action'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "phase1")
    logger.info("Starting Phase 1: Edit Planning")

    if args.catalog:
        catalog = PartCatalog.load(args.catalog)
    else:
        labels_path = args.labels or (
            cfg["phase0"]["cache_dir"] + "/semantic_labels.jsonl")
        logger.info(f"Building catalog from {labels_path}")
        from partcraft.io.hy3d_loader import HY3DPartDataset
        dataset = HY3DPartDataset(
            cfg["data"]["image_npz_dir"],
            cfg["data"]["mesh_npz_dir"],
            cfg["data"]["shards"],
        )
        catalog = PartCatalog.from_phase0_output(labels_path, dataset=dataset)

    logger.info(catalog.summary())

    specs = run_phase1(cfg, catalog, output_suffix=args.suffix)
    logger.info(f"Phase 1 generated {len(specs)} edit specifications")

    catalog_path = cfg["phase1"]["cache_dir"] + f"/part_catalog{args.suffix}.json"
    catalog.save(catalog_path)
    logger.info(f"Catalog saved to {catalog_path}")


if __name__ == "__main__":
    main()
