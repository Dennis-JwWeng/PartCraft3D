#!/usr/bin/env python3
"""Run Phase 0: VLM semantic labeling for all objects in the dataset."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.hy3d_loader import HY3DPartDataset
from partcraft.phase0_semantic.labeler import run_phase0


def main():
    parser = argparse.ArgumentParser(description="Phase 0: VLM Semantic Labeling")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Max objects to process")
    parser.add_argument("--shards", nargs="+", default=None, help="Override shards to process")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run: delete existing output and re-label all objects")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.shards:
        cfg["data"]["shards"] = args.shards

    logger = setup_logging(cfg, "phase0")
    logger.info("Starting Phase 0: Semantic Labeling")

    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )
    logger.info(f"Dataset: {len(dataset)} objects across shards {cfg['data']['shards']}")

    output_path = run_phase0(cfg, dataset, limit=args.limit, force=args.force)
    logger.info(f"Phase 0 output: {output_path}")


if __name__ == "__main__":
    main()
