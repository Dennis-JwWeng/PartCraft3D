#!/usr/bin/env python3
"""Run Phase 2: Mesh assembly — execute edit specs to produce mesh pairs."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.phase2_assembly.assembler import run_phase2


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Mesh Assembly")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max edit specs to process")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "phase2")
    logger.info("Starting Phase 2: Mesh Assembly")

    results = run_phase2(cfg, limit=args.limit, max_workers=args.workers)
    logger.info(f"Phase 2 produced {len(results)} assembled pairs")


if __name__ == "__main__":
    main()
