#!/usr/bin/env python3
"""Enrich PartObjaverse-Tiny semantic labels via VLM (1 image + text).

Uses pre-rendered views from HY3D-Part NPZ as visual grounding,
combined with category + part labels, to generate rich edit prompts.

Usage:
    python scripts/run_enrich.py --limit 5                    # test 5 objects
    python scripts/run_enrich.py                               # all 200 objects
    python scripts/run_enrich.py --prompt-style action         # 3DEditVerse-style
    python scripts/run_enrich.py --prompt-style action --limit 5  # test action style
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.phase1_planning.enricher import enrich_semantic_labels


def main():
    parser = argparse.ArgumentParser(
        description="Enrich semantic labels via VLM (1 image + text)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--semantic-json", type=str, default=None,
                        help="Override path to semantic.json")
    parser.add_argument("--prompt-style", type=str, default="default",
                        choices=["default", "action"],
                        help="Prompt style: 'default' (detailed) or "
                             "'action' (3DEditVerse-style short prompts)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "enrich")

    # Find semantic.json
    semantic_json = args.semantic_json
    if not semantic_json:
        candidates = [
            Path(cfg["data"]["image_npz_dir"]).parent / "source" / "semantic.json",
            Path("data/partobjaverse_tiny/source/semantic.json"),
        ]
        for c in candidates:
            if c.exists():
                semantic_json = str(c)
                break

    if not semantic_json or not Path(semantic_json).exists():
        print("ERROR: semantic.json not found. Specify with --semantic-json")
        sys.exit(1)

    # Image NPZ dir for pre-rendered thumbnails
    image_npz_dir = cfg["data"]["image_npz_dir"]

    # Output to phase0 cache dir
    cache_dir = Path(cfg["phase0"]["cache_dir"])
    output_path = cache_dir / "semantic_labels.jsonl"

    print(f"Semantic JSON: {semantic_json}")
    print(f"Image NPZ dir: {image_npz_dir}")
    print(f"Output: {output_path}")
    print(f"VLM model: {cfg['phase0'].get('vlm_model', '?')}")
    print(f"Prompt style: {args.prompt_style}")

    # Use separate output for action-style prompts
    if args.prompt_style != "default":
        output_path = cache_dir / f"semantic_labels_{args.prompt_style}.jsonl"
        print(f"Output (style={args.prompt_style}): {output_path}")

    enrich_semantic_labels(
        cfg=cfg,
        semantic_json_path=semantic_json,
        output_path=output_path,
        image_npz_dir=image_npz_dir,
        shard=cfg["data"]["shards"][0],
        limit=args.limit,
        max_workers=args.workers,
        prompt_style=args.prompt_style,
    )


if __name__ == "__main__":
    main()
