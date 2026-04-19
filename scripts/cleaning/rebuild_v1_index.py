#!/usr/bin/env python3
"""Rebuild ``data/partverse_edit_v1/index/{objects,edits}.jsonl``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from partcraft.cleaning.v1.indexer import rebuild_index
from partcraft.cleaning.v1.layout import V1Layout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", type=Path,
                    default=Path("configs/cleaning/promote_v1.yaml"))
    ap.add_argument("--v1-root", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    if args.v1_root:
        v1_root = args.v1_root
    else:
        v1_root = Path(yaml.safe_load(args.rules.read_text())["v1_root"])
    summary = rebuild_index(V1Layout(root=v1_root))
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
