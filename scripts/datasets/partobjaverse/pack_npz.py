#!/usr/bin/env python3
"""Pack Vinedresser3D prerender outputs into PartCraft NPZ format.

Reads from data/partobjaverse_tiny/img_Enc/ + source/ and writes
data/partobjaverse_tiny/images/ + mesh/ NPZ shards consumed by the pipeline.

Usage:
    python scripts/datasets/partobjaverse/pack_npz.py --config configs/local_sglang.yaml
    python scripts/datasets/partobjaverse/pack_npz.py --config configs/local_sglang.yaml --limit 3 --force
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from partcraft.io.partcraft_loader import PartCraftDataset
from partcraft.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Pack prerender into PartCraft NPZ format")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_dir = Path(cfg["data"].get("data_dir", "data/partobjaverse_tiny"))
    if not data_dir.is_absolute():
        data_dir = _PROJECT_ROOT / data_dir
    if not data_dir.exists():
        img_dir = Path(cfg["data"]["image_npz_dir"])
        if not img_dir.is_absolute():
            img_dir = _PROJECT_ROOT / img_dir
        data_dir = img_dir.parent

    img_enc_base = data_dir / "img_Enc"
    source_dir   = data_dir / "source"
    shard        = cfg["data"]["shards"][0]

    print(f"Source:  {img_enc_base}")
    print(f"Render:  {cfg['data']['image_npz_dir']}/{shard}")
    print(f"Mesh:    {cfg['data']['mesh_npz_dir']}/{shard}")

    result = PartCraftDataset.prepare_from_prerender(
        img_enc_base=str(img_enc_base),
        source_dir=str(source_dir),
        render_out_dir=cfg["data"]["image_npz_dir"],
        mesh_out_dir=cfg["data"]["mesh_npz_dir"],
        shard=shard,
        limit=args.limit,
        force=args.force,
    )
    print(f"\nDone: {result['packed']} packed, "
          f"{result['skipped']} skipped, {result['total']} total")


if __name__ == "__main__":
    main()
