#!/usr/bin/env python3
"""Pack Vinedresser3D prerender outputs into PartCraft NPZ format.

Thin wrapper around PartCraftDataset.prepare_from_prerender().

Usage:
    python scripts/pack_prerender_npz.py --config configs/partobjaverse.yaml
    python scripts/pack_prerender_npz.py --config configs/partobjaverse.yaml --limit 3 --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    project_root = Path(__file__).resolve().parents[1]
    img_enc_base = project_root / "data" / "img_Enc"

    data_dir = Path(cfg["data"].get("data_dir", "data/partobjaverse_tiny"))
    if not data_dir.exists():
        data_dir = Path(cfg["data"]["image_npz_dir"]).parent
    source_dir = data_dir / "source"

    shard = cfg["data"]["shards"][0]

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
