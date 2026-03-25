#!/usr/bin/env python3
"""Pre-render PartObjaverse-Tiny objects + encode into SLAT.

Reads source GLBs from data/partobjaverse_tiny/source/mesh.zip (one entry per
object ID), runs Blender GPU rendering (150 views) + Open3D voxelization, then
DINOv2 + SLAT encoding.

Outputs (under data/partobjaverse_tiny/):
    img_Enc/{obj_id}/
        000.png .. 149.png
        transforms.json
        mesh.ply
        voxels.ply
    slat/
        {obj_id}_feats.pt
        {obj_id}_coords.pt

Usage:
    # Render on 4 GPUs, then encode on GPU 0
    CUDA_VISIBLE_DEVICES=0,1,2,3 ATTN_BACKEND=xformers \\
        python scripts/datasets/partobjaverse/prerender.py \\
        --config configs/local_sglang.yaml --render-workers 4

    # Encode only (multi-GPU)
    CUDA_VISIBLE_DEVICES=0,1,2,3 ATTN_BACKEND=xformers \\
        python scripts/datasets/partobjaverse/prerender.py \\
        --config configs/local_sglang.yaml --encode-only --num-gpus 4

    # Test: first 3 objects
    CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers \\
        python scripts/datasets/partobjaverse/prerender.py \\
        --config configs/local_sglang.yaml --limit 3
"""

import argparse
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_THIRD_PARTY  = _PROJECT_ROOT / "third_party"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIRD_PARTY))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from scripts.datasets.prerender_common import (
    get_available_gpus,
    launch_multi_gpu_encode,
    print_summary,
    run_encode,
    run_render,
)


# ---------------------------------------------------------------------------
# Dataset paths (resolved from config)
# ---------------------------------------------------------------------------

def _get_data_dir(cfg: dict) -> Path:
    data_dir = Path(cfg["data"].get("data_dir", "data/partobjaverse_tiny"))
    if not data_dir.is_absolute():
        data_dir = _PROJECT_ROOT / data_dir
    if not data_dir.exists():
        img_dir = Path(cfg["data"]["image_npz_dir"])
        if not img_dir.is_absolute():
            img_dir = _PROJECT_ROOT / img_dir
        data_dir = img_dir.parent
    return data_dir


# ---------------------------------------------------------------------------
# GLB access: extract from source/mesh.zip
# ---------------------------------------------------------------------------

def _make_glb_getter(mesh_zip: Path, tmp_dir: str):
    """Return a callable(obj_id) -> Path that extracts GLBs from mesh.zip."""
    def getter(obj_id: str) -> Path:
        out_path = Path(tmp_dir) / f"{obj_id}.glb"
        if out_path.exists():
            return out_path
        with zipfile.ZipFile(mesh_zip) as zf:
            matches = [n for n in zf.namelist()
                       if obj_id in n and n.endswith(".glb")]
            if not matches:
                return out_path  # caller checks .exists()
            with zf.open(matches[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        return out_path
    return getter


# ---------------------------------------------------------------------------
# Object discovery
# ---------------------------------------------------------------------------

def get_all_obj_ids(cfg: dict) -> list[str]:
    mesh_dir = Path(cfg["data"]["mesh_npz_dir"])
    if not mesh_dir.is_absolute():
        mesh_dir = _PROJECT_ROOT / mesh_dir
    shards = cfg["data"].get("shards", ["00"])
    obj_ids = []
    for shard in shards:
        shard_dir = mesh_dir / shard
        if not shard_dir.exists():
            continue
        obj_ids.extend(f.stem for f in sorted(shard_dir.iterdir())
                       if f.suffix == ".npz")
    return obj_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-render PartObjaverse-Tiny + encode SLAT")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--obj-ids", nargs="*", default=None,
                        help="Specific object IDs (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N objects (0 = all)")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs exist")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Parallel GPUs for encoding (0 = single process)")
    parser.add_argument("--render-workers", type=int, default=1,
                        help="Parallel Blender workers, each on a dedicated GPU")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "prerender_partobjaverse")

    data_dir   = _get_data_dir(cfg)
    mesh_zip   = data_dir / "source" / "mesh.zip"
    img_enc_dir = data_dir / "img_Enc"
    slat_dir   = data_dir / "slat"

    if not mesh_zip.exists():
        logger.error(f"source/mesh.zip not found at {mesh_zip}")
        sys.exit(1)

    img_enc_dir.mkdir(parents=True, exist_ok=True)
    slat_dir.mkdir(parents=True, exist_ok=True)

    # Determine object IDs
    if args.obj_ids:
        obj_ids = args.obj_ids
    else:
        obj_ids = get_all_obj_ids(cfg)
        if args.limit > 0:
            obj_ids = obj_ids[:args.limit]

    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Total objects: {len(obj_ids)}")

    # Multi-GPU encode
    if args.num_gpus > 1 and not args.render_only:
        launch_multi_gpu_encode(obj_ids, slat_dir,
                                Path(__file__).resolve(),
                                args.num_gpus, args.force, logger)
        print_summary(obj_ids, img_enc_dir, slat_dir, logger)
        return

    # Use a persistent temp dir so GLBs survive across the render loop
    with tempfile.TemporaryDirectory(prefix="partcraft_glb_") as tmp_dir:
        glb_getter = _make_glb_getter(mesh_zip, tmp_dir)

        if not args.encode_only:
            run_render(obj_ids, glb_getter, img_enc_dir, _THIRD_PARTY,
                       args.force, args.render_workers,
                       Path(__file__).resolve(), logger)

    if not args.render_only:
        run_encode(obj_ids, img_enc_dir, slat_dir, _THIRD_PARTY,
                   args.force, logger)

    print_summary(obj_ids, img_enc_dir, slat_dir, logger)


if __name__ == "__main__":
    main()
