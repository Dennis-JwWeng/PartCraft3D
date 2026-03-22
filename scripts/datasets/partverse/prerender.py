#!/usr/bin/env python3
"""Pre-render PartVerse objects + encode into SLAT.

PartVerse ships pre-normalized GLBs in data/partverse/source/normalized_glbs/.
This script reads them directly, skipping the mesh.zip format used by
PartObjaverse-Tiny.

Outputs (under data/partverse/):
    img_Enc/{obj_id}/
        000.png .. 149.png
        transforms.json
        mesh.ply
        voxels.ply
    slat/
        {obj_id}_feats.pt
        {obj_id}_coords.pt

Shard support: 12030 objects can be split into N shards (e.g. 10 shards of
~1203 objects each) and processed independently — on different machines or
sequentially. SLAT output is always flat under slat/, regardless of shard.

Usage:
    # Process shard 00 of 10 on 4 GPUs (render + encode)
    CUDA_VISIBLE_DEVICES=0,1,2,3 ATTN_BACKEND=xformers \\
        python scripts/datasets/partverse/prerender.py \\
        --shard 00 --num-shards 10 --render-workers 4

    # Render only, shard 01
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
        python scripts/datasets/partverse/prerender.py \\
        --shard 01 --num-shards 10 --render-only --render-workers 4

    # Encode only, shard 02, multi-GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 ATTN_BACKEND=xformers \\
        python scripts/datasets/partverse/prerender.py \\
        --shard 02 --num-shards 10 --encode-only --num-gpus 4

    # Process all objects (no sharding), single GPU
    CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers \\
        python scripts/datasets/partverse/prerender.py

    # Test: first 5 objects
    CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers \\
        python scripts/datasets/partverse/prerender.py --limit 5
"""

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT  = Path(__file__).resolve().parents[3]
_THIRD_PARTY   = _PROJECT_ROOT / "third_party"
_PARTVERSE_DIR = _PROJECT_ROOT / "data" / "partverse"
_GLB_DIR       = _PARTVERSE_DIR / "source" / "normalized_glbs"
_IMG_ENC_DIR   = _PARTVERSE_DIR / "img_Enc"
_SLAT_DIR      = _PARTVERSE_DIR / "slat"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIRD_PARTY))

from scripts.datasets.prerender_common import (
    ensure_outputs_symlink,
    launch_multi_gpu_encode,
    print_summary,
    run_encode,
    run_render,
    select_shard,
)


# ---------------------------------------------------------------------------
# Object discovery
# ---------------------------------------------------------------------------

def get_all_obj_ids() -> list[str]:
    return sorted(p.stem for p in _GLB_DIR.glob("*.glb"))


# ---------------------------------------------------------------------------
# GLB access: direct file lookup
# ---------------------------------------------------------------------------

def _glb_getter(obj_id: str) -> Path:
    return _GLB_DIR / f"{obj_id}.glb"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("prerender_partverse")

    parser = argparse.ArgumentParser(
        description="Pre-render PartVerse + encode SLAT")

    # Object selection (mutually exclusive priority: --obj-ids > --shard > all)
    sel = parser.add_argument_group("object selection")
    sel.add_argument("--obj-ids", nargs="*", default=None,
                     help="Explicit object IDs to process")
    sel.add_argument("--shard", type=str, default=None,
                     help="Shard to process, zero-padded string e.g. '00', '03'. "
                          "Requires --num-shards.")
    sel.add_argument("--num-shards", type=int, default=10,
                     help="Total number of shards (default: 10). "
                          "With 12030 objects each shard is ~1203 objects.")
    sel.add_argument("--limit", type=int, default=0,
                     help="Cap to first N objects after shard selection (0 = all). "
                          "Useful for quick tests.")

    # Mode
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist")

    # Parallelism
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Parallel GPUs for encoding (0 = single process)")
    parser.add_argument("--render-workers", type=int, default=1,
                        help="Parallel Blender workers, each on a dedicated GPU. "
                             "Should not exceed the number of available GPUs.")
    args = parser.parse_args()

    ensure_outputs_symlink(_THIRD_PARTY, _PARTVERSE_DIR, logger)
    _IMG_ENC_DIR.mkdir(parents=True, exist_ok=True)
    _SLAT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Determine object list ----
    if args.obj_ids:
        obj_ids = list(args.obj_ids)
        logger.info(f"Explicit --obj-ids: {len(obj_ids)} objects")
    else:
        all_ids = get_all_obj_ids()
        if args.shard is not None:
            obj_ids = select_shard(all_ids, args.shard, args.num_shards)
            logger.info(f"Shard {args.shard}/{args.num_shards}: "
                        f"{len(obj_ids)}/{len(all_ids)} objects "
                        f"(idx {all_ids.index(obj_ids[0])}–{all_ids.index(obj_ids[-1])})")
        else:
            obj_ids = all_ids
            logger.info(f"All objects: {len(obj_ids)}")

    if args.limit > 0:
        obj_ids = obj_ids[:args.limit]
        logger.info(f"--limit: capped to {len(obj_ids)} objects")

    logger.info(f"PartVerse GLB dir: {_GLB_DIR}")

    # ---- Multi-GPU encode mode (launches subprocesses with --obj-ids) ----
    if args.num_gpus > 1 and not args.render_only:
        launch_multi_gpu_encode(obj_ids, _SLAT_DIR,
                                Path(__file__).resolve(),
                                args.num_gpus, args.force, logger)
        print_summary(obj_ids, _IMG_ENC_DIR, _SLAT_DIR, logger)
        return

    # ---- Single-process or parallel-render mode ----
    if not args.encode_only:
        run_render(obj_ids, _glb_getter, _IMG_ENC_DIR, _THIRD_PARTY,
                   args.force, args.render_workers,
                   Path(__file__).resolve(), logger)

    if not args.render_only:
        run_encode(obj_ids, _IMG_ENC_DIR, _SLAT_DIR, _THIRD_PARTY,
                   args.force, logger)

    print_summary(obj_ids, _IMG_ENC_DIR, _SLAT_DIR, logger)


if __name__ == "__main__":
    main()
