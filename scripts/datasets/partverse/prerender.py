#!/usr/bin/env python3
"""Pre-render PartVerse objects + pack into NPZ + encode into SLAT.

PartVerse ships pre-normalized GLBs in data/partverse/source/normalized_glbs/.
This script reads them directly, skipping the mesh.zip format used by
PartObjaverse-Tiny.

Outputs (under data/partverse/):
    images/{shard}/{obj_id}.npz        — render NPZ (PNGs + transforms + split_mesh)
    mesh/{shard}/{obj_id}.npz          — mesh NPZ (full.ply + per-part PLYs)
    img_Enc/{obj_id}/voxels.ply        — voxel point cloud for SLAT encode
    slat/{shard}/{obj_id}_feats.pt     — SLAT features
    slat/{shard}/{obj_id}_coords.pt    — SLAT coordinates

Shard support: 12030 objects can be split into N shards (e.g. 10 shards of
~1203 objects each) and processed independently — on different machines or
sequentially. SLAT output is organized per-shard under slat/{shard}/.

Usage:
    # Process shard 00 of 10 on 4 GPUs (render + pack + encode)
    CUDA_VISIBLE_DEVICES=0,1,2,3 ATTN_BACKEND=xformers \\
        python scripts/datasets/partverse/prerender.py \\
        --shard 00 --num-shards 10 --render-workers 4

    # Render only, shard 01 (4 parallel Blender workers)
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
        python scripts/datasets/partverse/prerender.py \\
        --shard 01 --num-shards 10 --render-only --render-workers 4

    # Pack only, shard 01 (after render, before or without encode)
    python scripts/datasets/partverse/prerender.py \\
        --shard 01 --num-shards 10 --pack-only

    # Encode only, shard 02, multi-GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 ATTN_BACKEND=xformers \\
        python scripts/datasets/partverse/prerender.py \\
        --shard 02 --num-shards 10 --encode-only --num-gpus 4

    # Test: first 5 objects
    CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers \\
        python scripts/datasets/partverse/prerender.py --limit 5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT  = Path(__file__).resolve().parents[3]
_THIRD_PARTY   = _PROJECT_ROOT / "third_party"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIRD_PARTY))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from scripts.datasets.prerender_common import (
    launch_multi_gpu_encode,
    print_summary,
    run_encode,
    run_render,
    select_shard,
)


# ---------------------------------------------------------------------------
# Object discovery
# ---------------------------------------------------------------------------

def get_all_obj_ids(glb_dir: Path) -> list[str]:
    return sorted(p.stem for p in glb_dir.glob("*.glb"))


# ---------------------------------------------------------------------------
# GLB access: direct file lookup
# ---------------------------------------------------------------------------

def _glb_getter(glb_dir: Path, obj_id: str) -> Path:
    return glb_dir / f"{obj_id}.glb"


# ---------------------------------------------------------------------------
# Pack step: render outputs → images/ + mesh/ NPZ, clean up img_Enc PNGs
# ---------------------------------------------------------------------------

def _run_pack(
    obj_ids: list[str],
    shard: str,
    captions: dict,
    force: bool,
    logger: logging.Logger,
    *,
    img_enc_dir: Path,
    images_dir: Path,
    mesh_dir: Path,
):
    """Pack rendered img_Enc outputs into images/{shard}/ + mesh/{shard}/ NPZ.

    After successful packing, PNGs / transforms.json / mesh.ply are removed
    from img_Enc — only voxels.ply is kept (used as encode cache marker).
    """
    from scripts.datasets.partverse.pack_npz import _pack_one, PACK_VIEWS

    render_out = images_dir / shard
    mesh_out   = mesh_dir   / shard
    render_out.mkdir(parents=True, exist_ok=True)
    mesh_out.mkdir(parents=True, exist_ok=True)

    total = len(obj_ids)
    ok = skip = fail = 0

    for i, obj_id in enumerate(obj_ids):
        out_r = render_out / f"{obj_id}.npz"
        out_m = mesh_out   / f"{obj_id}.npz"
        if out_r.exists() and out_m.exists() and not force:
            skip += 1
            continue

        one_img_enc_dir = img_enc_dir / obj_id
        if not one_img_enc_dir.exists():
            skip += 1
            continue

        result = _pack_one(obj_id, one_img_enc_dir, render_out, mesh_out, captions,
                           keep_views=PACK_VIEWS)
        if result["status"] == "ok":
            ok += 1
            logger.info(f"[pack {i+1}/{total}] {obj_id}: "
                        f"{result['views']} views, {result['parts']} parts")
            # Keep all 150 PNGs in img_Enc — only 16 selected views go into NPZ.
            # img_Enc serves as the full render archive; images/ NPZ is the slim
            # pipeline input.
        else:
            fail += 1
            logger.warning(f"[pack {i+1}/{total}] {obj_id}: skip — {result['reason']}")

    logger.info(f"Pack: {ok} packed, {skip} skipped, {fail} failed / {total} total")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-render PartVerse + pack NPZ + encode SLAT")
    parser.add_argument("--config", type=str,
                        default="configs/prerender_partverse.yaml",
                        help="Prerender config path")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Deprecated: override paths.dataset_root for this run")

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

    # Mode (mutually exclusive steps; default = all three)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true",
                      help="Render only (write img_Enc/), skip pack and encode")
    mode.add_argument("--pack-only", action="store_true",
                      help="Pack only (img_Enc/ → images/ + mesh/ NPZ), skip render and encode")
    mode.add_argument("--encode-only", action="store_true",
                      help="Encode only (img_Enc/ → slat/), skip render and pack")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist")

    # Parallelism
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Parallel GPUs for encoding (0 = single process)")
    parser.add_argument("--render-workers", type=int, default=1,
                        help="Parallel Blender workers, each on a dedicated GPU. "
                             "Should not exceed the number of available GPUs.")
    args = parser.parse_args()

    cfg = load_config(args.config, for_prerender=True, prerender_mode="partverse")
    logger = setup_logging(cfg, "prerender_partverse")

    if args.data_root:
        raise ValueError(
            "[CONFIG_ERROR] cli.--data-root disabled runtime "
            "use config paths.dataset_root instead"
        )

    paths = cfg["paths"]
    partverse_dir = Path(paths["dataset_root"])
    glb_dir = Path(paths["source_glb_dir"])
    captions_path = Path(paths["captions_json"])
    img_enc_dir = Path(paths["img_enc_dir"])
    slat_root_dir = Path(paths["slat_dir"])
    images_dir = Path(paths["images_npz_dir"])
    mesh_dir = Path(paths["mesh_npz_dir"])

    if not glb_dir.exists():
        raise FileNotFoundError(f"Missing paths.source_glb_dir: {glb_dir}")

    img_enc_dir.mkdir(parents=True, exist_ok=True)
    slat_root_dir.mkdir(parents=True, exist_ok=True)

    # ---- Determine object list and shard ----
    if args.obj_ids:
        obj_ids = list(args.obj_ids)
        shard = args.shard if args.shard is not None else "00"
        logger.info(f"Explicit --obj-ids: {len(obj_ids)} objects → shard {shard}")
    else:
        all_ids = get_all_obj_ids(glb_dir)
        if not all_ids:
            raise RuntimeError(
                f"[CONFIG_ERROR] paths.source_glb_dir {glb_dir} config "
                "contains zero .glb objects"
            )
        if args.shard is not None:
            obj_ids = select_shard(all_ids, args.shard, args.num_shards)
            shard = args.shard
            if obj_ids:
                i0, i1 = all_ids.index(obj_ids[0]), all_ids.index(obj_ids[-1])
                span = f"(idx {i0}–{i1})"
            else:
                span = "(empty shard)"
            logger.info(f"Shard {shard}/{args.num_shards}: "
                        f"{len(obj_ids)}/{len(all_ids)} objects {span}")
            if not obj_ids:
                raise RuntimeError(
                    f"[CONFIG_ERROR] shard.{shard} empty runtime "
                    f"num_shards={args.num_shards} produces zero objects"
                )
        else:
            obj_ids = all_ids
            shard = "00"
            logger.info(f"All objects: {len(obj_ids)} → shard {shard}")

    if args.limit > 0:
        obj_ids = obj_ids[:args.limit]
        logger.info(f"--limit: capped to {len(obj_ids)} objects")

    # SLAT output goes into a per-shard subdirectory for easy batched compression.
    slat_shard_dir = slat_root_dir / shard
    slat_shard_dir.mkdir(parents=True, exist_ok=True)

    # Migrate any flat legacy files (slat/{obj_id}_*.pt) into the shard subdir.
    _migrated = 0
    for oid in obj_ids:
        src_f = slat_root_dir / f"{oid}_feats.pt"
        src_c = slat_root_dir / f"{oid}_coords.pt"
        if src_f.exists():
            src_f.rename(slat_shard_dir / f"{oid}_feats.pt")
            src_c.rename(slat_shard_dir / f"{oid}_coords.pt")
            _migrated += 1
    if _migrated:
        logger.info(f"Migrated {_migrated} flat SLAT files → slat/{shard}/")

    logger.info("PartVerse dataset root: %s", partverse_dir)
    logger.info("PartVerse GLB dir: %s", glb_dir)

    # ---- Load part captions (needed by pack step) ----
    captions: dict = {}
    if captions_path.exists():
        with open(captions_path) as f:
            captions = json.load(f)
        logger.info(f"Loaded captions for {len(captions)} objects")
    else:
        raise FileNotFoundError(
            f"[CONFIG_ERROR] paths.captions_json {captions_path} config missing file"
        )

    do_render = not args.pack_only and not args.encode_only
    do_encode = not args.render_only and not args.pack_only
    do_pack   = not args.render_only and not args.encode_only

    # ---- Multi-GPU encode shortcut ----
    if args.num_gpus > 1 and do_encode:
        extra_shard_args = ["--shard", shard, "--num-shards", str(args.num_shards)]
        if do_render:
            run_render(
                obj_ids,
                lambda oid: _glb_getter(glb_dir, oid),
                img_enc_dir,
                _THIRD_PARTY,
                args.force,
                args.render_workers,
                Path(__file__).resolve(),
                logger,
                extra_worker_args=extra_shard_args,
                dataset_root=partverse_dir,
            )
        launch_multi_gpu_encode(obj_ids, slat_shard_dir,
                                Path(__file__).resolve(),
                                args.num_gpus, args.force, logger,
                                extra_args=extra_shard_args,
                                dataset_root=partverse_dir)
        if do_pack:
            _run_pack(
                obj_ids,
                shard,
                captions,
                args.force,
                logger,
                img_enc_dir=img_enc_dir,
                images_dir=images_dir,
                mesh_dir=mesh_dir,
            )
        print_summary(obj_ids, img_enc_dir, slat_shard_dir, logger)
        return

    # ---- Step 1: Render ----
    if do_render:
        extra_worker_args = ["--shard", shard, "--num-shards", str(args.num_shards)]
        run_render(
            obj_ids,
            lambda oid: _glb_getter(glb_dir, oid),
            img_enc_dir,
            _THIRD_PARTY,
            args.force,
            args.render_workers,
            Path(__file__).resolve(),
            logger,
            extra_worker_args=extra_worker_args,
            dataset_root=partverse_dir,
        )

    # ---- Step 2: Encode (needs raw PNGs + transforms.json) ----
    if do_encode:
        run_encode(
            obj_ids,
            img_enc_dir,
            slat_shard_dir,
            _THIRD_PARTY,
            args.force,
            logger,
            dataset_root=partverse_dir,
        )

    # ---- Step 3: Pack (img_Enc → NPZ; can delete raw files safely now) ----
    if do_pack:
        _run_pack(
            obj_ids,
            shard,
            captions,
            args.force,
            logger,
            img_enc_dir=img_enc_dir,
            images_dir=images_dir,
            mesh_dir=mesh_dir,
        )

    print_summary(obj_ids, img_enc_dir, slat_shard_dir, logger)


if __name__ == "__main__":
    main()
