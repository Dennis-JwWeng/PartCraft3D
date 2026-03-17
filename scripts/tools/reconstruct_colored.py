#!/usr/bin/env python3
"""Reconstruct colored 3D meshes from HY3D-Part objects using TRELLIS.

For each object:
  1. Load 42 pre-rendered RGBA views + camera transforms from image NPZ
  2. Load geometry-only mesh from mesh NPZ
  3. Encode views → DINOv2 features → project to voxels → SLAT
  4. Decode SLAT → colored triangle mesh via FlexiCubes mesh decoder
  5. Export as colored PLY (vertices + faces + vertex RGB)

This produces colored versions of the original HY3D-Part objects, which are
stored as geometry-only (no vertex color) in the dataset.

Usage:
    ATTN_BACKEND=xformers python scripts/reconstruct_colored.py
    ATTN_BACKEND=xformers python scripts/reconstruct_colored.py --limit 10
    ATTN_BACKEND=xformers python scripts/reconstruct_colored.py --obj-ids abc123 def456
    ATTN_BACKEND=xformers python scripts/reconstruct_colored.py --resume
    ATTN_BACKEND=xformers python scripts/reconstruct_colored.py --fmt glb  # textured GLB (slow)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.hy3d_loader import HY3DPartDataset


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct colored 3D meshes from HY3D-Part using TRELLIS")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max objects to process")
    parser.add_argument("--obj-ids", nargs="+", type=str, default=None,
                        help="Specific object IDs to reconstruct")
    parser.add_argument("--fmt", type=str, default="glb", choices=["ply", "glb"],
                        help="Output format: glb (textured, high quality) or ply (vertex color only, low quality)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed objects")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "reconstruct")
    p25_cfg = cfg.get("phase2_5", {})

    # Add Vinedresser3D to path
    vinedresser_path = p25_cfg.get(
        "vinedresser_path", "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main")
    sys.path.insert(0, vinedresser_path)

    from partcraft.phase2_assembly.trellis_refine import TrellisRefiner

    # ---- Load dataset ----
    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )

    # ---- Setup output ----
    output_dir = Path(cfg["data"]["output_dir"]) / "colored_meshes"
    cache_dir = Path(cfg["data"]["output_dir"]) / p25_cfg.get("cache_dir", "cache/phase2_5")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"

    # ---- Resume support ----
    done_ids: set[str] = set()
    if args.resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["obj_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    # ---- Determine which objects to process ----
    if args.obj_ids:
        # Specific objects
        objects = []
        for oid in args.obj_ids:
            for shard in cfg["data"]["shards"]:
                try:
                    obj = dataset.load_object(shard, oid)
                    objects.append((shard, oid))
                    break
                except Exception:
                    continue
    else:
        # All objects (with optional limit)
        objects = []
        for obj in dataset:
            objects.append((obj.shard, obj.obj_id))
            obj.close()
            if args.limit and len(objects) >= args.limit:
                break

    pending = [(s, o) for s, o in objects if o not in done_ids]
    logger.info(f"Reconstruct: {len(pending)} objects to process "
                f"({len(done_ids)} already done)")

    if not pending:
        logger.info("Nothing to do")
        return

    # ---- Initialize refiner (reuses TRELLIS models) ----
    refiner = TrellisRefiner(
        device="cuda",
        cache_dir=str(cache_dir),
        vinedresser_path=vinedresser_path,
        trellis_text_ckpt=p25_cfg.get("trellis_text_ckpt"),
        trellis_image_ckpt=p25_cfg.get("trellis_image_ckpt"),
    )
    refiner.load_models()

    # ---- Process ----
    ext = args.fmt
    success, fail = 0, 0
    with open(manifest_path, "a") as out_fp:
        for i, (shard, obj_id) in enumerate(pending):
            logger.info(f"[{i+1}/{len(pending)}] {shard}/{obj_id}")
            try:
                obj = dataset.load_object(shard, obj_id)

                # Encode to SLAT (cached)
                slat = refiner._get_slat(obj)

                # Decode to colored mesh
                out_path = output_dir / shard / f"{obj_id}.{ext}"
                if ext == "glb":
                    refiner.slat_to_glb(slat, out_path)
                else:
                    refiner.slat_to_mesh_ply(slat, out_path)

                record = {
                    "obj_id": obj_id,
                    "shard": shard,
                    "mesh_path": str(out_path),
                    "format": ext,
                    "status": "success",
                }
                out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fp.flush()
                success += 1

                obj.close()

            except Exception as e:
                logger.error(f"Failed {obj_id}: {e}")
                out_fp.write(json.dumps({
                    "obj_id": obj_id, "shard": shard, "status": "failed",
                    "error": str(e),
                }) + "\n")
                out_fp.flush()
                fail += 1

    logger.info(f"Reconstruct complete: {success} succeeded, {fail} failed")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
