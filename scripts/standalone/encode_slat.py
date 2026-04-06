#!/usr/bin/env python3
"""Pre-encode PartObjaverse objects into SLAT using Vinedresser3D's pipeline.

Extracts original GLB from source/mesh.zip, then runs:
  1. Blender rendering (150 views) + Open3D voxelization
  2. DINOv2 feature extraction + SLAT encoding

Output: cache/phase2_5/slat_cache/{obj_id}_feats.pt, {obj_id}_coords.pt

These cached SLATs are loaded by Phase 2.5's TrellisRefiner directly,
bypassing the broken HY3D-Part view-based encoding.

Usage:
    CUDA_VISIBLE_DEVICES=2 ATTN_BACKEND=xformers python scripts/encode_slat.py \
        --config configs/partobjaverse.yaml

    # Encode specific objects only
    CUDA_VISIBLE_DEVICES=2 ATTN_BACKEND=xformers python scripts/encode_slat.py \
        --config configs/partobjaverse.yaml \
        --obj-ids 00aee5c2fef743d69421bb642d446a5b
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging


def extract_glb(mesh_zip_path: str, obj_id: str, out_dir: str) -> str:
    """Extract a single GLB from source/mesh.zip."""
    with zipfile.ZipFile(mesh_zip_path) as zf:
        # Find the GLB (may be in a subdirectory)
        matches = [n for n in zf.namelist() if obj_id in n and n.endswith('.glb')]
        if not matches:
            raise FileNotFoundError(f"GLB for {obj_id} not found in {mesh_zip_path}")
        glb_name = matches[0]
        out_path = os.path.join(out_dir, f"{obj_id}.glb")
        with zf.open(glb_name) as src, open(out_path, 'wb') as dst:
            dst.write(src.read())
        return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Pre-encode objects into SLAT using Vinedresser3D pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--obj-ids", nargs="*", default=None,
                        help="Specific object IDs to encode (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if cached SLAT exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "encode_slat")
    p25_cfg = cfg.get("phase2_5", {})
    project_root = Path(__file__).resolve().parents[2]

    # Paths (resolve early so mesh.zip / metadata exist regardless of cwd)
    data_dir = Path(cfg["data"].get("data_dir", "data/partobjaverse_tiny"))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    data_dir = data_dir.resolve()
    mesh_zip = data_dir / "source" / "mesh.zip"
    if not mesh_zip.exists():
        logger.error(f"source/mesh.zip not found at {mesh_zip}")
        sys.exit(1)

    cache_dir = Path(p25_cfg.get("cache_dir", "cache/phase2_5"))
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    cache_dir = cache_dir.resolve()
    slat_cache_dir = cache_dir / "slat_cache"
    slat_cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which objects to encode
    if args.obj_ids:
        obj_ids = args.obj_ids
    else:
        meta = json.load(open(data_dir / "metadata.json"))
        obj_ids = [obj["obj_id"] for obj in meta["objects"]]

    # Skip already-encoded
    if not args.force:
        pending = []
        for oid in obj_ids:
            feats_path = slat_cache_dir / f"{oid}_feats.pt"
            coords_path = slat_cache_dir / f"{oid}_coords.pt"
            if feats_path.exists() and coords_path.exists():
                logger.info(f"Skipping {oid} (cached)")
            else:
                pending.append(oid)
        obj_ids = pending

    if not obj_ids:
        logger.info("All objects already encoded")
        return

    logger.info(f"Encoding {len(obj_ids)} objects into SLAT")

    os.environ["PARTCRAFT_DATASET_ROOT"] = str(data_dir)
    third_party = str(project_root / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    os.chdir(str(data_dir))

    from encode_asset.render_img_for_enc import renderImg_voxelize
    from encode_asset.encode_into_SLAT import encode_into_SLAT

    # Temporary directory for extracted GLBs
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, oid in enumerate(obj_ids):
            logger.info(f"[{i+1}/{len(obj_ids)}] Encoding {oid}...")

            try:
                # 1. Extract GLB
                glb_path = extract_glb(str(mesh_zip), oid, tmp_dir)
                logger.info(f"  Extracted GLB: {glb_path}")

                # 2. Blender render + voxelize
                renderImg_voxelize(glb_path)
                logger.info(f"  Rendered 150 views + voxelized")

                # 3. DINOv2 encode → SLAT
                encode_into_SLAT(oid)
                logger.info(f"  SLAT encoded")

                # 4. Copy SLAT to our cache
                import shutil
                src_feats = data_dir / "slat" / f"{oid}_feats.pt"
                src_coords = data_dir / "slat" / f"{oid}_coords.pt"
                if src_feats.exists() and src_coords.exists():
                    shutil.copy2(str(src_feats), str(slat_cache_dir / f"{oid}_feats.pt"))
                    shutil.copy2(str(src_coords), str(slat_cache_dir / f"{oid}_coords.pt"))
                    logger.info(f"  Cached to {slat_cache_dir}")
                else:
                    logger.error(f"  SLAT files not found after encoding!")

            except Exception as e:
                logger.error(f"  Failed: {e}")
                import traceback
                traceback.print_exc()

    logger.info("Done")


if __name__ == "__main__":
    main()
