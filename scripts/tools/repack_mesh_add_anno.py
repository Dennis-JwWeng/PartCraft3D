#!/usr/bin/env python3
"""Repack existing mesh.npz files to embed annotation data (making each shard self-contained).

Adds two new keys into mesh/{shard}/{id}.npz:
    anno_info.json     ← source/anno_infos/anno_infos/{id}/{id}_info.json
                         (peer groups, structural parts, part levels, bboxes)
    part_captions.json ← text_captions.json[{id}]
                         (per-part semantic short+long captions from PartVerse)

After repacking, configs no longer need:
    data.anno_dir          — read from mesh.npz["anno_info.json"]
    data.normalized_glb_dir — s5b reads mesh.npz["full.glb"] directly

Usage:
    python scripts/tools/repack_mesh_add_anno.py \\
        --shard 08 \\
        --mesh-dir /mnt/zsn/data/partverse/inputs/mesh \\
        --anno-dir /mnt/zsn/data/partverse/source/anno_infos/anno_infos \\
        --captions /mnt/zsn/data/partverse/partverse_zip/text_captions.json \\
        [--workers 8] [--force] [--dry-run] [--limit N]

    # All shards:
    for s in 00 02 05 06 07 08 09; do
        python scripts/tools/repack_mesh_add_anno.py --shard $s --workers 16 \\
            --captions /mnt/zsn/data/partverse/partverse_zip/text_captions.json
    done
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

LOG = logging.getLogger("repack_anno")


def repack_one(
    obj_id: str,
    mesh_npz_path: Path,
    anno_dir: Path,
    captions_index: dict[str, bytes] | None,
    force: bool = False,
) -> dict:
    """Add anno_info.json (and optionally part_captions.json) to one mesh.npz."""
    info_path = anno_dir / obj_id / f"{obj_id}_info.json"

    if not info_path.is_file():
        return {"status": "skip", "reason": f"missing {info_path}"}

    try:
        existing = np.load(str(mesh_npz_path), allow_pickle=True)
        data = {k: existing[k] for k in existing.files}
    except Exception as e:
        return {"status": "error", "reason": f"load failed: {e}"}

    already_anno = "anno_info.json" in data
    already_captions = "part_captions.json" in data
    if not force and already_anno and (captions_index is None or already_captions):
        return {"status": "skip", "reason": "already repacked"}

    added_bytes = 0

    if force or not already_anno:
        try:
            raw = info_path.read_bytes()
            data["anno_info.json"] = np.frombuffer(raw, dtype=np.uint8)
            added_bytes += len(raw)
        except Exception as e:
            return {"status": "error", "reason": f"read anno_info failed: {e}"}

    if captions_index is not None and (force or not already_captions):
        cap_bytes = captions_index.get(obj_id)
        if cap_bytes is not None:
            data["part_captions.json"] = np.frombuffer(cap_bytes, dtype=np.uint8)
            added_bytes += len(cap_bytes)

    # Atomic write: temp file → rename
    try:
        with tempfile.NamedTemporaryFile(
            dir=mesh_npz_path.parent, suffix=".npztmp", delete=False
        ) as tf:
            tmp = Path(tf.name)
        np.savez_compressed(str(tmp), **data)
        tmp_npz = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp
        if tmp_npz.exists() and tmp_npz != tmp:
            tmp.unlink(missing_ok=True)
            tmp = tmp_npz
        tmp.replace(mesh_npz_path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"status": "error", "reason": f"write failed: {e}"}

    return {"status": "ok", "added_bytes": added_bytes}


def _load_captions_index(captions_path: Path) -> dict[str, bytes]:
    """Load text_captions.json and return per-object bytes for embedding."""
    LOG.info("Loading captions from %s ...", captions_path)
    all_captions: dict = json.loads(captions_path.read_text())
    # Pre-serialise each object's entry to bytes to avoid repeated JSON dumps
    index: dict[str, bytes] = {
        obj_id: json.dumps(obj_caps, ensure_ascii=False).encode()
        for obj_id, obj_caps in all_captions.items()
    }
    LOG.info("Loaded captions for %d objects", len(index))
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard",     required=True, help="Shard id, e.g. 08")
    ap.add_argument("--mesh-dir",  default="/mnt/zsn/data/partverse/inputs/mesh",
                    help="Root of mesh/ inputs (shard subdirs underneath)")
    ap.add_argument("--anno-dir",
                    default="/mnt/zsn/data/partverse/source/anno_infos/anno_infos",
                    help="Root of anno_infos per-object dirs")
    ap.add_argument("--captions",  default=None,
                    help="Path to text_captions.json (embeds part_captions.json if given)")
    ap.add_argument("--workers",   type=int, default=8)
    ap.add_argument("--force",     action="store_true",
                    help="Re-add even if already present")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Show plan without writing")
    ap.add_argument("--limit",     type=int, default=0,
                    help="Process only first N objects (for testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    shard_str  = args.shard.zfill(2)
    mesh_shard = Path(args.mesh_dir) / shard_str
    anno_root  = Path(args.anno_dir)

    if not mesh_shard.is_dir():
        LOG.error("Mesh shard dir not found: %s", mesh_shard)
        raise SystemExit(1)

    captions_index: dict[str, bytes] | None = None
    if args.captions:
        captions_index = _load_captions_index(Path(args.captions))

    obj_ids = sorted(p.stem for p in mesh_shard.glob("*.npz"))
    if args.limit > 0:
        obj_ids = obj_ids[:args.limit]

    LOG.info("Shard %s: %d objects, workers=%d, force=%s, dry_run=%s, captions=%s",
             shard_str, len(obj_ids), args.workers, args.force, args.dry_run,
             "yes" if captions_index else "no")

    if args.dry_run:
        LOG.info("[dry-run] Would repack %d mesh.npz in %s", len(obj_ids), mesh_shard)
        for oid in obj_ids[:5]:
            npz = mesh_shard / f"{oid}.npz"
            ex = np.load(str(npz), allow_pickle=True)
            has_anno = "anno_info.json" in ex.files
            has_cap  = "part_captions.json" in ex.files
            info_ok  = (anno_root / oid / f"{oid}_info.json").is_file()
            cap_ok   = captions_index is not None and oid in captions_index
            LOG.info("  %-38s  anno=%s cap=%s  info_exists=%s cap_exists=%s",
                     oid, has_anno, has_cap, info_ok, cap_ok)
        return

    n_ok = n_skip = n_err = 0
    added_mb = 0.0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(repack_one, oid,
                        mesh_shard / f"{oid}.npz",
                        anno_root, captions_index, args.force): oid
            for oid in obj_ids
        }
        for i, fut in enumerate(as_completed(futures), 1):
            oid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"status": "error", "reason": str(e)}

            status = r.get("status")
            if status == "ok":
                n_ok += 1
                added_mb += r.get("added_bytes", 0) / 1024 / 1024
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                LOG.warning("ERROR %s: %s", oid, r.get("reason"))

            if i % 200 == 0 or i == len(obj_ids):
                LOG.info("Progress %d/%d  ok=%d skip=%d err=%d added=%.1f MB",
                         i, len(obj_ids), n_ok, n_skip, n_err, added_mb)

    LOG.info("Done. ok=%d  skip=%d  err=%d  total_added=%.1f MB",
             n_ok, n_skip, n_err, added_mb)
    if n_err > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
