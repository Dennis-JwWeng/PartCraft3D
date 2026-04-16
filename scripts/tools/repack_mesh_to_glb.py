#!/usr/bin/env python3
"""Pack / repack mesh NPZs to GLB format, independent of images/render data.

vd_scale and vd_offset are computed directly from the normalized_glb
bounding box (Y-up → Z-up axis swap, then fit-to-[-0.5,0.5]^3).
No img_Enc or images NPZ is required.

Object discovery: reads UUID subdirs from <anno-infos-dir>/ and applies
range-based shard selection (same as prerender_common.select_shard).

Usage
-----
Repack existing shards in-place (GLB format, keep what's already GLB):
    python scripts/tools/repack_mesh_to_glb.py \\
        --data-root /mnt/zsn/data/partverse \\
        --shards 00 02 06 07 08 \\
        --workers 8

Pack new shards from scratch (no existing mesh NPZ needed):
    python scripts/tools/repack_mesh_to_glb.py \\
        --data-root /mnt/zsn/data/partverse \\
        --shards 01 03 04 09 \\
        --workers 8

Dry run:
    python scripts/tools/repack_mesh_to_glb.py \\
        --data-root /mnt/zsn/data/partverse \\
        --shards 08 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("repack_mesh_glb")


def _is_int_stem(p: Path) -> bool:
    try:
        int(p.stem)
        return True
    except ValueError:
        return False


def _compute_vd_transform(norm_glb_path: Path) -> tuple[float, np.ndarray] | None:
    """Compute vd_scale and vd_offset from normalized_glb bounding box.

    Replicates blender_render.normalize_scene() in trimesh:
      1. Load GLB (Y-up source coords)
      2. Apply Y-up → Z-up axis swap: (x,y,z) → (x,-z,y)
      3. Fit bounding box to [-0.5, 0.5]^3:
           scale  = 1 / max(bbox_max - bbox_min)
           offset = -(bbox_min + bbox_max) / 2
    """
    try:
        import trimesh
        scene = trimesh.load(str(norm_glb_path))
        mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
        v = np.asarray(mesh.vertices, dtype=np.float64)
        # Y-up → Z-up
        vd = np.empty_like(v)
        vd[:, 0] = v[:, 0]
        vd[:, 1] = -v[:, 2]
        vd[:, 2] = v[:, 1]
        bbox_min = vd.min(axis=0)
        bbox_max = vd.max(axis=0)
        max_extent = float(np.max(bbox_max - bbox_min))
        if max_extent < 1e-6:
            return None
        scale = 1.0 / max_extent                         # target: [-0.5, 0.5]^3
        offset = -((bbox_min + bbox_max) / 2.0)
        return scale, offset
    except Exception:
        return None


def pack_one(
    obj_id: str,
    mesh_npz: Path,
    textured_part_glbs_dir: Path,
    normalized_glb_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """Pack / repack a single mesh NPZ to GLB format. Returns status string."""
    # Already GLB and not forcing?
    if not force and mesh_npz.exists():
        try:
            existing = np.load(mesh_npz)
            if "full.glb" in existing:
                return "skip:already_glb"
        except Exception:
            pass  # corrupt NPZ — re-pack it

    norm_glb = normalized_glb_dir / f"{obj_id}.glb"
    part_root = textured_part_glbs_dir / obj_id

    if not norm_glb.exists():
        return "error:no_normalized_glb"
    if not part_root.exists():
        return "error:no_part_glbs_dir"

    part_files = sorted(
        (p for p in part_root.glob("*.glb") if _is_int_stem(p)),
        key=lambda p: int(p.stem),
    )
    if not part_files:
        return "error:no_part_glbs"

    if dry_run:
        return f"dry_run:ok:{len(part_files)}parts"

    # Compute vd_scale / vd_offset from GLB bounding box (no render data needed)
    vd = _compute_vd_transform(norm_glb)
    if vd is None:
        return "error:vd_transform_failed"
    vd_scale, vd_offset = vd

    # Build NPZ data
    data: dict[str, np.ndarray] = {}
    try:
        data["full.glb"] = np.frombuffer(norm_glb.read_bytes(), dtype=np.uint8)
    except OSError as e:
        return f"error:read_full_glb:{e}"

    for pp in part_files:
        pid = int(pp.stem)
        try:
            data[f"part_{pid}.glb"] = np.frombuffer(pp.read_bytes(), dtype=np.uint8)
        except OSError:
            continue

    data["vd_scale"] = np.array([vd_scale], dtype=np.float64)
    data["vd_offset"] = np.array(vd_offset, dtype=np.float64)

    # Atomic write: savez_compressed auto-appends .npz, so use stem-only tmp path
    mesh_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_stem = mesh_npz.parent / f".tmp_{obj_id}"
    tmp_npz  = mesh_npz.parent / f".tmp_{obj_id}.npz"
    try:
        np.savez_compressed(str(tmp_stem), **data)
        os.replace(tmp_npz, mesh_npz)
    except Exception as e:
        tmp_npz.unlink(missing_ok=True)
        return f"error:write:{e}"

    return f"ok:{len(part_files)}parts"


# ── worker ────────────────────────────────────────────────────────────────

_ctx: dict = {}


def _init(ctx: dict) -> None:
    global _ctx
    _ctx = ctx


def _do(obj_id: str) -> tuple[str, str]:
    c = _ctx
    return obj_id, pack_one(
        obj_id,
        c["mesh_dir"] / f"{obj_id}.npz",
        c["textured_part_glbs_dir"],
        c["normalized_glb_dir"],
        force=c["force"],
        dry_run=c["dry_run"],
    )


# ── shard runner ──────────────────────────────────────────────────────────

def select_shard(all_ids: list[str], shard: str, num_shards: int) -> list[str]:
    idx = int(shard)
    n = len(all_ids)
    chunk = (n + num_shards - 1) // num_shards
    start = idx * chunk
    end = min(start + chunk, n)
    return all_ids[start:end]


def pack_shard(
    shard: str,
    data_root: Path,
    anno_infos_dir: Path,
    textured_part_glbs_dir: Path,
    normalized_glb_dir: Path,
    num_shards: int,
    *,
    workers: int = 4,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    mesh_dir = data_root / "inputs" / "mesh" / shard

    # Discover obj_ids from anno_infos (works even without existing mesh files)
    all_ids = sorted(p.name for p in anno_infos_dir.iterdir() if p.is_dir())
    obj_ids = select_shard(all_ids, shard, num_shards)
    log.info("Shard %s: %d objects (from anno_infos)", shard, len(obj_ids))

    ctx = {
        "mesh_dir": mesh_dir,
        "textured_part_glbs_dir": textured_part_glbs_dir,
        "normalized_glb_dir": normalized_glb_dir,
        "force": force,
        "dry_run": dry_run,
    }

    stats: dict[str, int] = {}
    done = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(ctx,)) as pool:
        futures = {pool.submit(_do, oid): oid for oid in obj_ids}
        for fut in as_completed(futures):
            oid, status = fut.result()
            key = status.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            done += 1
            if done % 200 == 0 or done == len(obj_ids):
                log.info("  Shard %s: %d/%d  stats=%s",
                         shard, done, len(obj_ids), stats)
            if key == "error":
                log.warning("  [%s] %s", oid[:12], status)

    log.info("Shard %s done: %s", shard, stats)
    return stats


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack/repack mesh NPZs to GLB (no render data required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--shards", nargs="+", required=True,
                        help="e.g. 00 02 06 07 08  or  01 03 04 09")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--anno-infos-dir", default=None,
                        help="Defaults to <data-root>/source/anno_infos/anno_infos")
    parser.add_argument("--textured-part-glbs-dir", default=None,
                        help="Defaults to <data-root>/source/textured_part_glbs")
    parser.add_argument("--normalized-glb-dir", default=None,
                        help="Defaults to <data-root>/source/normalized_glbs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true",
                        help="Re-pack even if already GLB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    anno_dir = Path(args.anno_infos_dir) if args.anno_infos_dir \
        else data_root / "source" / "anno_infos" / "anno_infos"
    tpg_dir = Path(args.textured_part_glbs_dir) if args.textured_part_glbs_dir \
        else data_root / "source" / "textured_part_glbs"
    ngl_dir = Path(args.normalized_glb_dir) if args.normalized_glb_dir \
        else data_root / "source" / "normalized_glbs"

    log.info("data_root:          %s", data_root)
    log.info("anno_infos:         %s  (%d objs)",
             anno_dir, len(list(anno_dir.iterdir())) if anno_dir.exists() else 0)
    log.info("textured_part_glbs: %s", tpg_dir)
    log.info("normalized_glbs:    %s", ngl_dir)
    log.info("shards:             %s  (num_shards=%d)", args.shards, args.num_shards)
    log.info("workers:            %d  force=%s  dry_run=%s",
             args.workers, args.force, args.dry_run)

    total: dict[str, int] = {}
    for shard in args.shards:
        s = pack_shard(shard, data_root, anno_dir, tpg_dir, ngl_dir,
                       args.num_shards, workers=args.workers,
                       force=args.force, dry_run=args.dry_run)
        for k, v in s.items():
            total[k] = total.get(k, 0) + v

    log.info("All shards done: %s", total)


if __name__ == "__main__":
    main()
