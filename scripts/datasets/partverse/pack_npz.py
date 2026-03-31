#!/usr/bin/env python3
"""Pack PartVerse prerender outputs into PartCraft NPZ format.

PartVerse layout (under ``PARTVERSE_DATA_ROOT`` / ``data/partverse/``) and
where **per-part semantic text** lives:

1. ``source/text_captions.json``  (**唯一**按部件 ID 存自然语言描述)
   - Top level: ``{ "<object_uuid>": { "<part_id>": [caption0, caption1, …], … }, … }``
   - ``part_id`` 为字符串 ``"0"``, ``"1"``, …，与 ``face2label`` 里的部件整数 ID 一致。
   - 每条 ``caption*`` 为一句/一段英文；通常 ``[0]`` 为短标题句，``[1+]`` 为更长 VLM 风格段落。
   - ``pack_npz`` 用 ``_label_from_part_captions`` 取**第一条非空**字符串写入 NPZ 的 ``part_id_to_name``。

2. ``source/anno_infos/<uuid>/<uuid>_face2label.json``
   - 仅 ``{ "<face_index>": <part_id_int>, … }``，**没有**文字语义；与 ``segmented.glb`` 面片一一对应。

3. ``source/anno_infos/<uuid>/<uuid>_info.json``
   - 几何/顺序元数据（如 ``bboxes``, ``ordered_face_label``, ``weights`` 等），**不是**主要文本描述来源。

4. ``normalized_glbs/<uuid>.glb``、``img_Enc/<uuid>/``
   - 归一化整模与预渲染结果；语义标签仍来自 (1)。

Reads from:
    source/anno_infos/{uuid}/{uuid}_segmented.glb — mesh for splitting (with part groups)
    source/anno_infos/{uuid}/{uuid}_face2label.json — per-face → part_id (integers only)
    img_Enc/{uuid}/                               — rendered views + transforms
    source/text_captions.json                     — per-part text captions (semantic labels)

Writes:
    data/partverse/images/{shard}/{uuid}.npz      — render NPZ (pipeline input)
    data/partverse/mesh/{shard}/{uuid}.npz        — mesh NPZ (pipeline input)

Shard support mirrors prerender.py: --shard 00 --num-shards 10 processes
~1203 objects of the 12030 total.

Usage:
    # Pack shard 00 of 10
    python scripts/datasets/partverse/pack_npz.py --shard 00 --num-shards 10

    # Pack shard 01, skip already-packed objects
    python scripts/datasets/partverse/pack_npz.py --shard 01 --num-shards 10

    # Test: first 5 objects only
    python scripts/datasets/partverse/pack_npz.py --limit 5

    # Re-pack everything (overwrite)
    python scripts/datasets/partverse/pack_npz.py --force
"""

import argparse
import io
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import os

_PROJECT_ROOT  = Path(__file__).resolve().parents[3]
_PARTVERSE_DIR = Path(os.environ.get(
    "PARTVERSE_DATA_ROOT", str(_PROJECT_ROOT / "data" / "partverse")))
_ANNO_DIR      = _PARTVERSE_DIR / "source" / "anno_infos"
_CAPTIONS_PATH = _PARTVERSE_DIR / "source" / "text_captions.json"
_IMG_ENC_DIR   = _PARTVERSE_DIR / "img_Enc"
_IMAGES_DIR    = _PARTVERSE_DIR / "images"
_MESH_DIR      = _PARTVERSE_DIR / "mesh"

sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.datasets.prerender_common import select_shard
from partcraft.io.partcraft_loader import (
    _align_source_to_vd,
    _split_mesh,
    _to_ply,
)

# Views selected from the 150-view Hammersley sequence to pack into images NPZ.
# Four layers: bottom / low-diagonal / horizontal(VLM) / upper-diagonal.
# transforms.json is always packed in full (all 150 frames needed for mask rendering).
PACK_VIEWS: list[int] = [
    8, 9, 10, 11,          # bottom   (pitch ≈ -52° to -45°)
    23, 24, 25, 26,        # low      (pitch ≈ -23° to -18°)
    32, 33, 34, 35,        # horiz    (pitch ≈  -8° to  -4°) ← VLM labeling views
    89, 90, 91, 100,       # upper    (pitch ≈  27° to  34°)
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_from_part_captions(cap_list: list) -> str | None:
    """Semantic part label from one part's caption list in text_captions.json.

    Uses the **first non-empty** caption verbatim (whitespace normalized).
    PartVerse convention: index 0 is the short curated line; later entries
    are longer VLM-style paragraphs — the short line is what we want for
    pipeline / VLM prompts.
    """
    if not cap_list:
        return None
    for c in cap_list:
        if not isinstance(c, str):
            continue
        s = " ".join(c.split())
        if s:
            return s
    return None


def _load_face2label(obj_id: str, anno_dir: Path | None = None) -> np.ndarray | None:
    """Load face2label.json and return per-face part-id array."""
    base = anno_dir if anno_dir is not None else _ANNO_DIR
    path = base / obj_id / f"{obj_id}_face2label.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    if not d:
        return None
    max_face = max(int(k) for k in d)
    arr = np.zeros(max_face + 1, dtype=np.int32)
    for k, v in d.items():
        arr[int(k)] = int(v)
    return arr


def _load_source_mesh(obj_id: str, anno_dir: Path | None = None):
    """Load segmented GLB from anno_infos/.

    face2label.json indices correspond to the segmented.glb faces (not the
    normalized_glb, which has a different tesselation). Both share identical
    bounding boxes, so _align_source_to_vd (using transforms.json offset/scale
    recorded during rendering of normalized_glb) applies equally well here.
    """
    try:
        import trimesh
    except ImportError:
        raise RuntimeError("trimesh is required — pip install trimesh")

    base = anno_dir if anno_dir is not None else _ANNO_DIR
    seg_path = base / obj_id / f"{obj_id}_segmented.glb"
    if not seg_path.exists():
        return None
    mesh = trimesh.load(str(seg_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    return mesh


def _pack_one(obj_id: str, img_enc_dir: Path,
               render_out: Path, mesh_out: Path,
               captions: dict,
               keep_views: list[int] | None = None,
               anno_dir: Path | None = None) -> dict:
    """Pack one PartVerse object into render + mesh NPZ.

    Args:
        keep_views: View indices to include in the render NPZ. None = all views.
                    transforms.json is always packed in full regardless.
        anno_dir:   Override for _ANNO_DIR (source/anno_infos) when source data
                    lives on a different mount than the output dataset_root.
    """
    transforms_path = img_enc_dir / "transforms.json"
    if not transforms_path.exists():
        return {"status": "skip", "reason": "no transforms.json"}

    with open(transforms_path) as f:
        transforms = json.load(f)
    frames = transforms["frames"]

    # ---- Collect rendered PNGs (only selected views) ----
    view_set = set(keep_views) if keep_views is not None else set(range(len(frames)))
    render_data: dict[str, np.ndarray] = {}
    found = 0
    for i in range(len(frames)):
        if i not in view_set:
            continue
        png = img_enc_dir / f"{i:03d}.png"
        if not png.exists():
            continue
        with open(png, "rb") as f:
            render_data[f"{i:03d}.png"] = np.frombuffer(f.read(), dtype=np.uint8)
        found += 1

    if found == 0:
        return {"status": "skip", "reason": "no PNGs"}

    render_data["transforms.json"] = np.frombuffer(
        json.dumps(transforms).encode("utf-8"), dtype=np.uint8)

    # ---- Load source mesh and per-face part labels ----
    instance_gt = _load_face2label(obj_id, anno_dir=anno_dir)
    if instance_gt is None:
        return {"status": "skip", "reason": "no face2label.json"}

    source_mesh = _load_source_mesh(obj_id, anno_dir=anno_dir)
    if source_mesh is None:
        return {"status": "skip", "reason": "no source GLB"}

    if len(instance_gt) != len(source_mesh.faces):
        return {"status": "skip",
                "reason": (f"face2label ({len(instance_gt)}) != "
                           f"source faces ({len(source_mesh.faces)})")}

    # ---- Align source mesh to VD coordinate space ----
    source_mesh = _align_source_to_vd(source_mesh, transforms)

    # ---- Part labels: full best caption (first non-empty line per part) ----
    obj_caps = captions.get(obj_id, {})
    n_parts = int(instance_gt.max()) + 1
    labels = []
    for pid in range(n_parts):
        cap_list = obj_caps.get(str(pid), [])
        label = _label_from_part_captions(cap_list)
        if not label:
            label = f"part_{pid}"
        labels.append(label)

    # ---- Split mesh ----
    parts, split_mesh_json = _split_mesh(source_mesh, instance_gt, labels)

    render_data["split_mesh.json"] = np.frombuffer(
        json.dumps(split_mesh_json).encode("utf-8"), dtype=np.uint8)

    mesh_data: dict[str, np.ndarray] = {
        "full.ply": np.frombuffer(_to_ply(source_mesh), dtype=np.uint8),
    }
    for pid, label, sub in parts:
        mesh_data[f"part_{pid}.ply"] = np.frombuffer(_to_ply(sub), dtype=np.uint8)

    np.savez_compressed(str(render_out / f"{obj_id}.npz"), **render_data)
    np.savez_compressed(str(mesh_out / f"{obj_id}.npz"), **mesh_data)

    return {"status": "ok", "views": found, "parts": len(parts)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pack_npz_partverse")

    parser = argparse.ArgumentParser(
        description="Pack PartVerse prerender into PartCraft NPZ format")
    parser.add_argument("--data-root", type=str, default=None,
                        help="PartVerse data root (overrides PARTVERSE_DATA_ROOT env var)")

    sel = parser.add_argument_group("object selection")
    sel.add_argument("--obj-ids", nargs="*", default=None,
                     help="Explicit object IDs (overrides --shard)")
    sel.add_argument("--shard", type=str, default=None,
                     help="Shard to pack, e.g. '00'. Requires --num-shards.")
    sel.add_argument("--num-shards", type=int, default=10,
                     help="Total number of shards (default: 10)")
    sel.add_argument("--limit", type=int, default=0,
                     help="Cap to first N objects (0 = all)")

    parser.add_argument("--force", action="store_true",
                        help="Re-pack even if output NPZs already exist")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel CPU workers (default: 1)")
    args = parser.parse_args()

    global _PARTVERSE_DIR, _ANNO_DIR, _CAPTIONS_PATH, _IMG_ENC_DIR, _IMAGES_DIR, _MESH_DIR
    if args.data_root:
        _PARTVERSE_DIR = Path(args.data_root)
        _ANNO_DIR      = _PARTVERSE_DIR / "source" / "anno_infos"
        _CAPTIONS_PATH = _PARTVERSE_DIR / "source" / "text_captions.json"
        _IMG_ENC_DIR   = _PARTVERSE_DIR / "img_Enc"
        _IMAGES_DIR    = _PARTVERSE_DIR / "images"
        _MESH_DIR      = _PARTVERSE_DIR / "mesh"

    # ---- Determine object list ----
    if args.obj_ids:
        obj_ids = list(args.obj_ids)
        shard = "00"
        logger.info(f"Explicit --obj-ids: {len(obj_ids)} objects → shard {shard}")
    else:
        all_ids = sorted(p.name for p in _ANNO_DIR.iterdir() if p.is_dir())
        if args.shard is not None:
            obj_ids = select_shard(all_ids, args.shard, args.num_shards)
            shard = args.shard
            logger.info(f"Shard {shard}/{args.num_shards}: "
                        f"{len(obj_ids)}/{len(all_ids)} objects")
        else:
            obj_ids = all_ids
            shard = "00"
            logger.info(f"All objects: {len(obj_ids)} → shard {shard}")

    if args.limit > 0:
        obj_ids = obj_ids[:args.limit]
        logger.info(f"--limit: capped to {len(obj_ids)} objects")

    # ---- Load part captions (optional) ----
    captions: dict = {}
    if _CAPTIONS_PATH.exists():
        with open(_CAPTIONS_PATH) as f:
            captions = json.load(f)
        logger.info(f"Loaded captions for {len(captions)} objects")
    else:
        logger.warning(f"text_captions.json not found at {_CAPTIONS_PATH} "
                       "— using generic part names")

    # ---- Output directories ----
    render_out = _IMAGES_DIR / shard
    mesh_out   = _MESH_DIR / shard
    render_out.mkdir(parents=True, exist_ok=True)
    mesh_out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output → images/{shard}/ and mesh/{shard}/")

    # ---- Pack ----
    total = len(obj_ids)

    pending = []
    pre_skip = 0
    for obj_id in obj_ids:
        out_r = render_out / f"{obj_id}.npz"
        out_m = mesh_out   / f"{obj_id}.npz"
        if out_r.exists() and out_m.exists() and not args.force:
            pre_skip += 1
            continue
        img_enc_dir = _IMG_ENC_DIR / obj_id
        if not img_enc_dir.exists():
            logger.warning(f"{obj_id}: no img_Enc dir, skip")
            pre_skip += 1
            continue
        pending.append(obj_id)

    logger.info(f"Pack: {len(pending)} pending, {pre_skip} skipped / {total} total "
                f"(workers={args.workers})")

    if not pending:
        logger.info("Nothing to pack.")
        return

    def _do_pack(oid: str) -> dict:
        return _pack_one(oid, _IMG_ENC_DIR / oid, render_out, mesh_out, captions,
                         keep_views=PACK_VIEWS)

    ok = fail = 0
    if args.workers <= 1:
        for i, obj_id in enumerate(pending):
            result = _do_pack(obj_id)
            if result["status"] == "ok":
                ok += 1
                logger.info(f"[{i+1}/{len(pending)}] {obj_id}: "
                            f"{result['views']} views, {result['parts']} parts")
            else:
                fail += 1
                logger.warning(f"[{i+1}/{len(pending)}] {obj_id}: SKIP — {result['reason']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_do_pack, oid): oid for oid in pending}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                oid = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "ok":
                        ok += 1
                        if done_count % 50 == 0 or done_count == len(pending):
                            logger.info(f"[{done_count}/{len(pending)}] packed {oid}")
                    else:
                        fail += 1
                        logger.warning(f"[{done_count}/{len(pending)}] {oid}: SKIP — {result['reason']}")
                except Exception as e:
                    fail += 1
                    logger.error(f"[{done_count}/{len(pending)}] {oid}: ERROR — {e}")

    logger.info(f"\nDone: {ok} packed, {pre_skip} skipped, {fail} failed / {total} total")


if __name__ == "__main__":
    main()
