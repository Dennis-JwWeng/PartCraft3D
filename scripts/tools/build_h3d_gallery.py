#!/usr/bin/env python3
"""build_h3d_gallery.py
=========================

Export a gallery of BEFORE/AFTER GLBs + edit instructions for a list of
hand-picked edits (``<obj>/<edit_id>`` strings -- the same picks format
consumed by ``build_showcase_hero.py``).

Per-pick layout (under ``--out``)::

    <edit_id>/
        before.glb
        after.glb
        meta.json   # {edit_id, edit_type, obj_id, shard, prompt,
                    #  target_part_desc, part_labels, source_del_id?,
                    #  before_source, after_source, before_frame, after_frame}

Provenance rules (matching
``docs/runbooks/showcase-pick-workflow.md`` section 5):

- ``deletion``        : BEFORE = ``full.glb`` from
                        ``data/partverse/inputs/mesh/<shard>/<obj>.npz`` ;
                        AFTER  = copy of ``edits_3d/<eid>/after_new.glb``.
- ``addition``        : BEFORE = copy of
                        ``edits_3d/<source_del_id>/after_new.glb`` ;
                        AFTER  = ``full.glb`` from input mesh NPZ.
- ``mod/scl/mat/clr/glb`` (FLUX): BEFORE = ``full.glb`` from input mesh
                        NPZ ; AFTER = decoded from
                        ``edits_3d/<eid>/after.npz`` via TRELLIS
                        ``decode_slat`` + ``to_glb``.

NOTE on frames:
- ``full.glb`` / ``after_new.glb`` live in the pipeline's **pre-vd-scale**
  frame (bounds roughly in [-1, 1]).
- TRELLIS-decoded GLBs live in TRELLIS's **normalized cube** frame
  ([-0.5, 0.5]) after z-up -> y-up rotation baked in by ``to_glb``.
- ``meta.json.before_frame`` / ``after_frame`` record which one applies,
  so downstream scene-assembly code can scale them on a uniform shelf.

Resume-safe: skips a pick when all three output files already exist.

Usage::

    PARTCRAFT_CKPT_ROOT=$PWD/checkpoints \
    PYTHONPATH=.:third_party \
    /mnt/zsn/miniconda3/envs/vinedresser3d/bin/python \
      scripts/tools/build_h3d_gallery.py \
        --picks reports/h3d_gallery_picks_union.json \
        --root  outputs/partverse/shard08/mode_e_text_align \
        --shard 08 \
        --mesh-root data/partverse/inputs/mesh \
        --out outputs/H3D_gallery
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("h3d_gallery")

# -- keep import paths symmetric with decode_inspect.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
for p in (_PROJECT_ROOT, _THIRD_PARTY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


EID_PFX = {"deletion": "del", "modification": "mod", "scale": "scl",
           "material": "mat", "color": "clr", "global": "glb",
           "addition": "add"}
PFX_TO_TYPE = {v: k for k, v in EID_PFX.items()}
FLUX_TYPES = {"modification", "scale", "material", "color", "global"}
FLUX_PREFIXES = {EID_PFX[t] for t in FLUX_TYPES}


def _eid_type(eid: str) -> str:
    return PFX_TO_TYPE.get(eid.split("_", 1)[0], "?")


# ───────────────────── meta lookup (mirrors hero script) ─────────────────────

def _load_meta_for_pick(root: Path, shard: str, oid: str, eid: str) -> dict:
    """Pull prompt / target_part_desc / part_labels / source_del_id for
    one pick. Same plumbing as build_showcase_hero._load_meta_for_pick."""
    sf = root / "objects" / shard / oid / "edit_status.json"
    parsed_path = root / "objects" / shard / oid / "phase1" / "parsed.json"
    meta: dict = {
        "prompt": "", "target_part_desc": "", "part_labels": [],
        "source_del_id": "",
        "score": None, "gate_e_reason": "",
    }
    et = _eid_type(eid)

    if sf.is_file():
        try:
            stat = json.loads(sf.read_text())
            ed = (stat.get("edits") or {}).get(eid) or {}
            ge_vlm = ((ed.get("gates") or {}).get("E") or {}).get("vlm") or {}
            meta["score"] = ge_vlm.get("score")
            meta["gate_e_reason"] = ge_vlm.get("reason") or ""
        except Exception:
            pass

    if et == "addition":
        mp = root / "objects" / shard / oid / "edits_3d" / eid / "meta.json"
        if mp.is_file():
            try:
                m = json.loads(mp.read_text())
                meta.update({
                    "prompt": m.get("prompt") or "",
                    "target_part_desc": m.get("target_part_desc") or "",
                    "part_labels": list(m.get("part_labels") or []),
                    "source_del_id": m.get("source_del_id") or "",
                })
            except Exception:
                pass
    elif parsed_path.is_file():
        try:
            pj = json.loads(parsed_path.read_text())
            edits_list = pj.get("parsed", {}).get("edits", [])
            flux_seq = 0
            del_seq = 0
            for ed in edits_list:
                t = ed.get("edit_type")
                pfx = EID_PFX.get(t)
                if not pfx:
                    continue
                if t == "deletion":
                    seq = del_seq
                    del_seq += 1
                elif t in FLUX_TYPES:
                    seq = flux_seq
                    flux_seq += 1
                else:
                    continue
                cand_eid = f"{pfx}_{oid}_{seq:03d}"
                if cand_eid == eid:
                    meta.update({
                        "prompt": ed.get("prompt") or "",
                        "target_part_desc": ed.get("target_part_desc") or "",
                        "part_labels": list(ed.get("part_labels") or []),
                    })
                    break
        except Exception:
            pass

    return meta


# ───────────────────── GLB helpers for del / add ─────────────────────

_MESH_CACHE: dict = {}


def _extract_full_glb(mesh_root: Path, shard: str, obj_id: str,
                      out_path: Path) -> None:
    """Write ``full.glb`` extracted from the input mesh NPZ."""
    if out_path.is_file():
        return
    src = mesh_root / shard / f"{obj_id}.npz"
    if not src.is_file():
        raise FileNotFoundError(f"input mesh NPZ missing: {src}")
    import numpy as np
    key = (str(src),)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = np.load(src)
    d = _MESH_CACHE[key]
    if "full.glb" not in d.files:
        raise KeyError(f"'full.glb' missing in {src} -- keys={list(d.files)}")
    out_path.write_bytes(bytes(d["full.glb"].tobytes()))


def _copy_glb(src: Path, dst: Path) -> None:
    if dst.is_file():
        return
    if not src.is_file():
        raise FileNotFoundError(f"GLB source missing: {src}")
    shutil.copy2(src, dst)


# ───────────────────── TRELLIS lazy pipeline loader ─────────────────────

_PIPELINE = None


def _load_pipeline(ckpt: str):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    log.info("Loading TRELLIS pipeline from %s ...", ckpt)
    from trellis.pipelines import TrellisTextTo3DPipeline  # type: ignore
    pipe = TrellisTextTo3DPipeline.from_pretrained(ckpt)
    pipe.cuda()
    log.info("TRELLIS pipeline ready")
    _PIPELINE = pipe
    return pipe


def _load_slat(npz_path: Path):
    import numpy as np
    import torch
    from trellis.modules import sparse as sp  # type: ignore
    d = np.load(str(npz_path))
    feats = torch.from_numpy(d["slat_feats"]).float().cuda()
    coords = torch.from_numpy(d["slat_coords"]).int().cuda()
    return sp.SparseTensor(feats=feats, coords=coords)


def _decode_slat_to_glb(pipeline, slat, *, simplify: float,
                       texture_size: int, fill_holes: bool,
                       fill_holes_resolution: int,
                       fill_holes_num_views: int,
                       bake_mode: str,
                       render_nviews: int,
                       render_resolution: int,
                       vd_scale: float | None = None,
                       vd_offset=None):
    """Decode a single SLAT SparseTensor to a textured ``trimesh.Trimesh``.

    This is a thinned copy of
    ``third_party/trellis/utils/postprocessing_utils.to_glb`` with
    configurable postprocess/bake knobs so the gallery build stays
    fast even across 40+ FLUX picks.
    """
    import numpy as np
    import torch  # noqa: F401  -- ensure cuda is live
    import trimesh
    import trimesh.visual
    from PIL import Image
    from trellis.utils.postprocessing_utils import (  # type: ignore
        postprocess_mesh, parametrize_mesh, bake_texture,
    )
    from trellis.utils.render_utils import render_multiview  # type: ignore

    outputs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]

    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()

    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=0.04,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
        fill_holes_resolution=fill_holes_resolution,
        fill_holes_num_views=fill_holes_num_views,
        verbose=False,
    )
    vertices, faces, uvs = parametrize_mesh(vertices, faces)

    observations, extrinsics, intrinsics = render_multiview(
        gs, resolution=render_resolution, nviews=render_nviews,
    )
    masks = [np.any(obs > 0, axis=-1) for obs in observations]
    extrinsics = [e.cpu().numpy() for e in extrinsics]
    intrinsics = [i.cpu().numpy() for i in intrinsics]
    texture = bake_texture(
        vertices, faces, uvs, observations, masks, extrinsics, intrinsics,
        texture_size=texture_size, mode=bake_mode,
        lambda_tv=0.01, verbose=False,
    )
    texture = Image.fromarray(texture)

    # z-up -> y-up (same rotation to_glb bakes in)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Inverse VD normalisation so the decoded mesh lands in the same
    # y-up, pre-vd-scale frame as ``full.glb`` in the input mesh NPZ
    # (forward transform in partcraft/render/overview.py:106-107 is
    #   v_dec = (v_raw + inv_off) * vd_scale,   inv_off = [ox, oz, -oy]).
    if vd_scale is not None and vd_offset is not None:
        ox, oy, oz = float(vd_offset[0]), float(vd_offset[1]), float(vd_offset[2])
        inv_off = np.array([ox, oz, -oy], dtype=np.float64)
        vertices = vertices / float(vd_scale) - inv_off

    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    out = trimesh.Trimesh(
        vertices, faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
    )
    return out


# ───────────────────── per-pick export ─────────────────────

def _export_pick(pick: str, *, root: Path, shard: str, mesh_root: Path,
                 out_root: Path, pipeline_factory,
                 decode_kwargs: dict, force: bool) -> dict:
    oid, eid = pick.split("/", 1)
    et = _eid_type(eid)
    if et == "?":
        raise ValueError(f"unknown eid prefix: {eid}")
    item_dir = out_root / eid
    item_dir.mkdir(parents=True, exist_ok=True)
    before_glb = item_dir / "before.glb"
    after_glb = item_dir / "after.glb"
    meta_path = item_dir / "meta.json"

    edit_dir = root / "objects" / shard / oid / "edits_3d" / eid
    lookup = _load_meta_for_pick(root, shard, oid, eid)

    record = {
        "edit_id": eid,
        "obj_id": oid,
        "shard": shard,
        "edit_type": et,
        "prompt": lookup.get("prompt") or "",
        "target_part_desc": lookup.get("target_part_desc") or "",
        "part_labels": lookup.get("part_labels") or [],
        "source_del_id": lookup.get("source_del_id") or "",
        "vlm_score": lookup.get("score"),
        "gate_e_reason": lookup.get("gate_e_reason") or "",
    }

    fully_done = (before_glb.is_file() and after_glb.is_file()
                  and meta_path.is_file())
    if fully_done and not force:
        record["status"] = "skip"
        try:
            record.update(json.loads(meta_path.read_text()))
        except Exception:
            pass
        record["status"] = "skip"
        return record

    def _rel(pth: Path) -> str:
        try:
            return str(pth.resolve().relative_to(_PROJECT_ROOT))
        except ValueError:
            return str(pth)

    mesh_npz = mesh_root / shard / (oid + ".npz")
    if et == "deletion":
        _extract_full_glb(mesh_root, shard, oid, before_glb)
        _copy_glb(edit_dir / "after_new.glb", after_glb)
        record["before_source"] = f"{_rel(mesh_npz)}::full.glb"
        record["after_source"] = f"{_rel(edit_dir)}/after_new.glb"
        record["before_frame"] = "vd_pre_scale"
        record["after_frame"] = "vd_pre_scale"
    elif et == "addition":
        src_del = record["source_del_id"]
        if not src_del:
            raise RuntimeError(
                f"addition {eid} has no source_del_id in meta.json")
        src_del_dir = (
            root / "objects" / shard / oid / "edits_3d" / src_del
        )
        _copy_glb(src_del_dir / "after_new.glb", before_glb)
        _extract_full_glb(mesh_root, shard, oid, after_glb)
        record["before_source"] = f"{_rel(src_del_dir)}/after_new.glb"
        record["after_source"] = f"{_rel(mesh_npz)}::full.glb"
        record["before_frame"] = "vd_pre_scale"
        record["after_frame"] = "vd_pre_scale"
    else:
        # FLUX: decode after.npz, BEFORE = full.glb from input NPZ
        after_npz = edit_dir / "after.npz"
        if not after_npz.is_file():
            raise FileNotFoundError(
                f"FLUX {eid}: after.npz missing at {after_npz}")
        _extract_full_glb(mesh_root, shard, oid, before_glb)
        # Pull vd_scale / vd_offset so decoded mesh is placed in the same
        # frame as before.glb (see _decode_slat_to_glb).
        import numpy as _np
        _npz = _MESH_CACHE.get((str(mesh_npz),))
        if _npz is None:
            _npz = _np.load(mesh_npz)
            _MESH_CACHE[(str(mesh_npz),)] = _npz
        vd_scale = float(_npz["vd_scale"][0]) if "vd_scale" in _npz.files else None
        vd_offset = _np.array(_npz["vd_offset"]) if "vd_offset" in _npz.files else None
        pipe = pipeline_factory()
        import torch
        with torch.no_grad():
            slat = _load_slat(after_npz)
            tm = _decode_slat_to_glb(
                pipe, slat,
                vd_scale=vd_scale, vd_offset=vd_offset,
                **decode_kwargs,
            )
        tm.export(str(after_glb))
        del slat, tm
        torch.cuda.empty_cache()
        record["before_source"] = f"{_rel(mesh_npz)}::full.glb"
        record["after_source"] = f"{_rel(after_npz)} (TRELLIS decoded)"
        record["before_frame"] = "vd_pre_scale"
        record["after_frame"] = (
            "vd_pre_scale" if vd_scale is not None and vd_offset is not None
            else "trellis_normalized_y_up"
        )
        record["vd_scale"] = vd_scale
        record["vd_offset"] = (
            [float(x) for x in vd_offset] if vd_offset is not None else None
        )
        record["decode_kwargs"] = decode_kwargs

    meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    record["status"] = "ok"
    return record


# ───────────────────── driver ─────────────────────

def run(picks: list[str], *, root: Path, shard: str, mesh_root: Path,
        out_root: Path, ckpt: str, decode_kwargs: dict,
        force: bool) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "picks.json").write_text(json.dumps(picks, indent=2) + "\n")

    has_flux = any(eid.split("/", 1)[1].split("_", 1)[0] in FLUX_PREFIXES
                   for eid in picks if "/" in eid)

    def _factory():
        return _load_pipeline(ckpt)

    records = []
    n_ok = n_fail = n_skip = 0
    t0 = time.time()

    for i, pick in enumerate(picks, 1):
        t_pick = time.time()
        try:
            rec = _export_pick(
                pick, root=root, shard=shard, mesh_root=mesh_root,
                out_root=out_root, pipeline_factory=_factory,
                decode_kwargs=decode_kwargs, force=force,
            )
            records.append(rec)
            if rec.get("status") == "skip":
                n_skip += 1
                log.info("[%02d/%02d] SKIP   %s  (already exported)",
                         i, len(picks), pick)
            else:
                n_ok += 1
                log.info("[%02d/%02d] ok     %-11s %s  (%.1fs)",
                         i, len(picks), rec["edit_type"], pick,
                         time.time() - t_pick)
        except Exception as e:
            n_fail += 1
            log.exception("[%02d/%02d] FAIL   %s: %s",
                          i, len(picks), pick, e)
            records.append({"edit_id": pick, "status": "fail",
                            "error": str(e)})

    (out_root / "index.json").write_text(
        json.dumps({
            "shard": shard,
            "n_picks": len(picks),
            "n_ok": n_ok,
            "n_skip": n_skip,
            "n_fail": n_fail,
            "wall_s": round(time.time() - t0, 2),
            "decode_kwargs": decode_kwargs if has_flux else None,
            "records": records,
        }, indent=2, ensure_ascii=False) + "\n"
    )
    log.info("done: ok=%d skip=%d fail=%d  wall=%.1fs  out=%s",
             n_ok, n_skip, n_fail, time.time() - t0, out_root)
    return 0 if n_fail == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--picks", required=True, type=Path,
                    help="JSON list of '<obj>/<edit_id>' strings.")
    ap.add_argument("--root", required=True, type=Path,
                    help="Pipeline_v3 shard root, e.g. "
                         "outputs/partverse/shard08/mode_e_text_align")
    ap.add_argument("--shard", required=True,
                    help="Shard number string (e.g. '08').")
    ap.add_argument("--mesh-root", required=True, type=Path,
                    help="Input mesh NPZ root (data/partverse/inputs/mesh).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Gallery output directory (e.g. outputs/H3D_gallery).")
    ap.add_argument("--ckpt", default="checkpoints/TRELLIS-text-xlarge",
                    help="TRELLIS pipeline checkpoint path.")
    ap.add_argument("--force", action="store_true",
                    help="Re-export even when outputs exist.")
    # Decode quality knobs (FLUX only). Defaults chosen for ~30 s/pick on L20X
    # and good-enough quality for visual gallery review. Bump for hero-grade.
    ap.add_argument("--simplify", type=float, default=0.95)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--fill-holes", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--fill-holes-resolution", type=int, default=512)
    ap.add_argument("--fill-holes-num-views", type=int, default=200)
    ap.add_argument("--bake-mode", default="fast", choices=["fast", "opt"])
    ap.add_argument("--render-nviews", type=int, default=60)
    ap.add_argument("--render-resolution", type=int, default=768)
    args = ap.parse_args()

    picks = json.loads(args.picks.read_text())
    if not isinstance(picks, list):
        log.error("--picks must be a JSON array of strings")
        return 2
    log.info("loaded %d picks from %s", len(picks), args.picks)

    decode_kwargs = dict(
        simplify=args.simplify,
        texture_size=args.texture_size,
        fill_holes=args.fill_holes,
        fill_holes_resolution=args.fill_holes_resolution,
        fill_holes_num_views=args.fill_holes_num_views,
        bake_mode=args.bake_mode,
        render_nviews=args.render_nviews,
        render_resolution=args.render_resolution,
    )

    return run(
        picks, root=args.root, shard=args.shard, mesh_root=args.mesh_root,
        out_root=args.out, ckpt=args.ckpt, decode_kwargs=decode_kwargs,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
