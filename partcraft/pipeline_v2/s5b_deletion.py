"""Step s5b / s6b — Deletion GLB masking + SLAT/SS re-encode.

Two halves of the deletion path, kept in one module because they share
the same per-object loop and ``edits_3d/<edit_id>/`` output dir:

* :func:`run_mesh_delete` (s5b) — CPU only. For every deletion spec,
  builds ``after_new.glb`` by KD-tree face-centroid matching from the
  normalized source GLB (UV-textured, high quality). Also writes
  ``add_*/meta.json`` for the inverse addition backfill.
  Requires ``normalized_glb_dir`` and ``anno_dir`` in pipeline config.
  Legacy PLY path (``TrellisRefiner.direct_delete_mesh``) is retained
  as fallback when those dirs are not configured.

* :func:`run_reencode` (s6b) — GPU. Runs after VLM filtering. For
  every surviving deletion edit, re-encodes ``after.ply`` (or GLB)
  via Blender 40-view render → DINOv2 → SLAT encoder → SS encoder →
  ``after.npz``. Also writes ``before.npz`` and hardlinks NPZ/PNG to
  the paired ``add_*`` directory.

The two phases write distinct status entries (``s5b_del_mesh`` and
``s6b_del_reencode``) so the orchestrator can resume / retry them
independently.
"""
from __future__ import annotations

import errno as _errno
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "tools"))

from .paths import ObjectContext
from .specs import EditSpec, iter_deletion_specs
from .status import update_step, STATUS_OK, STATUS_FAIL
from .qc_io import is_edit_qc_failed, is_gate_a_failed
from .edit_status_io import edit_needs_step, update_edit_stage, obj_needs_stage
from . import services_cfg as psvc
from .addition_utils import invert_delete_prompt


# ─────────────────── s5b: mesh-direct delete (CPU) ────────────────────

@dataclass
class DelMeshResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0


def _backfill_add(ctx, del_spec, add_seq, *, force=False, logger=None):
    """Create add_*/meta.json. NPZ/PNG links deferred to s6b."""
    import logging as _l
    log = logger or _l.getLogger("pipeline_v2.s5b")
    add_id = ctx.edit_id("addition", add_seq)
    add_dir = ctx.edit_3d_dir(add_id)
    meta_path = add_dir / "meta.json"
    if meta_path.is_file() and not force:
        return False
    try:
        add_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({
            "edit_id": add_id, "edit_type": "addition",
            "obj_id": ctx.obj_id, "shard": ctx.shard,
            "source_del_id": del_spec.edit_id,
            "selected_part_ids": list(del_spec.selected_part_ids),
            "view_index": del_spec.view_index,
            "prompt": invert_delete_prompt(del_spec.prompt),
            "target_part_desc": del_spec.target_part_desc,
            "object_desc": del_spec.object_desc,
            "part_labels": list(del_spec.part_labels),
            "rationale": f"inverse of {del_spec.edit_id}",
        }, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        log.warning("[s5b] add backfill seq=%d: %s", add_seq, e)
        return False



def _build_deletion_glb(
    obj_id: str,
    selected_part_ids: list,
    pair_dir: "Path",
    normalized_glb_dir: "Path",
    anno_dir: "Path",
    *,
    force: bool = False,
    logger: "logging.Logger | None" = None,
) -> bool:
    """Build after_new.glb by masking selected parts from the normalized GLB.

    Uses KD-tree face-centroid matching: segmented.glb (annotated, coarse) ->
    normalized GLB (high-quality UV). Returns True on success, False on error.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    log = logger or logging.getLogger("pipeline_v2.s5b")

    norm_path    = normalized_glb_dir / f"{obj_id}.glb"
    anno_obj_dir = anno_dir / obj_id
    seg_path     = anno_obj_dir / f"{obj_id}_segmented.glb"
    f2l_path     = anno_obj_dir / f"{obj_id}_face2label.json"

    for p in (norm_path, seg_path, f2l_path):
        if not p.is_file():
            log.warning("[s5b] _build_deletion_glb: missing %s", p)
            return False

    after_glb  = pair_dir / "after_new.glb"
    before_glb = pair_dir / "before_new.glb"

    if after_glb.is_file() and not force:
        return True

    try:
        norm_scene = trimesh.load(str(norm_path), force='scene')
        meshes = list(norm_scene.geometry.values())
        if len(meshes) == 1:
            norm_mesh = meshes[0]
            _visual = norm_mesh.visual
        else:
            norm_mesh = trimesh.util.concatenate(meshes)
            _visual = None  # visual is unreliable after concatenation
        seg_mesh = trimesh.load(str(seg_path), force='mesh')
        if len(seg_mesh.faces) == 0 or len(norm_mesh.faces) == 0:
            log.warning(
                "[s5b] _build_deletion_glb %s: empty mesh (seg=%d faces, norm=%d faces)",
                obj_id, len(seg_mesh.faces), len(norm_mesh.faces),
            )
            return False
        with open(f2l_path) as _f:
            f2l = {int(k): int(v) for k, v in json.load(_f).items()}

        seg_centroids  = seg_mesh.vertices[seg_mesh.faces].mean(axis=1)
        norm_centroids = norm_mesh.vertices[norm_mesh.faces].mean(axis=1)
        _, nn_idxs     = cKDTree(seg_centroids).query(norm_centroids, k=1)

        face_labels = np.array([f2l.get(int(i), -1) for i in nn_idxs])
        mask_keep   = ~np.isin(face_labels, selected_part_ids)

        if (~mask_keep).sum() == 0:
            log.warning(
                "[s5b] _build_deletion_glb %s: no faces deleted for part_ids=%s",
                obj_id, selected_part_ids,
            )
        if mask_keep.sum() == 0:
            log.warning(
                "[s5b] _build_deletion_glb %s: all faces deleted, skipping export",
                obj_id,
            )
            return False

        masked = trimesh.Trimesh(
            vertices=norm_mesh.vertices,
            faces=norm_mesh.faces[mask_keep],
            visual=_visual,
            process=False,
        )
        masked.remove_unreferenced_vertices()
        masked.export(str(after_glb))

        if not before_glb.exists():
            try:
                before_glb.symlink_to(norm_path.resolve())
            except OSError:
                pass

        log.info("[s5b] GLB del=%d keep=%d -> %s",
                 (~mask_keep).sum(), mask_keep.sum(), after_glb.name)
        return True
    except Exception as exc:
        log.warning("[s5b] _build_deletion_glb %s: %s", obj_id, exc)
        return False


def _build_deletion_from_npz(
    mesh_npz: "Path",
    selected_part_ids: "list[int]",
    pair_dir: "Path",
    *,
    force: bool = False,
    logger: "logging.Logger | None" = None,
) -> bool:
    """Build deletion result by concatenating non-deleted part GLBs from mesh NPZ.

    Returns True if successful, False if NPZ doesn't have GLB format or is missing.
    Writes after_new.glb to pair_dir.
    """
    import io
    import re
    import numpy as np
    import trimesh
    from pathlib import Path as _Path

    log = logger or logging.getLogger("pipeline_v2.s5b")
    mesh_npz = _Path(mesh_npz)
    pair_dir = _Path(pair_dir)
    out_path = pair_dir / "after_new.glb"

    if not force and out_path.exists():
        return True

    if not mesh_npz.exists():
        return False

    npz = np.load(mesh_npz, allow_pickle=False)

    glb_part_keys = [k for k in npz.files if k.startswith("part_") and k.endswith(".glb")]
    if not glb_part_keys:
        return False  # PLY format — caller should use the KD-tree path

    all_pids = sorted(
        int(re.search(r'\d+', k).group())
        for k in glb_part_keys
        if re.search(r'\d+', k)
    )
    keep_pids = [pid for pid in all_pids if pid not in selected_part_ids]

    if not keep_pids:
        log.warning("[s5b] _build_deletion_from_npz: all parts selected for deletion")
        return False

    # Detect if NPZ stores raw Y-up GLBs (new format) needing VD transform
    has_vd_transform = "vd_scale" in npz.files
    if has_vd_transform:
        vd_scale = float(npz["vd_scale"][0])
        vd_offset = np.array(npz["vd_offset"])

    meshes = []
    for pid in keep_pids:
        key = f"part_{pid}.glb"
        if key not in npz.files:
            continue
        raw = bytes(npz[key])
        scene = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene")
        if isinstance(scene, trimesh.Scene):
            geoms = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            meshes.extend(geoms)
        elif isinstance(scene, trimesh.Trimesh):
            meshes.append(scene)

    if not meshes:
        return False

    result = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    # Apply VD-space transform (Y-up → Z-up + scale/offset) for new-format NPZs
    if has_vd_transform:
        v = np.array(result.vertices)
        vd = np.empty_like(v)
        vd[:, 0] = v[:, 0]
        vd[:, 1] = -v[:, 2]
        vd[:, 2] = v[:, 1]
        result.vertices = (vd + vd_offset) * vd_scale
    pair_dir.mkdir(parents=True, exist_ok=True)
    result.export(str(out_path))
    log.info("[s5b] _build_deletion_from_npz: wrote %s (%d parts kept)", out_path, len(keep_pids))
    return True


def run_mesh_delete_for_object(
    ctx: ObjectContext,
    *,
    dataset=None,          # only used in legacy PLY path (normalized_glb_dir not set)
    normalized_glb_dir: "Path | None" = None,
    anno_dir: "Path | None" = None,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> DelMeshResult:
    log = logger or logging.getLogger("pipeline_v2.s5b")
    res = DelMeshResult(obj_id=ctx.obj_id)

    specs = list(iter_deletion_specs(ctx))
    if not specs:
        update_step(ctx, "s5b_del_mesh", status=STATUS_OK, n=0,
                    reason="no_deletions")
        return res

    use_glb = bool(normalized_glb_dir and anno_dir)

    # Legacy PLY path: load heavy dataset only when GLB dirs are not configured.
    obj_record = None
    if not use_glb:
        from partcraft.trellis.refiner import TrellisRefiner
        if dataset is None:
            log.error("[s5b] %s: dataset required for PLY path but not provided", ctx.obj_id)
            update_step(ctx, "s5b_del_mesh", status=STATUS_FAIL,
                        error="dataset_required_for_ply_path")
            res.n_fail = len(specs)
            return res
        try:
            obj_record = dataset.load_object(ctx.shard, ctx.obj_id)
        except Exception as e:
            log.error("[s5b] %s load failed: %s", ctx.obj_id, e)
            update_step(ctx, "s5b_del_mesh", status=STATUS_FAIL, error=str(e))
            res.n_fail = len(specs)
            return res

    add_seq = 0
    for spec in specs:
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s5b", prereq_map, force=force):
                res.n_skip += 1
                add_seq += 1
                continue
        elif is_gate_a_failed(ctx, spec.edit_id):
            log.info("[s5b] skip %s (gate_a_fail)", spec.edit_id)
            res.n_skip += 1
            add_seq += 1
            continue
        pair_dir = ctx.edit_3d_dir(spec.edit_id)

        if use_glb:
            # ── GLB path (primary) ─────────────────────────────────────
            after_glb = pair_dir / "after_new.glb"
            if after_glb.is_file() and not force:
                _backfill_add(ctx, spec, add_seq, force=False, logger=log)
                res.n_skip += 1
                add_seq += 1
                continue
            pair_dir.mkdir(parents=True, exist_ok=True)
            # Primary: NPZ-based deletion (no KD-tree needed)
            ok = False
            if ctx.mesh_npz is not None:
                ok = _build_deletion_from_npz(
                    ctx.mesh_npz,
                    list(spec.selected_part_ids),
                    pair_dir,
                    force=force,
                    logger=log,
                )
            # Fallback: KD-tree matching from normalized GLB
            if not ok:
                ok = _build_deletion_glb(
                    ctx.obj_id, list(spec.selected_part_ids),
                    pair_dir, normalized_glb_dir, anno_dir,
                    force=force, logger=log,
                )
            if ok:
                _backfill_add(ctx, spec, add_seq, force=force, logger=log)
                res.n_ok += 1
                update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5b", status="done")
            else:
                res.n_fail += 1
                update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5b",
                                  status="error", reason="glb_build_failed")
        else:
            # ── Legacy PLY path (fallback when GLB dirs not configured) ─
            a_ply = pair_dir / "after.ply"
            if a_ply.is_file() and not force:
                _backfill_add(ctx, spec, add_seq, force=False, logger=log)
                res.n_skip += 1
                add_seq += 1
                continue
            try:
                pair_dir.mkdir(parents=True, exist_ok=True)
                TrellisRefiner.direct_delete_mesh(
                    obj_record, spec.selected_part_ids, pair_dir, export_ply=True,
                )
                _backfill_add(ctx, spec, add_seq, force=force, logger=log)
                res.n_ok += 1
                update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5b", status="done")
            except Exception as e:
                log.error("[s5b] %s failed: %s", spec.edit_id, e)
                res.n_fail += 1
                update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5b",
                                  status="error", reason=str(e)[:200])

        add_seq += 1

    if obj_record is not None:
        obj_record.close()
    update_step(
        ctx, "s5b_del_mesh",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
    )
    return res


def _needs_glb_backfill(ctx: ObjectContext) -> bool:
    """Return True if any deletion edit is missing after_new.glb."""
    from .specs import iter_deletion_specs as _iter_del
    from .qc_io import is_gate_a_failed as _gate_a_fail
    for spec in _iter_del(ctx):
        if _gate_a_fail(ctx, spec.edit_id):
            continue
        if not (ctx.edit_3d_dir(spec.edit_id) / "after_new.glb").is_file():
            return True
    return False


def run_mesh_delete(
    ctxs: Iterable[ObjectContext],
    *,
    cfg: dict,
    images_root: "Path | None" = None,
    mesh_root: "Path | None" = None,
    shard: str = "01",
    normalized_glb_dir: "Path | None" = None,
    anno_dir: "Path | None" = None,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[DelMeshResult]:
    """Sequential per-object loop.

    When ``normalized_glb_dir`` and ``anno_dir`` are set (GLB path), the
    heavy ``HY3DPartDataset`` is not loaded.  Already-done objects that are
    still missing ``after_new.glb`` are automatically re-processed (backfill).
    """
    log = logger or logging.getLogger("pipeline_v2.s5b")

    use_glb = bool(normalized_glb_dir and anno_dir)

    # Only load the dataset when the legacy PLY path is needed.
    dataset = None
    if not use_glb:
        from partcraft.io.hy3d_loader import HY3DPartDataset
        if images_root is None or mesh_root is None:
            raise ValueError("images_root and mesh_root are required for the PLY path")
        dataset = HY3DPartDataset(str(images_root), str(mesh_root), [shard])

    out: list[DelMeshResult] = []
    for ctx in list(ctxs):
        del_ids = [sp.edit_id for sp in iter_deletion_specs(ctx)]
        _all_done = (not force and del_ids
                     and not obj_needs_stage(ctx, del_ids, "s5b", prereq_map or {}, force=force))
        if _all_done:
            # For GLB path: still run if any edit is missing its GLB (backfill).
            if use_glb and _needs_glb_backfill(ctx):
                pass  # fall through to run_mesh_delete_for_object
            else:
                out.append(DelMeshResult(ctx.obj_id))
                continue
        out.append(run_mesh_delete_for_object(
            ctx, dataset=dataset,
            normalized_glb_dir=normalized_glb_dir,
            anno_dir=anno_dir,
            prereq_map=prereq_map,
            force=force, logger=log,
        ))
    return out


# ─────────────────── s6b: PLY → DINOv2 → SLAT/SS reencode (GPU) ───────

@dataclass
class DelReencodeResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0


def _hardlink(src: Path, dst: Path) -> None:
    """Hard-link src -> dst; fall back to shutil.copy2 on cross-device.

    Unlinks existing dst so re-runs always reflect the current src.
    """
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno == _errno.EXDEV:
            shutil.copy2(src, dst)
        else:
            raise


def _write_before_npz(ctx, slat_dir: Path, before_npz: Path) -> None:
    """Load original SLAT .pt files and save as before.npz."""
    import torch
    import numpy as np
    coords = torch.load(
        slat_dir / ctx.shard / f"{ctx.obj_id}_coords.pt", map_location="cpu"
    ).numpy()
    feats = torch.load(
        slat_dir / ctx.shard / f"{ctx.obj_id}_feats.pt", map_location="cpu"
    ).numpy()
    np.savez(str(before_npz), slat_coords=coords, slat_feats=feats)


def _link_add_pair(
    ctx, spec, add_seq: int, pair_dir: Path, *, logger=None
) -> None:
    """Hardlink del/{after,before}.{npz,png} -> add/{before,after}.{npz,png}."""
    import logging as _l
    log = logger or _l.getLogger("pipeline_v2.s6b")
    add_id = ctx.edit_id("addition", add_seq)
    add_dir = ctx.edit_3d_dir(add_id)
    if not (add_dir / "meta.json").is_file():
        return
    # NPZ: del after -> add before, del before -> add after
    for del_name, add_name in [("after.npz", "before.npz"), ("before.npz", "after.npz")]:
        src = pair_dir / del_name
        if src.is_file():
            _hardlink(src, add_dir / add_name)
    # PNG: del after -> add before, del before -> add after
    for del_name, add_name in [("after.png", "before.png"), ("before.png", "after.png")]:
        src = pair_dir / del_name
        if src.is_file():
            _hardlink(src, add_dir / add_name)


def run_reencode_for_object(
    ctx: ObjectContext,
    *,
    ss_encoder,
    blender_path: str,
    slat_dir: Path,
    work_dir: Path,
    num_views: int = 40,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> DelReencodeResult:
    """For each deletion edit render → encode → write ``after.npz``.

    GLB path (primary): reads ``after_new.glb`` produced by s5b and passes it
    directly to encode_asset's Blender renderer (which supports GLB natively).
    PLY path (fallback): reads ``after.ply`` for legacy pipeline compatibility.
    """
    import numpy as np
    from migrate_slat_to_npz import _render_and_full_encode  # type: ignore

    log = logger or logging.getLogger("pipeline_v2.s6b")
    res = DelReencodeResult(obj_id=ctx.obj_id)
    specs = list(iter_deletion_specs(ctx))
    if not specs:
        update_step(ctx, "s6b_del_reencode", status=STATUS_OK, n=0)
        return res

    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    add_seq = 0
    for spec in specs:
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s6b", prereq_map, force=force):
                res.n_skip += 1
                add_seq += 1
                continue
        elif is_gate_a_failed(ctx, spec.edit_id):
            log.info("[s6b] skip %s (gate_a_fail)", spec.edit_id)
            res.n_skip += 1
            add_seq += 1
            continue
        pair_dir = ctx.edit_3d_dir(spec.edit_id)
        a_glb = pair_dir / "after_new.glb"
        a_ply = pair_dir / "after.ply"
        a_npz = pair_dir / "after.npz"

        # Determine mesh source: GLB preferred (textured, no PLY artefacts).
        if a_glb.is_file():
            mesh_path = a_glb
        elif a_ply.is_file():
            mesh_path = a_ply
        else:
            log.warning("[s6b] %s: no after_new.glb or after.ply", spec.edit_id)
            res.n_fail += 1
            add_seq += 1
            continue

        # Skip if a_npz already has the full encoded payload (per-edit done check).
        if a_npz.is_file() and not force:
            try:
                d = np.load(a_npz)
                if "ss" in d.files and d["slat_feats"].shape[0] > 0 and \
                        not edit_needs_step(ctx, spec.edit_id, "s6b", prereq_map or {}, force=force):
                    res.n_skip += 1
                    add_seq += 1
                    continue
            except Exception:
                pass
        try:
            # encode_asset render() accepts both .glb and .ply via --object arg;
            # the blender_script determines import method from file extension.
            payload = _render_and_full_encode(
                mesh_path, f"after_{spec.edit_id}", work_dir,
                ss_encoder, "cuda",
                num_views=num_views, blender_path=blender_path,
            )
            np.savez(a_npz, **payload)
            before_npz = pair_dir / "before.npz"
            if not before_npz.is_file() or force:
                _write_before_npz(ctx, slat_dir, before_npz)
            _link_add_pair(ctx, spec, add_seq, pair_dir, logger=log)
            res.n_ok += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s6b", status="done")
        except Exception as e:
            log.warning("[s6b] %s: %s", spec.edit_id, e)
            res.n_fail += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s6b",
                              status="error", reason=str(e)[:200])
        add_seq += 1

    update_step(
        ctx, "s6b_del_reencode",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


def run_reencode(
    ctxs: Iterable[ObjectContext],
    *,
    cfg: dict,
    blender_path: str,
    work_dir: Path | None = None,
    num_views: int = 40,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[DelReencodeResult]:
    """Single-GPU entrypoint. Loads only the SS encoder
    (``_render_and_full_encode`` lazily loads the SLAT encoder + DINOv2)."""
    log = logger or logging.getLogger("pipeline_v2.s6b")
    log.info("[s6b] CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_VISIBLE_DEVICES"))

    from partcraft.io.npz_utils import load_ss_encoder
    ss_encoder = load_ss_encoder(Path(cfg.get("ckpt_root", "checkpoints")), "cuda")

    data_cfg = cfg.get("data") or {}
    slat_dir = Path(data_cfg.get("slat_dir", ""))

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="pcv2_s6b_"))
    log.info("[s6b] work_dir=%s", work_dir)

    out: list[DelReencodeResult] = []
    for ctx in list(ctxs):
        del_ids = [sp.edit_id for sp in iter_deletion_specs(ctx)]
        if del_ids and not force and not obj_needs_stage(
            ctx, del_ids, "s6b", prereq_map or {}, force=force
        ):
            out.append(DelReencodeResult(ctx.obj_id))
            continue
        out.append(run_reencode_for_object(
            ctx, ss_encoder=ss_encoder, blender_path=blender_path,
            slat_dir=slat_dir, work_dir=work_dir,
            num_views=num_views, prereq_map=prereq_map,
            force=force, logger=log,
        ))
    return out


__all__ = [
    "DelMeshResult", "DelReencodeResult",
    "_build_deletion_from_npz",
    "run_mesh_delete", "run_mesh_delete_for_object",
    "run_reencode", "run_reencode_for_object",
]
