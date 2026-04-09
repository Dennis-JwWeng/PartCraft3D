"""Step s5b / s6b — Deletion mesh-direct delete + PLY → SLAT/SS re-encode.

Two halves of the deletion path, kept in one module because they share
the same per-object loop and ``edits_3d/<edit_id>/`` output dir:

* :func:`run_mesh_delete` (s5b) — CPU only. For every deletion spec
  assemble the GT meshes minus the removed parts → ``before.ply`` /
  ``after.ply``. Also writes a quick mask-based ``before.npz`` /
  ``after.npz`` so the dataset can already use the pair while we wait
  for the proper Phase 5 re-encode.

* :func:`run_reencode` (s6b) — GPU. For every deletion edit whose
  ``after.ply`` exists, render 40 views via Blender → DINOv2 voxel
  features → SLAT encoder → SS encoder → overwrite ``after.npz`` with
  the proper full re-encode (matches the legacy
  ``migrate_slat_to_npz.py`` Phase 5).

The two phases write distinct status entries (``s5b_del_mesh`` and
``s6b_del_reencode``) so the orchestrator can resume / retry them
independently.
"""
from __future__ import annotations

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
from .status import update_step, STATUS_OK, STATUS_FAIL, step_done
from .s5_trellis_3d import _ensure_refiner


# ─────────────────── s5b: mesh-direct delete (CPU) ────────────────────

@dataclass
class DelMeshResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0


def run_mesh_delete_for_object(
    ctx: ObjectContext,
    *,
    refiner,        # may be None — only needed for the mask-based npz
    dataset,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> DelMeshResult:
    from partcraft.trellis.refiner import TrellisRefiner
    log = logger or logging.getLogger("pipeline_v2.s5b")
    res = DelMeshResult(obj_id=ctx.obj_id)

    specs = list(iter_deletion_specs(ctx))
    if not specs:
        update_step(ctx, "s5b_del_mesh", status=STATUS_OK, n=0,
                    reason="no_deletions")
        return res

    try:
        obj_record = dataset.load_object(ctx.shard, ctx.obj_id)
    except Exception as e:
        log.error("[s5b] %s load failed: %s", ctx.obj_id, e)
        update_step(ctx, "s5b_del_mesh", status=STATUS_FAIL, error=str(e))
        res.n_fail = len(specs); return res

    # SLAT once for the cheap mask-based after.npz (overwritten in s6b).
    ori_slat = None
    if refiner is not None:
        try:
            ori_slat = refiner.encode_object(None, ctx.obj_id)
        except Exception as e:
            log.warning("[s5b] %s SLAT preload failed: %s "
                        "(PLY pair will still be written)", ctx.obj_id, e)

    for spec in specs:
        pair_dir = ctx.edit_3d_dir(spec.edit_id)
        a_ply = pair_dir / "after.ply"
        if a_ply.is_file() and not force:
            res.n_skip += 1
            continue
        try:
            pair_dir.mkdir(parents=True, exist_ok=True)
            TrellisRefiner.direct_delete_mesh(
                obj_record, spec.selected_part_ids, pair_dir, export_ply=True,
            )
            # Rough SLAT pair (s6b will overwrite after.npz).
            if refiner is not None and ori_slat is not None:
                try:
                    mask, _ = refiner.build_part_mask(
                        ctx.obj_id, obj_record, spec.selected_part_ids,
                        ori_slat, "Deletion",
                    )
                    refiner.export_deletion_pair(ori_slat, mask, pair_dir)
                except Exception as e:
                    log.warning("[s5b] %s mask npz failed: %s", spec.edit_id, e)
            res.n_ok += 1
        except Exception as e:
            log.error("[s5b] %s failed: %s", spec.edit_id, e)
            res.n_fail += 1

    obj_record.close()
    update_step(
        ctx, "s5b_del_mesh",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
    )
    return res


def run_mesh_delete(
    ctxs: Iterable[ObjectContext],
    *,
    cfg: dict,
    images_root: Path,
    mesh_root: Path,
    shard: str = "01",
    force: bool = False,
    use_refiner: bool = True,
    logger: logging.Logger | None = None,
) -> list[DelMeshResult]:
    """Sequential per-object loop. ``use_refiner=False`` skips the
    rough mask-based npz (PLY-only mode for pure CPU machines)."""
    log = logger or logging.getLogger("pipeline_v2.s5b")

    from partcraft.io.hy3d_loader import HY3DPartDataset
    dataset = HY3DPartDataset(str(images_root), str(mesh_root), [shard])

    refiner = None
    if use_refiner:
        p25 = cfg.get("phase2_5") or {}
        data_cfg = cfg.get("data") or {}
        refiner = _ensure_refiner(
            p25, cfg.get("ckpt_root"),
            data_cfg.get("slat_dir"), data_cfg.get("img_enc_dir"),
            False, log,
        )

    out: list[DelMeshResult] = []
    for ctx in list(ctxs):
        if not force and step_done(ctx, "s5b_del_mesh"):
            out.append(DelMeshResult(ctx.obj_id))
            continue
        out.append(run_mesh_delete_for_object(
            ctx, refiner=refiner, dataset=dataset, force=force, logger=log,
        ))
    return out


# ─────────────────── s6b: PLY → DINOv2 → SLAT/SS reencode (GPU) ───────

@dataclass
class DelReencodeResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0


def run_reencode_for_object(
    ctx: ObjectContext,
    *,
    ss_encoder,
    blender_path: str,
    work_dir: Path,
    num_views: int = 40,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> DelReencodeResult:
    """For each deletion ``after.ply`` in this object, render → encode
    → overwrite ``after.npz``. Reuses the legacy phase5 helper."""
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
    for spec in specs:
        pair_dir = ctx.edit_3d_dir(spec.edit_id)
        a_ply = pair_dir / "after.ply"
        a_npz = pair_dir / "after.npz"
        if not a_ply.is_file():
            res.n_fail += 1
            continue
        # Skip if a_npz already has the full DINOv2-encoded payload.
        # Heuristic: full reencode produces a `dino_voxel_mean` field; the
        # legacy export_deletion_pair does not. Cheaper: just skip when
        # status entry says ok.
        if a_npz.is_file() and not force:
            try:
                d = np.load(a_npz)
                if "ss" in d.files and d["slat_feats"].shape[0] > 0 and \
                        step_done(ctx, "s6b_del_reencode"):
                    res.n_skip += 1
                    continue
            except Exception:
                pass
        try:
            payload = _render_and_full_encode(
                a_ply, f"after_{spec.edit_id}", work_dir,
                ss_encoder, "cuda",
                num_views=num_views, blender_path=blender_path,
            )
            np.savez(a_npz, **payload)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6b] %s: %s", spec.edit_id, e)
            res.n_fail += 1

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
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[DelReencodeResult]:
    """Single-GPU entrypoint. Loads only the SS encoder
    (``_render_and_full_encode`` lazily loads the SLAT encoder + DINOv2)."""
    log = logger or logging.getLogger("pipeline_v2.s6b")
    log.info("[s6b] CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_VISIBLE_DEVICES"))

    from migrate_slat_to_npz import load_ss_encoder  # type: ignore
    ss_encoder = load_ss_encoder(Path(cfg.get("ckpt_root", "checkpoints")), "cuda")

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="pcv2_s6b_"))
    log.info("[s6b] work_dir=%s", work_dir)

    out: list[DelReencodeResult] = []
    for ctx in list(ctxs):
        if not force and step_done(ctx, "s6b_del_reencode"):
            out.append(DelReencodeResult(ctx.obj_id))
            continue
        out.append(run_reencode_for_object(
            ctx, ss_encoder=ss_encoder, blender_path=blender_path,
            work_dir=work_dir, num_views=num_views, force=force, logger=log,
        ))
    return out


__all__ = [
    "DelMeshResult", "DelReencodeResult",
    "run_mesh_delete", "run_mesh_delete_for_object",
    "run_reencode", "run_reencode_for_object",
]
