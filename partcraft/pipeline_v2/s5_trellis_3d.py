"""Step s5 — TRELLIS 3D editing for flux types (object-centric).

For each ``ObjectContext`` we run only the **GPU** edit types
(modification / scale / material / global). Deletion has its own
mesh-direct path in s5b; addition is backfilled in s7; identity is
skipped entirely.

Per object the runner does the legacy ``process_object_edits`` GPU
branch with object-centric paths::

    1. dataset.load_object(shard, obj_id)
    2. refiner.encode_object(...)        # SLAT once
    3. refiner.decode_to_gaussian(slat)  # gaussian once
    4. for spec in flux_specs:
         a. resolve_2d_conditioning   ← reads ctx.edits_2d_dir / "{edit_id}_edited.png"
         b. refiner.build_part_mask
         c. refiner.edit
         d. refiner.export_pair        ← writes ctx.edit_3d_dir(edit_id)/{before,after}.npz

The runner is **single-GPU**: it inherits whatever ``CUDA_VISIBLE_DEVICES``
the caller sets. The orchestrator (step 12) is responsible for
splitting a list of contexts across GPUs by spawning subprocess
workers, the same shape as the legacy multi-gpu dispatcher.

``resolve_2d_conditioning`` is hardwired to look at
``cache_dir / edit_dir / "{edit_id}_edited.png"``, so we set
``cache_dir = ctx.dir`` and ``edit_dir = "edits_2d"`` and reuse the
helper unchanged.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.pipeline_v2.s5_3d_utils import resolve_2d_conditioning  # noqa: E402
from .paths import ObjectContext
from . import services_cfg as psvc

from .specs import EditSpec, iter_flux_specs
from .status import update_step, STATUS_OK, STATUS_FAIL, step_done
from .qc_io import is_edit_qc_failed, is_gate_a_failed
from .edit_status_io import edit_needs_step, update_edit_stage


GPU_TYPES = frozenset({"modification", "scale", "material", "global"})


@dataclass
class Trellis3DResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0
    error: str | None = None


# ─────────────────── per-object processing ───────────────────────────

def _ensure_refiner(p25_cfg: dict, ckpt_root: str | None,
                    slat_dir: str | None, img_enc_dir: str | None,
                    debug: bool, logger):
    from partcraft.trellis.refiner import TrellisRefiner
    refiner = TrellisRefiner(
        cache_dir=str(Path(p25_cfg.get("cache_dir", "/tmp/pcv2_trellis"))),
        device="cuda",
        image_edit_model=p25_cfg.get("image_edit_model", ""),
        ckpt_dir=ckpt_root,
        image_edit_backend=p25_cfg.get("image_edit_backend", "local_diffusers"),
        image_edit_base_url=str(p25_cfg.get("image_edit_base_url", "")),
        debug=debug,
        slat_dir=slat_dir,
        img_enc_dir=img_enc_dir,
    )
    refiner.load_models()
    logger.info("[s5] TRELLIS models loaded")
    return refiner




def run_for_object(
    ctx: ObjectContext,
    *,
    refiner,
    dataset,
    p25_cfg: dict,
    seed: int = 1,
    use_2d: bool = True,
    debug: bool = False,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> Trellis3DResult:
    """Edit all GPU specs for one object using a pre-loaded ``refiner``."""
    from partcraft.trellis.refiner import build_prompts_from_spec

    log = logger or logging.getLogger("pipeline_v2.s5")
    res = Trellis3DResult(obj_id=ctx.obj_id)

    all_specs = []
    pending: list[EditSpec] = []
    for spec in iter_flux_specs(ctx):
        all_specs.append(spec)
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s5", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, spec.edit_id):
                log.debug("[s5] skip %s (gate_a_fail)", spec.edit_id)
                res.n_skip += 1
                continue
            before = ctx.edit_3d_npz(spec.edit_id, "before")
            after = ctx.edit_3d_npz(spec.edit_id, "after")
            if before.is_file() and after.is_file() and not force:
                res.n_skip += 1
                continue
        pending.append(spec)

    if not all_specs:
        update_step(ctx, "s5_trellis", status=STATUS_OK, n=0, reason="no_specs")
        return res
    if not pending:
        update_step(ctx, "s5_trellis", status=STATUS_OK,
                    n=res.n_skip, n_skip=res.n_skip)
        return res

    try:
        obj_record = dataset.load_object(ctx.shard, ctx.obj_id)
    except Exception as e:
        log.error("[s5] %s load_object failed: %s", ctx.obj_id, e)
        update_step(ctx, "s5_trellis", status=STATUS_FAIL,
                    error=f"load_object: {e}")
        res.error = str(e); res.n_fail = len(pending); return res

    try:
        ori_slat = refiner.encode_object(None, ctx.obj_id)
        ori_gaussian = refiner.decode_to_gaussian(ori_slat)
    except Exception as e:
        log.error("[s5] %s encode/decode failed: %s", ctx.obj_id, e)
        obj_record.close()
        update_step(ctx, "s5_trellis", status=STATUS_FAIL,
                    error=f"encode_decode: {e}")
        res.error = str(e); res.n_fail = len(pending); return res

    large_part_threshold = float(p25_cfg.get("large_part_threshold", 0.35))
    promote_scale_to_global = bool(p25_cfg.get("promote_scale_to_global", False))
    scale_large = p25_cfg.get("scale_large_part_threshold")
    scale_large = float(scale_large) if scale_large is not None else None
    repaint_mode = str(p25_cfg.get("repaint_mode", "interleaved"))

    t0 = time.time()
    for spec in pending:
        et_cap = spec.edit_type.capitalize()
        if et_cap in ("Modification", "Scale"):
            edit_part_ids = (spec.selected_part_ids
                             or ([spec.selected_part_ids[0]] if spec.selected_part_ids else []))
        elif et_cap == "Material":
            edit_part_ids = [spec.selected_part_ids[0]] if spec.selected_part_ids else []
        elif et_cap == "Global":
            edit_part_ids = []
        else:
            res.n_fail += 1
            continue

        try:
            mask, effective_type = refiner.build_part_mask(
                ctx.obj_id, obj_record, edit_part_ids, ori_slat, et_cap,
                large_part_threshold=large_part_threshold,
                promote_scale_to_global=promote_scale_to_global,
                scale_large_part_threshold=scale_large,
            )
            if mask.sum() == 0:
                log.warning("[s5] %s empty mask", spec.edit_id)
                res.n_fail += 1
                continue

            prompts = build_prompts_from_spec(spec, override_type=effective_type)

            img_cond = resolve_2d_conditioning(
                spec=spec,
                obj_id=ctx.obj_id,
                obj_record=obj_record,
                ori_gaussian=ori_gaussian,
                refiner=refiner,
                vlm_client=None,
                p25_cfg=p25_cfg,
                cache_dir=ctx.dir,
                edit_dir="edits_2d",
                cache_only_2d=True,   # never call edit server here; s4 already did
                use_2d=use_2d,
                image_edit_backend=p25_cfg.get("image_edit_backend", "local_diffusers"),
                logger=log,
                prompts=prompts,
            )

            edit_results = refiner.edit(
                ori_slat, mask, prompts,
                img_cond=img_cond, seed=seed, combinations=None,
                repaint_mode=repaint_mode,
            )
            if not edit_results:
                raise RuntimeError("no edited SLATs")
            best = edit_results[0]

            pair_dir = ctx.edit_3d_dir(spec.edit_id)
            pair_dir.mkdir(parents=True, exist_ok=True)
            refiner.export_pair(
                ori_slat, best["slat"], pair_dir,
                z_s_before=best.get("z_s_before"),
                z_s_after=best.get("z_s_after"),
            )
            res.n_ok += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5", status="done")
            log.info("[s5] %s ok", spec.edit_id)
        except Exception as e:
            log.error("[s5] %s failed: %s", spec.edit_id, e)
            res.n_fail += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5",
                              status="error", reason=str(e)[:200])

    obj_record.close()
    update_step(
        ctx, "s5_trellis",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


# ─────────────────── batch entrypoint (single GPU) ───────────────────

def run(
    ctxs: Iterable[ObjectContext],
    *,
    cfg: dict,
    images_root: Path,
    mesh_root: Path,
    shard: str = "01",
    seed: int = 1,
    debug: bool = False,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[Trellis3DResult]:
    """Sequentially process objects on whatever GPU is currently visible.

    The orchestrator (step 12) calls this once per GPU subprocess with
    a sliced list of contexts and a per-process ``CUDA_VISIBLE_DEVICES``.
    """
    log = logger or logging.getLogger("pipeline_v2.s5")
    log.info("[s5] CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))

    p25_cfg = psvc.trellis_image_edit_flat(cfg)
    ckpt_root = cfg.get("ckpt_root")
    data_cfg = cfg.get("data") or {}
    slat_dir = data_cfg.get("slat_dir")
    img_enc_dir = data_cfg.get("img_enc_dir")

    from partcraft.io.hy3d_loader import HY3DPartDataset
    dataset = HY3DPartDataset(str(images_root), str(mesh_root), [shard])

    refiner = _ensure_refiner(p25_cfg, ckpt_root, slat_dir, img_enc_dir,
                              debug, log)

    results: list[Trellis3DResult] = []
    for ctx in list(ctxs):
        if not force and step_done(ctx, "s5_trellis"):
            results.append(Trellis3DResult(ctx.obj_id))
            continue
        results.append(run_for_object(
            ctx, refiner=refiner, dataset=dataset, p25_cfg=p25_cfg,
            seed=seed, use_2d=True, debug=debug,
            prereq_map=prereq_map, force=force, logger=log,
        ))
    return results


__all__ = ["GPU_TYPES", "Trellis3DResult", "run_for_object", "run"]
