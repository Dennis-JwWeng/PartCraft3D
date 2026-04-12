"""Step s6p — 5-view preview renders for all edit types (pre-VLM-gate).

Renders preview_{0..4}.png for every non-identity edit using the same
VIEW_INDICES cameras as the phase1 VLM overview.  These previews are
consumed by sq3 (VLM quality gate) without needing a full 40-view render.

Route by type:
  deletion   → Blender renders after.ply (after state = object minus parts)
  addition   → Blender renders source_del's before.ply (after state = original)
  mod/scl/mat/glb → TRELLIS decode+render from after.npz

Output per edit: edits_3d/<edit_id>/preview_{0..4}.png
Step key: s6p_preview
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "standalone"))
sys.path.insert(0, str(_ROOT / "third_party"))

from .paths import ObjectContext
from .specs import VIEW_INDICES
from .status import update_step, STATUS_OK, STATUS_FAIL, step_done
from .qc_io import is_gate_a_failed


@dataclass
class PreviewResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0
    error: str | None = None


def _build_pipeline(ckpt: str, logger: logging.Logger):
    """Load TRELLIS pipeline onto GPU."""
    from trellis.pipelines import TrellisTextTo3DPipeline  # type: ignore
    logger.info("[s6p] loading TRELLIS %s", ckpt)
    pipe = TrellisTextTo3DPipeline.from_pretrained(ckpt)
    pipe.cuda()
    logger.info("[s6p] pipeline ready")
    return pipe


def _previews_exist(edit_dir: Path, n: int = 5) -> bool:
    """Return True if all preview_{0..n-1}.png files exist."""
    return all((edit_dir / f"preview_{i}.png").is_file() for i in range(n))


def _save_previews(edit_dir: Path, imgs: list[np.ndarray]) -> None:
    """Save list of RGB images as preview_{0..}.png."""
    for i, img in enumerate(imgs):
        Image.fromarray(img).save(str(edit_dir / f"preview_{i}.png"))


def _render_ply_views(
    ply_path: Path,
    frames: list[dict],
    blender: str,
    resolution: int,
) -> list[np.ndarray]:
    """Render a single PLY file at the given camera frames using Blender.

    Uses a temporary directory with a copy of the PLY as part_0.ply to satisfy
    run_blender's expected parts_dir layout (part_*.ply convention).
    """
    from partcraft.render.overview import run_blender as _run_blender
    with tempfile.TemporaryDirectory(prefix="pcv2_s6p_ply_") as tmp:
        tmp_path = Path(tmp)
        # run_blender expects part_*.ply files in parts_dir
        shutil.copy2(ply_path, tmp_path / "part_0.ply")
        # use_vertex_colors=True: detects vertex color attribute name
        # dynamically (Blender 4.x compatible) and renders with Principled
        # BSDF + 3-point lighting, matching the dataset prerender quality.
        imgs = _run_blender(
            tmp_path, blender, resolution,
            [[128, 128, 128]],   # palette unused in vertex-color mode
            frames,
            use_vertex_colors=True,
        )
    return imgs


def _render_slat_views(
    npz_path: Path,
    pipeline,
    frames: list[dict],
    resolution: int,
) -> list[np.ndarray]:
    """Render 5 views from a SLAT npz using TRELLIS pipeline."""
    from render_phase1v2_3d_results import render_one_view as _render_one_view, load_slat as _load_slat  # type: ignore
    slat = _load_slat(npz_path)
    imgs = []
    for frame in frames:
        img = _render_one_view(pipeline, slat, frame, resolution)
        imgs.append(img)
    return imgs


def _iter_add_edits(ctx: ObjectContext):
    """Yield (edit_id, meta_dict) for addition edits discovered from meta.json on disk."""
    if not ctx.edits_3d_dir.is_dir():
        return
    for add_dir in sorted(ctx.edits_3d_dir.iterdir()):
        if not add_dir.is_dir():
            continue
        meta_path = add_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("edit_type") == "addition":
            yield add_dir.name, meta


def run_for_object(
    ctx: ObjectContext,
    *,
    pipeline,          # TRELLIS pipeline (None if no SLAT edits present)
    blender: str,
    resolution: int = 518,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> PreviewResult:
    """Render preview_{0..4}.png for every non-identity edit in this object."""
    log = logger or logging.getLogger("pipeline_v2.s6p")
    res = PreviewResult(obj_id=ctx.obj_id)

    if ctx.image_npz is None or not ctx.image_npz.is_file():
        update_step(ctx, "s6p_preview", status=STATUS_FAIL,
                    error="missing_image_npz")
        res.error = "missing_image_npz"
        return res

    from partcraft.render.overview import load_views_from_npz
    _, frames = load_views_from_npz(ctx.image_npz, VIEW_INDICES)

    if not ctx.edits_3d_dir.is_dir():
        update_step(ctx, "s6p_preview", status=STATUS_OK, n=0,
                    reason="no_edits_3d")
        return res

    t0 = time.time()

    # --- deletion edits ---
    from .specs import iter_deletion_specs
    for spec in iter_deletion_specs(ctx):
        if is_gate_a_failed(ctx, spec.edit_id):
            res.n_skip += 1
            continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        if _previews_exist(edit_dir) and not force:
            res.n_skip += 1
            continue
        a_ply = edit_dir / "after.ply"
        if not a_ply.is_file():
            log.warning("[s6p] del %s: after.ply missing", spec.edit_id)
            res.n_fail += 1
            continue
        try:
            imgs = _render_ply_views(a_ply, frames, blender, resolution)
            _save_previews(edit_dir, imgs)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] del %s: %s", spec.edit_id, e)
            res.n_fail += 1

    # --- addition edits (use source_del's before.ply as after state) ---
    for add_id, meta in _iter_add_edits(ctx):
        if is_gate_a_failed(ctx, add_id):
            res.n_skip += 1
            continue
        add_dir = ctx.edit_3d_dir(add_id)
        if _previews_exist(add_dir) and not force:
            res.n_skip += 1
            continue
        source_del_id = meta.get("source_del_id")
        if not source_del_id:
            log.warning("[s6p] add %s: no source_del_id in meta", add_id)
            res.n_fail += 1
            continue
        before_ply = ctx.edit_3d_dir(source_del_id) / "before.ply"
        if not before_ply.is_file():
            log.warning("[s6p] add %s: source before.ply missing", add_id)
            res.n_fail += 1
            continue
        try:
            imgs = _render_ply_views(before_ply, frames, blender, resolution)
            _save_previews(add_dir, imgs)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] add %s: %s", add_id, e)
            res.n_fail += 1

    # --- SLAT-based edits (mod, scl, mat, glb) ---
    from .specs import iter_all_specs
    PLY_TYPES = {"deletion", "addition", "identity"}
    for spec in iter_all_specs(ctx):
        if spec.edit_type in PLY_TYPES:
            continue
        if is_gate_a_failed(ctx, spec.edit_id):
            res.n_skip += 1
            continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        if _previews_exist(edit_dir) and not force:
            res.n_skip += 1
            continue
        a_npz = edit_dir / "after.npz"
        if not a_npz.is_file():
            log.warning("[s6p] %s %s: after.npz missing", spec.edit_type, spec.edit_id)
            res.n_fail += 1
            continue
        if pipeline is None:
            log.error("[s6p] TRELLIS pipeline not loaded but needed for %s", spec.edit_id)
            res.n_fail += 1
            continue
        try:
            imgs = _render_slat_views(a_npz, pipeline, frames, resolution)
            _save_previews(edit_dir, imgs)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] %s %s: %s", spec.edit_type, spec.edit_id, e)
            res.n_fail += 1

    update_step(
        ctx, "s6p_preview",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


def _has_trellis_edits(ctxs: list[ObjectContext]) -> bool:
    """Check whether any object has SLAT-based (non-PLY) edits needing TRELLIS."""
    from .specs import iter_all_specs
    PLY_TYPES = {"deletion", "addition", "identity"}
    for ctx in ctxs:
        for spec in iter_all_specs(ctx):
            if spec.edit_type not in PLY_TYPES:
                a_npz = ctx.edit_3d_dir(spec.edit_id) / "after.npz"
                if a_npz.is_file():
                    return True
    return False


def run(
    ctxs: Iterable[ObjectContext],
    *,
    ckpt: str = "checkpoints/TRELLIS-text-xlarge",
    blender: str = "blender",
    resolution: int = 518,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[PreviewResult]:
    """Batch entry point. Lazily loads TRELLIS only if SLAT edits are present."""
    log = logger or logging.getLogger("pipeline_v2.s6p")
    log.info("[s6p] CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_VISIBLE_DEVICES"))

    ctx_list = list(ctxs)
    # Filter out already-done objects
    pending = [c for c in ctx_list
               if force or not step_done(c, "s6p_preview")]
    pending_set = set(pending)
    done = [c for c in ctx_list if c not in pending_set]

    pipeline = None
    if _has_trellis_edits(pending):
        pipeline = _build_pipeline(ckpt, log)

    results: list[PreviewResult] = [PreviewResult(c.obj_id) for c in done]
    for ctx in pending:
        results.append(run_for_object(
            ctx, pipeline=pipeline, blender=blender,
            resolution=resolution, force=force, logger=log,
        ))
    return results


__all__ = ["PreviewResult", "run_for_object", "run"]
