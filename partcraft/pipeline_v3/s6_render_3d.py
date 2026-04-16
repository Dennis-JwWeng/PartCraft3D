"""Step s6 — Re-render the s5 SLAT pairs at the original phase1 view.

For every edit that has ``edits_3d/<edit_id>/{before,after}.npz`` we
decode the SLAT to a Gaussian and render exactly one frame at the
camera matching ``edit.view_index`` (i.e. the same view phase1_v2 used
to pick the part). The rendered images go next to the npz::

    edits_3d/<edit_id>/before.png
    edits_3d/<edit_id>/after.png

This makes the report (s8) able to show "3D before / 3D after" at the
same camera as the original / highlight / FLUX edit columns.

The runner is single-GPU and re-uses the legacy
``frame_to_extrinsic_intrinsic`` + ``render_one_view`` helpers from
[scripts/standalone/render_phase1v2_3d_results.py](../../scripts/standalone/render_phase1v2_3d_results.py).
``ATTN_BACKEND=xformers`` should be set before importing trellis if
flash-attn is unavailable in the active env (vinedresser3d).
"""
from __future__ import annotations

import json
import logging
import os
import shutil as _shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "standalone"))
sys.path.insert(0, str(_ROOT / "third_party"))

from .paths import ObjectContext
from .specs import VIEW_INDICES
from .status import update_step, STATUS_OK, STATUS_FAIL
from .edit_status_io import edit_needs_step, update_edit_stage, obj_needs_stage


@dataclass
class Render3DResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0
    error: str | None = None



def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src -> dst; fall back to copy on cross-device."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        _shutil.copy2(src, dst)


def _lazy_helpers():
    """Import the trellis-dependent helpers on first use."""
    from render_phase1v2_3d_results import (  # type: ignore
        frame_to_extrinsic_intrinsic, render_one_view, load_slat,
    )
    from partcraft.render.overview import load_views_from_npz
    return frame_to_extrinsic_intrinsic, render_one_view, load_slat, load_views_from_npz


def _build_pipeline(ckpt: str, logger: logging.Logger):
    from trellis.pipelines import TrellisTextTo3DPipeline
    logger.info("[s6] loading TRELLIS %s", ckpt)
    pipe = TrellisTextTo3DPipeline.from_pretrained(ckpt)
    pipe.cuda()
    logger.info("[s6] pipeline ready")
    return pipe


# ─────────────────── per-object processing ───────────────────────────

def run_for_object(
    ctx: ObjectContext,
    *,
    pipeline,
    resolution: int = 518,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> Render3DResult:
    """Decode + render every present SLAT pair under ``ctx.edits_3d_dir``."""
    log = logger or logging.getLogger("pipeline_v3.s6")
    res = Render3DResult(obj_id=ctx.obj_id)

    if ctx.image_npz is None or not ctx.image_npz.is_file():
        update_step(ctx, "s6_render_3d", status=STATUS_FAIL,
                    error="missing_image_npz")
        res.error = "missing_image_npz"; return res
    if not ctx.parsed_path.is_file():
        update_step(ctx, "s6_render_3d", status=STATUS_FAIL,
                    error="missing_parsed_json")
        res.error = "missing_parsed_json"; return res

    _, _, load_slat, load_views_from_npz = _lazy_helpers()
    from render_phase1v2_3d_results import render_one_view  # type: ignore

    _, frames = load_views_from_npz(ctx.image_npz, VIEW_INDICES)

    parsed = (json.loads(ctx.parsed_path.read_text())
              .get("parsed") or {}).get("edits") or []

    # Map edit_idx → view_index from parsed.json (authoritative).
    # We then walk the on-disk edit_3d dirs and look up by edit_id.
    # The edit_id format encodes the type so we can locate parsed entry by
    # rebuilding seq numbers (mirrors specs.py).
    from .specs import iter_all_specs
    spec_by_edit_id = {s.edit_id: s for s in iter_all_specs(ctx)}

    if not ctx.edits_3d_dir.is_dir():
        update_step(ctx, "s6_render_3d", status=STATUS_OK, n=0,
                    reason="no_edits_3d")
        return res

    t0 = time.time()
    for edit_dir in sorted(ctx.edits_3d_dir.iterdir()):
        if not edit_dir.is_dir():
            continue
        edit_id = edit_dir.name
        spec = spec_by_edit_id.get(edit_id)
        if spec is None:
            log.warning("[s6] %s: no spec for %s", ctx.obj_id, edit_id)
            continue
        # gate_e enforcement (D3): skip edits that failed final quality gate
        if prereq_map is not None:
            if not edit_needs_step(ctx, edit_id, "s6", prereq_map, force=force):
                res.n_skip += 1
                continue
        if not (0 <= spec.view_index < len(frames)):
            res.n_fail += 1
            continue
        frame = frames[spec.view_index]

        _edit_ok = True
        for which in ("before", "after"):
            npz = edit_dir / f"{which}.npz"
            png = edit_dir / f"{which}.png"
            if not npz.is_file():
                continue
            if png.is_file() and not force:
                res.n_skip += 1
                continue
            if which == "after":
                preview = edit_dir / f"preview_{spec.view_index}.png"
                if preview.is_file():
                    try:
                        _hardlink_or_copy(preview, png)
                        res.n_ok += 1
                    except Exception as e:
                        log.warning("[s6] %s/after preview link: %s", edit_id, e)
                        res.n_fail += 1
                    continue
            try:
                slat = load_slat(npz)
                rgb = render_one_view(pipeline, slat, frame, resolution)
                Image.fromarray(rgb).save(str(png))
                res.n_ok += 1
            except Exception as e:
                log.warning("[s6] %s/%s: %s", edit_id, which, e)
                res.n_fail += 1
                _edit_ok = False

        if _edit_ok:
            update_edit_stage(ctx, edit_id, spec.edit_type, "s6", status="done")
        else:
            update_edit_stage(ctx, edit_id, spec.edit_type, "s6",
                              status="error", reason="render_failed")

    update_step(
        ctx, "s6_render_3d",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


# ─────────────────── batch entrypoint ─────────────────────────────────

def run(
    ctxs: Iterable[ObjectContext],
    *,
    ckpt: str = "checkpoints/TRELLIS-text-xlarge",
    resolution: int = 518,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[Render3DResult]:
    """Sequentially render all objects on the current GPU."""
    log = logger or logging.getLogger("pipeline_v3.s6")
    log.info("[s6] CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_VISIBLE_DEVICES"))
    pipeline = _build_pipeline(ckpt, log)

    results: list[Render3DResult] = []
    from .specs import iter_flux_specs as _iter_flux_s6
    for ctx in list(ctxs):
        flux_ids = [sp.edit_id for sp in _iter_flux_s6(ctx)]
        if flux_ids and not force and not obj_needs_stage(
            ctx, flux_ids, "s6", prereq_map or {}, force=force
        ):
            results.append(Render3DResult(ctx.obj_id))
            continue
        results.append(run_for_object(
            ctx, pipeline=pipeline, resolution=resolution,
            prereq_map=prereq_map, force=force, logger=log,
        ))
    return results


__all__ = ["Render3DResult", "run_for_object", "run"]
