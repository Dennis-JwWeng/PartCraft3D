"""Step s2 — Highlight rendering for every parsed edit (object-centric).

For each edit in ``ctx.parsed_path`` we re-render the chosen view with
the selected parts in the magenta highlight color and the rest in gray
(byte-identical to the inputs used by the phase 1.5 VLM scoring step,
since both call the same :func:`render_part_highlight.render_highlight`).

Output:
    ctx.highlights_dir / e{idx:02d}.png

Skipped cases:
* edit has no ``selected_part_ids`` (e.g. ``global``) → write the plain
  original view as the "highlight" so the report still has 5 columns
  populated.
* edit failed Gate A (QC-A) → skip; s4/s5 will never consume this edit.
* render failure for one edit → log + continue (does not abort the obj).

The runner is per-object: blender startup is the dominant cost so doing
many edits per object in one runner call lets us amortize that. We do
NOT batch across objects here — keep object isolation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from partcraft.render.highlight import render_highlight
from partcraft.render.overview import VIEW_INDICES, load_views_from_npz

from .paths import ObjectContext
from .qc_io import is_gate_a_failed, load_qc, update_edit_gate
from .specs import iter_all_specs
from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP, step_done

def _count_highlight_pixels(bgr: np.ndarray) -> int:
    """Count highlight-colored pixels in a rendered highlight image (BGR, uint8).

    The nominal HIGHLIGHT color is RGB [230, 40, 200] but Blender's sRGB
    pipeline shifts values to roughly BGR [184, 42, 197] in the saved PNG.
    Rather than hardcode an exact range, we detect "not-gray" pixels:
    background and non-selected parts are white (255,255,255) or near-gray
    (all channels within 30 of each other), whereas highlight pixels have a
    saturated hue with G << B ≈ R.

    Criterion: G < 100  AND  max(B, G, R) - min(B, G, R) > 60
    """
    b = bgr[..., 0].astype(np.int16)
    g = bgr[..., 1].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)
    chroma = (np.maximum(np.maximum(b, g), r)
              - np.minimum(np.minimum(b, g), r))
    mask = (g < 100) & (chroma > 60)
    return int(np.sum(mask))


@dataclass
class HighlightResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip_global: int = 0
    n_invisible: int = 0      # edits where target part has 0 visible pixels in chosen view
    error: str | None = None


def _write_image(path: Path, img) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), img))


def run_one(
    ctx: ObjectContext,
    *,
    blender: str,
    force: bool = False,
) -> HighlightResult:
    """Render one highlight per edit in ``ctx.parsed_path``."""
    if ctx.mesh_npz is None or ctx.image_npz is None:
        raise ValueError(f"{ctx} missing mesh_npz/image_npz")
    if not ctx.parsed_path.is_file():
        update_step(ctx, "s2_highlights", status=STATUS_FAIL,
                    error="missing_parsed_json")
        return HighlightResult(ctx.obj_id, error="missing_parsed_json")

    parsed = json.loads(ctx.parsed_path.read_text())
    edits = (parsed.get("parsed") or {}).get("edits") or []
    if not edits:
        update_step(ctx, "s2_highlights", status=STATUS_SKIP, n=0,
                    reason="no_edits")
        return HighlightResult(ctx.obj_id)

    # Build per-index spec map and gate-A-blocked set in one pass.
    idx_to_spec = {spec.edit_idx: spec for spec in iter_all_specs(ctx)}
    gate_a_blocked: set[int] = {
        idx for idx, spec in idx_to_spec.items()
        if is_gate_a_failed(ctx, spec.edit_id)
    }

    ctx.highlights_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = HighlightResult(ctx.obj_id)

    # cache loaded views once for global / fallback paths
    _orig_views = None

    def _origs():
        nonlocal _orig_views
        if _orig_views is None:
            _orig_views, _ = load_views_from_npz(ctx.image_npz, VIEW_INDICES)
        return _orig_views

    for idx, e in enumerate(edits):
        out_path = ctx.highlight_path(idx)
        if out_path.is_file() and not force:
            res.n_ok += 1
            continue

        if idx in gate_a_blocked:
            continue  # Gate A failed → s4/s5 will never use this highlight

        vi = int(e.get("view_index", 0))
        pids = list(e.get("selected_part_ids") or [])

        try:
            if not pids:
                # global / no-target edits: store the plain view
                if not (0 <= vi < len(VIEW_INDICES)):
                    raise ValueError(f"bad view_index={vi}")
                hl = _origs()[vi]
                res.n_skip_global += 1
            else:
                _, hl = render_highlight(
                    ctx.mesh_npz, ctx.image_npz, vi, pids, blender,
                )
                # Hard rule: if the target part has zero visible pixels from
                # this view, the edit is invalid — s4/s5 would edit blind.
                # Mark Gate A as failed and skip writing the image.
                if _count_highlight_pixels(hl) == 0:
                    spec = idx_to_spec.get(idx)
                    if spec:
                        existing_vlm = (
                            (load_qc(ctx).get("edits") or {})
                            .get(spec.edit_id, {})
                            .get("gates", {})
                            .get("A") or {}
                        ).get("vlm")
                        update_edit_gate(
                            ctx, spec.edit_id, spec.edit_type, "A",
                            rule_result={
                                "pass": False,
                                "checks": {"zero_visible_pixels":
                                    f"part(s) {pids} not visible from "
                                    f"view_index={vi} (frame {VIEW_INDICES[vi]})"},
                            },
                            vlm_result=existing_vlm,
                        )
                        print(f"  [invisible] {ctx.obj_id} e{idx}: "
                              f"part(s) {pids} zero pixels at view {vi} "
                              f"→ Gate A fail")
                    res.n_invisible += 1
                    continue   # do not write a useless all-gray image
            if not _write_image(out_path, hl):
                raise RuntimeError("imwrite failed")
            res.n_ok += 1
        except Exception as ex:
            print(f"  [hl fail] {ctx.obj_id} e{idx}: {ex}")
            res.n_fail += 1

    update_step(
        ctx, "s2_highlights",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n=res.n_ok, n_fail=res.n_fail, n_global=res.n_skip_global,
        n_invisible=res.n_invisible,
        wall_s=round(time.time() - t0, 2),
    )
    return res


def run_many(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    force: bool = False,
) -> list[HighlightResult]:
    """Sequential per-object loop. Blender is GPU/CPU bound and already
    parallel internally; running many objects concurrently risks OOM."""
    out: list[HighlightResult] = []
    for ctx in ctxs:
        if not force and step_done(ctx, "s2_highlights"):
            out.append(HighlightResult(ctx.obj_id))
            continue
        out.append(run_one(ctx, blender=blender, force=force))
    return out


__all__ = ["HighlightResult", "run_one", "run_many"]
