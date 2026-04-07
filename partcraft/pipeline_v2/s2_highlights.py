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
* render failure for one edit → log + continue (does not abort the obj).

The runner is per-object: blender startup is the dominant cost so doing
many edits per object in one runner call lets us amortize that. We do
NOT batch across objects here — keep object isolation.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "tools"))
from render_part_highlight import render_highlight  # noqa: E402
from render_part_overview import (  # noqa: E402
    VIEW_INDICES, load_views_from_npz,
)

from .paths import ObjectContext
from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP, step_done


@dataclass
class HighlightResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip_global: int = 0
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
