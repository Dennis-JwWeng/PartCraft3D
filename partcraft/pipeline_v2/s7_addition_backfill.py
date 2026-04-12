"""Step s7 — Addition backfill from deletion (object-centric).

addition is **not** produced by the VLM and not run through FLUX /
TRELLIS. For every successful deletion edit (``before.npz`` and
``after.npz`` both present) we materialize an ``add_<obj>_<seq>``
sibling whose ``before/after`` are the deletion's swapped pair —
hardlinked, not copied — plus a small ``meta.json`` describing the
inverse prompt.

This mirrors the legacy ``migrate_slat_to_npz.py`` Phase 3 logic but
uses a per-object ``add`` sequence and the new object-centric layout.

Result layout::

    edits_3d/del_<obj>_000/{before,after}.{npz,ply,png}
    edits_3d/add_<obj>_000/
        before.npz → ../del_<obj>_000/after.npz   (hardlink)
        after.npz  → ../del_<obj>_000/before.npz  (hardlink)
        before.png → ../del_<obj>_000/after.png   (hardlink, if present)
        after.png  → ../del_<obj>_000/before.png  (hardlink, if present)
        meta.json   (source_del_id, edit_id, prompt, target_part_desc, ...)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import ObjectContext
from .specs import iter_deletion_specs
from .status import update_step, STATUS_OK, STATUS_FAIL, step_done


# ─────────────────── prompt inversion ────────────────────────────────

# Verb-level rule rewrites. Order matters — longer matches first.
_INVERT_VERBS: list[tuple[str, str]] = [
    ("delete", "add"),
    ("remove", "add"),
    ("get rid of", "add"),
    ("take away", "add"),
    ("strip", "add"),
    ("erase", "add"),
]


def invert_delete_prompt(prompt: str) -> str:
    """Convert a deletion imperative into an addition imperative.

    Pure rule-based; the goal is "good enough for training labels". The
    add path can later be replaced with a VLM rewrite if needed.
    """
    if not prompt:
        return prompt
    p = prompt.strip()
    low = p.lower()
    for old, new in _INVERT_VERBS:
        if low.startswith(old):
            return new.capitalize() + p[len(old):]
        # also catch "Please remove ..." style
        idx = low.find(" " + old + " ")
        if idx >= 0:
            return p[:idx + 1] + new + p[idx + 1 + len(old):]
    # Fallback: prepend "Add back"
    return "Add back " + p[0].lower() + p[1:]


# ─────────────────── core ─────────────────────────────────────────────

@dataclass
class AdditionBackfillResult:
    obj_id: str
    n_ok: int = 0
    n_skip: int = 0
    n_fail: int = 0


def _hardlink(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        # Cross-fs fallback: copy
        import shutil
        shutil.copy2(src, dst)
    return True


def run_for_object(
    ctx: ObjectContext,
    *,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> AdditionBackfillResult:
    log = logger or logging.getLogger("pipeline_v2.s7")
    res = AdditionBackfillResult(obj_id=ctx.obj_id)

    del_specs = list(iter_deletion_specs(ctx))
    if not del_specs:
        update_step(ctx, "s7_add_backfill", status=STATUS_OK, n=0,
                    reason="no_deletions")
        return res

    add_seq = 0
    for spec in del_specs:
        del_dir = ctx.edit_3d_dir(spec.edit_id)
        d_before = del_dir / "before.npz"
        d_after = del_dir / "after.npz"
        if not (d_before.is_file() and d_after.is_file()):
            res.n_skip += 1
            continue

        add_id = ctx.edit_id("addition", add_seq)
        add_seq += 1
        add_dir = ctx.edit_3d_dir(add_id)
        a_before = add_dir / "before.npz"
        a_after = add_dir / "after.npz"
        meta_path = add_dir / "meta.json"

        if (a_before.is_file() and a_after.is_file()
                and meta_path.is_file() and not force):
            res.n_skip += 1
            continue

        try:
            add_dir.mkdir(parents=True, exist_ok=True)
            # swap before / after
            _hardlink(d_after, a_before)
            _hardlink(d_before, a_after)
            # mirror png pair if present
            _hardlink(del_dir / "after.png", add_dir / "before.png")
            _hardlink(del_dir / "before.png", add_dir / "after.png")

            inv_prompt = invert_delete_prompt(spec.prompt)
            meta_path.write_text(json.dumps({
                "edit_id":          add_id,
                "edit_type":        "addition",
                "obj_id":           ctx.obj_id,
                "shard":            ctx.shard,
                "source_del_id":    spec.edit_id,
                "selected_part_ids": list(spec.selected_part_ids),
                "view_index":       spec.view_index,
                "prompt":           inv_prompt,
                "target_part_desc": spec.target_part_desc,
                "rationale":        f"inverse of {spec.edit_id}",
            }, ensure_ascii=False, indent=2))
            res.n_ok += 1
        except Exception as e:
            log.warning("[s7] %s ← %s: %s", add_id, spec.edit_id, e)
            res.n_fail += 1

    update_step(
        ctx, "s7_add_backfill",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_skip=res.n_skip, n_fail=res.n_fail,
    )
    return res


def run(
    ctxs,
    *,
    force: bool = False,
    logger=None,
):
    """No-op: addition backfill is now inline in s5b (invert_delete_prompt)."""
    import logging as _l
    log = logger or _l.getLogger("pipeline_v2.s7")
    log.info("[s7] no-op: addition backfill moved to s5b")
    return []


__all__ = [
    "AdditionBackfillResult", "invert_delete_prompt",
    "run_for_object", "run",
]
