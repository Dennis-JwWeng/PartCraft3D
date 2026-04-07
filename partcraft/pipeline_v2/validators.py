"""Per-step product checks (file-system only, no content parsing).

A step is considered ``ok`` iff its expected output files all exist and
are non-empty. Each validator returns a :class:`StepCheck` describing
the result; the orchestrator uses it to flip ``status.json`` after a
step completes (so the next run resumes only the truly-incomplete
objects).

Rules are intentionally minimal — file existence + size > 0 + count
match against the parsed edit list. No image decode, no npz parse, no
trellis import. Anything heavier should live in a separate
``--validate`` pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import ObjectContext
from .specs import iter_all_specs, iter_deletion_specs, iter_flux_specs
from .status import (
    STATUS_OK, STATUS_FAIL, load_status, save_status,
)


@dataclass
class StepCheck:
    step: str
    ok: bool
    expected: int = 0
    found: int = 0
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "expected": self.expected,
            "found": self.found,
            "missing": self.missing[:10],   # cap noise
        }


def _exists_nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def _check_files(step: str, paths: list[tuple[str, Path]]) -> StepCheck:
    missing = [name for name, p in paths if not _exists_nonempty(p)]
    return StepCheck(
        step=step,
        ok=not missing,
        expected=len(paths),
        found=len(paths) - len(missing),
        missing=missing,
    )


# ─────────────────── per-step validators ─────────────────────────────

def check_s1(ctx: ObjectContext) -> StepCheck:
    return _check_files("s1_phase1", [
        ("parsed.json", ctx.parsed_path),
        ("overview.png", ctx.overview_path),
    ])


def check_s2(ctx: ObjectContext) -> StepCheck:
    if not ctx.parsed_path.is_file():
        return StepCheck("s2_highlights", ok=False, missing=["parsed.json"])
    edits = (json.loads(ctx.parsed_path.read_text())
             .get("parsed") or {}).get("edits") or []
    return _check_files("s2_highlights", [
        (f"e{i:02d}.png", ctx.highlight_path(i)) for i in range(len(edits))
    ])


def check_s4(ctx: ObjectContext) -> StepCheck:
    return _check_files("s4_flux_2d", [
        (f"{s.edit_id}_edited.png", ctx.edit_2d_output(s.edit_id))
        for s in iter_flux_specs(ctx)
    ])


def check_s5(ctx: ObjectContext) -> StepCheck:
    paths = []
    for s in iter_flux_specs(ctx):
        paths.append((f"{s.edit_id}/before.npz", ctx.edit_3d_npz(s.edit_id, "before")))
        paths.append((f"{s.edit_id}/after.npz",  ctx.edit_3d_npz(s.edit_id, "after")))
    return _check_files("s5_trellis", paths)


def check_s5b(ctx: ObjectContext) -> StepCheck:
    paths = []
    for s in iter_deletion_specs(ctx):
        d = ctx.edit_3d_dir(s.edit_id)
        paths.append((f"{s.edit_id}/before.ply", d / "before.ply"))
        paths.append((f"{s.edit_id}/after.ply",  d / "after.ply"))
    return _check_files("s5b_del_mesh", paths)


def check_s6(ctx: ObjectContext) -> StepCheck:
    paths = []
    for s in iter_flux_specs(ctx):
        paths.append((f"{s.edit_id}/before.png", ctx.edit_3d_png(s.edit_id, "before")))
        paths.append((f"{s.edit_id}/after.png",  ctx.edit_3d_png(s.edit_id, "after")))
    return _check_files("s6_render_3d", paths)


def check_s6b(ctx: ObjectContext) -> StepCheck:
    return _check_files("s6b_del_reencode", [
        (f"{s.edit_id}/after.npz", ctx.edit_3d_npz(s.edit_id, "after"))
        for s in iter_deletion_specs(ctx)
    ])


def check_s7(ctx: ObjectContext) -> StepCheck:
    paths = []
    add_seq = 0
    for s in iter_deletion_specs(ctx):
        # only expect an add if the source deletion has both npz files
        d = ctx.edit_3d_dir(s.edit_id)
        if not ((d / "before.npz").is_file() and (d / "after.npz").is_file()):
            continue
        add_id = ctx.edit_id("addition", add_seq); add_seq += 1
        ad = ctx.edit_3d_dir(add_id)
        paths.append((f"{add_id}/before.npz", ad / "before.npz"))
        paths.append((f"{add_id}/after.npz",  ad / "after.npz"))
        paths.append((f"{add_id}/meta.json",  ad / "meta.json"))
    return _check_files("s7_add_backfill", paths)


VALIDATORS: dict[str, Callable[[ObjectContext], StepCheck]] = {
    "s1":  check_s1,
    "s2":  check_s2,
    "s4":  check_s4,
    "s5":  check_s5,
    "s5b": check_s5b,
    "s6":  check_s6,
    "s6b": check_s6b,
    "s7":  check_s7,
}


# ─────────────────── status flip ─────────────────────────────────────

def apply_check(ctx: ObjectContext, step_short: str) -> StepCheck:
    """Run the validator and update ``status.json`` to reflect reality.

    If the check fails, the step's status is forced to ``fail`` so the
    next orchestrator run will retry it. If it passes, status stays
    ``ok``. Either way, a ``validation`` field is attached.
    """
    fn = VALIDATORS[step_short]
    rep = fn(ctx)
    s = load_status(ctx)
    steps = s.setdefault("steps", {})
    entry = steps.get(rep.step) or {"status": "?"}
    entry["validation"] = rep.to_dict()
    if not rep.ok:
        entry["status"] = STATUS_FAIL
    elif entry.get("status") not in (STATUS_OK, STATUS_FAIL):
        entry["status"] = STATUS_OK
    steps[rep.step] = entry
    save_status(ctx, s)
    return rep


__all__ = ["StepCheck", "VALIDATORS", "apply_check"]
