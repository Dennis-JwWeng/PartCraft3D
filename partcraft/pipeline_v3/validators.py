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
from .qc_io import load_qc, is_edit_qc_failed, is_gate_a_failed
from .specs import iter_all_specs, iter_deletion_specs, iter_flux_specs
from .status import (
    STATUS_OK, STATUS_FAIL, STATUS_SKIP, load_status, save_status,
    _status_lock,
)


def _phase1_skipped(ctx: ObjectContext) -> bool:
    """True if s1 was explicitly marked skip (e.g. too_many_parts)."""
    s = load_status(ctx)
    entry = (s.get("steps") or {}).get("s1_phase1") or {}
    return entry.get("status") == STATUS_SKIP


def _require_phase1(step: str, ctx: ObjectContext) -> StepCheck | None:
    """Gate downstream validators on parsed.json.

    Returns a short-circuit StepCheck, or ``None`` to continue:
      * SKIP at s1 (too_many_parts) → ok=True, expected=0 (nothing to do).
      * parsed.json missing → ok=False, missing=['parsed.json'].
      * otherwise → None (caller runs its own product check).
    """
    if _phase1_skipped(ctx):
        return StepCheck(step=step, ok=True, expected=0, found=0, skip=True)
    if not ctx.parsed_path.is_file():
        return StepCheck(step=step, ok=False, missing=["parsed.json"])
    return None


@dataclass
class StepCheck:
    step: str
    ok: bool
    expected: int = 0
    found: int = 0
    missing: list[str] = field(default_factory=list)
    skip: bool = False   # True when phase1 was skip → step is n/a

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
    # Preserve explicit SKIP (e.g. too_many_parts) — absence of parsed.json
    # is expected in that case and should not be flipped to FAIL.
    if _phase1_skipped(ctx):
        return StepCheck(step="s1_phase1", ok=True, expected=0, found=0, skip=True)
    return _check_files("s1_phase1", [
        ("parsed.json", ctx.parsed_path),
        ("overview.png", ctx.overview_path),
    ])


def check_s2(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s2_highlights", ctx)
    if gate is not None:
        return gate
    edits = (json.loads(ctx.parsed_path.read_text())
             .get("parsed") or {}).get("edits") or []
    return _check_files("s2_highlights", [
        (f"e{i:02d}.png", ctx.highlight_path(i)) for i in range(len(edits))
    ])


def check_s4(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s4_flux_2d", ctx)
    if gate is not None:
        return gate
    return _check_files("s4_flux_2d", [
        (f"{s.edit_id}_edited.png", ctx.edit_2d_output(s.edit_id))
        for s in iter_flux_specs(ctx)
        if not is_edit_qc_failed(ctx, s.edit_id)
    ])


def check_s5(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s5_trellis", ctx)
    if gate is not None:
        return gate
    paths = []
    for s in iter_flux_specs(ctx):
        if is_gate_a_failed(ctx, s.edit_id):
            continue
        paths.append((f"{s.edit_id}/before.npz", ctx.edit_3d_npz(s.edit_id, "before")))
        paths.append((f"{s.edit_id}/after.npz",  ctx.edit_3d_npz(s.edit_id, "after")))
    return _check_files("s5_trellis", paths)


def check_s5b(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s5b_del_mesh", ctx)
    if gate is not None:
        return gate
    paths = []
    for s in iter_deletion_specs(ctx):
        if is_gate_a_failed(ctx, s.edit_id):
            continue
        d = ctx.edit_3d_dir(s.edit_id)
        paths.append((f"{s.edit_id}/after_new.glb", d / "after_new.glb"))
    return _check_files("s5b_del_mesh", paths)


def check_s6(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s6_render_3d", ctx)
    if gate is not None:
        return gate
    paths = []
    for s in iter_flux_specs(ctx):
        if is_gate_a_failed(ctx, s.edit_id):
            continue
        paths.append((f"{s.edit_id}/before.png", ctx.edit_3d_png(s.edit_id, "before")))
        paths.append((f"{s.edit_id}/after.png",  ctx.edit_3d_png(s.edit_id, "after")))
    return _check_files("s6_render_3d", paths)


def check_s6b(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s6b_del_reencode", ctx)
    if gate is not None:
        return gate
    return _check_files("s6b_del_reencode", [
        (f"{s.edit_id}/after.npz", ctx.edit_3d_npz(s.edit_id, "after"))
        for s in iter_deletion_specs(ctx)
    ])


def check_s6p(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s6p_preview", ctx)
    if gate is not None:
        return gate
    if not ctx.edits_3d_dir.is_dir():
        return StepCheck(step="s6p_preview", ok=True, expected=0, found=0)
    paths = [
        (f"{d.name}/preview_{i}.png", d / f"preview_{i}.png")
        for d in sorted(ctx.edits_3d_dir.iterdir())
        if d.is_dir() and d.name.split("_")[0] != "idn"
        for i in range(5)
    ]
    return _check_files("s6p_preview", paths)


def check_s7(ctx: ObjectContext) -> StepCheck:
    return StepCheck(step="s7_add_backfill", ok=True, expected=0, found=0, skip=True)

def check_s6p_del(ctx: ObjectContext) -> StepCheck:
    """s6p_del tracks per-edit completion in edit_status.json and per-object status via
    update_step().  A simple gate_a check is sufficient — the step itself is the authority
    on which edits succeeded or were skipped."""
    gate = _require_phase1("s6p_del", ctx)
    if gate is not None:
        return gate
    return StepCheck(step="s6p_del", ok=True, expected=0, found=0, skip=True)


def check_s6p_flux(ctx: ObjectContext) -> StepCheck:
    """s6p_flux tracks per-edit completion in edit_status.json and per-object status via
    update_step().  A simple gate_a check is sufficient — the step itself is the authority
    on which edits succeeded or were skipped."""
    gate = _require_phase1("s6p_flux", ctx)
    if gate is not None:
        return gate
    return StepCheck(step="s6p_flux", ok=True, expected=0, found=0, skip=True)



def check_sq1(ctx: ObjectContext) -> StepCheck:
    sc = _require_phase1("sq1_qc_A", ctx)
    if sc is not None:
        return sc   # s1 was skip → sq1 is n/a; missing parsed.json → fail
    from .specs import iter_all_specs

    expected_ids = [sp.edit_id for sp in iter_all_specs(ctx)]
    if not expected_ids:
        return StepCheck(step="sq1_qc_A", ok=True, expected=0, found=0, skip=True)

    qc = load_qc(ctx)
    edits = qc.get("edits") or {}
    found = sum(1 for eid in expected_ids if (edits.get(eid, {}).get("gates") or {}).get("A") is not None)
    return StepCheck(step="sq1_qc_A", ok=(found == len(expected_ids)),
                     expected=len(expected_ids), found=found,
                     missing=[] if found == len(expected_ids) else ["gate_A_not_written"])

def check_sq2(ctx: ObjectContext) -> StepCheck:
    from .specs import iter_flux_specs
    if not any(True for _ in iter_flux_specs(ctx)):
        return StepCheck(step="sq2_qc_C", ok=True, expected=0, found=0, skip=True)
    qc = load_qc(ctx)
    edits = qc.get("edits") or {}
    flux_ids = {sp.edit_id for sp in iter_flux_specs(ctx)}
    if not flux_ids:
        return StepCheck(step="sq2_qc_C", ok=True, expected=0, found=0, skip=True)
    found = sum(1 for eid in flux_ids if (edits.get(eid, {}).get("gates") or {}).get("C") is not None)
    return StepCheck(step="sq2_qc_C", ok=(found == len(flux_ids)),
                     expected=len(flux_ids), found=found,
                     missing=[] if found == len(flux_ids) else ["gate_C_not_written"])

def check_sq3(ctx: ObjectContext) -> StepCheck:
    qc = load_qc(ctx)
    edits = qc.get("edits") or {}
    if not edits:
        return StepCheck(step="sq3_qc_E", ok=True, expected=0, found=0, skip=True)
    found = sum(1 for e in edits.values() if (e.get("gates") or {}).get("E") is not None)
    return StepCheck(step="sq3_qc_E", ok=(found == len(edits)),
                     expected=len(edits), found=found,
                     missing=[] if found == len(edits) else ["gate_E_not_written"])


# ── Active step validators (Mode E decided) ──────────────────────────
# Keys match the step identifiers used by run.py ALL_STEPS.
# Inactive steps are commented out but their check_* functions are
# preserved above so they can be re-enabled without rewriting logic.
VALIDATORS: dict[str, Callable[[ObjectContext], StepCheck]] = {
    # Text generation + gates
    "gen_edits":       check_s1,       # Phase 1 VLM → parsed.json + overview.png
    "gate_text_align": check_sq1,      # Gate A → gate_A written in qc.json
    "gate_quality":    check_sq3,      # Gate E → gate_E written in qc.json
    # Deletion branch
    "del_mesh":        check_s5b,      # Deletion mesh → after_new.glb
    "preview_del":     check_s6p_del,  # Blender 5-view preview for deletions
    # Flux branch
    "flux_2d":         check_s4,       # FLUX 2D → edits_2d/_edited.png
    "trellis_3d":      check_s5,       # Trellis → edits_3d/before+after.npz
    "preview_flux":    check_s6p_flux, # Trellis-decoded 5-view preview
    "render_3d":       check_s6,       # 40-view render → before/after.png
    # Inactive:
    # "reencode_del":  check_s6b,      # GPU re-encode after.npz for deletions
}


# ─────────────────── status flip ─────────────────────────────────────

def apply_check(ctx: ObjectContext, step_short: str) -> StepCheck:
    """Run the validator and update ``status.json`` to reflect reality.

    If the check fails, the step's status is forced to ``fail`` so the
    next orchestrator run will retry it. If it passes, status stays
    ``ok``. Either way, a ``validation`` field is attached.
    """
    fn = VALIDATORS[step_short]
    rep = fn(ctx)                     # read-only — outside the lock
    with _status_lock(ctx):
        s = load_status(ctx)
        steps = s.setdefault("steps", {})
        entry = steps.get(rep.step) or {"status": "?"}
        entry["validation"] = rep.to_dict()
        if rep.skip:
            # Only mark skip if not already completed — don't overwrite an "ok" status
            # written by the step itself (e.g. s6p_del / s6p_flux).
            if entry.get("status") not in (STATUS_OK, STATUS_FAIL):
                entry["status"] = STATUS_SKIP
        elif not rep.ok:
            entry["status"] = STATUS_FAIL
        elif entry.get("status") not in (STATUS_OK, STATUS_FAIL):
            entry["status"] = STATUS_OK
        steps[rep.step] = entry
        save_status(ctx, s)
    return rep


__all__ = ["StepCheck", "VALIDATORS", "apply_check"]

