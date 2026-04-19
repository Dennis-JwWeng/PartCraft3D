"""Gate-status acceptance rules for H3D_v1 promotion.

Reads pipeline_v3's per-object ``edit_status.json`` schema and decides
whether an edit qualifies for inclusion in the dataset. See spec §5 for
the canonical rule table.

Key principle: ``required-null-rejects``. If a required gate is missing,
``None``, ``"fail"``, or ``"error"``, the edit is rejected — only an
explicit ``"pass"`` admits the edit. This is stricter than the
pipeline's own ``final_pass`` field, which can be lenient when a gate
inherits from another (e.g. deletion's gate_E inheriting from gate_A is
still a real ``"pass"`` and goes through, but anything missing fails).

Source of truth for a gate's outcome is ``edits.<edit_id>.stages.<gate_key>.status``.
That field is what s4/s4q/s4t etc. write after considering both the
deterministic rule and the VLM judgement. The richer
``edits.<edit_id>.gates.<X>`` block is only consulted by ``gate_summary``
for the per-edit ``meta.json``.

Addition filtering is *not* handled here — addition's eligibility is a
function of dataset state (paired deletion already promoted), not of
``edit_status.json``. See ``promoter.promote_addition``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Map gate letter → ``stages.<key>`` entry.
_STAGE_KEY: dict[str, str] = {"A": "gate_a", "C": "gate_c", "E": "gate_e"}


@dataclass(frozen=True)
class AcceptDecision:
    ok: bool
    reason: str | None  # ``None`` iff ``ok``; otherwise short human-readable text.


def _stage_status(edit_status: Mapping[str, Any], edit_id: str, stage_key: str) -> str | None:
    """Return ``stages.<stage_key>.status`` string for the edit, or ``None`` if absent."""
    edit = edit_status.get("edits", {}).get(edit_id)
    if not isinstance(edit, Mapping):
        return None
    stage = edit.get("stages", {}).get(stage_key)
    if not isinstance(stage, Mapping):
        return None
    val = stage.get("status")
    return val if isinstance(val, str) else None


def accept_deletion(edit_status: Mapping[str, Any], edit_id: str) -> AcceptDecision:
    """Deletion accepted iff ``stages.gate_a.status == "pass"``."""
    status = _stage_status(edit_status, edit_id, _STAGE_KEY["A"])
    if status == "pass":
        return AcceptDecision(True, None)
    return AcceptDecision(False, f"gate_A={status!r}")


def accept_flux(edit_status: Mapping[str, Any], edit_id: str) -> AcceptDecision:
    """Flux edit accepted iff ``gate_a`` and ``gate_e`` both ``pass``.

    ``gate_c`` is informational on this pipeline (often ``None``); not enforced.
    """
    a = _stage_status(edit_status, edit_id, _STAGE_KEY["A"])
    if a != "pass":
        return AcceptDecision(False, f"gate_A={a!r}")
    e = _stage_status(edit_status, edit_id, _STAGE_KEY["E"])
    if e != "pass":
        return AcceptDecision(False, f"gate_E={e!r}")
    return AcceptDecision(True, None)


def gate_summary(edit_status: Mapping[str, Any], edit_id: str) -> dict[str, Any]:
    """Return a flat summary suitable for embedding in ``meta.json``.

    Captures, per gate (A/C/E):
      - ``status``: the canonical pass/fail/error/None.
      - ``ts``: stage timestamp (``None`` if stage not run).
      - ``vlm_pass`` + ``vlm_score``: when present in ``gates.<X>.vlm``.

    Plus the pipeline's own ``final_pass`` for cross-validation.
    """
    edit = edit_status.get("edits", {}).get(edit_id, {}) or {}
    stages = edit.get("stages", {}) or {}
    gates = edit.get("gates", {}) or {}

    out: dict[str, Any] = {}
    for letter, stage_key in _STAGE_KEY.items():
        stage_entry = stages.get(stage_key, {}) if isinstance(stages.get(stage_key), Mapping) else {}
        gate_entry = gates.get(letter)
        vlm = gate_entry.get("vlm") if isinstance(gate_entry, Mapping) else None
        out[f"gate_{letter}"] = {
            "status": stage_entry.get("status") if isinstance(stage_entry, Mapping) else None,
            "ts": stage_entry.get("ts") if isinstance(stage_entry, Mapping) else None,
            "vlm_pass": vlm.get("pass") if isinstance(vlm, Mapping) else None,
            "vlm_score": vlm.get("score") if isinstance(vlm, Mapping) else None,
        }
    out["final_pass"] = edit.get("final_pass")
    return out


__all__ = [
    "AcceptDecision",
    "accept_deletion",
    "accept_flux",
    "gate_summary",
]
