"""Unit tests for ``partcraft.cleaning.h3d_v1.filter``.

Table-driven coverage of the gate-status acceptance matrix per spec §5.
"""
from __future__ import annotations

import pytest

from partcraft.cleaning.h3d_v1.filter import (
    AcceptDecision,
    accept_deletion,
    accept_flux,
    gate_summary,
)


def _make(stages: dict[str, str | None] | None, *, gates: dict | None = None,
          final_pass: bool | None = None) -> dict:
    """Build a minimal edit_status doc with one edit ``e0``.

    ``stages``: ``{"gate_a": "pass", "gate_e": None, ...}``. A value of
    ``None`` means the stage entry is omitted entirely (mimicking the
    real pipeline behaviour when a gate hasn't run).
    """
    stage_block = {k: {"status": v} for k, v in (stages or {}).items() if v is not None}
    edit: dict = {"edit_type": "x", "stages": stage_block}
    if gates is not None:
        edit["gates"] = gates
    if final_pass is not None:
        edit["final_pass"] = final_pass
    return {"edits": {"e0": edit}}


# ── deletion ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "gate_a, ok",
    [("pass", True), ("fail", False), ("error", False), (None, False)],
)
def test_accept_deletion(gate_a: str | None, ok: bool) -> None:
    es = _make({"gate_a": gate_a})
    decision = accept_deletion(es, "e0")
    assert isinstance(decision, AcceptDecision)
    assert decision.ok is ok
    if ok:
        assert decision.reason is None
    else:
        assert decision.reason is not None and "gate_A" in decision.reason


def test_accept_deletion_missing_edit() -> None:
    decision = accept_deletion({"edits": {}}, "e0")
    assert decision.ok is False
    assert "gate_A" in (decision.reason or "")


# ── flux ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "gate_a, gate_e, ok, fail_letter",
    [
        ("pass", "pass", True, None),
        ("pass", "fail", False, "E"),
        ("pass", None, False, "E"),
        ("pass", "error", False, "E"),
        ("fail", "pass", False, "A"),
        (None, "pass", False, "A"),
        (None, None, False, "A"),
    ],
)
def test_accept_flux(gate_a: str | None, gate_e: str | None, ok: bool, fail_letter: str | None) -> None:
    es = _make({"gate_a": gate_a, "gate_e": gate_e})
    decision = accept_flux(es, "e0")
    assert decision.ok is ok
    if ok:
        assert decision.reason is None
    else:
        assert fail_letter is not None
        assert f"gate_{fail_letter}" in (decision.reason or "")


def test_accept_flux_ignores_gate_c() -> None:
    """gate_C is informational; not enforced even when None."""
    es = _make({"gate_a": "pass", "gate_e": "pass"})  # no gate_c at all
    assert accept_flux(es, "e0").ok is True


# ── gate_summary ───────────────────────────────────────────────────────
def test_gate_summary_reads_all_three_gates() -> None:
    es = _make(
        {"gate_a": "pass", "gate_e": "pass"},
        gates={
            "A": {"vlm": {"pass": True, "score": 1.0}, "rule": {"pass": True}},
            "C": None,
            "E": {"vlm": {"pass": True, "score": 0.8}},
        },
        final_pass=True,
    )
    summary = gate_summary(es, "e0")
    assert summary["gate_A"]["status"] == "pass"
    assert summary["gate_A"]["vlm_pass"] is True
    assert summary["gate_A"]["vlm_score"] == 1.0
    assert summary["gate_C"]["status"] is None
    assert summary["gate_C"]["vlm_pass"] is None
    assert summary["gate_E"]["vlm_score"] == 0.8
    assert summary["final_pass"] is True


def test_gate_summary_handles_missing_edit() -> None:
    summary = gate_summary({"edits": {}}, "e0")
    for letter in ("A", "C", "E"):
        assert summary[f"gate_{letter}"]["status"] is None
        assert summary[f"gate_{letter}"]["vlm_pass"] is None
    assert summary["final_pass"] is None
