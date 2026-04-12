"""Unit tests for s1_vlm_core changes: quota, prompt, validate."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partcraft.pipeline_v2.s1_vlm_core import quota_for, validate, USER_PROMPT_TEMPLATE


# ── Task 1: quota_for() scale cap ────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("n_parts", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
def test_quota_scale_always_one(n_parts):
    q = quota_for(n_parts)
    assert q["scale"] == 1, f"n_parts={n_parts}: expected scale=1, got {q['scale']}"

# ── Task 2: scale factor range in prompt ─────────────────────────────────────

@pytest.mark.unit
def test_prompt_scale_factor_range_shrink_only():
    assert "[0.3, 0.85]" in USER_PROMPT_TEMPLATE, \
        "scale factor range should be [0.3, 0.85] (shrink-only)"
    assert "Shrink only" in USER_PROMPT_TEMPLATE, \
        "prompt should say 'Shrink only' for scale edits"
    assert "2.5" not in USER_PROMPT_TEMPLATE, \
        "old factor upper bound 2.5 must be removed"

# ── Task 3: modification shape-only constraint in prompt ──────────────────────

@pytest.mark.unit
def test_prompt_has_modification_shape_only_section():
    assert "MODIFICATION EDITS" in USER_PROMPT_TEMPLATE, \
        "prompt must have MODIFICATION EDITS section header"
    assert "STRICTLY FORBIDDEN" in USER_PROMPT_TEMPLATE, \
        "prompt must forbid color-only modifications"
    assert "curved saber blade" in USER_PROMPT_TEMPLATE, \
        "prompt should include saber blade shape example"


@pytest.mark.unit
def test_prompt_has_r9_hard_rule():
    assert "R9." in USER_PROMPT_TEMPLATE, \
        "prompt must have Hard Rule R9 for modification shape constraint"
    assert "A blue sphere" in USER_PROMPT_TEMPLATE, \
        "R9 must include the wrong example (color-only)"
    assert "A flattened disc" in USER_PROMPT_TEMPLATE, \
        "R9 must include the right example (shape change)"

