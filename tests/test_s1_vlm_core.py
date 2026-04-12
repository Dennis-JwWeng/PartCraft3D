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

# ── Task 4: validate() R2 cross-edit check ───────────────────────────────────

def _base_edit(edit_type, part_ids, extra_params=None):
    """Minimal valid edit dict for validate() testing."""
    e = {
        "edit_type": edit_type,
        "selected_part_ids": part_ids,
        "prompt": "Change the widget.",
        "view_index": 0,
        "edit_params": extra_params or {},
        "after_desc_full": "After full." if edit_type != "deletion" else None,
        "after_desc_stage1": "After s1." if edit_type != "deletion" else None,
        "after_desc_stage2": "After s2." if edit_type != "deletion" else None,
    }
    return e


@pytest.mark.unit
def test_validate_r2_flags_duplicate_edit_type_and_parts():
    """Two modification edits on the same part_ids should produce an R2 warning."""
    edits = [
        _base_edit("modification", [0], {"new_part_desc": "A cube."}),
        _base_edit("modification", [0], {"new_part_desc": "A sphere."}),
    ]
    parsed = {
        "object": {"full_desc": "x", "full_desc_stage1": "x", "full_desc_stage2": "x", "parts": []},
        "edits": edits,
    }
    result = validate(parsed, valid_pids={0})
    warning_texts = [str(w) for w in result["warnings"]]
    assert any("R2" in t for t in warning_texts), \
        f"Expected R2 warning for duplicate (modification, [0]), got: {result['warnings']}"


@pytest.mark.unit
def test_validate_r2_no_false_positive_different_parts():
    """Two modification edits on DIFFERENT parts should NOT trigger R2."""
    edits = [
        _base_edit("modification", [0], {"new_part_desc": "A cube."}),
        _base_edit("modification", [1], {"new_part_desc": "A sphere."}),
    ]
    parsed = {
        "object": {"full_desc": "x", "full_desc_stage1": "x", "full_desc_stage2": "x", "parts": []},
        "edits": edits,
    }
    result = validate(parsed, valid_pids={0, 1})
    warning_texts = [str(w) for w in result["warnings"]]
    assert not any("R2" in t for t in warning_texts), \
        f"Unexpected R2 warning for different parts: {result['warnings']}"


@pytest.mark.unit
def test_validate_r2_material_spam():
    """Four material edits all with same parts (cloud pattern) should flag 3 R2 warnings."""
    edits = [
        _base_edit("material", [0, 1], {"target_material": "chrome"}),
        _base_edit("material", [0, 1], {"target_material": "rubber"}),
        _base_edit("material", [0, 1], {"target_material": "gold"}),
        _base_edit("material", [0, 1], {"target_material": "stone"}),
    ]
    parsed = {
        "object": {"full_desc": "x", "full_desc_stage1": "x", "full_desc_stage2": "x", "parts": []},
        "edits": edits,
    }
    result = validate(parsed, valid_pids={0, 1})
    r2_warnings = [w for w in result["warnings"] if any("R2" in str(p) for p in w.get("problems", []))]
    assert len(r2_warnings) == 3, \
        f"Expected 3 R2 warnings for 4 identical material edits, got {len(r2_warnings)}: {r2_warnings}"


# ── R2 fix: global exemption + invalid-edit skip ─────────────────────────────

@pytest.mark.unit
def test_validate_r2_global_edits_exempt():
    """Two global edits should NOT trigger R2 (globals are exempt)."""
    edits = [
        {
            "edit_type": "global", "selected_part_ids": [],
            "prompt": "Make it futuristic.", "view_index": 0,
            "edit_params": {"target_style": "futuristic"},
            "after_desc_full": "After.", "after_desc_stage1": "After.", "after_desc_stage2": "After.",
        },
        {
            "edit_type": "global", "selected_part_ids": [],
            "prompt": "Make it retro.", "view_index": 0,
            "edit_params": {"target_style": "retro"},
            "after_desc_full": "After.", "after_desc_stage1": "After.", "after_desc_stage2": "After.",
        },
    ]
    parsed = {
        "object": {"full_desc": "x", "full_desc_stage1": "x", "full_desc_stage2": "x", "parts": []},
        "edits": edits,
    }
    result = validate(parsed, valid_pids=set())
    r2_warnings = [w for w in result["warnings"] if any("R2" in str(p) for p in w.get("problems", []))]
    assert len(r2_warnings) == 0, f"Global edits should be R2-exempt, got: {r2_warnings}"


@pytest.mark.unit
def test_validate_r2_skips_already_invalid_edits():
    """Two edits with invalid edit_type should not trigger R2 (already invalid)."""
    bad_edit = {
        "edit_type": "INVALID_TYPE", "selected_part_ids": [0],
        "prompt": "Change the widget.", "view_index": 0,
        "edit_params": {}, "after_desc_full": None,
        "after_desc_stage1": None, "after_desc_stage2": None,
    }
    parsed = {
        "object": {"full_desc": "x", "full_desc_stage1": "x", "full_desc_stage2": "x", "parts": []},
        "edits": [bad_edit, dict(bad_edit)],
    }
    result = validate(parsed, valid_pids={0})
    r2_warnings = [w for w in result["warnings"] if any("R2" in str(p) for p in w.get("problems", []))]
    assert len(r2_warnings) == 0, f"Invalid edits should be R2-exempt, got: {r2_warnings}"
