# VLM Edit Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four systematic VLM output quality issues in `s1_vlm_core.py`: scale count/direction, modification=color drift, and missing R2 validation.

**Architecture:** All changes are confined to `partcraft/pipeline_v2/s1_vlm_core.py`. Three functions are touched: `quota_for()` (quota cap), `USER_PROMPT_TEMPLATE` (string constant with two insertions and one edit), and `validate()` (new post-loop R2 check). New unit tests go in `tests/test_s1_vlm_core.py`.

**Tech Stack:** Python 3.x, pytest. No GPU, no network, no Blender needed for any tests.

---

## File Map

- Modify: `partcraft/pipeline_v2/s1_vlm_core.py`
  - `quota_for()` lines ~215–228: cap scale=1 across all branches
  - `USER_PROMPT_TEMPLATE` string: 3 targeted edits to the prompt text
  - `validate()` lines ~270–290: add R2 cross-edit check after the per-edit loop
- Create: `tests/test_s1_vlm_core.py` (new file for all new tests)

---

## Task 1: Cap scale quota at 1 in quota_for()

**Files:**
- Modify: `partcraft/pipeline_v2/s1_vlm_core.py` (quota_for function)
- Create: `tests/test_s1_vlm_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_s1_vlm_core.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python -m pytest tests/test_s1_vlm_core.py::test_quota_scale_always_one -v
```

Expected: FAIL — current quota_for returns scale>1 for n_parts>=4.

- [ ] **Step 3: Implement — edit quota_for() in s1_vlm_core.py**

In `partcraft/pipeline_v2/s1_vlm_core.py`, replace the entire `quota_for` function body:

```python
def quota_for(n_parts: int) -> dict:
    """Per-edit-type quotas based on number of (valid) parts. Deletion is the
    cheapest + most useful task, so it gets the largest share. Scale is capped
    at 1 per object to avoid redundancy; prefer shrinking large parts."""
    if n_parts <= 2:  return {"deletion":1,  "modification":1,  "scale":1, "material":1, "global":1}
    if n_parts == 3:  return {"deletion":3,  "modification":3,  "scale":1, "material":1, "global":1}
    if n_parts == 4:  return {"deletion":4,  "modification":4,  "scale":1, "material":2, "global":1}
    if n_parts == 5:  return {"deletion":5,  "modification":5,  "scale":1, "material":2, "global":1}
    if n_parts == 6:  return {"deletion":6,  "modification":6,  "scale":1, "material":2, "global":1}
    if n_parts <= 8:  return {"deletion":8,  "modification":8,  "scale":1, "material":3, "global":1}
    if n_parts <= 10: return {"deletion":10, "modification":10, "scale":1, "material":4, "global":1}
    if n_parts <= 12: return {"deletion":12, "modification":12, "scale":1, "material":4, "global":2}
    if n_parts <= 14: return {"deletion":14, "modification":14, "scale":1, "material":5, "global":2}
    return                   {"deletion":16, "modification":16, "scale":1, "material":5, "global":2}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_s1_vlm_core.py::test_quota_scale_always_one -v
```

Expected: all 15 parametrized cases PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/pipeline_v2/s1_vlm_core.py tests/test_s1_vlm_core.py
git commit -m "fix: cap scale quota at 1 per object in quota_for()"
```

---

## Task 2: Tighten scale factor range in prompt

**Files:**
- Modify: `partcraft/pipeline_v2/s1_vlm_core.py` (USER_PROMPT_TEMPLATE string)
- Modify: `tests/test_s1_vlm_core.py` (add prompt content test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_s1_vlm_core.py`:

```python
# ── Task 2: scale factor range in prompt ─────────────────────────────────────

@pytest.mark.unit
def test_prompt_scale_factor_range_shrink_only():
    assert "[0.3, 0.85]" in USER_PROMPT_TEMPLATE, \
        "scale factor range should be [0.3, 0.85] (shrink-only)"
    assert "Shrink only" in USER_PROMPT_TEMPLATE, \
        "prompt should say 'Shrink only' for scale edits"
    assert "2.5" not in USER_PROMPT_TEMPLATE, \
        "old factor upper bound 2.5 must be removed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_s1_vlm_core.py::test_prompt_scale_factor_range_shrink_only -v
```

Expected: FAIL — current prompt has `[0.3, 2.5]` and no "Shrink only".

- [ ] **Step 3: Implement — edit scale edit_params line in USER_PROMPT_TEMPLATE**

In `partcraft/pipeline_v2/s1_vlm_core.py`, find this line in `USER_PROMPT_TEMPLATE`:

```
                      scale:        {{"factor": float in [0.3, 2.5]}}
```

Replace with:

```
                      scale:        {{"factor": float in [0.3, 0.85]}}
                                    Shrink only. Prefer large/dominant parts (main body, primary limbs).
                                    Do NOT enlarge small decorative parts.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_s1_vlm_core.py::test_prompt_scale_factor_range_shrink_only -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/pipeline_v2/s1_vlm_core.py tests/test_s1_vlm_core.py
git commit -m "fix: restrict scale factor to [0.3, 0.85] shrink-only in prompt"
```

---

## Task 3: Add MODIFICATION EDITS section and Hard Rule R9 to prompt

**Files:**
- Modify: `partcraft/pipeline_v2/s1_vlm_core.py` (USER_PROMPT_TEMPLATE string)
- Modify: `tests/test_s1_vlm_core.py` (add prompt content tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_s1_vlm_core.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_s1_vlm_core.py::test_prompt_has_modification_shape_only_section tests/test_s1_vlm_core.py::test_prompt_has_r9_hard_rule -v
```

Expected: both FAIL.

- [ ] **Step 3: Implement — insert MODIFICATION EDITS block before edit_params line**

In `USER_PROMPT_TEMPLATE`, find this exact line (the start of the edit_params schema block):

```
  edit_params         deletion: {{}}
```

Insert the following block IMMEDIATELY BEFORE that line (add a blank line before and after):

```
# MODIFICATION EDITS — SHAPE, FORM AND FUNCTION ONLY
  A modification replaces the geometry, silhouette, or functional role of a part.
  Think creatively: what shape or form would be surprising yet meaningful?
  Examples:
    • straight sword blade  →  curved saber blade
    • cylindrical barrel    →  hexagonal prism barrel
    • spherical head        →  cubic head
    • upright rabbit ears   →  floppy drooping ears
  STRICTLY FORBIDDEN in modification: changing only color, surface finish, or
  material. Those belong exclusively to "material" or "global" edit types.
  The new_part_desc MUST describe a geometry or silhouette change.

```

- [ ] **Step 4: Implement — append Hard Rule R9 after R8**

In `USER_PROMPT_TEMPLATE`, find the last Hard Rule line (currently ends with R8). Find:

```
R8. If an edit uses anatomical left/right, view_index ∈ {{F, (F+2) mod 4}}
    where F = frontal_view_index, and the rationale must cite the mirror
    reasoning explicitly.
```

Append immediately after:

```
R9. modification edit_params.new_part_desc MUST describe a shape, silhouette, or
    functional change — NOT a color or material change.
    Wrong: "A blue sphere"         (color only → use material instead)
    Right: "A flattened disc"      (shape change)
    Right: "A curved saber blade"  (functional + shape change)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_s1_vlm_core.py::test_prompt_has_modification_shape_only_section tests/test_s1_vlm_core.py::test_prompt_has_r9_hard_rule -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add partcraft/pipeline_v2/s1_vlm_core.py tests/test_s1_vlm_core.py
git commit -m "feat: add MODIFICATION EDITS shape-only section and R9 to prompt"
```

---

## Task 4: Add R2 duplicate check to validate()

**Files:**
- Modify: `partcraft/pipeline_v2/s1_vlm_core.py` (validate function)
- Modify: `tests/test_s1_vlm_core.py` (add validate tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_s1_vlm_core.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_s1_vlm_core.py::test_validate_r2_flags_duplicate_edit_type_and_parts tests/test_s1_vlm_core.py::test_validate_r2_no_false_positive_different_parts tests/test_s1_vlm_core.py::test_validate_r2_material_spam -v
```

Expected: all 3 FAIL — current validate() has no R2 check.

- [ ] **Step 3: Implement — add R2 check in validate()**

In `partcraft/pipeline_v2/s1_vlm_core.py`, in the `validate()` function, find the line:

```python
    out["n_kept_edits"] = kept
```

Insert the following block IMMEDIATELY AFTER that line (before the `out["type_counts"]` line):

```python
    # R2 cross-edit check: no two edits with same (edit_type, selected_part_ids)
    seen_signatures: set[tuple] = set()
    for i, e in enumerate(edits):
        et = e.get("edit_type")
        pids = tuple(sorted(e.get("selected_part_ids", [])))
        sig = (et, pids)
        if sig in seen_signatures:
            out["warnings"].append({
                "edit_index": i,
                "problems": [f"R2 violation: duplicate (edit_type={et}, selected_part_ids={list(pids)})"],
            })
        else:
            seen_signatures.add(sig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_s1_vlm_core.py::test_validate_r2_flags_duplicate_edit_type_and_parts tests/test_s1_vlm_core.py::test_validate_r2_no_false_positive_different_parts tests/test_s1_vlm_core.py::test_validate_r2_material_spam -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run the full new test file to confirm no regressions**

```bash
python -m pytest tests/test_s1_vlm_core.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add partcraft/pipeline_v2/s1_vlm_core.py tests/test_s1_vlm_core.py
git commit -m "fix: add R2 duplicate (edit_type, parts) check to validate()"
```

---

## Task 5: Smoke test — verify prompt token count is acceptable

**Files:**
- No code changes. Verification only.

- [ ] **Step 1: Run token count check**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python3 - << 'EOF'
import sys
sys.path.insert(0, '.')
from partcraft.pipeline_v2.s1_vlm_core import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT
filled = USER_PROMPT_TEMPLATE.replace('{part_menu}','[MENU]').replace('{n_total}','17').replace('{n_deletion}','6').replace('{n_modification}','6').replace('{n_scale}','1').replace('{n_material}','2').replace('{n_global}','1')
total = len(SYSTEM_PROMPT) + len(filled)
approx_tokens = total // 4
print(f"Total chars: {total}, approx tokens: {approx_tokens}")
assert approx_tokens < 2500, f"Prompt grew too large: {approx_tokens} tokens"
print("OK — within 2500 token budget")
EOF
```

Expected output: `OK — within 2500 token budget`

- [ ] **Step 2: Run full existing test suite to catch regressions**

```bash
python -m pytest tests/ -v --tb=short -x 2>&1 | tail -30
```

Expected: all pre-existing tests still PASS (new tests also PASS).

- [ ] **Step 3: Final commit if anything was adjusted**

If no adjustments needed, skip. Otherwise:

```bash
git add -A
git commit -m "chore: post-implementation cleanup"
```

---

## Spec Coverage Check

| Spec requirement | Covered by |
|---|---|
| Scale quota capped at 1 | Task 1 |
| Scale factor range [0.3, 0.85] shrink-only | Task 2 |
| MODIFICATION EDITS front-matter section with examples | Task 3 |
| Hard Rule R9 (new_part_desc must be shape change) | Task 3 |
| validate() R2 cross-edit check | Task 4 |
| Prompt token budget <2400 | Task 5 |
