# VLM Edit Quality Improvements — Design Spec

**Date:** 2026-04-13  
**Scope:** `partcraft/pipeline_v2/s1_vlm_core.py`  
**Status:** Approved

---

## Problem Statement

Analysis of Phase-1 VLM outputs across multiple objects revealed four systematic quality issues:

| # | Issue | Evidence |
|---|---|---|
| 1 | **Scale count too high** | Rabbit: 3 scale edits; Cloud: 3 scale edits; target is max 1 per object |
| 2 | **Scale direction wrong** | Rabbit body factor=1.5 (UP); Car tail light factor=1.5 (UP); small parts enlarged |
| 3 | **Modification = color change** | Cloud object: all 8 modifications are pure color changes ("bright red sphere", "purple cloud"); should be handled by `material`/`global` types |
| 4 | **R2 never validated** | Cloud: 4 material edits with identical `selected_part_ids`; Rabbit: same part modified 3 times; `validate()` does not enforce R2 |

Root cause: (3) modification has no explicit shape-only constraint in the prompt; (4) `validate()` is missing the R2 uniqueness check.

---

## Design

### Change 1 — quota_for(): Cap scale at 1

In every n_parts branch, set "scale": 1 regardless of part count.

Before (selected branches):
  if n_parts <= 8:  return {"deletion":8,  "modification":8,  "scale":3, ...}
  if n_parts <= 10: return {"deletion":10, "modification":10, "scale":3, ...}
  if n_parts <= 12: return {"deletion":12, "modification":12, "scale":4, ...}
  if n_parts <= 14: return {"deletion":14, "modification":14, "scale":4, ...}
  return                   {"deletion":16, "modification":16, "scale":5, ...}

After — all scale values become 1.

---

### Change 2 — USER_PROMPT_TEMPLATE: Scale factor range + guidance

In the edit_params schema line for scale, change factor range and add shrink-bias note.

Before:
  scale: {"factor": float in [0.3, 2.5]}

After:
  scale: {"factor": float in [0.3, 0.85]}
         Shrink only. Prefer large/dominant parts (main body, primary limbs).
         Do NOT enlarge small decorative parts.

---

### Change 3 — USER_PROMPT_TEMPLATE: MODIFICATION front-matter section

Insert a dedicated block immediately before the edit_params schema definitions.
This ensures the VLM reads the constraint before encountering the modification schema entry.

  MODIFICATION EDITS — SHAPE, FORM AND FUNCTION ONLY
    A modification replaces the geometry, silhouette, or functional role of a part.
    Think creatively: what shape or form would be surprising yet meaningful?
    Examples:
      - straight sword blade  ->  curved saber blade
      - cylindrical barrel    ->  hexagonal prism barrel
      - spherical head        ->  cubic head
      - upright rabbit ears   ->  floppy drooping ears
    STRICTLY FORBIDDEN: changing only color, surface finish, or material in a
    modification edit. Those belong exclusively to "material" or "global" types.
    The new_part_desc MUST describe a geometry or silhouette change.

---

### Change 4 — USER_PROMPT_TEMPLATE: Hard Rule R9

Append after the existing R8:

  R9. modification edit_params.new_part_desc MUST describe a shape, silhouette, or
      functional change — NOT a color or material change.
      Wrong: "A blue sphere"         (color only -> use material instead)
      Right: "A flattened disc"      (shape change)
      Right: "A curved saber blade"  (functional + shape change)

---

### Change 5 — validate(): Enforce R2 (duplicate edit_type + parts)

After the per-edit loop, add a cross-edit check. For each edit compute
sig = (edit_type, tuple(sorted(selected_part_ids))). If sig already seen,
append an R2 warning. This catches cloud-style color-spam modifications and
rabbit-style material all-same-parts duplicates.

---

## Files Changed

  partcraft/pipeline_v2/s1_vlm_core.py  — all 5 changes

No other files are touched.

---

## Success Criteria

1. Every object in Phase-1 output has exactly 1 scale edit with factor in [0.3, 0.85]
2. No modification edit has a new_part_desc that is solely a color/material description
3. validate() flags duplicate (edit_type, selected_part_ids) pairs as warnings
4. Prompt total token count remains under ~2400 tokens (was ~2220; delta ~+150)

---

## Out of Scope

- Fixing the R3 (structural body deletion) issue
- Splitting modification into sub-types
- Any changes outside s1_vlm_core.py

---

## Implementation Notes (from spec self-review)

### Insertion point for Change 3
Insert the MODIFICATION EDITS block immediately before the line that begins with
"  edit_params" in the USER_PROMPT_TEMPLATE (the line listing edit_params for each type).
This is the natural reading-order position for the VLM.

### validate() R2 check placement (Change 5)
The cross-edit R2 check runs as a second pass AFTER the per-edit for loop ends
and BEFORE the `out["ok"]` line. Duplicate-signature edits are added to
`out["warnings"]` but are NOT subtracted from `kept` (the first occurrence is
already counted; the duplicate simply gets a warning, and downstream QC will
handle filtering).
