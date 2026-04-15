# Design: Addition Backfill Inline + Preview-Gate Pipeline Restructure

**Date:** 2026-04-12  
**Status:** Draft  
**Branch:** feature/prompt-driven-part-selection

---

## Problem Statement

Current pipeline has two structural issues:

1. **addition (`add`) edits bypass the VLM final gate.**  
   `s7` (stage F) runs *after* `sq3` (E_qc). `add` edits are never judged; `final_pass` is vacuously true from gate A alone.

2. **Expensive rendering & encoding runs before the quality gate.**  
   `s6b` (40-view Blender render → DINOv2 → SLAT/SS encode) executes for *all* deletion edits, even those that will fail `sq3`. This wastes GPU/CPU on edits that will be discarded. The rough NPZ (`export_deletion_pair`) in `s5b` is also wasted because `s6b` overwrites it.

---

## Goals

- `add` is created synchronously with `del` (same step, not a separate stage).
- `add` passes through the VLM final gate (`sq3`).
- Expensive rendering/encoding (`s6b`, full `s6`) runs **only for survivors** of the gate.
- Camera parameters are identical between before/after comparison views → no perspective mismatch.
- No computation is wasted on edits that will fail the gate.

---

## Non-Goals

- Changing the FLUX 2D edit flow (s4, sq2).
- Changing gate A (sq1) or gate C (sq2) logic.
- Modifying training dataset loading.

---

## Design

### New Stage Order

```
A:      s1 + sq1      VLM phase1 + gate A                        (unchanged)
C:      s4            FLUX 2D edit                               (unchanged)
D:      s5            TRELLIS 3D edit (mod/scl/mat/glb)          (unchanged)
D2:     s5b           Deletion PLY only + inline add backfill    (CHANGED: no rough NPZ)
E_pre:  s6p           Preview render — 5 VIEW_INDICES views      (NEW)
E_qc:   sq3           VLM final gate on 5-view collage           (CHANGED: covers add; multi-view)
E:      s6 + s6b      Full render + del re-encode (survivors)    (REORDERED: after gate)
```

Stage F (`s7`) is **removed** from the config. Its logic moves into `s5b`.

---

## Component Changes

### 1. `s5b_deletion.py` — Delete rough NPZ; inline add backfill

**Remove:**
- `build_part_mask` + `export_deletion_pair` (rough NPZ generation)
- `ori_slat` loading via refiner in s5b (move to s6b instead)

**Add** after each successful deletion PLY export:
```python
_backfill_add(ctx, spec, log)
```

**New helper `_backfill_add`:** writes `edits_3d/add_NNN/meta.json` with:
- `edit_id`, `edit_type: "addition"`, `obj_id`, `shard`
- `source_del_id`: the paired deletion edit_id
- `selected_part_ids`, `view_index`: same as del spec
- `prompt`: `invert_delete_prompt(spec.prompt)` (improved; see §6)
- `target_part_desc`, `rationale`

No NPZ or PNG created in s5b. NPZ hardlinks deferred to s6b (after proper encode exists).

**Step key:** `s5b_del_mesh` (unchanged).

---

### 2. New `s6_preview.py` (step `s6p`, stage `E_pre`)

Renders 5 preview images for *every* edit using `VIEW_INDICES = [89, 90, 91, 100, 8]`
(same cameras as `overview.png` and phase1 VLM — loaded from `image_npz`).

**Output per edit:** `edits_3d/<edit_id>/preview_{0..4}.png` (after-state renders only)

**Before images:** Not saved — loaded directly from `image_npz[VIEW_INDICES]` on demand (zero cost).

**Render method by edit type:**

| Type | Source | Renderer |
|------|--------|----------|
| `mod`, `scl`, `mat`, `glb` | `after.npz` SLAT | `render_one_view(pipeline, slat, frame)` × 5 (TRELLIS) |
| `deletion` | `after.ply` | Blender at VIEW_INDICES frames |
| `addition` | del's `before.ply` (= add's after state) | Blender at VIEW_INDICES frames |

**Camera consistency:** `load_views_from_npz(ctx.image_npz, VIEW_INDICES)` returns 5 frame
dicts with `transform_matrix` + `camera_angle_x`. The same dicts go to both
`frame_to_extrinsic_intrinsic` (TRELLIS path) and Blender (deletion path).
Identical extrinsic/intrinsic → zero perspective mismatch between before/after.

**Step key:** `s6p_preview` in `status.json`.

---

### 3. `sq3_qc_e.py` — Multi-view collage; cover addition edits

**Before state:** 5 images loaded from `image_npz[VIEW_INDICES]` — always available, zero render cost.

**After state:** 5 `preview_{i}.png` files from `edits_3d/<edit_id>/`.

**Collage layout:** 2 rows × 5 columns (row 0 = before, row 1 = after).

**Coverage extended to `addition` edits:** new iterator reads `edits_3d/add_*/meta.json` on disk.
Addition edits use the same `_passes()` logic with existing `_DEFS["addition"]` threshold.

**Step key:** `sq3_qc_E` (unchanged).

---

### 4. `s6_render_3d.py` — Reuse preview frame for canonical PNG

In `run_for_object`, for each edit:
```python
preview = edit_dir / f"preview_{spec.view_index}.png"
png = edit_dir / f"{which}.png"
if which == "after" and preview.is_file() and not force:
    # hardlink preview as canonical PNG, skip TRELLIS re-render
    _hardlink_or_copy(preview, png)
    res.n_skip += 1
    continue
```

Since `preview_{i}` is already the correct camera, this eliminates the redundant
TRELLIS decode+render for the canonical view of survivors.

**Step key:** `s6_render_3d` (unchanged).

---

### 5. `s5b_deletion.py` (s6b section) — Proper encode + NPZ hardlinks for add

After `_render_and_full_encode` for `del/after.npz`:

1. Generate `del/before.npz` from original SLAT (`encode_ss` — fast, CPU-only).
2. Hardlink NPZ and PNG to `add_*` (swapping before/after):

```
del/after.npz   → add/before.npz
del/before.npz  → add/after.npz
del/after.png   → add/before.png
del/before.png  → add/after.png
del/preview_{i}.png  → NOT linked (add has its own preview from s6p)
```

**Step key:** `s6b_del_reencode` (unchanged).

---

### 6. Prompt Inversion Fix (`invert_delete_prompt`)

**Current bug:** `"Remove the X from the Y"` → `"Add the X from the Y"` (wrong preposition).

**Fix:** After verb substitution, replace the first ` from ` with ` to `:

```python
# after verb swap:
result = result.replace(" from ", " to ", 1)
```

Example transformations:
- `"Remove the engine from the car"` → `"Add the engine to the car"` ✓
- `"Delete the antenna from the robot"` → `"Add the antenna to the robot"` ✓
- `"Remove the wheel"` (no "from") → `"Add the wheel"` ✓

Function moves to `partcraft/pipeline_v2/addition_utils.py` (imported by s5b).

---

### 7. Config Changes (all shard YAMLs)

Remove stage `F`. Add stage `E_pre` before `E_qc`:

```yaml
stages:
  - {name: A,     steps: [s1, sq1]}
  - {name: C,     steps: [s4]}
  - {name: D,     steps: [s5]}
  - {name: D2,    steps: [s5b]}
  - {name: E_pre, steps: [s6p]}     # NEW
  - {name: E_qc,  steps: [sq3]}
  - {name: E,     steps: [s6, s6b]}
  # F removed
```

`ALL_STEPS` in `run.py`:
```python
ALL_STEPS = ("s1", "s2", "sq1", "s4", "s5", "s5b", "s6p", "sq2", "sq3", "s6", "s6b")
# s7 removed
```

---

### 8. `s7_addition_backfill.py` — No-op stub

```python
def run(ctxs, *, force=False, logger=None):
    """No-op: addition backfill is now inline in s5b."""
    log = logger or logging.getLogger("pipeline_v2.s7")
    log.info("[s7] skipped: addition backfill moved to s5b")
    return []
```

---

## Data Flow Summary

```
s5b:   del_*/before.ply + after.ply  (PLY only, no NPZ)
       add_*/meta.json               (inv prompt + source_del_id)

s6p:   del_*/preview_{0..4}.png     (Blender, VIEW_INDICES cameras from image_npz)
       mod_*/preview_{0..4}.png     (TRELLIS render_one_view, same frame dicts)
       add_*/preview_{0..4}.png     (Blender, del before.ply, same cameras)

sq3:   reads image_npz[VIEW_INDICES] → before row (free)
             edits_3d/<id>/preview_{0..4}.png → after row
       updates qc.json gate E for: del, mod, scl, mat, glb, add

s6b:   [survivors only]
       del_*/after.npz   (40-view Blender render → DINOv2 → SS encode)
       del_*/before.npz  (original SLAT encode_ss)
       add_*/{before,after}.npz  (hardlinked from del, swapped)
       add_*/{before,after}.png  (hardlinked from del, swapped)

s6:    [survivors only]
       */{before,after}.png  (canonical PNG; reuses preview_{view_index}.png if exists)
```

---

## Error Handling

- `s6p` render failure → mark edit preview as failed in gate E metadata → sq3 marks as fail.
- `add_*/meta.json` missing at sq3 time → skip (source del already failed; no noise in qc.json).
- Gate E fail → `s6b` / `s6` skip the edit (`is_edit_qc_failed` check before each spec).
- Blender crash in s6p → only affects the specific edit, not the whole object batch.

---

## Testing

- **Unit** `tests/test_addition_utils.py`: `invert_delete_prompt` with "from"→"to" cases.
- **Unit** `tests/test_sq3_passes.py`: existing tests still pass; addition thresholds unchanged.
- **Integration** `tests/test_pipeline_smoke.py`: s6p importable; s7 importable as no-op; ALL_STEPS updated; E_pre stage in config loads cleanly.

---

## Self-Review Notes

### Finding 1 — `s5b` already has `use_refiner=False` mode (line 136)

`run_mesh_delete_for_object` accepts `use_refiner: bool = True`. The rough NPZ path
is already guarded by `if refiner is not None`. Simplest change: make `use_refiner`
default to `False` and remove all refiner/mask/export_deletion_pair code paths entirely
(they will not be needed post-redesign). No caller outside s5b passes `use_refiner`.

### Finding 2 — `ALL_STEPS` order must be updated carefully

Current: `("s1", "s2", "sq1", "s4", "s5", "s5b", "sq2", "s6", "s6b", "s7", "sq3")`

Note that `sq3` currently appears at the END, after `s6b`. That's the bug — gate runs
after full encode. New order:

```python
ALL_STEPS = ("s1", "s2", "sq1", "s4", "s5", "s5b", "s6p", "sq2", "sq3", "s6", "s6b")
```

Also update `_STATUS_KEYS` dict (line 478) to add `"s6p": "s6p_preview"` and remove `"s7": "s7_add_backfill"`.

### Finding 3 — New `check_s6p` validator needed in `validators.py`

`check_sq3` (line 192) only checks `qc.json`. Need to add `check_s6p` that verifies
at least one `preview_0.png` exists per non-failed edit directory.  
Suggested: same pattern as `check_s5b` but glob `edits_3d/*/preview_0.png`.
