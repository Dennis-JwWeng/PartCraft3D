# Per-Type QC Pipeline Design

**Date**: 2026-04-09  
**Branch**: feature/prompt-driven-part-selection  
**Status**: Approved, pending implementation

---

## Background & Goals

The core pipeline goal is to produce **~100K high-quality 3D part-editing pairs** where:

1. Edit instructions have clear semantics — unambiguous action verb, well-specified target part.
2. The 3D result is a semantically local edit — only the target region changes.
3. Each edit type has its own cleaning/validation mechanism.

Geometric/mesh-level QC (locality IoU, penetration ratio) is deferred to Phase 2. This spec covers **Phase 1: semantic QC only**.

---

## Edit Type Taxonomy

Source of truth: `partcraft/edit_types.py`

| Type | Execution path | Notes |
|------|---------------|-------|
| `deletion` | Mesh-only (s5b) | No FLUX, no TRELLIS gen |
| `addition` | Backfill from deletion (s7) | Not VLM-produced |
| `modification` | FLUX 2D → TRELLIS S1+S2 | Large-part may promote → Global (see §Promotion) |
| `scale` | FLUX 2D → TRELLIS S1+S2 | Same as modification |
| `material` | FLUX 2D → TRELLIS S2 only | Appearance only |
| `global` | FLUX 2D → TRELLIS S2 only | Full-object, intended whole-object change |
| `identity` | No-op | Not produced by VLM, skipped |

### Large-part promotion

In `partcraft/trellis/refiner.py`, if `edit_type == Modification` and the target part covers `> large_part_threshold` (default 0.35) of total SLAT voxels, execution is silently promoted to `Global`. The `parsed.json` field `edit_type` is **never updated**.

**QC decision**: Let promoted modifications fail QC-E under `modification` thresholds. A modification that covers >35% of the object is not meaningfully local and should not enter the 100K dataset. This is correct behaviour, not a false positive. The `effective_type` field is deferred to a future analytics pass.

---

## QC Architecture

### Stage topology

```
A → A_qc → B → C → C_qc → D → D2 → E → E_qc → F
```

Changes vs current config:
- `B` (`s2_highlights`): remove `optional: true` — required for QC-C inputs.
- Add three new stages: `A_qc`, `C_qc`, `E_qc`, each containing one new step (`sq1`, `sq2`, `sq3`).
- All three QC stages use `servers: vlm` (reuse same VLM pool; no extra server lifecycle).

### Failure semantics

| Failure at | Effect |
|-----------|--------|
| QC-A (sq1) | Edit flagged `qc_fail`; s4 and s5 skip this edit |
| QC-C (sq2) | Edit flagged `qc_fail`; s5 / s5b skip this edit |
| QC-E (sq3) | Edit flagged `qc_fail`; pipeline continues, edit excluded at export |

A QC failure on one edit never aborts the object or other edits.

---

## Data Contract: `qc.json`

Location: `objects/<shard>/<obj_id>/qc.json` (alongside `status.json`).

```json
{
  "obj_id": "ABCDE",
  "shard": "02",
  "updated": "2026-04-09T12:00:00",
  "edits": {
    "del_000": {
      "edit_type": "deletion",
      "gates": {
        "A": {
          "rule": { "pass": true, "checks": {} },
          "vlm":  { "pass": true, "score": 0.9, "reason": "" }
        },
        "C": null,
        "E": { "vlm": { "pass": true, "score": 0.85, "reason": "" } }
      },
      "final_pass": true
    },
    "mod_000": {
      "edit_type": "modification",
      "gates": {
        "A": {
          "rule": { "pass": false, "checks": { "new_desc_missing": true } },
          "vlm":  null
        },
        "C": null,
        "E": null
      },
      "final_pass": false,
      "fail_gate": "A",
      "fail_reason": "new_desc_missing"
    }
  }
}
```

### Schema notes

- `gates.<gate>`: `null` = not applicable to this type (e.g. deletion has no C gate).
- `gates.<gate>.rule.checks`: only failing rule codes; empty `{}` = all rules passed.
- `gates.<gate>.vlm`: `null` if rule layer failed (VLM not called).
- `vlm.score`: float 0–1, threshold in YAML `qc.vlm_score_threshold`.
- `final_pass`: `true` iff all applicable gates passed. Export reads only this field.
- `fail_gate` / `fail_reason`: top-level for fast aggregation.

### Integration with `status.json`

```json
"sq1_qc_A": { "status": "ok", "n_pass": 14, "n_fail": 2, "ts": "..." },
"sq2_qc_C": { "status": "ok", "n_pass": 11, "n_fail": 1, "ts": "..." },
"sq3_qc_E": { "status": "ok", "n_pass": 12, "n_fail": 0, "ts": "..." }
```

---

## Step Specifications

### `sq1` — QC-A: Instruction & Structure Check

**Inputs**: `phase1/overview.png` (5×2 VLM grid) + `phase1/parsed.json`

**Phase 1 — Rule layer** (synchronous, full coverage, no VLM):

| Rule code | Check | Types |
|-----------|-------|-------|
| `prompt_empty` | `prompt` empty or < 8 chars | all |
| `parts_missing` | `selected_part_ids` empty | deletion / modification / scale / material |
| `parts_invalid` | Any ID not in `object.parts` table | same |
| `new_desc_missing` | `new_parts_desc` empty | modification |
| `target_desc_missing` | `target_part_desc` empty | modification / scale / material |
| `stage_decomp_missing` | Both `new_parts_desc_stage1` and `stage2` empty | modification |
| `verb_conflict` | Prompt main verb contradicts `edit_type` | all |

Any rule failure → record fail, skip VLM call for this edit.

**Phase 2 — VLM instruction clarity** (async concurrent, reuses `call_vlm_async` pattern):

Input: `overview.png` + text (`edit_type`, `prompt`, `target_part_desc`, `part_labels`).

Response (~200 output tokens):
```json
{ "instruction_clear": true, "part_identifiable": true, "type_consistent": true, "reason": "..." }
```

Pass: all three boolean fields `true`. Score = (sum of true) / 3.

---

### `sq2` — QC-C: 2D Region Alignment Check

**Applies to**: `FLUX_TYPES` only (modification, scale, material, global).  
deletion / addition → C gate = `null`.

**Inputs**:
- `highlights/e{idx:02d}.png` — Stage B highlight (magenta = target part)
- `edits_2d/{edit_id}_edited.png` — Stage C FLUX output

**Execution**: `asyncio.gather` all FLUX edits for one object; `max_tokens=128`.

Prompt:
```
Image 1: magenta = target part to edit. Image 2: 2D edit result.
Did the edit happen primarily in the highlighted region?
Reply ONLY: {"region_match": true/false, "reason": "one short phrase"}
```

Pass: `region_match == true`.

---

### `sq3` — QC-E: Final 3D Quality Check

**Inputs**: `edits_3d/{edit_id}/before.png` + `after.png`

**Implementation**: reuses `cleaning/vlm_filter.py` → `build_judge_prompt` + `call_vlm_judge` directly (8-dim response schema).

**Pass thresholds by type** (configurable via `qc.thresholds_by_type`):

| Type | Pass condition |
|------|---------------|
| `deletion` | `edit_executed=true` AND `correct_region=true` AND `visual_quality≥3` |
| `modification` | above AND `preserve_other=true` |
| `scale` | above AND `preserve_other=true` |
| `material` | `edit_executed=true` AND `visual_quality≥3` |
| `global` | `edit_executed=true` AND `visual_quality≥3` |
| `addition` | `edit_executed=true` AND `visual_quality≥3` |

**Large-part promotion**: modifications promoted to Global in s5 will fail `preserve_other=true` naturally. Intentional — they are not local edits. No special handling.

---

## New Files

```
partcraft/pipeline_v2/
  sq1_qc_a.py    — QC-A runner (rule layer + async VLM instruction check)
  sq2_qc_c.py    — QC-C runner (lightweight 2D region alignment)
  sq3_qc_e.py    — QC-E runner (final 3D quality, reuses vlm_filter judge)
  qc_io.py       — qc.json read/write helpers (atomic write, load, is_edit_qc_failed)
  qc_rules.py    — Rule layer pure functions (no IO, unit-testable)
```

## Modified Files

| File | Change |
|------|--------|
| `run.py` | Add `sq1/sq2/sq3` to `ALL_STEPS`; add elif branches in `run_step` |
| `validators.py` | Add `check_sq1/sq2/sq3` (verify `qc.json` written) |
| `paths.py` | Add `ctx.qc_path` → `<obj_dir>/qc.json` |
| `s4_flux_2d.py` | Skip edits where `qc_io.is_edit_qc_failed(ctx, edit_id)` |
| `s5_trellis_3d.py` | Same skip check |
| `s5b_deletion.py` | Same skip check |
| `configs/pipeline_v2_shard*.yaml` | Add QC stages; remove `optional: true` on B; add `qc:` block |

---

## YAML Config Addition

```yaml
pipeline:
  stages:
  - { name: A,     desc: "phase1 VLM",        servers: vlm,  steps: [s1] }
  - { name: A_qc,  desc: "QC-A instruction",  servers: vlm,  steps: [sq1] }
  - { name: B,     desc: "highlights",         servers: none, steps: [s2] }
  - { name: C,     desc: "FLUX 2D",            servers: flux, steps: [s4] }
  - { name: C_qc,  desc: "QC-C 2D region",    servers: vlm,  steps: [sq2] }
  - { name: D,     desc: "TRELLIS 3D edit",    servers: none, steps: [s5],      use_gpus: true }
  - { name: D2,    desc: "deletion mesh",      servers: none, steps: [s5b] }
  - { name: E,     desc: "3D rerender",        servers: none, steps: [s6, s6b], use_gpus: true }
  - { name: E_qc,  desc: "QC-E final quality", servers: vlm,  steps: [sq3] }
  - { name: F,     desc: "addition backfill",  servers: none, steps: [s7] }

qc:
  vlm_score_threshold: 0.7
  thresholds_by_type:
    deletion:     { min_visual_quality: 3 }
    modification: { min_visual_quality: 3, require_preserve_other: true }
    scale:        { min_visual_quality: 3, require_preserve_other: true }
    material:     { min_visual_quality: 3 }
    global:       { min_visual_quality: 3 }
    addition:     { min_visual_quality: 3 }
```

---

## Deferred / Out of Scope

- Geometric / mesh-level QC (locality IoU, penetration ratio) — Phase 2.
- `effective_type` tracking for large-part promotion analytics — add when needed.
- Global `qc_manifest.jsonl` — derivable from per-object `qc.json` on demand.
- Automated threshold tuning — manual iteration on pass-rate stats first.
