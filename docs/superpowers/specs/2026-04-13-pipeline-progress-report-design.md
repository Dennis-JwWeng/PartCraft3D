# Pipeline Progress Report — Design Spec

**Date:** 2026-04-13  
**Status:** Approved  
**Scope:** `scripts/tools/show_progress.py` — read-only progress snapshot for any shard

---

## Problem

Current tooling makes it hard to know the real pipeline state:

- `--count-pending` gives only an object count; `fail` and `absent` look the same
- `manifest_summary()` dumps a raw dict with no edit-level breakdown
- A `fail` object can mean "completely failed" or "mostly done, one edit missing" — the object-level number alone doesn't distinguish these
- Edit-level throughput (how many of the ~20k edits actually made it through each stage) is invisible

## Goal

A single command that reads **only existing on-disk data** (`status.json` + `qc.json`)  
and prints two clear tables — no validator calls, no writes.

```bash
python scripts/tools/show_progress.py \
    --config configs/pipeline_v2_shard02.yaml \
    --shard 02
```

---

## Output Format

### Table 1 — Object layer

One row per pipeline stage (in `pipeline.stages` order from the config YAML).
Columns: stage name, step key, object counts (ok / fail / absent), edit-level aggregates from
`status.json` (`n_ok` / `n_fail` / `n_skip` or `n_pass`), and top-3 fail reasons.

```
Shard 02 — 1213 objects  (phase1-skip=62, net=1151)
──────────────────────────────────────────────────────────────────────────────────────
Stage   step   obj:ok  obj:fail  obj:absent  │ edit:ok  edit:fail  edit:skip  fail-reason (top 3)
──────────────────────────────────────────────────────────────────────────────────────
D       s5      1103      48          —       │  12970      70          —      no_specs×10 | missing_npz×28 | …
D2      s5b     1069      82          —       │   7072       0          —      no_deletions×10 | missing_ply×62 | …
E_pre   s6p       18      10       1183       │    184       0        178      missing_image_npz×10
E_qc    sq3       18       0       1195       │    229     125          8      —
──────────────────────────────────────────────────────────────────────────────────────
```

**Definitions:**

| Column | Source |
|---|---|
| obj:ok | steps.<step>.status == "ok" |
| obj:fail | steps.<step>.status == "fail" |
| obj:absent | step key not present in steps |
| phase1-skip | steps.s1_phase1.status == "skip" — skipped objects excluded from obj counts |
| edit:ok | sum of steps.<step>.n_ok across all objects |
| edit:fail | sum of steps.<step>.n_fail across all objects |
| edit:skip | sum of steps.<step>.n_skip across all objects |
| edit:pass/fail/skip | for sq3: sum of n_pass / n_fail / n_skip |
| fail-reason | aggregate steps.<step>.reason field + prefix of validation.missing entries |

`—` means the field is not recorded for that step (not "zero").

### Table 2 — Edit layer (sq3 scope)

```
Edit throughput (sq3 — final QC gate)
  s1 kept edits (planned total)   :  20151
  reached sq3                     :    362  (  1.8%)
    ├─ pass                       :    229  ( 63.3%)
    ├─ fail                       :    125  ( 34.5%)
    └─ skip                       :      8  (  2.2%)
  not yet reached sq3             :  19789
```

Source: sum of steps.s1_phase1.n_kept (planned total) and steps.sq3_qc_E.n_pass/n_fail/n_skip.

---

## Implementation

### File

`scripts/tools/show_progress.py` — standalone, no new modules.

### Dependencies

- Standard library only: json, glob, argparse, collections.Counter
- yaml (already in env) to read config and resolve data.output_dir + pipeline.stages
- No imports from partcraft.* — avoids loading heavy deps

### Logic

```
1. Load config YAML → resolve output_dir, shard, pipeline.stages order
2. Glob all status.json under objects/<shard>/*/status.json
3. For each object:
   a. Read status.json
   b. Classify phase1-skip
   c. For each stage step: accumulate obj:ok/fail/absent, edit counts, fail reasons
4. Print Table 1
5. Print Table 2 (sq3 edit layer)
```

### Fail-reason aggregation

For each fail object at a given step:
1. Check steps.<step>.reason — if present, use directly (e.g. no_specs, no_deletions)
2. Otherwise extract file-prefix from validation.missing entries
3. Counter across all fail objects → top 3

### Step → status-key mapping

Reuse same mapping as run.py:
  s5→s5_trellis, s5b→s5b_del_mesh, s6p→s6p_preview, sq3→sq3_qc_E,
  s6→s6_render_3d, s6b→s6b_del_reencode, s1→s1_phase1, sq1→sq1_qc_A

### Edit-field mapping per step

  s5_trellis:   (n_ok, n_fail, n_skip)
  s5b_del_mesh: (n_ok, n_fail, n_skip)
  s6p_preview:  (n_ok, n_fail, n_skip)
  sq3_qc_E:     (n_pass, n_fail, n_skip)   # "pass" not "ok"

Steps without edit-level fields (s1, sq1, s4, s6, s6b) show — in edit columns.

### Optional --stages filter

Comma-separated stage names (e.g. D,D2,E_pre,E_qc) to limit which rows appear.

---

## Non-goals

- No validator calls (VALIDATORS[step](ctx)) — that is what apply_check is for
- No file existence checks — read only what status.json already recorded
- No writes of any kind
- No HTML output (use generate_qc_report.py for that)
- No live refresh / watch mode

---

## Usage examples

```bash
# shard02
python scripts/tools/show_progress.py --config configs/pipeline_v2_shard02.yaml --shard 02

# shard05 test run
python scripts/tools/show_progress.py --config configs/pipeline_v2_shard05_test.yaml --shard 05

# show only up to E_qc
python scripts/tools/show_progress.py --config configs/pipeline_v2_shard02.yaml --shard 02 \
    --stages D,D2,E_pre,E_qc
```
