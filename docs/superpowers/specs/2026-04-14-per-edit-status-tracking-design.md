# Per-Edit Status Tracking Design

**Date:** 2026-04-14  
**Status:** Approved for implementation  
**Scope:** `partcraft/pipeline_v2/` — adds `edit_status.json` as per-edit state source alongside existing `status.json` and `qc.json`

---

## Problem

The current pipeline tracks state at two incompatible granularities:

- **`status.json`** (step-level): "Did step s4 complete for this object?" Records aggregate counts (`n_ok`, `n_fail`, `n_skip`) but gives no per-edit detail.
- **`qc.json`** (gate-quality): "Did edit X pass Gate A/C/E?" Records quality signals but nothing about processing outcomes.

To know "what is the current state of edit X?", you must cross-reference three places. This creates:
1. **No single source of truth**: inconsistent answers across 3 files.
2. **Misleading step status**: `s4_flux_2d: fail` with `n_fail=0` (191 objects, 38% false-fail in shard06).
3. **Opaque resume logic**: different steps use different skip strategies.
4. **No funnel statistics**: can't answer "how many edits reached s5?" or "blocked at gate_a?".
5. **Gate gap**: `s6` and `s6b` do not check `gate_e` — edits failing final quality gate still get rendered.

---

## Design Decisions

**D1: Add `edit_status.json`, keep existing files.**  
`status.json` and `qc.json` continue unchanged. `edit_status.json` is new. Processing steps additionally write to it. Resume logic migrates to read from it.

**D2: Only write stages that actually ran.**  
Absence of a stage key = "not yet reached". No `gate_blocked` marker — derivable from predecessor gate status, and absence avoids polluting funnel denominators.

**D3: gate_e is a hard routing gate for s6/s6b.**  
`s6` and `s6b` must skip edits where `gate_e = fail`. This is not in current code — **fixing s6 and s6b is part of this work**.

**D4: gate_c is optional, config-driven.**  
`STAGE_PREREQ_GATE` is derived from active config steps. If `sq2` is added to a config, `s5`'s prereq automatically becomes `gate_c`.

**D5: addition edits require no gate_a.**  
Additions exist only when their parent deletion passed gate_a (`s5b` only creates `meta.json` for gate_a-passing deletions). Stage path: `s6p → gate_e`.

---

## Stage Paths per Edit Type

```
flux (modification / scale / material / global):
  gate_a → s4 → [gate_c?] → s5 → s6p → gate_e → s6

deletion:
  gate_a → s5b → s6p → gate_e → s6b

addition (gate_a guaranteed by del construction):
  s6p → gate_e
```

---

## Status Values

| Value | Applies to | Meaning | Resume |
|---|---|---|---|
| `"pass"` | gate_ stages | Quality OK | Continue downstream |
| `"fail"` | gate_ stages | Quality rejected | Stop; downstream not written |
| `"done"` | processing stages | Completed successfully | Skip |
| `"error"` + `reason` | processing stages | Technical failure | Retry |
| *(absent)* | any | Not yet reached | Run (if prereq gate passed) |

No `"gate_blocked"` value — blocked status inferred from absent key + failed predecessor gate.

---

## `edit_status.json` Schema

```
objects/<shard>/<obj_id>/
├── phase1/parsed.json
├── qc.json              ← unchanged (legacy gate quality signals)
├── status.json          ← unchanged (step-level aggregates)
└── edit_status.json     ← NEW: per-edit operational state source
```

```json
{
  "obj_id": "6ed5f68...",
  "shard": "06",
  "schema_version": 1,
  "updated": "2026-04-14T07:00:00",
  "edits": {
    "mod_..._001": {
      "edit_type": "modification",
      "stages": {
        "gate_a": {"status": "pass", "ts": "2026-04-14T01:24:19"},
        "s4":     {"status": "done", "ts": "2026-04-14T06:30:27"},
        "s5":     {"status": "done", "ts": "..."},
        "s6p":    {"status": "done", "ts": "..."},
        "gate_e": {"status": "pass", "ts": "..."},
        "s6":     {"status": "done", "ts": "..."}
      }
    },
    "del_..._000": {
      "edit_type": "deletion",
      "stages": {
        "gate_a": {"status": "fail", "reason": "zero_visible_pixels", "ts": "..."}
      }
    },
    "mod_..._000": {
      "edit_type": "modification",
      "stages": {
        "gate_a": {"status": "pass", "ts": "..."},
        "s4":     {"status": "error", "reason": "flux_server_timeout", "ts": "..."}
      }
    },
    "add_..._001": {
      "edit_type": "addition",
      "stages": {
        "s6p":    {"status": "done", "ts": "..."},
        "gate_e": {"status": "pass", "ts": "..."}
      }
    }
  }
}
```

---

## New Module: `partcraft/pipeline_v2/edit_status_io.py`

### Public API

```python
load_edit_status(ctx) -> dict
save_edit_status(ctx, data) -> None          # atomic write (tempfile + os.replace)

update_edit_stage(
    ctx, edit_id, edit_type, stage_key,
    *, status: str, reason: str | None = None
) -> None                                    # process-safe RMW (lockf + threading.Lock)

edit_needs_step(ctx, edit_id, stage_key, prereq_map: dict) -> bool
gate_already_done(ctx, edit_id, gate_key) -> bool

build_prereq_map(cfg: dict) -> dict[str, str | None]
```

### `build_prereq_map` — Config-Derived Gate Prerequisites

```python
def build_prereq_map(cfg: dict) -> dict[str, str | None]:
    active = {step for stage in cfg["pipeline"]["stages"]
                   for step in stage.get("steps", [])}
    return {
        "s4":  "gate_a" if "sq1" in active else None,
        "s5b": "gate_a" if "sq1" in active else None,
        "s5":  "gate_c" if "sq2" in active else
               "gate_a" if "sq1" in active else None,
        "s6p": "gate_a" if "sq1" in active else None,
        "s6":  "gate_e" if "sq3" in active else None,   # hard gate (code gap fix)
        "s6b": "gate_e" if "sq3" in active else None,   # hard gate (code gap fix)
    }
```

### `edit_needs_step` — Single Resume Function

```python
def edit_needs_step(ctx, edit_id, stage_key, prereq_map) -> bool:
    stages = (load_edit_status(ctx)
              .get("edits", {})
              .get(edit_id, {})
              .get("stages", {}))
    prereq = prereq_map.get(stage_key)
    if prereq:
        gate_status = stages.get(prereq, {}).get("status")
        if gate_status != "pass":   # absent or "fail" → cannot run
            return False
    own = stages.get(stage_key)
    if own is None:
        return True
    return own.get("status") == "error"   # error → retry; done → skip
```

---

## Step Integration

Each processing step replaces its file-existence resume check with `edit_needs_step` and adds `update_edit_stage` after running. Gate steps (sq1, sq3) add `update_edit_stage` alongside their existing `qc.json` write. Existing `qc.json` and `status.json` writes are **not removed**.

### Gate gap fix in s6 and s6b

**`s6_render_3d.py`** (per-edit loop, currently no gate check):
```python
# Add before processing each edit:
if not edit_needs_step(ctx, edit_id, "s6", prereq_map):
    res.n_skip += 1
    continue
```

**`s5b_deletion.py`** (`run_reencode_for_object`, currently checks only gate_a):
```python
# Replace: if is_gate_a_failed(ctx, spec.edit_id):
# With:
if not edit_needs_step(ctx, spec.edit_id, "s6b", prereq_map):
    res.n_skip += 1
    continue
```

---

## Backfill Script: `scripts/tools/backfill_edit_status.py`

Reconstructs `edit_status.json` for all objects in a shard from existing data. Idempotent.

### Inference Rules

| Stage | Source |
|---|---|
| `gate_a` | `qc.json → edits[id].gates.A.rule.pass` → `"pass"` / `"fail"` |
| `gate_c` | `qc.json → edits[id].gates.C` (if present) |
| `gate_e` | `qc.json → edits[id].gates.E` (if present) |
| `s4` | `edits_2d/{id}_edited.png` exists |
| `s5` | `edits_3d/{id}/before.npz` AND `after.npz` exist |
| `s5b` | `edits_3d/{id}/after_new.glb` exists |
| `s6p` | `edits_3d/{id}/preview_0.png` exists |
| `s6` | `edits_3d/{id}/before.png` AND `after.png` exist |
| `s6b` | `edits_3d/{id}/after.npz` (deletion, has `ss` key in npz) |
| addition | Discovered from `edits_3d/add_*/meta.json`; infer `s6p` / `gate_e` from files |

```bash
python scripts/tools/backfill_edit_status.py \
    --config configs/pipeline_v2_shard06.yaml \
    --shard 06 [--obj-ids id1,id2] [--force]
```

---

## Funnel Statistics: `scripts/tools/summarize_edit_status.py`

Config-aware. Only shows stages active in the config. Funnel denominator = edits that have an entry for that stage (i.e., the stage was attempted).

```
=== Funnel: pipeline_v2_shard06 ===  objects=856  edits=8234

stage     eligible   pass/done  fail   error  pending
────────  ────────   ─────────  ────   ─────  ───────
gate_a       8234        8022    212       0        0
  s4         7104        6980      0      24      100
  s5b         918         892      0      26        0
s5           6980        6820      0     160        0
s6p          7762        7700      0      62        0
gate_e       7700        7600    100       0        0
  s6         5834        5820      0      14        0
  s6b         750         748      0       2        0
```

---

## Files Changed

| File | Change | Description |
|---|---|---|
| `partcraft/pipeline_v2/edit_status_io.py` | **New** | Core I/O, `build_prereq_map`, `edit_needs_step`, `gate_already_done` |
| `partcraft/pipeline_v2/sq1_qc_a.py` | Extend | Write `gate_a` result to `edit_status.json` |
| `partcraft/pipeline_v2/sq2_qc_c.py` | Extend | Write `gate_c` result (for future config activation) |
| `partcraft/pipeline_v2/sq3_qc_e.py` | Extend | Write `gate_e` result to `edit_status.json` |
| `partcraft/pipeline_v2/s4_flux_2d.py` | Extend | `edit_needs_step` + write `s4` result |
| `partcraft/pipeline_v2/s5_trellis_3d.py` | Extend | `edit_needs_step` + write `s5` result |
| `partcraft/pipeline_v2/s5b_deletion.py` | Extend | `edit_needs_step` + write `s5b`/`s6b`; **fix gate_e gap in s6b** |
| `partcraft/pipeline_v2/s6_preview.py` | Extend | `edit_needs_step` + write `s6p` result |
| `partcraft/pipeline_v2/s6_render_3d.py` | Extend | `edit_needs_step` + write `s6`; **fix gate_e gap** |
| `scripts/tools/backfill_edit_status.py` | **New** | Reconstruct `edit_status.json` from existing shard data |
| `scripts/tools/summarize_edit_status.py` | **New** | Config-aware funnel statistics CLI |

**Not changed:** `status.json` write paths, `qc.json` write paths, `qc_io.py`, `validators.py`, scheduler, config YAML.

---

## Migration Plan

**Phase 0 (already done):** Fix `s4` `all_present` bug; fix `sq1` deletion zero_visible_pixels exclusion.

**Phase 1 — Core + gate writers:**  
Implement `edit_status_io.py`. Extend sq1 and sq3. Implement `backfill_edit_status.py`. Run backfill on all completed shards.

**Phase 2 — Processing step writers + gate_e fix:**  
Extend s4, s5, s5b, s6p, s6, s6b. Replace file-existence resume checks. Fix gate_e gap in s6 and s6b.

**Phase 3 — Statistics tool:**  
Implement `summarize_edit_status.py`. Validate funnel output against expected counts.

**Phase 4 (deferred):**  
After pipeline stable: mark `qc.json` gates as legacy read-only; clean up redundant file-existence checks.

---

## Invariants

1. `edit_status.json` only contains stages that actually ran — no speculative writes.
2. Gate stages written by sq steps; processing stages written by the step itself.
3. `status.json` and `qc.json` are never modified — full backward compatibility.
4. Backfill is idempotent.
5. `edit_needs_step` is the single authoritative resume function for all processing steps.
6. `build_prereq_map` always reads from config — no hardcoded gate constants.
