# Design: Parallel Execution of Stage D (s5/TRELLIS) and Stage D2 (s5b/deletion)

**Date:** 2026-04-13  
**Status:** Approved — pending implementation

---

## Background

The pipeline currently runs stages D and D2 sequentially:

- **Stage D** (`s5` / `s5_trellis_3d.py`): GPU-bound. Runs TRELLIS 3D editing for flux-type edits (`mod`, `scl`, `mat`, `glb`). Writes `edits_3d/{edit_id}/before.npz` and `after.npz`.
- **Stage D2** (`s5b` / `s5b_deletion.py`): CPU-only. Runs trimesh-direct mesh deletion for `deletion`-type edits. Writes `edits_3d/{del_xxx}/before.ply` and `after.ply`, plus `add_xxx/meta.json`.

The two stages process entirely disjoint sets of edit IDs (different prefixes: `mod/scl/mat/glb` vs `del/add`) and write to separate directories inside each object's `edits_3d/`. There is no data dependency between them; D2 does not read any output of D.

The only shared mutable state is **`status.json`** (one file per object), updated by both stages via `update_step()`. The current implementation is an unguarded read-modify-write. A stress test of 200 concurrent writes confirmed a 16.5% data-loss rate.

---

## Goal

Allow stages D and D2 to run as parallel OS processes, with the GPU pipeline (s5) and CPU pipeline (s5b) overlapping in time. The orchestrator shell script handles parallelism; the Python pipeline code is not restructured.

---

## Design

### 1. Fix: `status.json` Concurrent Write Safety

**Files:** `partcraft/pipeline_v2/status.py`, `partcraft/pipeline_v2/validators.py`

Add a per-object exclusive file lock using `fcntl.LOCK_EX` (Linux built-in, zero new dependencies). Lock file: `<object_dir>/status.json.lock`. The kernel releases the lock automatically on process exit or crash.

```python
# status.py — new internal helper
import fcntl
from contextlib import contextmanager

@contextmanager
def _status_lock(ctx: ObjectContext):
    lock_path = ctx.dir / "status.json.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        yield
```

`update_step()` wraps its entire read-modify-write inside `_status_lock`. `validators.apply_check()` also wraps its load-flip-save inside `_status_lock` (the read-only `CHECK_FNS[step](ctx)` call runs outside the lock). `save_status()` itself is unchanged.

### 2. Config: `parallel_group` Field

**Files:** All `configs/pipeline_v2_shard*.yaml` (7 files), `partcraft/pipeline_v2/scheduler.py`

Add an optional `parallel_group: <string>` field to stage entries. Stages sharing the same non-empty group string are executed concurrently by the shell orchestrator.

```yaml
  - name: D
    desc: TRELLIS 3D edit
    servers: none
    steps: [s5]
    use_gpus: true
    parallel_group: "D+D2"    # NEW
  - name: D2
    desc: deletion PLY + add meta
    servers: none
    steps: [s5b]
    parallel_group: "D+D2"    # NEW
```

**`scheduler.py` changes:**
- `Phase` dataclass gains `parallel_group: str = ""`.
- `stages_for()` reads the new field from YAML.
- New function `dump_stage_batches(cfg, stage_names) -> list[list[str]]` groups the requested stages into ordered batches, e.g. `[["A"],["C"],["D","D2"],["E"]]`. Used by the shell script.

**Constraint:** Both stages in a parallel group must have `servers: none`. If any stage in a group needs a server, the scheduler logs a warning and the group falls back to serial execution.

### 3. Shell Orchestrator: `run_parallel_group()`

**File:** `scripts/tools/run_pipeline_v2_shard.sh`

Replace the sequential `while` loop with a batch-aware main loop driven by `dump_stage_batches()` output.

```bash
run_parallel_group() {
    if [ "$#" -eq 1 ]; then
        run_pipeline_stage "$1"; return $?
    fi
    local pids=() names=("$@") rc=0
    echo "Parallel group [${names[*]}] — starting"
    for stage in "${names[@]}"; do
        run_pipeline_stage "$stage" >"$LOG_DIR/stage_${stage}.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"; local _rc=$?
        [ $_rc -ne 0 ] && { echo "[scheduler] stage ${names[$i]} FAILED (exit=$_rc)"; rc=$_rc; }
    done
    return $rc
}
```

Main loop calls `dump_stage_batches()` once, then iterates batches, calling `run_parallel_group` per batch. All parallel stage logs go to separate files (`stage_D.log`, `stage_D2.log`).

---

## Backward Compatibility

- YAML files without `parallel_group` produce single-element batches; behavior is identical to current serial execution.
- All existing `--stage <name>` direct Python invocations are unaffected.
- `STAGES=D2` env-var override continues to work.

---

## Testing

| Test | Location | Pass Criteria |
|---|---|---|
| Concurrent `update_step` no data loss | `tests/test_status_concurrent.py` (new) | 200 concurrent paired writes → 0 lost keys |
| `dump_stage_batches` groups correctly | same file | `[["A"],["C"],["D","D2"],["E_pre"]]` for shard05 config |
| Integration: D+D2 parallel run | Manual, shard05_test config | All objects have both `s5_trellis: ok` and `s5b_del_mesh: ok` |
| Regression: no-group yaml serial run | Manual | Output identical to pre-change serial run |

---

## File Change Summary

| File | Change |
|---|---|
| `partcraft/pipeline_v2/status.py` | Add `_status_lock()`, wrap `update_step()` |
| `partcraft/pipeline_v2/validators.py` | Wrap `apply_check()` save with `_status_lock()` |
| `partcraft/pipeline_v2/scheduler.py` | `Phase.parallel_group`, `dump_stage_batches()` |
| `scripts/tools/run_pipeline_v2_shard.sh` | `run_parallel_group()`, batch-aware main loop |
| `configs/pipeline_v2_shard{00,01,02,03,04,05,05_test}.yaml` | Add `parallel_group: "D+D2"` to D and D2 |
| `tests/test_status_concurrent.py` | New: concurrency correctness test |
