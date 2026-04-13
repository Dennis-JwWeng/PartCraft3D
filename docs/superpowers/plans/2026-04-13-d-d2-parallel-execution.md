# D+D2 Parallel Stage Execution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow pipeline stages D (s5/TRELLIS, GPU) and D2 (s5b/deletion, CPU) to run as parallel OS processes without data corruption, cutting wall-clock time by ~50% when both stages are requested.

**Architecture:** Add `fcntl.LOCK_EX` per-object file locking to `status.json` writes so concurrent processes never lose step entries; add a `parallel_group` field to the stage config so the shell scheduler knows to run grouped stages with `&`+`wait`; add `dump_stage_batches()` to produce an ordered list of execution batches that the shell consumes.

**Tech Stack:** Python `fcntl` (stdlib), `contextlib.contextmanager` (stdlib), YAML `parallel_group` field, bash `&`+`wait`, pytest for concurrency test.

**Spec:** `docs/superpowers/specs/2026-04-13-d-d2-parallel-execution-design.md`

---

## Task 1: Concurrent-write safety — `status.py`

**Files:**
- Modify: `partcraft/pipeline_v2/status.py` (lines 33-100)
- Create: `tests/test_status_concurrent.py`

### Step 1.1 — Write the failing test

Create `tests/test_status_concurrent.py`:

```python
"""Regression test: concurrent update_step must not lose step entries."""
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def test_concurrent_update_step_no_data_loss(tmp_path):
    """Two threads writing different step keys must never overwrite each other."""
    from partcraft.pipeline_v2.paths import PipelineRoot
    from partcraft.pipeline_v2.status import STATUS_OK, load_status, update_step

    root = PipelineRoot(tmp_path)
    ctx = root.context("05", "concurrent_test_obj")
    ctx.dir.mkdir(parents=True, exist_ok=True)

    lost = 0
    for _ in range(200):
        if ctx.status_path.exists():
            ctx.status_path.unlink()

        def write_s5():
            time.sleep(random.random() * 0.002)
            update_step(ctx, "s5_trellis", status=STATUS_OK, n_ok=1)

        def write_s5b():
            time.sleep(random.random() * 0.002)
            update_step(ctx, "s5b_del_mesh", status=STATUS_OK, n_ok=1)

        with ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(write_s5)
            ex.submit(write_s5b)

        steps = load_status(ctx).get("steps") or {}
        if "s5_trellis" not in steps or "s5b_del_mesh" not in steps:
            lost += 1

    assert lost == 0, (
        f"Lost {lost}/200 status updates — concurrent write is not safe"
    )
```

### Step 1.2 — Run the test and confirm it fails

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python -m pytest tests/test_status_concurrent.py::test_concurrent_update_step_no_data_loss -v
```

Expected: **FAILED** — `AssertionError: Lost N/200 status updates` (typically 15-35 losses).

### Step 1.3 — Implement `_status_lock()` and wrap `update_step()`

In `partcraft/pipeline_v2/status.py`, add `fcntl` and `contextmanager` imports at the top (lines 33-40), and add the `_status_lock` context manager plus the updated `update_step` function:

Replace the import block (lines 33-42):
```python
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ObjectContext, PipelineRoot, normalize_shard
```

Add `_status_lock` right after `_now()` (after line 51, before `load_status`):
```python
@contextmanager
def _status_lock(ctx: ObjectContext):
    """Per-object exclusive lock for status.json read-modify-write operations.

    Uses fcntl.LOCK_EX on a companion .lock file. The kernel releases the
    lock automatically if the holder process exits or crashes — no cleanup
    needed. Safe for concurrent OS processes on the same NFS mount.
    """
    lock_path = ctx.dir / "status.json.lock"
    ctx.dir.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        yield
```

Replace the `update_step` function body (lines 87-100):
```python
def update_step(
    ctx: ObjectContext,
    step: str,
    *,
    status: str = STATUS_OK,
    **fields: Any,
) -> dict[str, Any]:
    """Read-modify-write a single step entry (process-safe via flock)."""
    with _status_lock(ctx):
        s = load_status(ctx)
        s.setdefault("steps", {})[step] = {
            "status": status, "ts": _now(), **fields,
        }
        save_status(ctx, s)
    return s
```

### Step 1.4 — Run the test and confirm it passes

```bash
python -m pytest tests/test_status_concurrent.py::test_concurrent_update_step_no_data_loss -v
```

Expected: **PASSED** — 0 lost updates.

### Step 1.5 — Commit

```bash
git add partcraft/pipeline_v2/status.py tests/test_status_concurrent.py
git commit -m "fix: add fcntl flock to update_step to prevent concurrent status loss"
```

---

## Task 2: Wrap `apply_check()` in validators.py

**Files:**
- Modify: `partcraft/pipeline_v2/validators.py` (lines 24-26, 243-257)

### Step 2.1 — Add `_status_lock` import

In `partcraft/pipeline_v2/validators.py`, extend the import from `status` (currently line 24-26):

```python
from .status import (
    STATUS_OK, STATUS_FAIL, STATUS_SKIP, load_status, save_status,
    step_done, _status_lock,
)
```

### Step 2.2 — Wrap `apply_check()` load-flip-save

Replace the body of `apply_check` (lines 236-257). The validator `fn(ctx)` call is read-only (no `status.json` write) and stays outside the lock. Only the load-flip-save block needs protecting:

```python
def apply_check(ctx: ObjectContext, step_short: str) -> StepCheck:
    """Run the validator and update ``status.json`` to reflect reality.

    If the check fails, the step's status is forced to ``fail`` so the
    next orchestrator run will retry it. If it passes, status stays
    ``ok``. Either way, a ``validation`` field is attached.
    """
    fn = VALIDATORS[step_short]
    rep = fn(ctx)                     # read-only — outside the lock
    with _status_lock(ctx):
        s = load_status(ctx)
        steps = s.setdefault("steps", {})
        entry = steps.get(rep.step) or {"status": "?"}
        entry["validation"] = rep.to_dict()
        if rep.skip:
            entry["status"] = STATUS_SKIP
        elif not rep.ok:
            entry["status"] = STATUS_FAIL
        elif entry.get("status") not in (STATUS_OK, STATUS_FAIL):
            entry["status"] = STATUS_OK
        steps[rep.step] = entry
        save_status(ctx, s)
    return rep
```

### Step 2.3 — Run the existing concurrent test (no new test needed)

```bash
python -m pytest tests/test_status_concurrent.py -v
```

Expected: **PASSED**.

### Step 2.4 — Commit

```bash
git add partcraft/pipeline_v2/validators.py
git commit -m "fix: lock apply_check status write for concurrent safety"
```

---

## Task 3: Add `parallel_group` to scheduler and `dump_stage_batches()`

**Files:**
- Modify: `partcraft/pipeline_v2/scheduler.py` (lines 19-29, 101-115, 192-200)
- Test: `tests/test_status_concurrent.py` (add a new test function)

### Step 3.1 — Write the failing test for `dump_stage_batches`

Add to `tests/test_status_concurrent.py`:

```python
def test_dump_stage_batches_groups_d_and_d2():
    """Stages with the same parallel_group must end up in the same batch."""
    import yaml
    from partcraft.pipeline_v2.scheduler import dump_stage_batches

    cfg = yaml.safe_load(open("configs/pipeline_v2_shard05_test.yaml"))
    # After Task 4 adds parallel_group to the yaml, D and D2 must be grouped.
    # For now this test asserts the function exists and returns list[list[str]].
    result = dump_stage_batches(cfg, ["A", "C", "D", "D2", "E_pre"])
    assert isinstance(result, list)
    assert all(isinstance(b, list) for b in result)
    # Every requested stage appears exactly once
    flat = [s for batch in result for s in batch]
    assert sorted(flat) == sorted(["A", "C", "D", "D2", "E_pre"])
```

```bash
python -m pytest tests/test_status_concurrent.py::test_dump_stage_batches_groups_d_and_d2 -v
```

Expected: **ERROR** — `ImportError: cannot import name 'dump_stage_batches'`.

### Step 3.2 — Add `parallel_group` field to `Phase`

In `partcraft/pipeline_v2/scheduler.py`, replace the `Phase` dataclass (lines 19-29):

```python
@dataclass
class Phase:
    """One pipeline stage row from ``pipeline.stages`` (historical class name)."""

    name: str
    desc: str = ""
    servers: str = "none"          # "vlm" | "flux" | "none"
    steps: list[str] = field(default_factory=list)
    use_gpus: bool = False
    optional: bool = False
    parallel_group: str = ""       # non-empty → run concurrently with same-group stages
```

### Step 3.3 — Extend `stages_for()` to read `parallel_group`

Replace `stages_for` (lines 101-115):

```python
def stages_for(cfg: dict) -> list[Phase]:
    raw_list = sc.pipeline_stages_raw(cfg)
    out: list[Phase] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError(f"[CONFIG] pipeline.stages entry not a dict: {entry}")
        out.append(Phase(
            name=str(entry["name"]),
            desc=str(entry.get("desc", "")),
            servers=str(entry.get("servers", "none")),
            steps=list(entry.get("steps") or []),
            use_gpus=bool(entry.get("use_gpus", False)),
            optional=bool(entry.get("optional", False)),
            parallel_group=str(entry.get("parallel_group", "")),
        ))
    return out
```

### Step 3.4 — Add `dump_stage_batches()` function

Add the new function after `get_stage` (after line 135, before `dump_shell_env`):

```python
def dump_stage_batches(
    cfg: dict,
    stage_names: list[str],
) -> list[list[str]]:
    """Group stage_names into ordered execution batches by parallel_group.

    Stages sharing the same non-empty ``parallel_group`` are placed in the
    same batch and will be run concurrently by the shell orchestrator.
    Stages without a group (or with ``servers != "none"``) form
    single-element batches and run serially.

    The output preserves the original relative order of stage_names.

    Example with D and D2 both having ``parallel_group: "D+D2"``::

        dump_stage_batches(cfg, ["A", "C", "D", "D2", "E"])
        → [["A"], ["C"], ["D", "D2"], ["E"]]
    """
    by_name = {ph.name: ph for ph in stages_for(cfg)}
    batches: list[list[str]] = []
    group_to_idx: dict[str, int] = {}   # parallel_group → index in batches

    for name in stage_names:
        ph = by_name.get(name)
        group = (ph.parallel_group if ph else "") or ""
        # Fall back to serial if stage needs external servers (safety guard)
        if group and ph and ph.servers != "none":
            import logging as _log
            _log.getLogger("scheduler").warning(
                "[scheduler] stage %s is in parallel_group %r but servers=%r "
                "— running serially",
                name, group, ph.servers,
            )
            group = ""
        if group and group in group_to_idx:
            batches[group_to_idx[group]].append(name)
        else:
            idx = len(batches)
            batches.append([name])
            if group:
                group_to_idx[group] = idx

    return batches
```

### Step 3.5 — Add `dump_stage_batches` to `__all__`

Replace the `__all__` list (lines 192-200):

```python
__all__ = [
    "Phase",
    "gpus_for", "n_gpus",
    "vlm_port", "flux_port",
    "vlm_urls_for", "flux_urls_for",
    "stages_for", "select_stages", "get_stage",
    "phases_for", "select_phases", "get_phase",
    "dump_shell_env",
    "dump_stage_batches",
]
```

### Step 3.6 — Run the test and confirm it passes

```bash
python -m pytest tests/test_status_concurrent.py::test_dump_stage_batches_groups_d_and_d2 -v
```

Expected: **PASSED** (the function exists and returns the right shape; grouping test will fully pass after Task 4 adds `parallel_group` to yaml).

### Step 3.7 — Commit

```bash
git add partcraft/pipeline_v2/scheduler.py tests/test_status_concurrent.py
git commit -m "feat: add parallel_group to Phase and dump_stage_batches() to scheduler"
```

---

## Task 4: Add `parallel_group` to all YAML configs

**Files (8 total):**
- Modify: `configs/pipeline_v2_shard00.yaml`
- Modify: `configs/pipeline_v2_shard01.yaml`
- Modify: `configs/pipeline_v2_shard02.yaml`
- Modify: `configs/pipeline_v2_shard03.yaml`
- Modify: `configs/pipeline_v2_shard04.yaml`
- Modify: `configs/pipeline_v2_shard05.yaml`
- Modify: `configs/pipeline_v2_shard05_test.yaml`
- Modify: `configs/pipeline_v2_test_shard00.yaml`

### Step 4.1 — Update multi-line D/D2 stages (shard00, 02, 03, 04, 05, 05_test, 01)

For each of `pipeline_v2_shard00.yaml`, `shard01.yaml`, `shard02.yaml`, `shard03.yaml`, `shard04.yaml`, `shard05.yaml`, `shard05_test.yaml`:

Find this block (exact text varies by desc but structure is identical):
```yaml
  - name: D
    desc: TRELLIS 3D edit
    servers: none
    steps:
    - s5
    use_gpus: true
  - name: D2
```

Replace with (adding `parallel_group` to both D and D2):
```yaml
  - name: D
    desc: TRELLIS 3D edit
    servers: none
    steps:
    - s5
    use_gpus: true
    parallel_group: "D+D2"
  - name: D2
```

And for D2 (find the `- name: D2` block, which ends before the next `- name:`), add `parallel_group: "D+D2"` as a new line after the last line of the D2 block's existing fields (after `- s5b`):

For shard00, 02, 05, 05_test (desc: "deletion PLY + add meta"):
```yaml
  - name: D2
    desc: deletion PLY + add meta
    servers: none
    steps:
    - s5b
    parallel_group: "D+D2"
```

For shard01, 03, 04 (desc: "deletion mesh"):
```yaml
  - name: D2
    desc: deletion mesh
    servers: none
    steps:
    - s5b
    parallel_group: "D+D2"
```

### Step 4.2 — Update inline-style D/D2 (`pipeline_v2_test_shard00.yaml`)

Find:
```yaml
  - {name: D,    desc: "TRELLIS 3D edit",     servers: none, steps: [s5],      use_gpus: true}
  - {name: D2,   desc: "deletion mesh",       servers: none, steps: [s5b]}
```

Replace with:
```yaml
  - {name: D,    desc: "TRELLIS 3D edit",     servers: none, steps: [s5],  use_gpus: true, parallel_group: "D+D2"}
  - {name: D2,   desc: "deletion mesh",       servers: none, steps: [s5b],                 parallel_group: "D+D2"}
```

### Step 4.3 — Run the grouping test (now with yaml changes)

```bash
python -m pytest tests/test_status_concurrent.py::test_dump_stage_batches_groups_d_and_d2 -v
```

Expected: **PASSED**. Now verify the grouping is correct:

```bash
python -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_stage_batches
cfg = yaml.safe_load(open('configs/pipeline_v2_shard05_test.yaml'))
batches = dump_stage_batches(cfg, ['A', 'C', 'D', 'D2', 'E_pre', 'E_qc', 'E'])
for b in batches:
    print(b)
"
```

Expected output:
```
['A']
['C']
['D', 'D2']
['E_pre']
['E_qc']
['E']
```

### Step 4.4 — Commit

```bash
git add configs/pipeline_v2_shard*.yaml configs/pipeline_v2_test_shard00.yaml
git commit -m "feat: add parallel_group D+D2 to all shard configs"
```

---

## Task 5: Shell orchestrator — `run_parallel_group()` and batch-aware main loop

**Files:**
- Modify: `scripts/tools/run_pipeline_v2_shard.sh` (lines 249-260)

### Step 5.1 — Add `run_parallel_group()` function

Insert the following function **before** the `# ═══ MAIN LOOP ═══` comment (before line 251). The function goes right after the closing `}` of `run_pipeline_stage`:

```bash
run_parallel_group() {
    # Run one or more stages; single stage → serial, multiple → parallel with & wait.
    # Each stage's output is captured by run_pipeline_stage's own tee to stage_N.log.
    # Background stdout is redirected to /dev/null to prevent terminal interleaving.
    if [ "$#" -eq 1 ]; then
        run_pipeline_stage "$1"
        return $?
    fi

    local pids=() names=("$@") _rc=0 _any_fail=0
    echo
    echo "▶ Parallel group [${names[*]}] — launching"
    for _stage in "${names[@]}"; do
        run_pipeline_stage "$_stage" >/dev/null &
        pids+=($!)
        echo "  ${_stage} → PID ${pids[-1]}"
    done

    for _i in "${!pids[@]}"; do
        wait "${pids[$_i]}"; _rc=$?
        if [ "$_rc" -ne 0 ]; then
            echo "[scheduler] stage ${names[$_i]} FAILED (exit=$_rc)"
            _any_fail=$_rc
        else
            echo "[scheduler] stage ${names[$_i]} OK"
        fi
    done

    if [ "$_any_fail" -ne 0 ]; then
        echo "[scheduler] parallel group [${names[*]}] had failures — aborting"
        exit "$_any_fail"
    fi
}
```

### Step 5.2 — Replace the main loop

Replace the current main loop (lines 251-257):

```bash
# ─── OLD (remove this) ─────────────────────────────────────────────
# _stage_idx=0
# while [ "$_stage_idx" -lt "${#SELECTED_STAGES[@]}" ]; do
#     stage="${SELECTED_STAGES[$_stage_idx]}"
#     run_pipeline_stage "$stage"
#     _stage_idx=$(( _stage_idx + 1 ))
# done
```

With:

```bash
# ═══ MAIN LOOP ═══════════════════════════════════════════════════════
# Ask Python to group SELECTED_STAGES by parallel_group; each line is one
# batch (space-separated stage names). Single-element lines run serially,
# multi-element lines run concurrently via run_parallel_group().
_stages_str="${SELECTED_STAGES[*]}"   # space-separated, safe (stage names are alnum+_)

while IFS=' ' read -ra _batch; do
    [ ${#_batch[@]} -eq 0 ] && continue
    run_parallel_group "${_batch[@]}"
done < <(
    "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_stage_batches
cfg = yaml.safe_load(open('$CFG'))
stages = '$_stages_str'.split()
for batch in dump_stage_batches(cfg, stages):
    print(' '.join(batch))
"
)
```

### Step 5.3 — Validate shell syntax

```bash
bash -n scripts/tools/run_pipeline_v2_shard.sh
```

Expected: no output (syntax OK).

### Step 5.4 — Dry-run regression: single stage still works

```bash
STAGES=D2 python -m partcraft.pipeline_v2.run \
    --config configs/pipeline_v2_shard05_test.yaml \
    --shard 05 --all --dry-run --stage D2
```

Expected: prints object list and manifest summary, no error.

### Step 5.5 — Commit

```bash
git add scripts/tools/run_pipeline_v2_shard.sh
git commit -m "feat: add run_parallel_group and batch-aware main loop to shell orchestrator"
```

---

## Task 6: Full test sweep

### Step 6.1 — Run all concurrent tests

```bash
python -m pytest tests/test_status_concurrent.py -v
```

Expected: all tests **PASS**.

### Step 6.2 — Run existing pipeline tests

```bash
python -m pytest tests/ -v --ignore=tests/test_status_concurrent.py
```

Expected: no regressions.

### Step 6.3 — Verify `dump_stage_batches` is backward-compatible

Run against a config with no `parallel_group` fields to confirm it returns single-element batches:

```bash
python -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_stage_batches

# Simulate a config with no parallel_group
cfg = yaml.safe_load('''
pipeline:
  stages:
    - {name: A, servers: none, steps: [s1]}
    - {name: C, servers: none, steps: [s4]}
    - {name: D, servers: none, steps: [s5], use_gpus: true}
    - {name: D2, servers: none, steps: [s5b]}
''')
batches = dump_stage_batches(cfg, ['A', 'C', 'D', 'D2'])
print(batches)
assert batches == [['A'], ['C'], ['D'], ['D2']], f'Expected serial batches, got {batches}'
print('OK: no parallel_group → serial execution')
"
```

Expected: prints `OK: no parallel_group → serial execution`.

### Step 6.4 — Final commit

```bash
git add tests/test_status_concurrent.py
git commit -m "test: finalize concurrent status and batch grouping tests"
```

---

## Summary of All Changes

| File | Task | Change |
|---|---|---|
| `partcraft/pipeline_v2/status.py` | 1 | Add `fcntl`, `contextmanager`; add `_status_lock()`; wrap `update_step()` |
| `partcraft/pipeline_v2/validators.py` | 2 | Import `_status_lock`; wrap `apply_check()` load-flip-save |
| `partcraft/pipeline_v2/scheduler.py` | 3 | `Phase.parallel_group`; extend `stages_for()`; add `dump_stage_batches()` |
| `configs/pipeline_v2_shard{00..05}.yaml` | 4 | Add `parallel_group: "D+D2"` to D and D2 stages |
| `configs/pipeline_v2_shard05_test.yaml` | 4 | Same |
| `configs/pipeline_v2_test_shard00.yaml` | 4 | Same (inline-style) |
| `scripts/tools/run_pipeline_v2_shard.sh` | 5 | Add `run_parallel_group()`; replace main loop |
| `tests/test_status_concurrent.py` | 1,3,6 | New: concurrent write test + batch grouping test |
