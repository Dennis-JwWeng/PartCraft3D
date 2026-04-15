# Pipeline Progress Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `scripts/tools/show_progress.py` — a read-only script that prints a two-table progress snapshot from existing on-disk `status.json` files, without calling any validator logic.

**Architecture:** Single standalone script. Reads config YAML to get `data.output_dir` and `pipeline.stages` order. Globs all `status.json` under `objects/<shard>/*/`. Aggregates counters per stage into two tables: object layer (ok/fail/absent + edit counts + fail reasons) and edit throughput (sq3 pass/fail/skip vs s1_kept total).

**Tech Stack:** Python stdlib (`json`, `glob`, `argparse`, `collections.Counter`) + `PyYAML` (already in env). Zero imports from `partcraft.*`.

---

## Confirmed status.json field names (from actual disk data)

```
s1_phase1    → n_kept, n_edits, reason, validation  (status=skip means too_many_parts)
s5_trellis   → n_ok, n_fail, n_skip, reason, validation
s5b_del_mesh → n_ok, n_fail, n_skip, reason, validation
s6p_preview  → n_ok, n_fail, n_skip, error, validation
sq3_qc_E     → n_pass, n_fail, n_skip, validation
```

## File structure

- **Create:** `scripts/tools/show_progress.py`

No other files touched.

---

## Task 1: Script skeleton + CLI

**Files:**
- Create: `scripts/tools/show_progress.py`

- [ ] **Step 1: Create the file with CLI and top-level constants**

```python
#!/usr/bin/env python3
"""Read-only pipeline progress snapshot from on-disk status.json files.

Usage:
    python scripts/tools/show_progress.py \\
        --config configs/pipeline_v2_shard02.yaml \\
        --shard 02 \\
        [--stages D,D2,E_pre,E_qc]
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import yaml

# step short-name → status.json key
STEP_KEY: dict[str, str] = {
    "s1":  "s1_phase1",
    "sq1": "sq1_qc_A",
    "s2":  "s2_highlights",
    "s4":  "s4_flux_2d",
    "s5":  "s5_trellis",
    "s5b": "s5b_del_mesh",
    "s6p": "s6p_preview",
    "sq3": "sq3_qc_E",
    "s6":  "s6_render_3d",
    "s6b": "s6b_del_reencode",
}

# (ok_field, fail_field, skip_field) for edit-level aggregation.
# Only steps that actually record these fields are listed.
EDIT_FIELDS: dict[str, tuple[str, str, str]] = {
    "s5_trellis":    ("n_ok",   "n_fail", "n_skip"),
    "s5b_del_mesh":  ("n_ok",   "n_fail", "n_skip"),
    "s6p_preview":   ("n_ok",   "n_fail", "n_skip"),
    "sq3_qc_E":      ("n_pass", "n_fail", "n_skip"),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read-only pipeline progress snapshot."
    )
    ap.add_argument("--config", required=True, type=Path,
                    help="Pipeline YAML config (e.g. configs/pipeline_v2_shard02.yaml)")
    ap.add_argument("--shard",  required=True,
                    help="Shard id, e.g. 02")
    ap.add_argument("--stages", default=None,
                    help="Comma-separated stage names to show, e.g. D,D2,E_pre,E_qc")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    filter_names = (
        {s.strip() for s in args.stages.split(",")} if args.stages else None
    )
    stages = _resolve_stages(cfg, filter_names)
    if not stages:
        raise SystemExit("No matching stages found in config.")

    shard = args.shard.zfill(2)
    status_dir = _resolve_status_dir(cfg, shard)
    if not status_dir.is_dir():
        raise SystemExit(f"Status dir not found: {status_dir}")

    result = _collect(status_dir, stages)
    _print_report(result, shard)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file runs without error (no data yet)**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python scripts/tools/show_progress.py --config configs/pipeline_v2_shard02.yaml --shard 02
```

Expected: `NameError: name '_resolve_stages' is not defined`  
(Confirms CLI parsed OK, function stubs missing — expected at this stage.)

---

## Task 2: Config + directory resolution helpers

**Files:**
- Modify: `scripts/tools/show_progress.py` (add helpers before `main`)

- [ ] **Step 1: Add `_resolve_stages` and `_resolve_status_dir`**

Insert above `main()`:

```python
def _resolve_stages(cfg: dict, filter_names: set[str] | None) -> list[dict]:
    """Return ordered stage dicts from pipeline.stages, filtered if requested."""
    stages = (cfg.get("pipeline") or {}).get("stages") or []
    if filter_names:
        stages = [s for s in stages if s["name"] in filter_names]
    return stages


def _resolve_status_dir(cfg: dict, shard: str) -> Path:
    out = (cfg.get("data") or {}).get("output_dir") or \
          (cfg.get("data") or {}).get("pipeline_v2_root")
    if not out:
        raise SystemExit("[CONFIG] data.output_dir is required")
    return Path(out) / "objects" / shard
```

- [ ] **Step 2: Verify stages are loaded correctly**

```bash
python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/pipeline_v2_shard02.yaml").read_text())
stages = (cfg.get("pipeline") or {}).get("stages") or []
for s in stages:
    print(s["name"], s.get("steps"))
PY
```

Expected output:
```
A ['s1', 'sq1']
C ['s4']
D ['s5']
D2 ['s5b']
E_pre ['s6p']
E_qc ['sq3']
E ['s6', 's6b']
```

---

## Task 3: Data collection — accumulate counters from status.json

**Files:**
- Modify: `scripts/tools/show_progress.py` (add `_collect` and `_reason_from_entry`)

- [ ] **Step 1: Add `_reason_from_entry` helper**

Insert before `main()`:

```python
def _reason_from_entry(entry: dict) -> str | None:
    """Extract a short human-readable reason from a step's status.json entry.

    Priority:
    1. entry["reason"]  — e.g. "no_specs", "no_deletions", "missing_image_npz"
    2. entry["error"]   — e.g. "missing_image_npz" (s6p uses this key)
    3. First item of entry["validation"]["missing"] → strip obj-id, keep prefix
    """
    r = entry.get("reason") or entry.get("error")
    if r:
        return str(r)
    val = entry.get("validation")
    if isinstance(val, dict):
        missing = val.get("missing") or []
        if missing:
            m = str(missing[0])
            parts = m.split("/")
            if len(parts) == 2:
                # "del_abc123_000/before.ply" → "del/before.ply"
                prefix = parts[0].split("_")[0]
                return f"missing {prefix}/{parts[1]}"
            return f"missing {m[:50]}"
    return None
```

- [ ] **Step 2: Add `_collect` function**

Insert before `main()`:

```python
def _collect(status_dir: Path, stages: list[dict]) -> dict:
    """Scan all status.json files and accumulate progress counters.

    Returns a dict with keys:
        n_total        : int   — total objects found
        n_phase1_skip  : int   — objects where s1_phase1.status == "skip"
        rows           : list  — one entry per (stage_name, step_short) pair
            each entry: {stage, step, step_key, obj_ok, obj_fail, obj_absent,
                         edit_ok, edit_fail, edit_skip, reasons: Counter}
        s1_kept_total  : int   — sum of s1_phase1.n_kept across all objects
        sq3_pass       : int   — sum of sq3_qc_E.n_pass
        sq3_fail       : int   — sum of sq3_qc_E.n_fail
        sq3_skip       : int   — sum of sq3_qc_E.n_skip
    """
    status_files = list(status_dir.glob("*/status.json"))

    # Build flat list of (stage_name, step_short, step_key) rows
    rows: list[dict] = []
    for stg in stages:
        for step_short in (stg.get("steps") or []):
            step_key = STEP_KEY.get(step_short, step_short)
            rows.append({
                "stage": stg["name"],
                "step":  step_short,
                "step_key": step_key,
                "obj_ok":    0,
                "obj_fail":  0,
                "obj_absent": 0,
                "edit_ok":   0,
                "edit_fail": 0,
                "edit_skip": 0,
                "reasons": Counter(),
            })

    n_phase1_skip  = 0
    s1_kept_total  = 0
    sq3_pass = sq3_fail = sq3_skip = 0

    for sp in status_files:
        try:
            data = json.loads(sp.read_text())
        except Exception:
            continue
        steps = data.get("steps") or {}

        # phase1 skip detection
        s1 = steps.get("s1_phase1") or {}
        if s1.get("status") == "skip":
            n_phase1_skip += 1
        s1_kept_total += int(s1.get("n_kept") or 0)

        # sq3 edit-level totals
        sq3 = steps.get("sq3_qc_E") or {}
        sq3_pass += int(sq3.get("n_pass") or 0)
        sq3_fail += int(sq3.get("n_fail") or 0)
        sq3_skip += int(sq3.get("n_skip") or 0)

        # per-row accumulation
        for row in rows:
            entry = steps.get(row["step_key"])
            if entry is None:
                row["obj_absent"] += 1
                continue
            status = entry.get("status", "")
            if status == "ok":
                row["obj_ok"] += 1
            elif status == "fail":
                row["obj_fail"] += 1
                r = _reason_from_entry(entry)
                if r:
                    row["reasons"][r] += 1
            # "skip" at step level (phase1-skip) is already captured above;
            # don't double-count as absent

            ef = EDIT_FIELDS.get(row["step_key"])
            if ef:
                ok_f, fail_f, skip_f = ef
                row["edit_ok"]   += int(entry.get(ok_f)   or 0)
                row["edit_fail"] += int(entry.get(fail_f)  or 0)
                row["edit_skip"] += int(entry.get(skip_f)  or 0)

    return {
        "n_total":       len(status_files),
        "n_phase1_skip": n_phase1_skip,
        "rows":          rows,
        "s1_kept_total": s1_kept_total,
        "sq3_pass":      sq3_pass,
        "sq3_fail":      sq3_fail,
        "sq3_skip":      sq3_skip,
    }
```

- [ ] **Step 3: Quick smoke-test the collection on shard02**

```bash
python - <<'PY'
import yaml
from pathlib import Path
import sys; sys.path.insert(0, '.')
# inline-import the two functions by exec-ing the script so far
exec(open('scripts/tools/show_progress.py').read().split('def main')[0])

cfg = yaml.safe_load(Path('configs/pipeline_v2_shard02.yaml').read_text())
stages = _resolve_stages(cfg, {'D','D2','E_pre','E_qc'})
status_dir = _resolve_status_dir(cfg, '02')
result = _collect(status_dir, stages)
print('n_total', result['n_total'])
print('n_phase1_skip', result['n_phase1_skip'])
print('s1_kept_total', result['s1_kept_total'])
for row in result['rows']:
    print(row['stage'], row['step'], 'ok=%d fail=%d absent=%d edit_ok=%d edit_fail=%d edit_skip=%d' % (
        row['obj_ok'], row['obj_fail'], row['obj_absent'],
        row['edit_ok'], row['edit_fail'], row['edit_skip']))
PY
```

Expected (approximate, based on existing data):
```
n_total 1213
n_phase1_skip 62
s1_kept_total 20151
D  s5   ok=827  fail=324  absent=0   edit_ok=12970 edit_fail=70  edit_skip=...
D2 s5b  ok=1069 fail=82   absent=0   edit_ok=7072  edit_fail=0   edit_skip=...
E_pre s6p ok=18 fail=10   absent=1183 edit_ok=184  edit_fail=0   edit_skip=178
E_qc sq3  ok=18 fail=0    absent=1195 edit_ok=0    edit_fail=0   edit_skip=0
```

---

## Task 4: Print report — Table 1 + Table 2

**Files:**
- Modify: `scripts/tools/show_progress.py` (add `_print_report`)

- [ ] **Step 1: Add `_fmt` helper and `_print_report`**

Insert before `main()`:

```python
def _fmt(n: int | None, width: int = 7) -> str:
    """Right-justify an int, or '—' if None."""
    return "—".rjust(width) if n is None else str(n).rjust(width)


def _print_report(result: dict, shard: str) -> None:
    n     = result["n_total"]
    skip  = result["n_phase1_skip"]
    net   = n - skip

    # ── Table 1: object layer ─────────────────────────────────────────
    header = (
        f"{'Stage':<8} {'step':<5}"
        f" {'obj:ok':>8} {'obj:fail':>9} {'obj:absent':>11}"
        f"  │ {'edit:ok':>8} {'edit:fail':>10} {'edit:skip':>10}"
        f"  fail-reason (top 3)"
    )
    sep = "─" * 100

    print(f"\nShard {shard} — {n} objects  (phase1-skip={skip}, net={net})")
    print(sep)
    print(header)
    print(sep)

    prev_stage = None
    for row in result["rows"]:
        has_edit = row["step_key"] in EDIT_FIELDS
        reasons  = row["reasons"].most_common(3)
        reason_s = " | ".join(f"{r}×{c}" for r, c in reasons) or "—"

        stage_label = row["stage"] if row["stage"] != prev_stage else ""
        prev_stage  = row["stage"]

        e_ok   = _fmt(row["edit_ok"]   if has_edit else None, 8)
        e_fail = _fmt(row["edit_fail"] if has_edit else None, 10)
        e_skip = _fmt(row["edit_skip"] if has_edit else None, 10)

        print(
            f"{stage_label:<8} {row['step']:<5}"
            f" {row['obj_ok']:>8} {row['obj_fail']:>9} {row['obj_absent']:>11}"
            f"  │ {e_ok} {e_fail} {e_skip}"
            f"  {reason_s}"
        )

    print(sep)

    # ── Table 2: edit throughput (sq3) ────────────────────────────────
    kept    = result["s1_kept_total"]
    reached = result["sq3_pass"] + result["sq3_fail"] + result["sq3_skip"]
    not_yet = max(0, kept - reached)

    def pct(x: int, total: int) -> str:
        return f"{x/total:.1%}" if total else "—"

    print("\nEdit throughput  (sq3 — final QC gate)")
    print(f"  s1 kept edits (planned total)  : {kept:>7}")
    print(f"  reached sq3                    : {reached:>7}  ({pct(reached, kept)})")
    if reached:
        print(f"    ├─ pass                      : {result['sq3_pass']:>7}  ({pct(result['sq3_pass'], reached)})")
        print(f"    ├─ fail                      : {result['sq3_fail']:>7}  ({pct(result['sq3_fail'], reached)})")
        print(f"    └─ skip                      : {result['sq3_skip']:>7}  ({pct(result['sq3_skip'], reached)})")
    print(f"  not yet reached sq3            : {not_yet:>7}")
    print()
```

- [ ] **Step 2: Run the full script end-to-end**

```bash
python scripts/tools/show_progress.py \
    --config configs/pipeline_v2_shard02.yaml \
    --shard 02 \
    --stages D,D2,E_pre,E_qc
```

Expected output shape:
```
Shard 02 — 1213 objects  (phase1-skip=62, net=1151)
────────────────────────────────────────────────────────────────────────────────────────────────────
Stage    step   obj:ok  obj:fail  obj:absent  │   edit:ok   edit:fail   edit:skip  fail-reason (top 3)
────────────────────────────────────────────────────────────────────────────────────────────────────
D        s5        827       324           0  │    12970          70           —  no_specs×10 | missing mod/before.npz×... | …
D2       s5b      1069        82           0  │     7072           0           —  no_deletions×10 | …
E_pre    s6p        18        10        1183  │      184           0         178  missing_image_npz×10
E_qc     sq3        18         0        1195  │      229         125           8  —
────────────────────────────────────────────────────────────────────────────────────────────────────

Edit throughput  (sq3 — final QC gate)
  s1 kept edits (planned total)  :   20151
  reached sq3                    :     362  (1.8%)
    ├─ pass                      :     229  (63.3%)
    ├─ fail                      :     125  (34.5%)
    └─ skip                      :       8  (2.2%)
  not yet reached sq3            :   19789
```

- [ ] **Step 3: Run without --stages to show all stages**

```bash
python scripts/tools/show_progress.py \
    --config configs/pipeline_v2_shard02.yaml \
    --shard 02
```

Expected: Same two tables but with all stages (A, C, D, D2, E_pre, E_qc, E) visible. Steps without edit-level fields (s1, sq1, s4, s6, s6b) show `—` in edit columns.

- [ ] **Step 4: Test with a different shard config**

```bash
python scripts/tools/show_progress.py \
    --config configs/pipeline_v2_shard05_test.yaml \
    --shard 05 \
    --stages D,D2,E_pre,E_qc
```

Expected: No crash; numbers differ from shard02.

- [ ] **Step 5: Commit**

```bash
git add scripts/tools/show_progress.py \
        docs/superpowers/specs/2026-04-13-pipeline-progress-report-design.md \
        docs/superpowers/plans/2026-04-13-pipeline-progress-report.md
git commit -m "feat: add show_progress.py — read-only pipeline progress snapshot"
```

---

## Self-review checklist

| Spec requirement | Covered by |
|---|---|
| Read-only, no validator calls | `_collect` only reads json files |
| Object layer: ok/fail/absent per stage | Task 3 + Task 4 Table 1 |
| Edit layer: ok/fail/skip per stage | Task 3 EDIT_FIELDS + Task 4 Table 1 edit cols |
| fail distinct from absent | `obj_fail` vs `obj_absent` separate counters |
| fail reason top-3 | `_reason_from_entry` + Counter.most_common(3) |
| Edit throughput table (sq3) | Task 4 Table 2 |
| --stages filter | `_resolve_stages` filter_names |
| Generic across shards | `--config` + `--shard` args |
| No partcraft.* imports | Confirmed: only stdlib + yaml |
