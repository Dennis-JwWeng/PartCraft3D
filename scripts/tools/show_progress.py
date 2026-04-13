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


def _reason_from_entry(entry: dict) -> str | None:
    """Extract a short human-readable reason from a step's status.json entry.

    Priority:
    1. entry["reason"]  — e.g. "no_specs", "no_deletions"
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
                "stage":     stg["name"],
                "step":      step_short,
                "step_key":  step_key,
                "obj_ok":    0,
                "obj_fail":  0,
                "obj_absent": 0,
                "edit_ok":   0,
                "edit_fail": 0,
                "edit_skip": 0,
                "reasons":   Counter(),
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
        is_phase1_skip = s1.get("status") == "skip"
        if is_phase1_skip:
            n_phase1_skip += 1
        s1_kept_total += int(s1.get("n_kept") or 0)

        # sq3 edit-level totals (always accumulate regardless of stage filter)
        sq3 = steps.get("sq3_qc_E") or {}
        sq3_pass += int(sq3.get("n_pass") or 0)
        sq3_fail += int(sq3.get("n_fail") or 0)
        sq3_skip += int(sq3.get("n_skip") or 0)

        # Skip per-row accumulation entirely for phase1-skip objects
        if is_phase1_skip:
            continue

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
            # Step status "skip" is neither ok/fail/absent for obj_* columns.
            # Phase1-skip objects are excluded from this loop entirely (see above).

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


def _fmt(n: int | None, width: int = 7) -> str:
    """Right-justify an int, or '—' if None (field not recorded for this step)."""
    return "—".rjust(width) if n is None else str(n).rjust(width)


def _print_report(result: dict, shard: str) -> None:
    n    = result["n_total"]
    skip = result["n_phase1_skip"]
    net  = n - skip

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
        return f"{x / total:.1%}" if total else "—"

    print("\nEdit throughput  (sq3 — final QC gate)")
    print(f"  s1 kept edits (planned total)  : {kept:>7}")
    print(f"  reached sq3                    : {reached:>7}  ({pct(reached, kept)})")
    if reached:
        print(f"    ├─ pass                      : {result['sq3_pass']:>7}  ({pct(result['sq3_pass'], reached)})")
        print(f"    ├─ fail                      : {result['sq3_fail']:>7}  ({pct(result['sq3_fail'], reached)})")
        print(f"    └─ skip                      : {result['sq3_skip']:>7}  ({pct(result['sq3_skip'], reached)})")
    print(f"  not yet reached sq3            : {not_yet:>7}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read-only pipeline progress snapshot from on-disk status.json files."
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
