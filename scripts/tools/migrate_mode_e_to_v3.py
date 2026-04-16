#!/usr/bin/env python3
"""Migrate mode_e bench data to pipeline_v3 layout.

The mode_e text-align bench writes edit_status.json under
  objects/<shard>/<obj_id>/phase1/edit_status.json

Pipeline v3 expects it at the object root:
  objects/<shard>/<obj_id>/edit_status.json

This script copies (and lightly normalises) the file for every object
in the given bench output directory, then reports how many edits are
gate_a pass vs fail so you can verify before running downstream steps.

Usage::

    python scripts/tools/migrate_mode_e_to_v3.py \\
        --bench-dir data/partverse/outputs/partverse/bench_shard08/mode_e_text_align \\
        --shard 08 \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def migrate_one(obj_dir: Path, shard: str, dry_run: bool) -> dict:
    """Migrate one object's edit_status.json from phase1/ to root.

    Returns a summary dict with counts.
    """
    src = obj_dir / "phase1" / "edit_status.json"
    dst = obj_dir / "edit_status.json"
    obj_id = obj_dir.name

    if not src.is_file():
        return {"obj_id": obj_id, "status": "skip_no_src"}

    if dst.is_file() and not dry_run:
        # Already migrated — merge: keep existing root file, don't overwrite
        # unless phase1 has newer gate_a data.
        existing = json.loads(dst.read_text())
        if existing.get("schema_version", 0) >= 1 and "edits" in existing:
            return {"obj_id": obj_id, "status": "already_migrated",
                    "n_edits": len(existing.get("edits", {}))}

    data = json.loads(src.read_text())

    # Normalise: ensure required fields for pipeline_v3 consumers
    data.setdefault("shard", shard)
    data.setdefault("schema_version", 1)
    data.setdefault("steps", {})     # step-level status written by run.py
    if "updated" not in data or data["updated"] is None:
        data["updated"] = _now()

    edits = data.get("edits", {})
    n_pass = sum(
        1 for e in edits.values()
        if e.get("stages", {}).get("gate_a", {}).get("status") == "pass"
    )
    n_fail = len(edits) - n_pass

    if not dry_run:
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return {
        "obj_id": obj_id,
        "status": "dry_run" if dry_run else "migrated",
        "n_edits": len(edits),
        "n_gate_a_pass": n_pass,
        "n_gate_a_fail": n_fail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True, type=Path,
                    help="mode_e bench output root (contains objects/<shard>/)")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen without writing files")
    args = ap.parse_args()

    shard_dir = args.bench_dir / "objects" / args.shard
    if not shard_dir.is_dir():
        raise SystemExit(f"shard dir not found: {shard_dir}")

    obj_dirs = sorted(d for d in shard_dir.iterdir() if d.is_dir())
    print(f"Found {len(obj_dirs)} objects in {shard_dir}")
    print()

    total_pass = total_fail = total_edits = 0
    results = []
    for obj_dir in obj_dirs:
        r = migrate_one(obj_dir, args.shard, args.dry_run)
        results.append(r)
        total_edits += r.get("n_edits", 0)
        total_pass  += r.get("n_gate_a_pass", 0)
        total_fail  += r.get("n_gate_a_fail", 0)
        tag = "✓" if r["status"] in ("migrated", "dry_run", "already_migrated") else "!"
        print(f"  {tag} {r['obj_id'][:12]}  "
              f"edits={r.get('n_edits','-'):3}  "
              f"gate_a pass={r.get('n_gate_a_pass','-')}  "
              f"fail={r.get('n_gate_a_fail','-')}  "
              f"[{r['status']}]")

    print()
    print(f"Total: {len(obj_dirs)} objects  {total_edits} edits  "
          f"gate_a pass={total_pass}  fail={total_fail}  "
          f"yield={total_pass/total_edits*100:.1f}%")
    if args.dry_run:
        print("\n[DRY RUN] — re-run without --dry-run to write files")


if __name__ == "__main__":
    main()
