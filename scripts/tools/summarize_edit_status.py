#!/usr/bin/env python3
"""Config-aware funnel statistics from edit_status.json.

Reads all edit_status.json files in a shard and prints a funnel table
showing how many edits reached each stage and their outcomes.

Only stages active in the config are shown.  The funnel denominator
for each stage is the number of edits that have an entry for that stage
(i.e. the stage was attempted).

Usage:
    python scripts/tools/summarize_edit_status.py \
        --config configs/pipeline_v2_shard06.yaml \
        --shard 06 [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.pipeline_v2.paths import PipelineRoot
from partcraft.pipeline_v2.edit_status_io import (
    load_edit_status, build_prereq_map,
)


STAGE_ORDER = ["gate_a", "s4", "s5b", "gate_c", "s5", "s6p", "gate_e", "s6", "s6b"]

GATE_STAGES = frozenset({"gate_a", "gate_c", "gate_e"})


def _active_stages(cfg: dict) -> list[str]:
    """Return STAGE_ORDER filtered to stages implied by the config."""
    active_steps = {
        step
        for stage in cfg.get("pipeline", {}).get("stages", [])
        for step in stage.get("steps", [])
    }
    show = set()
    if "sq1" in active_steps:
        show.add("gate_a")
    if "sq2" in active_steps:
        show.add("gate_c")
    if "sq3" in active_steps:
        show.add("gate_e")
    if "s4" in active_steps:
        show.add("s4")
    if "s5" in active_steps:
        show.add("s5")
    if "s5b" in active_steps:
        show.add("s5b")
    if "s6p_del" in active_steps or "s6p_flux" in active_steps or "s6p" in active_steps:
        show.add("s6p")
    if "s6" in active_steps:
        show.add("s6")
    if "s6b" in active_steps:
        show.add("s6b")
    return [s for s in STAGE_ORDER if s in show]


def summarize(
    cfg: dict,
    shard: str,
) -> dict:
    """Collect funnel stats across all objects in a shard.

    Returns dict with keys: objects, edits, stages (list of dicts per stage).
    """
    output_dir = Path(cfg["data"]["output_dir"])
    root = PipelineRoot(output_dir)

    shard = str(shard).zfill(2)
    objects_root = root.objects_root / shard

    active = _active_stages(cfg)
    counters: dict[str, dict[str, int]] = {
        s: {"pass_done": 0, "fail": 0, "error": 0}
        for s in active
    }

    n_obj = 0
    n_edits = 0

    if not objects_root.is_dir():
        return {"objects": 0, "edits": 0, "stages": []}

    for obj_dir in sorted(objects_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        ctx = root.context(shard, obj_dir.name)
        es = load_edit_status(ctx)
        edits = es.get("edits") or {}
        if not edits:
            continue
        n_obj += 1
        n_edits += len(edits)

        for edit_id, edit_data in edits.items():
            stages = edit_data.get("stages") or {}
            for stage_key in active:
                entry = stages.get(stage_key)
                if entry is None:
                    continue
                st = entry.get("status")
                if stage_key in GATE_STAGES:
                    if st == "pass":
                        counters[stage_key]["pass_done"] += 1
                    elif st == "fail":
                        counters[stage_key]["fail"] += 1
                else:
                    if st == "done":
                        counters[stage_key]["pass_done"] += 1
                    elif st == "error":
                        counters[stage_key]["error"] += 1
                    elif st == "fail":
                        counters[stage_key]["fail"] += 1

    stage_rows = []
    for s in active:
        c = counters[s]
        eligible = c["pass_done"] + c["fail"] + c["error"]
        pending = n_edits - eligible if s == active[0] else 0
        stage_rows.append({
            "stage": s,
            "eligible": eligible,
            "pass_done": c["pass_done"],
            "fail": c["fail"],
            "error": c["error"],
            "pending": pending,
        })

    return {
        "objects": n_obj,
        "edits": n_edits,
        "stages": stage_rows,
    }


def print_table(data: dict, config_name: str) -> None:
    print(f"\n=== Funnel: {config_name} ===  "
          f"objects={data['objects']}  edits={data['edits']}\n")
    print(f"{'stage':<10} {'eligible':>8}   {'pass/done':>9}  {'fail':>5}   "
          f"{'error':>5}  {'pending':>7}")
    print("\u2500" * 60)
    for row in data["stages"]:
        indent = "  " if row["stage"] not in GATE_STAGES else ""
        print(f"{indent}{row['stage']:<{10 - len(indent)}} {row['eligible']:>8}   "
              f"{row['pass_done']:>9}  {row['fail']:>5}   "
              f"{row['error']:>5}  {row['pending']:>7}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Config-aware funnel statistics from edit_status.json."
    )
    ap.add_argument("--config", required=True, help="Pipeline YAML config")
    ap.add_argument("--shard", required=True, help="Shard ID, e.g. 06")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    cfg = __import__("yaml").safe_load(Path(args.config).read_text())
    data = summarize(cfg, args.shard)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_table(data, Path(args.config).stem)


if __name__ == "__main__":
    main()
