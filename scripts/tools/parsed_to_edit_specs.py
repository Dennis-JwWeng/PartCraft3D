#!/usr/bin/env python3
"""Convert phase1_v2 *.parsed.json into legacy edit_specs.jsonl so that the
existing scripts/run_2d_edit.py (FLUX backend) can consume them.

Only edits whose edit_type needs a 2D edit image are emitted:
    modification, scale, material, global
(Deletion uses direct mesh delete; addition is a copy of deletion.)

Usage:
    python scripts/tools/parsed_to_edit_specs.py \
        --in-dir outputs/_debug/phase1_v2_mirror5 \
        --shard 01 \
        --out outputs/_debug/phase1_v2_mirror5/edit_specs.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_part_overview import VIEW_INDICES  # noqa: E402

NEED_FLUX = {"modification", "scale", "material", "global"}
PREFIX = {"modification": "mod", "scale": "scl",
          "material": "mat", "global": "glb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = sorted(args.in_dir.glob("*.parsed.json"))
    n_in = n_out = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fp:
        for f in files:
            j = json.loads(f.read_text())
            obj_id = j["obj_id"]
            parsed = j.get("parsed") or {}
            obj = parsed.get("object") or {}
            full_desc = obj.get("full_desc", "") or ""
            parts = {p["part_id"]: p for p in (obj.get("parts") or [])}
            edits = parsed.get("edits") or []
            seq = 0
            for e in edits:
                n_in += 1
                et = e.get("edit_type")
                if et not in NEED_FLUX:
                    continue
                pids = list(e.get("selected_part_ids") or [])
                vi = int(e.get("view_index", 0))
                npz_view = (VIEW_INDICES[vi]
                            if 0 <= vi < len(VIEW_INDICES) else -1)
                first_pid = pids[0] if pids else -1
                first_label = (parts.get(first_pid, {}).get("name", "")
                               if first_pid >= 0 else "")
                labels = [parts.get(pid, {}).get("name", "") for pid in pids]
                new_desc = e.get("new_parts_desc") or \
                    e.get("target_part_desc") or ""

                spec = {
                    "edit_id": f"{PREFIX[et]}_{obj_id}_{seq:03d}",
                    "edit_type": et,
                    "obj_id": obj_id,
                    "shard": args.shard,
                    "object_desc": full_desc,
                    "before_desc": "",
                    "remove_part_ids": pids,
                    "remove_labels": labels,
                    "keep_part_ids": [],
                    "add_part_ids": [],
                    "add_labels": [],
                    "base_part_ids": [],
                    "old_part_id": first_pid,
                    "old_label": first_label,
                    "source_del_id": "",
                    "edit_prompt": e.get("prompt", "") or "",
                    "after_desc": new_desc,
                    "before_part_desc": e.get("target_part_desc", "") or "",
                    "after_part_desc": new_desc,
                    "mod_type": "",
                    "best_view": npz_view,
                }
                fp.write(json.dumps(spec, ensure_ascii=False) + "\n")
                seq += 1
                n_out += 1
    print(f"[OK] {n_out}/{n_in} edits → {args.out}")


if __name__ == "__main__":
    main()
