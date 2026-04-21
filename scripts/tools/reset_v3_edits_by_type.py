#!/usr/bin/env python3
"""
Reset per-edit FLUX to Trellis to preview state for one or more edit_type values.

Use when a type (e.g. color) needs a full 3D-path redo without re-running
gen_edits and other types:

  1) Remove stages s4, s5, s6p and optional gate E on matching edits
  2) Optionally delete edits_2d and edits_3d artifacts for those edit_ids

After a dry run, re-drive the stage chain (same v3 config), e.g.:

  STAGES=flux_2d,trellis_preview  SHARD=05  \\
    bash scripts/tools/run_pipeline_v3_shard.sh <tag> configs/pipeline_v3_shard05.yaml

Then run gate_quality (Gate E) in a separate invocation; set qc.gate_quality_types
or QC_ONLY_TYPES to include every type you want scored.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

STAGES_FLUX_3D = ("s4", "s5", "s6p")
# Object-level summary step keys stored under edit_status.json:"steps".
# The bash scheduler uses --count-pending -> load_status -> these keys.
# If we only clear per-edit stages.s4/s5/s6p but leave these as "ok", the
# scheduler will skip FLUX/VLM startup. We delete them so the next run
# sees the object as pending; step runners will recreate them at the end.
OBJ_STEP_KEYS_FLUX_3D = ("s4_flux_2d", "s5_trellis", "s6p_flux")
SCHEMA = 1


def _now() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .astimezone()
        .replace(microsecond=0)
        .isoformat()
    )


def _iter_edit_status_files(objects_root: Path) -> list[Path]:
    if not objects_root.is_dir():
        return []
    return sorted(objects_root.glob("*/edit_status.json"))


def _reset_one_file(
    path: Path,
    types_set: set[str],
    clear_gate_e: bool,
    delete_artifacts: bool,
    apply: bool,
) -> tuple[int, int, int]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0

    edits: dict = data.get("edits") or {}
    n_match = 0
    f_del = d_del = 0
    parent = path.parent
    to_touch = False
    for eid, rec in list(edits.items()):
        if not isinstance(rec, dict):
            continue
        et = (rec.get("edit_type") or "").strip().lower()
        if et not in types_set:
            continue
        n_match += 1
        to_touch = True
        st = rec.setdefault("stages", {})
        for k in STAGES_FLUX_3D:
            st.pop(k, None)
        if clear_gate_e:
            g = rec.get("gates")
            if isinstance(g, dict) and g.get("E") is not None:
                g["E"] = None
            rec.pop("final_pass", None)
        if delete_artifacts and apply:
            eid_s = str(eid)
            p2d = parent / "edits_2d"
            for suf in ("_input.png", "_edited.png"):
                f = p2d / f"{eid_s}{suf}"
                if f.is_file():
                    f.unlink()
                    f_del += 1
            d3d = parent / "edits_3d" / eid_s
            if d3d.is_dir():
                shutil.rmtree(d3d, ignore_errors=False)
                d_del += 1
    if to_touch:
        # Downgrade object-level summary so the scheduler's --count-pending
        # sees pending work on this object.  Step runners rewrite these keys.
        steps_root = data.get("steps") or {}
        if isinstance(steps_root, dict):
            for k in OBJ_STEP_KEYS_FLUX_3D:
                steps_root.pop(k, None)
            if clear_gate_e:
                steps_root.pop("sq3_qc_E", None)
            data["steps"] = steps_root
    if not to_touch or not apply:
        return n_match, f_del, d_del
    data["schema_version"] = int(data.get("schema_version") or SCHEMA)
    data["updated"] = _now()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return n_match, f_del, d_del


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "objects_root",
        type=Path,
        help="e.g. <output_dir>/objects/05 (per-shard object tree)",
    )
    ap.add_argument(
        "--edit-type",
        action="append",
        default=["color"],
        help="Edit type to reset (can repeat). Default: color",
    )
    ap.add_argument(
        "--clear-gate-e",
        action="store_true",
        help="Clear gates['E'] and final_pass for matched edits.",
    )
    ap.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="Remove edits_2d and edits_3d/<id>/ for matched edits (requires --apply).",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write; without this, dry-run counts only.",
    )
    args = ap.parse_args()

    raw: Iterable[str] = args.edit_type or []
    types_set = {t.strip().lower() for t in raw if t and str(t).strip()}
    if not types_set:
        print("no edit types", file=sys.stderr)
        return 2

    files = _iter_edit_status_files(args.objects_root)
    if not files:
        print(f"no edit_status.json under {args.objects_root}", file=sys.stderr)
        return 1

    tot_m = tot_f = tot_d = 0
    for fpath in files:
        m, fd, dd = _reset_one_file(
            fpath,
            types_set,
            args.clear_gate_e,
            args.delete_artifacts,
            args.apply,
        )
        tot_m += m
        tot_f += fd
        tot_d += dd

    mode = "APPLY" if args.apply else "DRY-RUN"
    del_on = bool(args.delete_artifacts and args.apply)
    print(
        f"[{mode}] types={sorted(types_set)}  objects={len(files)}  "
        f"edits_matched={tot_m}  png_deleted={tot_f}  "
        f"edits_3d_rmdirs={tot_d}  delete_artifacts={'on' if del_on else 'off'}"
    )
    if not args.apply:
        print("Re-run with --apply to modify state (and delete files if --delete-artifacts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
