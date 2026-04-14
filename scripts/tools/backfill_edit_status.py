#!/usr/bin/env python3
"""Backfill edit_status.json from existing qc.json + file existence.

Reconstructs per-edit operational state for all objects in a shard.
Idempotent: without --force, existing stage entries are preserved;
only absent entries are filled in. With --force, all entries are
recomputed from current file-system evidence.

Usage:
    python scripts/tools/backfill_edit_status.py \
        --config configs/pipeline_v2_shard06.yaml \
        --shard 06 [--obj-ids id1,id2] [--force] [-v]

Inference rules
---------------
gate_a  <- qc.json edits[id].gates.A  -> pass/fail
gate_c  <- qc.json edits[id].gates.C  -> pass/fail/null
gate_e  <- qc.json edits[id].gates.E  -> pass/fail/null
s4      <- edits_2d/{id}_edited.png    -> done
s5      <- edits_3d/{id}/before.npz + after.npz -> done  (flux types)
s5b     <- edits_3d/{id}/after_new.glb -> done           (deletion)
s6p     <- edits_3d/{id}/preview_0.png -> done
s6      <- edits_3d/{id}/before.png + after.png -> done  (flux types)
s6b     <- edits_3d/{id}/after.npz     -> done           (deletion)
addition <- edits_3d/add_*/meta.json   -> only s6p / gate_e inferred
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.pipeline_v2.paths import PipelineRoot, ObjectContext
from partcraft.pipeline_v2.specs import iter_all_specs
from partcraft.pipeline_v2.qc_io import load_qc
from partcraft.pipeline_v2.edit_status_io import (
    load_edit_status, save_edit_status, SCHEMA_VERSION,
)
FLUX_TYPES = frozenset({"modification", "scale", "material", "global"})


# ─────────────────── helpers ───────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _mtime_iso(path: Path) -> str:
    try:
        mt = os.path.getmtime(path)
        return datetime.fromtimestamp(mt, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    except OSError:
        return _now_iso()


def _gate_status(gate_dict: dict | None) -> str | None:
    """Return 'pass'/'fail'/None from a qc.json gates entry.

    None means the gate was not run (dict absent or completely empty).
    """
    if gate_dict is None:
        return None
    r = gate_dict.get("rule")
    v = gate_dict.get("vlm")
    if r is None and v is None:
        return None  # entry exists but nothing written yet
    if r is not None and not r.get("pass", True):
        return "fail"
    if v is not None and not v.get("pass", True):
        return "fail"
    return "pass"


# ─────────────────── per-edit inference ───────────────────────────────

def _infer_stages(
    ctx: ObjectContext,
    edit_id: str,
    edit_type: str,
    qc_entry: dict | None,
    qc_ts: str,
) -> dict[str, dict]:
    """Return dict of stage_key -> {status, ts} from file-system evidence."""
    stages: dict[str, dict] = {}
    ed3 = ctx.edit_3d_dir(edit_id)
    gates = (qc_entry or {}).get("gates") or {}

    # gate_a
    gate_a_st = _gate_status(gates.get("A"))
    if gate_a_st is not None:
        stages["gate_a"] = {"status": gate_a_st, "ts": qc_ts}
        if gate_a_st == "fail":
            return stages  # blocked — no downstream stages possible

    # gate_c (optional)
    gate_c_st = _gate_status(gates.get("C"))
    if gate_c_st is not None:
        stages["gate_c"] = {"status": gate_c_st, "ts": qc_ts}

    # s4: FLUX 2D
    edited_png = ctx.edits_2d_dir / f"{edit_id}_edited.png"
    if edited_png.is_file():
        stages["s4"] = {"status": "done", "ts": _mtime_iso(edited_png)}

    # s5b: GLB deletion mesh (deletion only)
    if edit_type == "deletion":
        after_glb = ed3 / "after_new.glb"
        if after_glb.is_file():
            stages["s5b"] = {"status": "done", "ts": _mtime_iso(after_glb)}

    # s5: Trellis 3D (flux types only)
    if edit_type in FLUX_TYPES:
        before_npz = ed3 / "before.npz"
        after_npz = ed3 / "after.npz"
        if before_npz.is_file() and after_npz.is_file():
            stages["s5"] = {"status": "done", "ts": _mtime_iso(after_npz)}

    # s6p: preview images
    preview0 = ed3 / "preview_0.png"
    if preview0.is_file():
        stages["s6p"] = {"status": "done", "ts": _mtime_iso(preview0)}

    # gate_e
    gate_e_st = _gate_status(gates.get("E"))
    if gate_e_st is not None:
        stages["gate_e"] = {"status": gate_e_st, "ts": qc_ts}

    # s6: render output (flux types — before.png + after.png)
    if edit_type in FLUX_TYPES:
        before_png = ed3 / "before.png"
        after_png = ed3 / "after.png"
        if before_png.is_file() and after_png.is_file():
            stages["s6"] = {"status": "done", "ts": _mtime_iso(after_png)}

    # s6b: re-encode NPZ (deletion — after.npz in del dir)
    if edit_type == "deletion":
        after_npz_del = ed3 / "after.npz"
        if after_npz_del.is_file():
            stages["s6b"] = {"status": "done", "ts": _mtime_iso(after_npz_del)}

    return stages


def _infer_addition_stages(
    ctx: ObjectContext,
    add_id: str,
    qc_entry: dict | None,
    qc_ts: str,
) -> dict[str, dict]:
    """Addition edits: only s6p and gate_e are inferred."""
    stages: dict[str, dict] = {}
    ed3 = ctx.edit_3d_dir(add_id)

    preview0 = ed3 / "preview_0.png"
    if preview0.is_file():
        stages["s6p"] = {"status": "done", "ts": _mtime_iso(preview0)}

    gates = (qc_entry or {}).get("gates") or {}
    gate_e_st = _gate_status(gates.get("E"))
    if gate_e_st is not None:
        stages["gate_e"] = {"status": gate_e_st, "ts": qc_ts}

    return stages


# ─────────────────── edit_status.json I/O ─────────────────────────────

# I/O delegated to partcraft.pipeline_v2.edit_status_io





# ─────────────────── per-object backfill ──────────────────────────────

def backfill_object(
    ctx: ObjectContext,
    *,
    force: bool = False,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Rebuild edit_status.json for one object. Returns (n_written, n_skipped)."""
    qc = load_qc(ctx)
    qc_edits = qc.get("edits") or {}
    qc_ts = qc.get("updated") or _now_iso()

    es = load_edit_status(ctx)
    new_edits: dict = {} if force else dict(es.get("edits") or {})

    n_written = n_skipped = 0

    # Non-addition edits via iter_all_specs
    for spec in iter_all_specs(ctx):
        edit_id = spec.edit_id
        if edit_id in new_edits and not force:
            n_skipped += 1
            continue

        stages = _infer_stages(
            ctx, edit_id, spec.edit_type,
            qc_edits.get(edit_id), qc_ts,
        )
        if stages:
            new_edits[edit_id] = {"edit_type": spec.edit_type, "stages": stages}
            n_written += 1

    # Addition edits discovered from edits_3d/add_*/meta.json
    if ctx.edits_3d_dir.is_dir():
        for add_dir in sorted(ctx.edits_3d_dir.iterdir()):
            if not add_dir.is_dir() or not add_dir.name.startswith("add_"):
                continue
            if not (add_dir / "meta.json").is_file():
                continue
            add_id = add_dir.name
            if add_id in new_edits and not force:
                n_skipped += 1
                continue

            stages = _infer_addition_stages(
                ctx, add_id, qc_edits.get(add_id), qc_ts,
            )
            if stages:
                new_edits[add_id] = {"edit_type": "addition", "stages": stages}
                n_written += 1

    es["obj_id"] = ctx.obj_id
    es["shard"] = ctx.shard
    es["schema_version"] = SCHEMA_VERSION
    es["edits"] = new_edits
    save_edit_status(ctx, es)
    return n_written, n_skipped


# ─────────────────── CLI ───────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct edit_status.json from qc.json + file existence."
    )
    ap.add_argument("--config", required=True, help="Pipeline YAML config")
    ap.add_argument("--shard", required=True, help="Shard ID, e.g. 06")
    ap.add_argument(
        "--obj-ids",
        help="Comma-separated obj_ids to restrict backfill (optional)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Recompute all entries, even existing ones",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("backfill_edit_status")

    cfg = (json.loads(Path(args.config).read_text())
           if args.config.endswith(".json")
           else __import__("yaml").safe_load(Path(args.config).read_text()))
    output_dir = Path(cfg["data"]["output_dir"])
    root = PipelineRoot(output_dir)

    shard = str(args.shard).zfill(2)
    objects_root = root.objects_root / shard
    if not objects_root.is_dir():
        log.error("No objects dir: %s", objects_root)
        sys.exit(1)

    obj_filter = set(args.obj_ids.split(",")) if args.obj_ids else None
    obj_dirs = sorted(objects_root.iterdir())
    log.info("Scanning %d object dirs for shard %s", len(obj_dirs), shard)

    n_obj = n_total_written = n_total_skipped = n_errors = 0
    for obj_dir in obj_dirs:
        if not obj_dir.is_dir():
            continue
        if obj_filter and obj_dir.name not in obj_filter:
            continue
        ctx = root.context(shard, obj_dir.name)
        n_obj += 1
        try:
            n_written, n_skipped = backfill_object(ctx, force=args.force, logger=log)
            n_total_written += n_written
            n_total_skipped += n_skipped
            if args.verbose and n_written:
                log.debug("%s: wrote=%d skipped=%d", ctx.obj_id, n_written, n_skipped)
        except Exception as e:
            log.warning("Error processing %s: %s", obj_dir.name, e)
            n_errors += 1

    log.info(
        "Done. objects=%d  entries_written=%d  entries_skipped=%d  errors=%d",
        n_obj, n_total_written, n_total_skipped, n_errors,
    )


if __name__ == "__main__":
    main()
