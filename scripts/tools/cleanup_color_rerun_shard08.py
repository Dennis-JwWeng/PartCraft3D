#!/usr/bin/env python3
"""Reset live shard08 color edits so they can be re-run end-to-end.

Background
----------
On 2026-04-20 we discovered a whitelist bug in
``partcraft/pipeline_v3/trellis_utils.py``: ``resolve_2d_conditioning``
silently returned ``None`` for ``edit_type == "Color"``, so TRELLIS
fell back to a blank white image in ``repaint_mode='image'``.  Every
``clr_*`` edit written to shard08 prior to the fix is a
colour-agnostic 3D reconstruction with no FLUX recolour applied.
``bench_color_fix.py`` validates the fix works; this script applies the
remediation to the LIVE shard08 tree so the normal pipeline can re-run
only the affected color edits.

What it does (in --execute mode; --dry-run prints the plan)
-----------------------------------------------------------
Under ``outputs/partverse/shard08/mode_e_text_align/objects/08/<obj>/``:
  1. Delete every ``edits_3d/clr_*`` directory.
  2. Rewrite ``edit_status.json`` -- for each ``clr_*`` edit, drop
     ``stages.s5 / s6 / s6p / gate_e`` and ``gates.E``.  ``gate_a`` and
     ``s4`` (FLUX 2D) stay intact so the per-edit resume logic will
     mark only clr_* as pending for the next run.

Under ``data/H3D_v1/``:
  3. Delete every ``color/08/<obj>/clr_*`` directory (release-side
     promotion of the corrupt results).
  4. Truncate ``manifests/color/08.jsonl`` (all lines are stale).

``manifests/all.jsonl`` is left untouched; it is rebuilt by
``build_h3d_v1_index`` after the next ``pull_flux`` run.

After this script exits, run (in order):

    python -m partcraft.pipeline_v3.run --config configs/pipeline_v3_shard08.yaml \\
        --shard 08 --all --step trellis_3d
    python -m partcraft.pipeline_v3.run --config configs/pipeline_v3_shard08.yaml \\
        --shard 08 --all --step preview_flux
    QC_ONLY_TYPES=color python -m partcraft.pipeline_v3.run \\
        --config configs/pipeline_v3_shard08.yaml \\
        --shard 08 --all --step gate_quality
    python -m scripts.cleaning.h3d_v1.pull_flux \\
        --pipeline-cfg configs/pipeline_v3_shard08.yaml \\
        --shard 08 --dataset-root data/H3D_v1 --types color

Usage
-----
    # Dry-run (default): nothing is written, prints a summary.
    python scripts/tools/cleanup_color_rerun_shard08.py

    # Apply for real after reviewing the dry-run output.
    python scripts/tools/cleanup_color_rerun_shard08.py --execute
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

LOG = logging.getLogger("cleanup_color_rerun_shard08")

STAGES_TO_DROP = ("s5", "s6", "s6p", "gate_e")
GATES_TO_DROP = ("E",)


def _size_bytes(p: Path) -> int:
    total = 0
    try:
        for child in p.rglob("*"):
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
    except Exception:
        pass
    return total


def _iter_obj_dirs(live_shard_dir: Path) -> list[Path]:
    if not live_shard_dir.is_dir():
        return []
    return sorted(p for p in live_shard_dir.iterdir() if p.is_dir())


def _process_live_obj(obj_dir: Path, *, execute: bool) -> dict:
    stats = {
        "obj_id": obj_dir.name,
        "clr_dirs_removed": 0,
        "bytes_freed": 0,
        "clr_edits_reset": 0,
        "edit_status_rewritten": False,
    }

    edits3d = obj_dir / "edits_3d"
    if edits3d.is_dir():
        for sub in sorted(edits3d.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("clr_"):
                continue
            stats["bytes_freed"] += _size_bytes(sub)
            stats["clr_dirs_removed"] += 1
            if execute:
                shutil.rmtree(sub)

    es_path = obj_dir / "edit_status.json"
    if es_path.is_file():
        try:
            data = json.loads(es_path.read_text())
        except json.JSONDecodeError:
            LOG.warning("%s: edit_status.json unparseable, skipped", obj_dir.name)
            return stats
        changed = False
        for eid, entry in (data.get("edits") or {}).items():
            if not eid.startswith("clr_"):
                continue
            stages = entry.get("stages") or {}
            for sk in STAGES_TO_DROP:
                if sk in stages:
                    stages.pop(sk, None)
                    changed = True
            gates = entry.get("gates") or {}
            for gk in GATES_TO_DROP:
                if gk in gates:
                    gates.pop(gk, None)
                    changed = True
            if "final_pass" in entry:
                entry.pop("final_pass", None)
                changed = True
            stats["clr_edits_reset"] += 1
        if changed:
            stats["edit_status_rewritten"] = True
            if execute:
                from datetime import datetime
                data["updated"] = datetime.now().isoformat(timespec="seconds")
                tmp = es_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                tmp.replace(es_path)
    return stats


def _process_release(release_root: Path, shard: str, *, execute: bool) -> dict:
    color_shard = release_root / "color" / shard
    manifest_path = release_root / "manifests" / "color" / f"{shard}.jsonl"

    rel = {
        "release_clr_dirs_removed": 0,
        "release_bytes_freed": 0,
        "release_obj_dirs_emptied": 0,
        "manifest_lines_removed": 0,
    }

    if color_shard.is_dir():
        for obj_dir in sorted(color_shard.iterdir()):
            if not obj_dir.is_dir():
                continue
            any_removed = False
            for sub in sorted(obj_dir.iterdir()):
                if sub.is_dir() and sub.name.startswith("clr_"):
                    rel["release_bytes_freed"] += _size_bytes(sub)
                    rel["release_clr_dirs_removed"] += 1
                    any_removed = True
                    if execute:
                        shutil.rmtree(sub)
            if any_removed:
                rel["release_obj_dirs_emptied"] += 1
                if execute and not any(obj_dir.iterdir()):
                    obj_dir.rmdir()

    if manifest_path.is_file():
        try:
            lines = manifest_path.read_text().splitlines()
        except OSError as e:
            LOG.warning("manifest %s read failed: %s", manifest_path, e)
            lines = []
        rel["manifest_lines_removed"] = sum(1 for ln in lines if ln.strip())
        if execute:
            manifest_path.write_text("")

    return rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-root", type=Path,
                    default=_ROOT / "outputs/partverse/shard08/mode_e_text_align")
    ap.add_argument("--release-root", type=Path,
                    default=_ROOT / "data/H3D_v1")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--execute", action="store_true",
                    help="Perform mutations. Default is a dry-run printout.")
    ap.add_argument("--log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING"))
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    LOG.info("mode=%s  live_root=%s  release_root=%s  shard=%s",
             mode, args.live_root, args.release_root, args.shard)

    live_shard_dir = args.live_root / "objects" / args.shard
    obj_dirs = _iter_obj_dirs(live_shard_dir)
    LOG.info("[live] %d object dirs under %s", len(obj_dirs), live_shard_dir)

    agg = {
        "objects_touched": 0,
        "clr_dirs_removed": 0,
        "bytes_freed": 0,
        "clr_edits_reset": 0,
        "edit_status_rewritten": 0,
    }
    for od in obj_dirs:
        s = _process_live_obj(od, execute=args.execute)
        if s["clr_dirs_removed"] or s["clr_edits_reset"]:
            agg["objects_touched"] += 1
        agg["clr_dirs_removed"] += s["clr_dirs_removed"]
        agg["bytes_freed"] += s["bytes_freed"]
        agg["clr_edits_reset"] += s["clr_edits_reset"]
        if s["edit_status_rewritten"]:
            agg["edit_status_rewritten"] += 1

    rel = _process_release(args.release_root, args.shard, execute=args.execute)

    gb = lambda n: n / (1024 ** 3)
    LOG.info("=" * 64)
    LOG.info("[live/outputs] mode=%s", mode)
    LOG.info("  objects touched (clr_* present):   %d / %d",
             agg["objects_touched"], len(obj_dirs))
    LOG.info("  edits_3d/clr_* dirs to remove:     %d",
             agg["clr_dirs_removed"])
    LOG.info("  disk space to free (edits_3d):     %.2f GB",
             gb(agg["bytes_freed"]))
    LOG.info("  clr_* entries in edit_status.json: %d across %d files",
             agg["clr_edits_reset"], agg["edit_status_rewritten"])
    LOG.info("-" * 64)
    LOG.info("[H3D_v1 release]")
    LOG.info("  color/%s/*/clr_* dirs to remove:   %d (across %d objects)",
             args.shard, rel["release_clr_dirs_removed"],
             rel["release_obj_dirs_emptied"])
    LOG.info("  release disk space to free:        %.2f GB",
             gb(rel["release_bytes_freed"]))
    LOG.info("  manifests/color/%s.jsonl lines:    %d (will be truncated)",
             args.shard, rel["manifest_lines_removed"])
    LOG.info("=" * 64)
    if not args.execute:
        LOG.info("Dry-run complete. Re-run with --execute to apply.")
    else:
        LOG.info("Cleanup complete. Proceed to trellis_3d re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
