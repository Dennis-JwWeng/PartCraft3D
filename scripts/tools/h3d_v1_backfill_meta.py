#!/usr/bin/env python3
"""Rewrite existing H3D_v1 per-edit ``meta.json`` files to schema v3 (final).

Applies the same transformations the updated promoter now does at
write time, so meta files produced by earlier `pull_*` runs match
records emitted by subsequent runs.  Transformations:

* ``instruction``: drop ``part_labels``, ``n_parts_selected``
* ``quality``: rename ``gate_A_score`` -> ``alignment_score``,
               ``gate_E_score`` -> ``quality_score``;
               keep ``final_pass``.
* ``lineage``:  keep only ``source_dataset`` + ``pipeline_version``
                (drops ``pipeline_config``, ``pipeline_git_sha``,
                ``promoted_at``, ``paired_edit_id``).
* ``stats``:    removed.
* ``views``:    added.  ``best_view_index`` is sourced from the
                pipeline's ``edit_status.json.gates.A.vlm.best_view``
                (pixel-mask argmax) when ``--pipeline-cfg`` is
                provided.  ``global`` edits use the same field when
                present; ``addition`` edits mirror their paired deletion;
                otherwise falls back to ``DEFAULT_FRONT_VIEW_INDEX``
                with a warning count.

Idempotent: re-running on an already-v3-final meta.json is a no-op.

Usage::

    # Recommended: provide pipeline cfg so best_view_index can be
    # pulled from edit_status.json (per-shard cfg).
    python -m scripts.tools.h3d_v1_backfill_meta \\
        --dataset-root data/H3D_v1 \\
        --shard 08 \\
        --pipeline-cfg configs/pipeline_v3_shard08.yaml

    # Dry-run to see diff counts without writing:
    python -m scripts.tools.h3d_v1_backfill_meta \\
        --dataset-root data/H3D_v1 --dry-run

    # No pipeline cfg: views.best_view_index falls back to default front.
    python -m scripts.tools.h3d_v1_backfill_meta \\
        --dataset-root data/H3D_v1 --shard 08
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from partcraft.cleaning.h3d_v1.layout import (
    EDIT_TYPES_ALL,
    H3DLayout,
    paired_edit_id,
)
from partcraft.cleaning.h3d_v1.promoter import DEFAULT_FRONT_VIEW_INDEX

LOG = logging.getLogger("h3d_v1.backfill_meta")

N_VIEWS = 5


# ── pipeline status lookup (lazy, cached per-obj) ─────────────────────
class _StatusCache:
    """Lazily load ``edit_status.json`` per obj from pipeline output."""

    def __init__(self, pipeline_objects_root: Path | None) -> None:
        self.root = pipeline_objects_root
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(self, shard: str, obj_id: str) -> dict[str, Any]:
        if self.root is None:
            return {}
        key = (shard, obj_id)
        if key in self._cache:
            return self._cache[key]
        p = self.root / shard / obj_id / "edit_status.json"
        if not p.is_file():
            self._cache[key] = {}
            return {}
        try:
            self._cache[key] = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("malformed %s: %s", p, exc)
            self._cache[key] = {}
        return self._cache[key]


def _best_view_from_status(es: dict[str, Any], edit_id: str) -> int | None:
    e = es.get("edits", {}).get(edit_id, {}) or {}
    gate_a = (e.get("gates") or {}).get("A")
    vlm = gate_a.get("vlm") if isinstance(gate_a, dict) else None
    bv = vlm.get("best_view") if isinstance(vlm, dict) else None
    if isinstance(bv, int) and 0 <= bv < N_VIEWS:
        return int(bv)
    return None


def _resolve_best_view(
    edit_type: str, edit_id: str, es: dict[str, Any], no_status: bool,
) -> tuple[int, str]:
    """Return ``(best_view_index, source_tag)``."""
    if edit_type == "global":
        if not no_status:
            bv = _best_view_from_status(es, edit_id)
            if bv is not None:
                return bv, "global_pipeline_best_view"
        return DEFAULT_FRONT_VIEW_INDEX, "global_default_front"
    if no_status:
        return DEFAULT_FRONT_VIEW_INDEX, "fallback_no_status"
    bv = _best_view_from_status(es, edit_id)
    if bv is None and edit_type == "addition":
        paired = paired_edit_id(edit_id)
        if paired is not None:
            bv = _best_view_from_status(es, paired)
            if bv is not None:
                return bv, "paired_deletion"
    if bv is not None:
        return bv, "pipeline_mask_argmax"
    return DEFAULT_FRONT_VIEW_INDEX, "fallback_missing_gate_a"


# ── meta record rewrite ──────────────────────────────────────────────
def _slim_instruction(instr: dict[str, Any]) -> dict[str, Any]:
    out = dict(instr)
    out.pop("part_labels", None)
    out.pop("n_parts_selected", None)
    return out


def _slim_quality(
    q: dict[str, Any],
    *,
    edit_type: str,
    edit_id: str,
    es: dict[str, Any],
) -> dict[str, Any]:
    """Migrate legacy quality keys to new names; also derive missing
    ``alignment_score`` for addition edits from the paired deletion's
    gate A (matches promoter._quality_block live behaviour).
    """
    out: dict[str, Any] = {}
    if "final_pass" in q:
        out["final_pass"] = bool(q["final_pass"])

    # alignment_score: legacy gate_A_score -> new name, otherwise keep already-new name.
    if "alignment_score" in q:
        out["alignment_score"] = q["alignment_score"]
    elif "gate_A_score" in q:
        out["alignment_score"] = q["gate_A_score"]
    # For addition, gate A is synthesised; mirror paired deletion's gate A score.
    if "alignment_score" not in out and edit_type == "addition" and es:
        paired = paired_edit_id(edit_id)
        if paired is not None:
            pe = es.get("edits", {}).get(paired, {}) or {}
            pg = (pe.get("gates") or {}).get("A")
            pv = pg.get("vlm") if isinstance(pg, dict) else None
            ps = pv.get("score") if isinstance(pv, dict) else None
            if isinstance(ps, (int, float)):
                out["alignment_score"] = float(ps)

    # quality_score: legacy gate_E_score -> new name, otherwise keep new name.
    if "quality_score" in q:
        out["quality_score"] = q["quality_score"]
    elif "gate_E_score" in q:
        out["quality_score"] = q["gate_E_score"]
    return out


def _slim_lineage(l: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_version": l.get("pipeline_version", "v3"),
        "source_dataset":   l.get("source_dataset",   "partverse"),
    }


def _rewrite(
    old: dict[str, Any], es: dict[str, Any], no_status: bool,
) -> tuple[dict[str, Any], str]:
    """Return ``(new_record, bv_source_tag)``."""
    edit_type = old.get("edit_type", "")
    edit_id   = old.get("edit_id", "")
    bv, src   = _resolve_best_view(edit_type, edit_id, es, no_status)

    # Preserve existing views.best_view_index if already schema v3 final and
    # we have no stronger signal (avoid rewriting valid data with fallback).
    existing_views = old.get("views") or {}
    if (no_status and isinstance(existing_views.get("best_view_index"), int)
            and 0 <= existing_views["best_view_index"] < N_VIEWS):
        bv = int(existing_views["best_view_index"])
        src = "preserved"

    return {
        "edit_id":        edit_id,
        "edit_type":      edit_type,
        "obj_id":         old.get("obj_id", ""),
        "shard":          old.get("shard", ""),
        "schema_version": 3,
        "instruction":    _slim_instruction(old.get("instruction") or {}),
        "views":          {"best_view_index": int(bv)},
        "quality":        _slim_quality(
            old.get("quality") or {},
            edit_type=edit_type,
            edit_id=edit_id,
            es=es,
        ),
        "lineage":        _slim_lineage(old.get("lineage") or {}),
    }, src


def _write_meta_atomic(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2))
    os.replace(tmp, path)


# ── walker ────────────────────────────────────────────────────────────
def _iter_meta_paths(
    layout: H3DLayout, shards: list[str] | None, edit_types: list[str],
) -> list[Path]:
    out: list[Path] = []
    for et in edit_types:
        et_root = layout.root / et
        if not et_root.is_dir():
            continue
        for shard_dir in sorted(et_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            if shards is not None and shard_dir.name not in shards:
                continue
            for obj_dir in sorted(shard_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                for edit_dir in sorted(obj_dir.iterdir()):
                    meta = edit_dir / "meta.json"
                    if meta.is_file():
                        out.append(meta)
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path,
                    help="Path to data/H3D_v1.")
    ap.add_argument("--pipeline-cfg", type=Path, default=None,
                    help="Pipeline v3 YAML; enables views.best_view_index lookup "
                         "from edit_status.json. Omit for default-front fallback.")
    ap.add_argument("--shard", action="append", default=None,
                    help='Limit to shard(s) (e.g. --shard 08). Repeatable. Default: all.')
    ap.add_argument("--edit-type", action="append", default=None,
                    choices=list(EDIT_TYPES_ALL),
                    help="Limit to one or more edit types. Default: all.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print a summary without writing any files.")
    ap.add_argument("--rebuild-manifests", action="store_true",
                    help="After rewriting meta.json files, regenerate "
                         "manifests/<edit_type>/<NN>.jsonl for the affected "
                         "(shard, edit_type) pairs by streaming the new "
                         "meta.json contents. Also rebuilds manifests/all.jsonl "
                         "if any shard was touched. Safe to run alone (no "
                         "meta rewrites) to bring manifests in sync with metas.")
    ap.add_argument("--log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    layout = H3DLayout(root=args.dataset_root)
    if not layout.root.is_dir():
        LOG.error("dataset-root not found: %s", layout.root)
        return 2

    pipeline_objects_root: Path | None = None
    if args.pipeline_cfg is not None:
        from partcraft.cleaning.h3d_v1.pipeline_io import resolve_paths
        paths = resolve_paths(args.pipeline_cfg)
        pipeline_objects_root = paths.objects_root
        LOG.info("pipeline objects root: %s", pipeline_objects_root)
    else:
        LOG.warning("--pipeline-cfg not provided: views.best_view_index will "
                    "fall back to DEFAULT_FRONT_VIEW_INDEX for flux/deletion edits")

    status_cache = _StatusCache(pipeline_objects_root)
    no_status_globally = pipeline_objects_root is None

    edit_types = args.edit_type or list(EDIT_TYPES_ALL)
    shards = args.shard
    metas = _iter_meta_paths(layout, shards, edit_types)
    LOG.info("found %d meta.json files (shards=%s, types=%s)",
              len(metas), shards or "all", edit_types)

    n_rewrite = n_unchanged = n_error = 0
    bv_src_counts: dict[str, int] = {}
    for meta_path in metas:
        try:
            old = json.loads(meta_path.read_text())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("failed to read %s: %s", meta_path, exc)
            n_error += 1
            continue
        shard = old.get("shard") or meta_path.parents[2].name
        obj_id = old.get("obj_id") or meta_path.parents[1].name
        es = {} if no_status_globally else status_cache.get(shard, obj_id)
        no_status_this = no_status_globally or not es
        try:
            new, src = _rewrite(old, es, no_status_this)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("rewrite failed for %s: %s", meta_path, exc)
            n_error += 1
            continue
        bv_src_counts[src] = bv_src_counts.get(src, 0) + 1
        if new == old:
            n_unchanged += 1
            continue
        n_rewrite += 1
        if not args.dry_run:
            _write_meta_atomic(meta_path, new)

    LOG.info("done: rewrite=%d unchanged=%d error=%d (dry_run=%s)",
              n_rewrite, n_unchanged, n_error, args.dry_run)
    LOG.info("best_view_index sources: %s",
              {k: bv_src_counts[k] for k in sorted(bv_src_counts)})

    if args.rebuild_manifests and not args.dry_run:
        _rebuild_manifests(layout, shards, edit_types)

    return 0 if n_error == 0 else 1


def _rebuild_manifests(
    layout: H3DLayout, shards: list[str] | None, edit_types: list[str],
) -> None:
    """Regenerate ``manifests/<edit_type>/<NN>.jsonl`` + ``manifests/all.jsonl``
    from the current on-disk ``meta.json`` files.

    The manifests are plain JSONL streams of per-edit records. Rebuilding
    is safe because ``build_h3d_v1_index`` (and any other consumer) treats
    them as derivable from the per-edit ``meta.json`` files. Order within
    each manifest: sorted by ``edit_id`` for reproducibility.
    """
    all_records: list[dict[str, Any]] = []
    for et in edit_types:
        et_root = layout.root / et
        if not et_root.is_dir():
            continue
        for shard_dir in sorted(et_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            if shards is not None and shard_dir.name not in shards:
                continue
            shard = shard_dir.name
            records: list[dict[str, Any]] = []
            for obj_dir in sorted(shard_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                for edit_dir in sorted(obj_dir.iterdir()):
                    meta_path = edit_dir / "meta.json"
                    if not meta_path.is_file():
                        continue
                    try:
                        rec = json.loads(meta_path.read_text())
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("skip malformed meta %s: %s", meta_path, exc)
                        continue
                    records.append(rec)
            records.sort(key=lambda r: r.get("edit_id", ""))
            mpath = layout.manifest_path(et, shard)
            tmp = mpath.with_suffix(mpath.suffix + ".tmp")
            mpath.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text("".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in records
            ))
            os.replace(tmp, mpath)
            LOG.info("rebuilt %s (%d records)", mpath, len(records))
            all_records.extend(records)

    # Aggregate manifest: rebuild from scratch across selected scope.
    # If caller restricted shards/types, we still merge the remainder of
    # the existing all.jsonl so un-selected data is not dropped.
    agg = layout.aggregated_manifest()
    if agg.is_file() and (shards is not None or len(edit_types) < 7):
        kept: list[dict[str, Any]] = []
        with open(agg, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                in_scope = (rec.get("edit_type") in edit_types
                            and (shards is None or rec.get("shard") in shards))
                if not in_scope:
                    kept.append(rec)
        all_records = kept + all_records

    all_records.sort(key=lambda r: (r.get("edit_type", ""),
                                    r.get("shard", ""),
                                    r.get("edit_id", "")))
    agg.parent.mkdir(parents=True, exist_ok=True)
    tmp = agg.with_suffix(agg.suffix + ".tmp")
    tmp.write_text("".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
        for r in all_records
    ))
    os.replace(tmp, agg)
    LOG.info("rebuilt %s (%d records total)", agg, len(all_records))


if __name__ == "__main__":
    sys.exit(main())
