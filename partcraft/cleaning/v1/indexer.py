"""Rebuild ``index/{objects,edits}.jsonl`` by scanning the v1 tree."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .layout import V1Layout


def _rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root))


def _scan_edit(edit_dir: Path, *, base_edit_id: str, suffix: str,
               obj_id: str, shard: str, layout: V1Layout) -> dict[str, Any]:
    spec_p = edit_dir / "spec.json"
    qc_p = edit_dir / "qc.json"
    spec = json.loads(spec_p.read_text()) if spec_p.is_file() else {}
    qc = json.loads(qc_p.read_text()) if qc_p.is_file() else {}
    after_npz = layout.after_npz_path(shard, obj_id, base_edit_id, suffix=suffix)
    after_views = layout.after_view_paths(shard, obj_id, base_edit_id, suffix=suffix)
    return {
        "obj_id": obj_id, "shard": shard,
        "edit_id": base_edit_id, "edit_dir_suffix": suffix,
        "edit_type": spec.get("edit_type", ""),
        "before_ss":   _rel(layout.before_ss_npz(shard, obj_id), layout.root),
        "before_slat": _rel(layout.before_slat_npz(shard, obj_id), layout.root),
        "before_views": [_rel(p, layout.root) for p in layout.before_view_paths(shard, obj_id)],
        "after_npz":   _rel(after_npz, layout.root) if after_npz.is_file() else None,
        "after_views": [_rel(p, layout.root) for p in after_views],
        "spec":        _rel(spec_p, layout.root) if spec_p.is_file() else None,
        "qc":          _rel(qc_p, layout.root) if qc_p.is_file() else None,
        "source_pipeline": qc.get("source", {}).get("pipeline_version", ""),
        "source_run_tag":  qc.get("source", {}).get("run_tag", ""),
    }


def _split_suffix(dir_name: str, *, edit_ids: set[str]) -> tuple[str, str]:
    if dir_name in edit_ids:
        return dir_name, ""
    if "__r" in dir_name:
        base, _, n = dir_name.rpartition("__r")
        if n.isdigit():
            return base, f"__r{n}"
    return dir_name, ""


def rebuild_index(layout: V1Layout) -> dict[str, int]:
    objects_rows: list[dict[str, Any]] = []
    edits_rows: list[dict[str, Any]] = []
    for obj_dir in layout.iter_object_dirs():
        shard = obj_dir.parent.name
        obj_id = obj_dir.name
        edits_root = obj_dir / "edits"
        if not edits_root.is_dir():
            continue
        edit_dirs = sorted([d for d in edits_root.iterdir() if d.is_dir()])
        canonical_ids: set[str] = set()
        for d in edit_dirs:
            spec_p = d / "spec.json"
            if spec_p.is_file():
                try:
                    canonical_ids.add(json.loads(spec_p.read_text()).get("edit_id", d.name))
                except Exception:
                    canonical_ids.add(d.name)
        type_counts: Counter[str] = Counter()
        for d in edit_dirs:
            base_id, suffix = _split_suffix(d.name, edit_ids=canonical_ids)
            row = _scan_edit(d, base_edit_id=base_id, suffix=suffix,
                             obj_id=obj_id, shard=shard, layout=layout)
            edits_rows.append(row)
            if row["edit_type"]:
                type_counts[row["edit_type"]] += 1
        objects_rows.append({
            "obj_id": obj_id, "shard": shard,
            "n_edits": len(edit_dirs),
            "edit_types": dict(type_counts),
        })
    layout.objects_jsonl().parent.mkdir(parents=True, exist_ok=True)
    layout.objects_jsonl().write_text(
        ("\n".join(json.dumps(r) for r in objects_rows) + "\n") if objects_rows else "")
    layout.edits_jsonl().write_text(
        ("\n".join(json.dumps(r) for r in edits_rows) + "\n") if edits_rows else "")
    summary = {
        "n_objects": len(objects_rows),
        "n_edits": len(edits_rows),
        "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    layout.last_rebuild_json().write_text(json.dumps(summary, indent=2))
    return summary
