from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.pipeline_jsonl import iter_jsonl


def diagnose_step1(labels_path: Path, shard: str | None) -> dict:
    rows = list(iter_jsonl(labels_path))
    ids = [r.get("obj_id") for r in rows if r.get("obj_id")]
    shard_counter = Counter(str(r.get("shard", "")).zfill(2) for r in rows)
    dup_count = len(ids) - len(set(ids))
    return {
        "stage": "step1_semantic",
        "file": str(labels_path),
        "count": len(rows),
        "unique_obj_ids": len(set(ids)),
        "duplicate_obj_id_rows": dup_count,
        "target_shard": shard,
        "by_shard": dict(shard_counter),
    }


def diagnose_step2(specs_path: Path) -> dict:
    rows = list(iter_jsonl(specs_path))
    edit_ids = [r.get("edit_id") for r in rows if r.get("edit_id")]
    type_counter = Counter(r.get("edit_type", "unknown") for r in rows)
    return {
        "stage": "step2_planning",
        "file": str(specs_path),
        "count": len(rows),
        "unique_edit_ids": len(set(edit_ids)),
        "duplicate_edit_id_rows": len(edit_ids) - len(set(edit_ids)),
        "by_edit_type": dict(type_counter),
    }


def diagnose_step3(manifest_path: Path) -> dict:
    rows = list(iter_jsonl(manifest_path))
    status_counter = Counter(r.get("status", "unknown") for r in rows)
    return {
        "stage": "step3_2d_edit",
        "file": str(manifest_path),
        "count": len(rows),
        "status": dict(status_counter),
    }


def diagnose_step4(results_path: Path, expected_edit_ids: set[str] | None = None) -> dict:
    rows = list(iter_jsonl(results_path))
    status_counter = Counter(r.get("status", "unknown") for r in rows)
    type_counter = Counter(r.get("edit_type", "unknown") for r in rows)
    fail_reason_counter = Counter(
        (r.get("reason") or "unknown")[:120]
        for r in rows
        if r.get("status") != "success"
    )
    result_ids = {r.get("edit_id") for r in rows if r.get("edit_id")}
    missing = []
    if expected_edit_ids:
        missing = sorted([eid for eid in expected_edit_ids if eid not in result_ids])
    return {
        "stage": "step4_3d_edit",
        "file": str(results_path),
        "count": len(rows),
        "status": dict(status_counter),
        "by_edit_type": dict(type_counter),
        "top_fail_reasons": fail_reason_counter.most_common(20),
        "expected_edit_ids": len(expected_edit_ids or []),
        "missing_edit_ids_count": len(missing),
        "missing_edit_ids_examples": missing[:50],
    }


def diagnose_step5(scores_path: Path) -> dict:
    rows = list(iter_jsonl(scores_path))
    tier_counter = Counter(r.get("quality_tier", "unknown") for r in rows)
    return {
        "stage": "step5_quality",
        "file": str(scores_path),
        "count": len(rows),
        "by_tier": dict(tier_counter),
    }


def diagnose_step6(export_path: Path) -> dict:
    rows = list(iter_jsonl(export_path))
    type_counter = Counter(r.get("edit_type", "unknown") for r in rows)
    return {
        "stage": "step6_export",
        "file": str(export_path),
        "count": len(rows),
        "by_edit_type": dict(type_counter),
    }
