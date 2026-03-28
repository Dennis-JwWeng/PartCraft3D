from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from pathlib import Path


def split_obj_groups(specs, n_buckets: int) -> list[list]:
    buckets = [[] for _ in range(n_buckets)]
    bucket_sizes = [0] * n_buckets
    obj_groups: OrderedDict[str, list] = OrderedDict()
    for s in specs:
        obj_groups.setdefault(s.obj_id, []).append(s)
    groups = sorted(obj_groups.values(), key=lambda g: len(g), reverse=True)
    for g in groups:
        idx = min(range(n_buckets), key=lambda i: bucket_sizes[i])
        buckets[idx].extend(g)
        bucket_sizes[idx] += len(g)
    return buckets


def assert_unique_dispatch(groups: list[tuple[str, list]], stage: str):
    seen: set[str] = set()
    dup: list[str] = []
    for _name, items in groups:
        for s in items:
            eid = getattr(s, "edit_id", None)
            if not eid:
                continue
            if eid in seen:
                dup.append(eid)
            else:
                seen.add(eid)
    if dup:
        sample = ", ".join(dup[:10])
        raise RuntimeError(
            f"{stage}: duplicate edit_id dispatched across groups: {sample} "
            f"(total duplicate assignments={len(dup)})"
        )


def wait_for_workers(procs: list[tuple[str, subprocess.Popen]], stage: str):
    failed = []
    for name, proc in procs:
        ret = proc.wait()
        if ret != 0:
            failed.append((name, ret))
    if failed:
        raise RuntimeError(f"{stage} workers failed: {failed}")


def validate_worker_jsonl_outputs(
    worker_paths: list[Path],
    *,
    pending_ids: set[str],
    id_key: str,
    stage: str,
) -> tuple[list[str], list[str], set[str]]:
    source_counts: dict[str, int] = {}
    unexpected_ids: set[str] = set()
    for rp in worker_paths:
        if not rp.exists():
            raise RuntimeError(f"{stage}: missing worker result file: {rp}")
        with open(rp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = rec.get(id_key)
                if not rid:
                    continue
                if rid in pending_ids:
                    source_counts[rid] = source_counts.get(rid, 0) + 1
                else:
                    unexpected_ids.add(rid)
    missing = sorted([rid for rid in pending_ids if source_counts.get(rid, 0) == 0])
    dup = sorted([rid for rid, count in source_counts.items() if count > 1])
    return missing, dup, unexpected_ids


def merge_jsonl_by_key(
    *,
    output_path: Path,
    worker_paths: list[Path],
    id_key: str,
) -> OrderedDict[str, dict]:
    merged: OrderedDict[str, dict] = OrderedDict()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = rec.get(id_key)
                if rid:
                    merged[rid] = rec
    for rp in worker_paths:
        if not rp.exists():
            continue
        with open(rp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = rec.get(id_key)
                if rid:
                    merged[rid] = rec
    return merged


def discover_step4_worker_results(cache_dir: Path, tag_suffix: str) -> list[Path]:
    """Discover historical Step4 worker result shards for resume precheck."""
    gpu_shards = sorted(
        cache_dir.glob(f"edit_results{tag_suffix}_gpu*.jsonl"),
        key=lambda p: p.name,
    )
    other_shards = sorted(
        cache_dir.glob(f"edit_results{tag_suffix}_w*.jsonl"),
        key=lambda p: p.name,
    )
    nongpu = cache_dir / f"edit_results{tag_suffix}_nongpu.jsonl"
    found: list[Path] = []
    seen: set[Path] = set()
    for p in [*gpu_shards, *other_shards, nongpu]:
        if p.is_file() and p not in seen:
            found.append(p)
            seen.add(p)
    return found


def reconcile_step4_results(
    *,
    output_path: Path,
    worker_paths: list[Path],
    expected_ids: set[str],
    strict: bool,
) -> tuple[OrderedDict[str, dict], dict]:
    """Merge canonical + worker shards and return diagnostics for resume precheck."""
    source_paths: list[Path] = []
    if output_path.is_file():
        source_paths.append(output_path)
    source_paths.extend([p for p in worker_paths if p.is_file()])

    source_counts: dict[str, int] = {}
    unexpected_ids: set[str] = set()
    bad_json_lines = 0
    missing_id_rows = 0
    total_rows = 0
    for path in source_paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    bad_json_lines += 1
                    continue
                rid = rec.get("edit_id")
                if not rid:
                    missing_id_rows += 1
                    continue
                source_counts[rid] = source_counts.get(rid, 0) + 1
                if rid not in expected_ids:
                    unexpected_ids.add(rid)

    duplicate_ids = sorted([rid for rid, n in source_counts.items() if n > 1])
    if strict:
        if bad_json_lines:
            raise RuntimeError(
                "Step4 resume precheck found invalid JSON lines: "
                f"{bad_json_lines}"
            )
        if missing_id_rows:
            raise RuntimeError(
                "Step4 resume precheck found rows without edit_id: "
                f"{missing_id_rows}"
            )
        if unexpected_ids:
            sample = ", ".join(sorted(unexpected_ids)[:10])
            raise RuntimeError(
                "Step4 resume precheck found unexpected edit_ids "
                f"(sample: {sample}, total={len(unexpected_ids)})"
            )

    merged = merge_jsonl_by_key(
        output_path=output_path,
        worker_paths=worker_paths,
        id_key="edit_id",
    )
    merged_ids = set(merged.keys())
    missing_expected = sorted([eid for eid in expected_ids if eid not in merged_ids])
    done_success_ids = {
        rid
        for rid, rec in merged.items()
        if rec.get("status") == "success"
    }

    stats = {
        "files_scanned": [str(p) for p in source_paths],
        "worker_shards_found": len(worker_paths),
        "total_rows_scanned": total_rows,
        "bad_json_lines": bad_json_lines,
        "missing_id_rows": missing_id_rows,
        "duplicate_rows_detected": len(duplicate_ids),
        "duplicate_ids_sample": duplicate_ids[:10],
        "unexpected_ids_count": len(unexpected_ids),
        "unexpected_ids_sample": sorted(unexpected_ids)[:10],
        "merged_records": len(merged),
        "done_success_ids": len(done_success_ids),
        "missing_expected_ids_count": len(missing_expected),
        "missing_expected_ids_sample": missing_expected[:10],
    }
    return merged, stats
