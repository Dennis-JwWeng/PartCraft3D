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
