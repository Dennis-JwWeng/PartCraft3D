#!/usr/bin/env python3
"""Tests for Step4 resume preflight shard reconcile."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline_dispatch import (
    discover_step4_worker_results,
    reconcile_step4_results,
)


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_discover_step4_worker_results_finds_gpu_and_nongpu(tmp_path: Path):
    cache_dir = tmp_path / "phase2_5"
    _write_jsonl(cache_dir / "edit_results_shard00_gpu7.jsonl", [{"edit_id": "a"}])
    _write_jsonl(cache_dir / "edit_results_shard00_gpu0.jsonl", [{"edit_id": "b"}])
    _write_jsonl(cache_dir / "edit_results_shard00_w2.jsonl", [{"edit_id": "c"}])
    _write_jsonl(cache_dir / "edit_results_shard00_nongpu.jsonl", [{"edit_id": "d"}])

    got = discover_step4_worker_results(cache_dir, "_shard00")
    names = [p.name for p in got]

    assert names == [
        "edit_results_shard00_gpu0.jsonl",
        "edit_results_shard00_gpu7.jsonl",
        "edit_results_shard00_w2.jsonl",
        "edit_results_shard00_nongpu.jsonl",
    ]


def test_reconcile_step4_results_non_strict_merges_and_reports(tmp_path: Path):
    cache_dir = tmp_path / "phase2_5"
    merged_path = cache_dir / "edit_results_shard00.jsonl"
    gpu0 = cache_dir / "edit_results_shard00_gpu0.jsonl"
    gpu1 = cache_dir / "edit_results_shard00_gpu1.jsonl"

    # canonical has one success row and one malformed row (tracked in stats)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"edit_id": "e1", "status": "success"}) + "\n")
        f.write("{bad json}\n")
        f.write(json.dumps({"status": "success"}) + "\n")

    # worker rows override by edit_id and add unexpected ID.
    _write_jsonl(
        gpu0,
        [
            {"edit_id": "e2", "status": "success"},
            {"edit_id": "e1", "status": "failed"},
        ],
    )
    _write_jsonl(
        gpu1,
        [
            {"edit_id": "e3", "status": "success"},
            {"edit_id": "x999", "status": "success"},
        ],
    )

    merged, stats = reconcile_step4_results(
        output_path=merged_path,
        worker_paths=[gpu0, gpu1],
        expected_ids={"e1", "e2", "e3", "e4"},
        strict=False,
    )

    assert set(merged.keys()) == {"e1", "e2", "e3", "x999"}
    assert stats["worker_shards_found"] == 2
    assert stats["bad_json_lines"] == 1
    assert stats["missing_id_rows"] == 1
    assert stats["duplicate_rows_detected"] >= 1
    assert stats["unexpected_ids_count"] == 1
    assert stats["missing_expected_ids_count"] == 1
    assert stats["done_success_ids"] == 3


def test_reconcile_step4_results_strict_raises_on_bad_or_unexpected(tmp_path: Path):
    cache_dir = tmp_path / "phase2_5"
    merged_path = cache_dir / "edit_results_shard00.jsonl"
    gpu0 = cache_dir / "edit_results_shard00_gpu0.jsonl"

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write("{bad json}\n")
    _write_jsonl(gpu0, [{"edit_id": "x999", "status": "success"}])

    with pytest.raises(RuntimeError):
        reconcile_step4_results(
            output_path=merged_path,
            worker_paths=[gpu0],
            expected_ids={"e1"},
            strict=True,
        )

