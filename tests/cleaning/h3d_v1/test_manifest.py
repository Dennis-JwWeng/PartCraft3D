"""Unit tests for ``partcraft.cleaning.h3d_v1.manifest``.

Pure tmp-dir tests covering the jsonl append/read/rewrite contract.
Concurrent-writer behaviour is exercised by spawning two threads that
race on the same path — they must each emit a complete line.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from partcraft.cleaning.h3d_v1.manifest import (
    append_jsonl,
    append_jsonl_many,
    read_jsonl,
    rewrite_jsonl,
)


def test_append_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "log.jsonl"
    append_jsonl(p, {"a": 1})
    assert p.is_file()
    assert json.loads(p.read_text().strip()) == {"a": 1}


def test_append_jsonl_many_writes_n(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    n = append_jsonl_many(p, ({"i": i} for i in range(5)))
    assert n == 5
    lines = [json.loads(s) for s in p.read_text().splitlines() if s]
    assert lines == [{"i": i} for i in range(5)]


def test_read_jsonl_missing_returns_empty(tmp_path: Path) -> None:
    assert list(read_jsonl(tmp_path / "absent.jsonl")) == []


def test_read_jsonl_skips_malformed(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(
        '{"ok": 1}\n'
        "not-json\n"
        '\n'
        '"bare-string"\n'
        '{"ok": 2}\n'
    )
    records = list(read_jsonl(p))
    assert records == [{"ok": 1}, {"ok": 2}]


def test_rewrite_jsonl_atomic(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text("stale\n")
    n = rewrite_jsonl(p, [{"k": "v1"}, {"k": "v2"}])
    assert n == 2
    assert not (tmp_path / "log.jsonl.tmp").exists()
    assert [json.loads(s) for s in p.read_text().splitlines()] == [{"k": "v1"}, {"k": "v2"}]



def test_concurrent_append_no_torn_lines(tmp_path: Path) -> None:
    """Two threads appending to the same file must each emit complete lines."""
    p = tmp_path / "race.jsonl"
    payload_a = {"who": "A", "data": "x" * 200}
    payload_b = {"who": "B", "data": "y" * 200}
    n_per_thread = 50

    def writer(payload: dict) -> None:
        for _ in range(n_per_thread):
            append_jsonl(p, payload)

    t1 = threading.Thread(target=writer, args=(payload_a,))
    t2 = threading.Thread(target=writer, args=(payload_b,))
    t1.start(); t2.start(); t1.join(); t2.join()

    records = list(read_jsonl(p))
    assert len(records) == n_per_thread * 2
    counts = {"A": 0, "B": 0}
    for r in records:
        counts[r["who"]] += 1
    assert counts == {"A": n_per_thread, "B": n_per_thread}
