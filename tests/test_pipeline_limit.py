"""Tests for LIMIT env trimming of pipeline_v2 object lists."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.pipeline_v2.paths import PipelineRoot
from partcraft.pipeline_v2 import run as runmod


def test_apply_obj_limit_trims(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = PipelineRoot(tmp_path / "pipeline_out")
    ctxs = [root.context("01", f"obj{i}") for i in range(5)]
    monkeypatch.setenv("LIMIT", "2")
    out = runmod._apply_obj_limit(ctxs)
    assert len(out) == 2
    assert [c.obj_id for c in out] == ["obj0", "obj1"]


def test_apply_obj_limit_noop_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = PipelineRoot(tmp_path / "pipeline_out")
    ctxs = [root.context("01", "a")]
    monkeypatch.delenv("LIMIT", raising=False)
    assert len(runmod._apply_obj_limit(ctxs)) == 1
