"""Tests for resolve_blender_executable (pipeline_v2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.pipeline_v2.paths import resolve_blender_executable


def test_tools_blender_path_wins() -> None:
    cfg = {"tools": {"blender_path": "/opt/blender"}, "blender": "/legacy/blender"}
    assert resolve_blender_executable(cfg) == "/opt/blender"


def test_legacy_top_level_blender(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    cfg = {"blender": "/usr/bin/blender"}
    assert resolve_blender_executable(cfg) == "/usr/bin/blender"


def test_default_is_blender_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    assert resolve_blender_executable({}) == "blender"
