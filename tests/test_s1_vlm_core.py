"""Unit tests for s1_vlm_core changes: quota, prompt, validate."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partcraft.pipeline_v2.s1_vlm_core import quota_for, validate, USER_PROMPT_TEMPLATE


# ── Task 1: quota_for() scale cap ────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("n_parts", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
def test_quota_scale_always_one(n_parts):
    q = quota_for(n_parts)
    assert q["scale"] == 1, f"n_parts={n_parts}: expected scale=1, got {q['scale']}"
