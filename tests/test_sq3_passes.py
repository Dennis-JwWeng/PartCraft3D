"""Unit tests for sq3_qc_e._passes() — threshold logic for all edit types.

These tests are pure-function: no IO, no images, no VLM calls.
They exercise every branch of the pass/fail matrix defined in the spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the internal helper directly via the module.
import importlib
_sq3 = importlib.import_module("partcraft.pipeline_v2.sq3_qc_e")
_passes = _sq3._passes

# Default thresholds matching the spec
_THR: dict = {}


# ─── deletion ────────────────────────────────────────────────────────

class TestDeletion:
    ET = "deletion"

    def test_pass(self):
        j = {"edit_executed": True, "correct_region": True, "visual_quality": 3}
        assert _passes(j, self.ET, _THR)

    def test_pass_high_quality(self):
        j = {"edit_executed": True, "correct_region": True, "visual_quality": 5}
        assert _passes(j, self.ET, _THR)

    def test_fail_not_executed(self):
        j = {"edit_executed": False, "correct_region": True, "visual_quality": 4}
        assert not _passes(j, self.ET, _THR)

    def test_fail_wrong_region(self):
        j = {"edit_executed": True, "correct_region": False, "visual_quality": 4}
        assert not _passes(j, self.ET, _THR)

    def test_fail_low_quality(self):
        j = {"edit_executed": True, "correct_region": True, "visual_quality": 2}
        assert not _passes(j, self.ET, _THR)

    def test_fail_quality_zero(self):
        j = {"edit_executed": True, "correct_region": True, "visual_quality": 0}
        assert not _passes(j, self.ET, _THR)

    def test_fail_missing_executed_key(self):
        j = {"correct_region": True, "visual_quality": 4}
        assert not _passes(j, self.ET, _THR)

    def test_quality_string_coercion(self):
        """visual_quality as string "3" should be handled."""
        j = {"edit_executed": True, "correct_region": True, "visual_quality": "3"}
        assert _passes(j, self.ET, _THR)

    def test_quality_bad_string(self):
        j = {"edit_executed": True, "correct_region": True, "visual_quality": "bad"}
        assert not _passes(j, self.ET, _THR)


# ─── modification ────────────────────────────────────────────────────

class TestModification:
    ET = "modification"

    def test_pass(self):
        j = {"edit_executed": True, "correct_region": True,
             "visual_quality": 3, "preserve_other": True}
        assert _passes(j, self.ET, _THR)

    def test_fail_preserve_other_false(self):
        j = {"edit_executed": True, "correct_region": True,
             "visual_quality": 4, "preserve_other": False}
        assert not _passes(j, self.ET, _THR)

    def test_fail_preserve_other_missing(self):
        j = {"edit_executed": True, "correct_region": True, "visual_quality": 4}
        assert not _passes(j, self.ET, _THR)

    def test_fail_not_executed(self):
        j = {"edit_executed": False, "correct_region": True,
             "visual_quality": 5, "preserve_other": True}
        assert not _passes(j, self.ET, _THR)

    def test_fail_low_quality(self):
        j = {"edit_executed": True, "correct_region": True,
             "visual_quality": 2, "preserve_other": True}
        assert not _passes(j, self.ET, _THR)


# ─── scale (same rules as modification) ─────────────────────────────

class TestScale:
    ET = "scale"

    def test_pass(self):
        j = {"edit_executed": True, "correct_region": True,
             "visual_quality": 3, "preserve_other": True}
        assert _passes(j, self.ET, _THR)

    def test_fail_no_preserve(self):
        j = {"edit_executed": True, "correct_region": True,
             "visual_quality": 4, "preserve_other": False}
        assert not _passes(j, self.ET, _THR)


# ─── material / global / addition (no correct_region, no preserve_other) ─────

@pytest.mark.parametrize("et", ["material", "global", "addition"])
class TestAppearanceTypes:
    def test_pass(self, et):
        j = {"edit_executed": True, "visual_quality": 3}
        assert _passes(j, et, _THR)

    def test_fail_not_executed(self, et):
        j = {"edit_executed": False, "visual_quality": 4}
        assert not _passes(j, et, _THR)

    def test_fail_low_quality(self, et):
        j = {"edit_executed": True, "visual_quality": 1}
        assert not _passes(j, et, _THR)

    def test_preserve_other_ignored(self, et):
        """material/global/addition should NOT require preserve_other."""
        j = {"edit_executed": True, "visual_quality": 3, "preserve_other": False}
        assert _passes(j, et, _THR)


# ─── custom threshold override ───────────────────────────────────────

class TestCustomThresholds:
    def test_override_min_visual_quality(self):
        thr = {"deletion": {"min_visual_quality": 4}}
        j = {"edit_executed": True, "correct_region": True, "visual_quality": 3}
        assert not _passes(j, "deletion", thr)

    def test_override_require_preserve_other_false(self):
        thr = {"modification": {"min_visual_quality": 3, "require_preserve_other": False}}
        j = {"edit_executed": True, "correct_region": True,
             "visual_quality": 3, "preserve_other": False}
        assert _passes(j, "modification", thr)

    def test_unknown_type_uses_defaults(self):
        """An unrecognized edit_type should use the built-in fallback."""
        j = {"edit_executed": True, "visual_quality": 3}
        # Should not raise; pass/fail depends on defaults
        result = _passes(j, "unknown_future_type", _THR)
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
