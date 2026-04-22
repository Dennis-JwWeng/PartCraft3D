"""Unit tests for preview_render best-view-only helpers (pipeline_v3 s6p_del).

Covers the pure-Python logic that picks the canonical preview slot per edit
from ``edit_status.json`` and the slot-aware existence check used to short
circuit re-renders.  The Blender render path is covered by the integration
pipeline run itself, not here.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from partcraft.pipeline_v3 import preview_render as pr
from partcraft.pipeline_v3.specs import VIEW_INDICES


def _fake_ctx():
    # _best_view_slot_for_edit only passes ctx to load_edit_status which we patch.
    return SimpleNamespace(obj_id="abc")


def _es(edits: dict) -> dict:
    return {"edits": edits}


class BestViewSlotResolve(unittest.TestCase):
    def test_reads_gate_a_best_view_when_present(self):
        es = _es({"del_abc_000": {"gates": {"A": {"vlm": {"best_view": 2}}}}})
        with patch.object(pr, "VIEW_INDICES", VIEW_INDICES), \
             patch("partcraft.pipeline_v3.edit_status_io.load_edit_status",
                   return_value=es):
            self.assertEqual(pr._best_view_slot_for_edit(_fake_ctx(), "del_abc_000"), 2)

    def test_falls_back_to_default_when_missing(self):
        es = _es({"del_abc_001": {"gates": {"A": {}}}})
        with patch("partcraft.pipeline_v3.edit_status_io.load_edit_status",
                   return_value=es):
            self.assertEqual(
                pr._best_view_slot_for_edit(_fake_ctx(), "del_abc_001"),
                pr.DEFAULT_FRONT_VIEW_INDEX,
            )

    def test_rejects_out_of_range(self):
        es = _es({"del_abc_002": {"gates": {"A": {"vlm": {"best_view": 99}}}}})
        with patch("partcraft.pipeline_v3.edit_status_io.load_edit_status",
                   return_value=es):
            self.assertEqual(
                pr._best_view_slot_for_edit(_fake_ctx(), "del_abc_002"),
                pr.DEFAULT_FRONT_VIEW_INDEX,
            )

    def test_addition_mirrors_paired_deletion(self):
        # add_* has synthesised gate A (null vlm); should mirror paired del_*.
        es = _es({
            "add_abc_003": {"gates": {"A": None}},
            "del_abc_003": {"gates": {"A": {"vlm": {"best_view": 1}}}},
        })
        with patch("partcraft.pipeline_v3.edit_status_io.load_edit_status",
                   return_value=es):
            self.assertEqual(
                pr._best_view_slot_for_edit(_fake_ctx(), "add_abc_003"), 1,
            )

    def test_addition_falls_back_when_paired_missing(self):
        es = _es({"add_abc_004": {"gates": {"A": None}}})  # no paired del entry
        with patch("partcraft.pipeline_v3.edit_status_io.load_edit_status",
                   return_value=es):
            self.assertEqual(
                pr._best_view_slot_for_edit(_fake_ctx(), "add_abc_004"),
                pr.DEFAULT_FRONT_VIEW_INDEX,
            )


class SlotExistsCheck(unittest.TestCase):
    def test_slot_previews_exist(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "preview_2.png").write_bytes(b"x")
            self.assertTrue(pr._slot_previews_exist(p, [2]))
            self.assertFalse(pr._slot_previews_exist(p, [2, 3]))
            self.assertFalse(pr._slot_previews_exist(p, []))


if __name__ == "__main__":
    unittest.main()
