from __future__ import annotations
import unittest

class TestQcRules(unittest.TestCase):
    def setUp(self):
        from partcraft.pipeline_v2.qc_rules import check_rules
        self.check = check_rules
        self.parts = {0: {"part_id": 0, "name": "leg"}, 1: {"part_id": 1, "name": "seat"}}

    def test_pass_deletion(self):
        edit = {"edit_type": "deletion", "prompt": "Remove the leg from the chair",
                "selected_part_ids": [0], "target_part_desc": "chair leg"}
        self.assertEqual(self.check(edit, self.parts), {})

    def test_prompt_too_short(self):
        edit = {"edit_type": "deletion", "prompt": "hi", "selected_part_ids": [0]}
        self.assertIn("prompt_too_short", self.check(edit, self.parts))

    def test_parts_missing(self):
        edit = {"edit_type": "deletion",
                "prompt": "Remove the leg from the chair", "selected_part_ids": []}
        self.assertIn("parts_missing", self.check(edit, self.parts))

    def test_parts_invalid(self):
        edit = {"edit_type": "deletion",
                "prompt": "Remove the leg from the chair", "selected_part_ids": [99]}
        self.assertIn("parts_invalid", self.check(edit, self.parts))

    def test_new_desc_missing_mod(self):
        edit = {"edit_type": "modification",
                "prompt": "Replace the leg with a metal rod",
                "selected_part_ids": [0], "target_part_desc": "wooden leg",
                "new_parts_desc": "", "new_parts_desc_stage1": "", "new_parts_desc_stage2": ""}
        fails = self.check(edit, self.parts)
        self.assertIn("new_desc_missing", fails)
        self.assertIn("stage_decomp_missing", fails)

    def test_target_desc_missing_scale(self):
        edit = {"edit_type": "scale", "prompt": "Make the leg taller",
                "selected_part_ids": [0], "target_part_desc": ""}
        self.assertIn("target_desc_missing", self.check(edit, self.parts))

    def test_verb_conflict_deletion(self):
        edit = {"edit_type": "deletion",
                "prompt": "Add a new leg to the chair", "selected_part_ids": [0]}
        self.assertIn("verb_conflict", self.check(edit, self.parts))

    def test_verb_conflict_modification(self):
        edit = {"edit_type": "modification",
                "prompt": "Remove the wooden leg completely",
                "selected_part_ids": [0], "target_part_desc": "wooden leg",
                "new_parts_desc": "nothing", "new_parts_desc_stage1": "none",
                "new_parts_desc_stage2": ""}
        self.assertIn("verb_conflict", self.check(edit, self.parts))

    def test_global_no_parts(self):
        edit = {"edit_type": "global", "prompt": "Change the style to industrial metal"}
        self.assertEqual(self.check(edit, {}), {})

    def test_pass_modification_full(self):
        edit = {"edit_type": "modification",
                "prompt": "Replace the wooden leg with a metal rod",
                "selected_part_ids": [0], "target_part_desc": "wooden leg",
                "new_parts_desc": "thin metal rod",
                "new_parts_desc_stage1": "metal rod geometry", "new_parts_desc_stage2": ""}
        self.assertEqual(self.check(edit, self.parts), {})

if __name__ == "__main__":
    unittest.main()
