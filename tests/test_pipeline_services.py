"""Tests for canonical pipeline services config access (PR-C)."""
from __future__ import annotations

import unittest

from partcraft.pipeline_v2 import services_cfg as sc
from partcraft.pipeline_v2.scheduler import stages_for, dump_shell_env


class TestPipelineServices(unittest.TestCase):
    def test_vlm_and_image_edit_required_shape(self):
        cfg = {
            "services": {
                "vlm": {"model": "m"},
                "image_edit": {"image_edit_backend": "local_diffusers"},
            },
            "pipeline": {"gpus": [0], "stages": [{"name": "A", "steps": ["s1"]}]},
        }
        self.assertEqual(sc.vlm_model_name(cfg), "m")
        flat = sc.trellis_image_edit_flat(cfg)
        self.assertEqual(flat["image_edit_backend"], "local_diffusers")

    def test_stages_for_reads_pipeline_stages(self):
        cfg = {
            "services": {"vlm": {"model": "x"}, "image_edit": {}},
            "pipeline": {
                "gpus": [0],
                "stages": [
                    {"name": "A", "servers": "vlm", "steps": ["s1"]},
                ],
            },
        }
        st = stages_for(cfg)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].name, "A")

    def test_dump_shell_env_emits_stages_only(self):
        cfg = {
            "services": {"vlm": {"model": "x"}, "image_edit": {}},
            "pipeline": {
                "gpus": [0, 1],
                "stages": [
                    {"name": "A", "servers": "vlm", "steps": ["s1"]},
                    {"name": "B", "servers": "none", "steps": ["s2"], "optional": True},
                ],
            },
        }
        out = dump_shell_env(cfg)
        self.assertIn("DEFAULT_STAGES=", out)
        self.assertIn("ALL_STAGES=", out)
        self.assertNotIn("DEFAULT_PHASES=", out)


if __name__ == "__main__":
    unittest.main()
