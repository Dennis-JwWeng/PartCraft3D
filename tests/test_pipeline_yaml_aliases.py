"""Tests for pipeline YAML alias merging."""
from __future__ import annotations

import copy
import unittest

from partcraft.utils.pipeline_yaml_aliases import apply_yaml_aliases


class TestPipelineYamlAliases(unittest.TestCase):
    def test_services_merge_into_legacy_phase_blocks(self):
        cfg = {
            "services": {
                "vlm": {"model": "/ckpt/vlm", "base_urls": ["http://localhost:8002/v1"]},
                "image_edit": {
                    "base_urls": ["http://localhost:8004"],
                    "workers_per_server": 3,
                },
            },
            "phase0": {"vlm_model": "old"},
            "phase2_5": {"workers_per_server": 1},
        }
        apply_yaml_aliases(cfg)
        self.assertEqual(cfg["phase0"]["vlm_model"], "/ckpt/vlm")
        self.assertEqual(cfg["phase0"]["vlm_base_urls"], ["http://localhost:8002/v1"])
        self.assertEqual(cfg["phase2_5"]["image_edit_base_urls"], ["http://localhost:8004"])
        self.assertEqual(cfg["phase2_5"]["workers_per_server"], 3)

    def test_pipeline_stages_copies_to_phases(self):
        cfg = {
            "pipeline": {
                "gpus": [0],
                "stages": [{"name": "A", "servers": "vlm", "steps": ["s1"]}],
            }
        }
        apply_yaml_aliases(cfg)
        self.assertEqual(cfg["pipeline"]["phases"][0]["name"], "A")

    def test_step_params_s5_merges_into_phase5(self):
        cfg = {"step_params": {"s5": {"num_views": 12}}, "phase5": {"num_views": 40}}
        apply_yaml_aliases(cfg)
        self.assertEqual(cfg["phase5"]["num_views"], 12)

    def test_scheduler_reads_stages_without_phases(self):
        from partcraft.pipeline_v2.scheduler import phases_for

        cfg = {
            "pipeline": {
                "gpus": [0],
                "stages": [
                    {
                        "name": "X",
                        "desc": "t",
                        "servers": "none",
                        "steps": ["s7"],
                        "use_gpus": False,
                        "optional": False,
                    }
                ],
            }
        }
        out = phases_for(cfg)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "X")

    def test_apply_idempotent(self):
        cfg = {
            "services": {"vlm": {"model": "m"}},
            "pipeline": {"gpus": [0], "stages": [{"name": "A", "steps": ["s1"]}]},
        }
        c1 = copy.deepcopy(cfg)
        apply_yaml_aliases(c1)
        apply_yaml_aliases(c1)
        apply_yaml_aliases(c1)
        self.assertEqual(c1["phase0"]["vlm_model"], "m")


if __name__ == "__main__":
    unittest.main()
