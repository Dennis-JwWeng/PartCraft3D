"""Tests for pipeline_v3 post-stage hooks (spec 2026-04-21)."""
from __future__ import annotations

import unittest

from partcraft.pipeline_v3.scheduler import Hook, hooks_for


def _cfg_with_stages(**extra) -> dict:
    base = {
        "services": {"vlm": {"model": "m"}, "image_edit": {}},
        "pipeline": {
            "gpus": [0],
            "stages": [
                {"name": "del_mesh", "servers": "none", "steps": ["del_mesh"]},
                {"name": "flux_2d", "servers": "flux", "steps": ["flux_2d"]},
            ],
        },
    }
    base["pipeline"].update(extra)
    return base


class TestHooksParsing(unittest.TestCase):
    def test_no_hooks_block_returns_empty(self):
        cfg = _cfg_with_stages()
        self.assertEqual(hooks_for(cfg), [])

    def test_minimal_hook_parses(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "pull_deletion_render",
            "after_stage": "del_mesh",
            "uses": "cpu",
            "command": ["echo", "{shard}"],
        }])
        out = hooks_for(cfg)
        self.assertEqual(len(out), 1)
        h = out[0]
        self.assertIsInstance(h, Hook)
        self.assertEqual(h.name, "pull_deletion_render")
        self.assertEqual(h.after_stage, "del_mesh")
        self.assertEqual(h.uses, "cpu")
        self.assertEqual(h.command, ["echo", "{shard}"])
        self.assertEqual(h.env_passthrough, [])

    def test_env_passthrough_preserved(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h1", "after_stage": "del_mesh", "uses": "none",
            "command": ["true"], "env_passthrough": ["PARTCRAFT_CKPT_ROOT", "HOME"],
        }])
        self.assertEqual(hooks_for(cfg)[0].env_passthrough,
                         ["PARTCRAFT_CKPT_ROOT", "HOME"])

    def test_missing_required_fields_raise(self):
        for bad in [
            {"after_stage": "del_mesh", "uses": "cpu", "command": ["x"]},
            {"name": "h", "uses": "cpu", "command": ["x"]},
            {"name": "h", "after_stage": "del_mesh", "command": ["x"]},
            {"name": "h", "after_stage": "del_mesh", "uses": "cpu"},
        ]:
            with self.subTest(bad=bad):
                cfg = _cfg_with_stages(hooks=[bad])
                with self.assertRaises(ValueError):
                    hooks_for(cfg)

    def test_unknown_after_stage_raises(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "nope",
            "uses": "cpu", "command": ["true"],
        }])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("after_stage", str(ctx.exception))
        self.assertIn("nope", str(ctx.exception))

    def test_unknown_uses_raises(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "del_mesh",
            "uses": "gpu", "command": ["true"],
        }])
        with self.assertRaises(ValueError):
            hooks_for(cfg)

    def test_unknown_field_raises(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
            "retry": 3,
        }])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("retry", str(ctx.exception))

    def test_duplicate_name_collides_with_stage(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "del_mesh",
            "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }])
        with self.assertRaises(ValueError):
            hooks_for(cfg)


if __name__ == "__main__":
    unittest.main()
