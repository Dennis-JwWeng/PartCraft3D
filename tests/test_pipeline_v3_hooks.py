"""Tests for pipeline_v3 post-stage hooks (spec 2026-04-21)."""
from __future__ import annotations

import unittest
from pathlib import Path

from partcraft.pipeline_v3.scheduler import (
    Hook,
    dump_hook_meta,
    dump_stage_chains,
    format_stage_chains_text,
    hooks_for,
    resolve_hook_command,
)


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


def _cfg_with_parallel_group() -> dict:
    """Minimal shard06-shaped config: del_mesh || (flux_2d > trellis_preview)."""
    return {
        "services": {"vlm": {"model": "m"}, "image_edit": {}},
        "pipeline": {
            "gpus": [0],
            "stages": [
                {"name": "text_gen_gate_a", "servers": "vlm", "steps": ["gen_edits"]},
                {"name": "del_mesh", "servers": "none", "steps": ["del_mesh"],
                 "parallel_group": "edit_branches"},
                {"name": "flux_2d", "servers": "flux", "steps": ["flux_2d"],
                 "parallel_group": "edit_branches",
                 "chain_id": "flux_chain", "chain_order": 0},
                {"name": "trellis_preview", "servers": "none", "steps": ["trellis_3d"],
                 "parallel_group": "edit_branches",
                 "chain_id": "flux_chain", "chain_order": 1},
                {"name": "gate_quality", "servers": "vlm", "steps": ["gate_quality"]},
            ],
        },
    }


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

    def test_hooks_non_list_scalar_raises(self):
        cfg = _cfg_with_stages(hooks="bad")
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("must be a list", str(ctx.exception))

    def test_hook_entry_not_mapping_raises(self):
        cfg = _cfg_with_stages(hooks=["not-a-mapping"])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("not a mapping", str(ctx.exception))

    def test_duplicate_hook_names_raise(self):
        cfg = _cfg_with_stages(hooks=[
            {"name": "h", "after_stage": "del_mesh", "uses": "cpu", "command": ["a"]},
            {"name": "h", "after_stage": "del_mesh", "uses": "cpu", "command": ["b"]},
        ])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_scalar_command_string_rejected(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "del_mesh",
            "uses": "cpu", "command": "echo hi",
        }])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("non-empty list of strings", str(ctx.exception))

    def test_non_list_command_raises_value_error_not_typeerror(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "del_mesh",
            "uses": "cpu", "command": 42,
        }])
        with self.assertRaises(ValueError):
            hooks_for(cfg)

    def test_env_passthrough_non_list_raises(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["x"],
            "env_passthrough": "PARTCRAFT_CKPT_ROOT",
        }])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("env_passthrough", str(ctx.exception))


class TestHookChainSplice(unittest.TestCase):
    def test_chain_splice_basic(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [{
            "name": "pull_deletion_render", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }]
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        batches = dump_stage_chains(cfg, stages)
        edit_batch = batches[1]
        chains = {c[0]: c for c in edit_batch}
        self.assertIn("del_mesh", chains)
        self.assertEqual(chains["del_mesh"],
                         ["del_mesh", "pull_deletion_render@hook"])
        self.assertIn("flux_2d", chains)
        self.assertEqual(chains["flux_2d"], ["flux_2d", "trellis_preview"])

    def test_chain_splice_format(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [{
            "name": "pull_deletion_render", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }]
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        text = format_stage_chains_text(dump_stage_chains(cfg, stages))
        self.assertIn("del_mesh>pull_deletion_render@hook", text)
        self.assertIn("flux_2d>trellis_preview", text)

    def test_hook_dropped_when_after_stage_not_selected(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [{
            "name": "pull_deletion_render", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }]
        batches = dump_stage_chains(cfg, ["flux_2d", "trellis_preview"])
        flat = [s for batch in batches for chain in batch for s in chain]
        self.assertNotIn("pull_deletion_render@hook", flat)

    def test_multi_hook_same_stage_appended_in_order(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [
            {"name": "h_a", "after_stage": "del_mesh",
             "uses": "cpu", "command": ["a"]},
            {"name": "h_b", "after_stage": "del_mesh",
             "uses": "cpu", "command": ["b"]},
        ]
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        with self.assertLogs("scheduler", level="WARNING") as cm:
            batches = dump_stage_chains(cfg, stages)
        del_chain = next(c for c in batches[1] if c[0] == "del_mesh")
        self.assertEqual(del_chain, ["del_mesh", "h_a@hook", "h_b@hook"])
        joined = "\n".join(cm.output)
        self.assertIn("share after_stage=del_mesh", joined)
        self.assertIn("h_a", joined)
        self.assertIn("h_b", joined)

    def test_no_hooks_matches_legacy_output(self):
        cfg = _cfg_with_parallel_group()
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        batches = dump_stage_chains(cfg, stages)
        flat = [s for batch in batches for chain in batch for s in chain]
        self.assertNotIn("@hook", " ".join(flat))

    def test_hook_after_stage_in_middle_of_chain_is_dropped(self):
        # Guards against a future regression that iterates `for s in chain`
        # instead of checking `chain[-1]`. flux_2d sits in the middle of
        # the flux chain; a hook on it must NOT attach.
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [{
            "name": "mid_hook", "after_stage": "flux_2d",
            "uses": "cpu", "command": ["true"],
        }]
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        batches = dump_stage_chains(cfg, stages)
        flat = [s for b in batches for c in b for s in c]
        self.assertNotIn("mid_hook@hook", flat)


class TestResolveHookCommand(unittest.TestCase):
    def _hook(self, command):
        return Hook(
            name="h", after_stage="del_mesh", uses="cpu",
            command=command, env_passthrough=[],
        )

    def _ctx(self, **over):
        base = dict(
            py_pipe="/usr/bin/python3",
            cfg_path=Path("configs/pipeline_v3_shard06.yaml"),
            shard="06",
            blender="/opt/blender/blender",
            h3d_dataset_root=Path("data/H3D_v1"),
            h3d_encode_work_dir=Path("outputs/h3d_v1_encode/06"),
        )
        base.update(over)
        return base

    def test_resolves_all_known_placeholders(self):
        h = self._hook([
            "{py_pipe}", "-m", "mod", "--cfg", "{cfg}", "--shard", "{shard}",
            "--dataset-root", "{h3d_dataset_root}",
            "--encode-work-dir", "{h3d_encode_work_dir}",
            "--blender", "{blender}",
        ])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, [
            "/usr/bin/python3", "-m", "mod",
            "--cfg", "configs/pipeline_v3_shard06.yaml",
            "--shard", "06",
            "--dataset-root", "data/H3D_v1",
            "--encode-work-dir", "outputs/h3d_v1_encode/06",
            "--blender", "/opt/blender/blender",
        ])

    def test_literal_argv_unchanged(self):
        h = self._hook(["echo", "no-placeholders-here"])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, ["echo", "no-placeholders-here"])

    def test_unknown_placeholder_raises(self):
        h = self._hook(["{unknown_key}"])
        with self.assertRaises(ValueError) as ctx:
            resolve_hook_command(h, **self._ctx())
        self.assertIn("unknown_key", str(ctx.exception))

    def test_partial_placeholder_like_string_is_literal(self):
        h = self._hook(["{", "}", "{not a placeholder}"])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, ["{", "}", "{not a placeholder}"])

    def test_multiple_placeholders_in_one_arg(self):
        h = self._hook(["{h3d_dataset_root}/shard-{shard}.tar"])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, ["data/H3D_v1/shard-06.tar"])


class TestDumpHookMeta(unittest.TestCase):
    def _cfg_with_hook(self):
        return {
            "services": {"vlm": {"model": "m"}, "image_edit": {}},
            "pipeline": {
                "gpus": [0],
                "stages": [
                    {"name": "del_mesh", "servers": "none", "steps": ["del_mesh"]},
                ],
                "hooks": [{
                    "name": "pull_deletion_render",
                    "after_stage": "del_mesh",
                    "uses": "cpu",
                    "command": [
                        "{py_pipe}", "-m", "scripts.cleaning.h3d_v1.pull_deletion",
                        "--shard", "{shard}",
                    ],
                    "env_passthrough": ["PARTCRAFT_CKPT_ROOT"],
                }],
            },
        }

    def test_dump_hook_meta_shape(self):
        meta = dump_hook_meta(self._cfg_with_hook(), "pull_deletion_render")
        self.assertEqual(meta["name"], "pull_deletion_render")
        self.assertEqual(meta["uses"], "cpu")
        self.assertEqual(meta["after_stage"], "del_mesh")
        self.assertEqual(meta["command"], [
            "{py_pipe}", "-m", "scripts.cleaning.h3d_v1.pull_deletion",
            "--shard", "{shard}",
        ])
        self.assertEqual(meta["env_passthrough"], ["PARTCRAFT_CKPT_ROOT"])

    def test_dump_hook_meta_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            dump_hook_meta(self._cfg_with_hook(), "nope")


if __name__ == "__main__":
    unittest.main()
