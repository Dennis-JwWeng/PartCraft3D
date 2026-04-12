"""Smoke tests: verify all pipeline_v2 modules import cleanly and key symbols exist.

These tests do NOT run any pipeline logic — they only:
  1. Import each module and assert expected public names exist.
  2. Verify the QC stages are wired into ALL_STEPS and config stages.
  3. Verify config loading + stage resolution works end-to-end.

No GPU, no network, no filesystem side-effects beyond tmp_path.
"""
from __future__ import annotations

import json
import sys
import textwrap
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── import smoke ────────────────────────────────────────────────────

@pytest.mark.smoke
class TestImports:
    def test_run_module(self):
        from partcraft.pipeline_v2 import run
        assert hasattr(run, "ALL_STEPS")
        assert hasattr(run, "main")

    def test_sq1_importable(self):
        from partcraft.pipeline_v2 import sq1_qc_a
        assert hasattr(sq1_qc_a, "run")

    def test_sq2_importable(self):
        from partcraft.pipeline_v2 import sq2_qc_c
        assert hasattr(sq2_qc_c, "run")

    def test_sq3_importable(self):
        from partcraft.pipeline_v2 import sq3_qc_e
        assert hasattr(sq3_qc_e, "run")

    def test_s6p_importable(self):
        from partcraft.pipeline_v2 import s6_preview
        assert hasattr(s6_preview, "run") and hasattr(s6_preview, "PreviewResult")

    def test_addition_utils_importable(self):
        from partcraft.pipeline_v2 import addition_utils
        assert hasattr(addition_utils, "invert_delete_prompt")

    def test_s7_is_noop(self):
        from partcraft.pipeline_v2 import s7_addition_backfill
        assert s7_addition_backfill.run([]) == []

    def test_qc_io_importable(self):
        from partcraft.pipeline_v2 import qc_io
        assert hasattr(qc_io, "load_qc")
        assert hasattr(qc_io, "save_qc")
        assert hasattr(qc_io, "update_edit_gate")
        assert hasattr(qc_io, "is_edit_qc_failed")

    def test_qc_rules_importable(self):
        from partcraft.pipeline_v2 import qc_rules
        assert hasattr(qc_rules, "check_rules")

    def test_validators_importable(self):
        from partcraft.pipeline_v2 import validators
        assert "sq1" in validators.VALIDATORS
        assert "sq2" in validators.VALIDATORS
        assert "sq3" in validators.VALIDATORS
        assert "s6p" in validators.VALIDATORS

    def test_paths_importable(self):
        from partcraft.pipeline_v2 import paths
        assert hasattr(paths, "ObjectContext")
        assert hasattr(paths, "PipelineRoot")

    def test_specs_importable(self):
        from partcraft.pipeline_v2 import specs
        assert hasattr(specs, "EditSpec")
        assert hasattr(specs, "iter_all_specs")
        assert hasattr(specs, "iter_flux_specs")
        assert hasattr(specs, "iter_deletion_specs")

    def test_scheduler_importable(self):
        from partcraft.pipeline_v2 import scheduler
        assert hasattr(scheduler, "stages_for")

    def test_status_importable(self):
        from partcraft.pipeline_v2 import status
        assert hasattr(status, "update_step")
        assert hasattr(status, "load_status")
        assert hasattr(status, "step_done")


# ─── ALL_STEPS contains QC steps ─────────────────────────────────────

@pytest.mark.smoke
class TestAllSteps:
    def test_sq1_in_all_steps(self):
        from partcraft.pipeline_v2.run import ALL_STEPS
        assert "sq1" in ALL_STEPS

    def test_sq2_in_all_steps(self):
        from partcraft.pipeline_v2.run import ALL_STEPS
        assert "sq2" in ALL_STEPS

    def test_sq3_in_all_steps(self):
        from partcraft.pipeline_v2.run import ALL_STEPS
        assert "sq3" in ALL_STEPS

    def test_all_steps_tuple_order(self):
        """s6p before sq3; sq3 before s6 (gate before encode); s7 removed."""
        from partcraft.pipeline_v2.run import ALL_STEPS
        steps = list(ALL_STEPS)
        assert steps.index("sq1") < steps.index("s4"),  "sq1 before s4"
        assert steps.index("sq2") > steps.index("s4"),  "sq2 after s4"
        assert steps.index("s6p") < steps.index("sq3"), "s6p before sq3"
        assert steps.index("sq3") < steps.index("s6"),  "sq3 before s6"
        assert "s7" not in steps, "s7 must not be in ALL_STEPS"


# ─── config loading + stage resolution ───────────────────────────────

_MINIMAL_YAML = textwrap.dedent("""\
    data:
      output_dir: {output_dir}
      mesh_root: /tmp/mesh
      images_root: /tmp/images
      slat_dir: /tmp/slat
    pipeline:
      gpus: [0]
      n_vlm_servers: 1
      vlm_port_base: 8002
      vlm_port_stride: 10
      flux_port_base: 8004
      flux_port_stride: 1
      stages:
        - {{name: A,     desc: "phase1 VLM",       servers: vlm,  steps: [s1]}}
        - {{name: A_qc,  desc: "QC-A instruction", servers: vlm,  steps: [sq1]}}
        - {{name: B,     desc: "highlights",        servers: none, steps: [s2]}}
        - {{name: C,     desc: "FLUX 2D",           servers: flux, steps: [s4]}}
        - {{name: C_qc,  desc: "QC-C 2D region",   servers: vlm,  steps: [sq2]}}
        - {{name: D,     desc: "TRELLIS 3D edit",   servers: none, steps: [s5], use_gpus: true}}
        - {{name: D2,    desc: "deletion mesh",     servers: none, steps: [s5b]}}
        - {{name: E_pre, desc: "5-view preview",    servers: none, steps: [s6p], use_gpus: true}}
        - {{name: E_qc,  desc: "QC-E final",        servers: vlm,  steps: [sq3]}}
        - {{name: E,     desc: "3D rerender",       servers: none, steps: [s6], use_gpus: true}}
    services:
      vlm:
        model: /tmp/fake-model
      image_edit:
        enabled: true
        image_edit_backend: local_diffusers
    qc:
      vlm_score_threshold: 0.7
      thresholds_by_type:
        deletion:     {{min_visual_quality: 3}}
        modification: {{min_visual_quality: 3, require_preserve_other: true}}
""")


@pytest.mark.smoke
class TestConfigLoading:
    def test_load_config_parses_yaml(self, tmp_path):
        from partcraft.pipeline_v2.run import load_config
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(_MINIMAL_YAML.format(output_dir=str(tmp_path / "out")))
        cfg = load_config(cfg_file)
        assert "pipeline" in cfg
        assert "services" in cfg
        assert "qc" in cfg

    def test_stages_for_returns_all_qc_stages(self, tmp_path):
        from partcraft.pipeline_v2.run import load_config
        from partcraft.pipeline_v2.scheduler import stages_for
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(_MINIMAL_YAML.format(output_dir=str(tmp_path / "out")))
        cfg = load_config(cfg_file)
        stages = stages_for(cfg)
        names = [s.name for s in stages]
        assert "A_qc" in names
        assert "C_qc" in names
        assert "E_qc" in names

    def test_qc_stage_steps_contain_sq_steps(self, tmp_path):
        from partcraft.pipeline_v2.run import load_config
        from partcraft.pipeline_v2.scheduler import stages_for
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(_MINIMAL_YAML.format(output_dir=str(tmp_path / "out")))
        cfg = load_config(cfg_file)
        stages = {s.name: s for s in stages_for(cfg)}
        assert "sq1" in stages["A_qc"].steps
        assert "sq2" in stages["C_qc"].steps
        assert "sq3" in stages["E_qc"].steps

    def test_qc_config_block_accessible(self, tmp_path):
        from partcraft.pipeline_v2.run import load_config
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(_MINIMAL_YAML.format(output_dir=str(tmp_path / "out")))
        cfg = load_config(cfg_file)
        assert cfg["qc"]["vlm_score_threshold"] == 0.7
        assert "deletion" in cfg["qc"]["thresholds_by_type"]
        assert "modification" in cfg["qc"]["thresholds_by_type"]

    def test_vlm_model_name_accessible(self, tmp_path):
        from partcraft.pipeline_v2.run import load_config
        from partcraft.pipeline_v2.services_cfg import vlm_model_name
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(_MINIMAL_YAML.format(output_dir=str(tmp_path / "out")))
        cfg = load_config(cfg_file)
        assert vlm_model_name(cfg) == "/tmp/fake-model"


# ─── PipelineRoot + ObjectContext paths ──────────────────────────────

@pytest.mark.smoke
class TestPipelineRootPaths:
    def test_pipeline_root_creates_dirs(self, tmp_path):
        from partcraft.pipeline_v2.paths import PipelineRoot
        root = PipelineRoot(tmp_path / "pipeline_out")
        root.ensure()
        assert (tmp_path / "pipeline_out").is_dir()

    def test_object_context_paths(self, tmp_path):
        from partcraft.pipeline_v2.paths import PipelineRoot
        root = PipelineRoot(tmp_path / "pipeline_out")
        ctx = root.context("01", "abc123")
        assert ctx.obj_id == "abc123"
        assert ctx.shard == "01"
        assert ctx.qc_path == ctx.dir / "qc.json"
        assert ctx.status_path == ctx.dir / "status.json"
        assert ctx.parsed_path == ctx.dir / "phase1" / "parsed.json"
        assert ctx.overview_path == ctx.dir / "phase1" / "overview.png"

    def test_highlight_path(self, tmp_path):
        from partcraft.pipeline_v2.paths import PipelineRoot
        root = PipelineRoot(tmp_path / "out")
        ctx = root.context("00", "obj1")
        assert ctx.highlight_path(0) == ctx.dir / "highlights" / "e00.png"
        assert ctx.highlight_path(9) == ctx.dir / "highlights" / "e09.png"

    def test_edit_3d_png_path(self, tmp_path):
        from partcraft.pipeline_v2.paths import PipelineRoot
        root = PipelineRoot(tmp_path / "out")
        ctx = root.context("00", "obj1")
        p = ctx.edit_3d_png("mod_000", "before")
        assert p == ctx.dir / "edits_3d" / "mod_000" / "before.png"


# ─── edit_types consistency ───────────────────────────────────────────

@pytest.mark.smoke
class TestEditTypes:
    def test_flux_types_importable(self):
        from partcraft.edit_types import FLUX_TYPES
        assert "modification" in FLUX_TYPES
        assert "scale" in FLUX_TYPES
        assert "material" in FLUX_TYPES
        assert "global" in FLUX_TYPES
        assert "deletion" not in FLUX_TYPES  # deletion is mesh-only

    def test_edit_type_prefix_importable(self):
        from partcraft.edit_types import EDIT_TYPE_PREFIX
        assert EDIT_TYPE_PREFIX.get("deletion") is not None
        assert EDIT_TYPE_PREFIX.get("modification") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
