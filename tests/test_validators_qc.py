"""Unit tests for sq1/sq2/sq3 validators under edit_status-only storage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.pipeline_v2.paths import PipelineRoot
from partcraft.pipeline_v2.validators import check_sq1, check_sq2, check_sq3, apply_check
from partcraft.pipeline_v2.status import load_status, STATUS_OK, STATUS_FAIL
from partcraft.pipeline_v2.qc_io import update_edit_gate


def _make_ctx(tmp: Path, obj_id: str = "obj001"):
    root = PipelineRoot(tmp / "pipeline_out")
    ctx = root.context("00", obj_id)
    (ctx.dir / "phase1").mkdir(parents=True, exist_ok=True)
    return ctx


def _write_parsed_flux(ctx) -> str:
    ctx.parsed_path.write_text(json.dumps({
        "parsed": {
            "object": {"parts": [{"part_id": 0, "name": "seat"}]},
            "edits": [{
                "edit_type": "modification",
                "prompt": "Change seat to metal",
                "selected_part_ids": [0],
                "view_index": 0,
                "target_part_desc": "seat",
                "new_parts_desc": "metal seat",
                "new_parts_desc_stage1": "metal seat geometry",
                "new_parts_desc_stage2": "",
            }],
        }
    }))
    return "mod_obj001_000"


def _write_parsed_deletion_only(ctx) -> str:
    ctx.parsed_path.write_text(json.dumps({
        "parsed": {
            "object": {"parts": [{"part_id": 0, "name": "leg"}]},
            "edits": [{
                "edit_type": "deletion",
                "prompt": "Remove the old wooden leg",
                "selected_part_ids": [0],
                "view_index": 0,
                "target_part_desc": "wooden leg",
            }],
        }
    }))
    return "del_obj001_000"


def _write_gate(ctx, edit_id: str, edit_type: str, gate: str, passed: bool = True):
    update_edit_gate(
        ctx,
        edit_id,
        edit_type,
        gate,
        vlm_result={"pass": passed, "reason": "test"},
    )


class TestCheckSq1:
    def test_missing_gate_a_fails(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        _write_parsed_flux(ctx)
        result = check_sq1(ctx)
        assert result.ok is False
        assert "gate_A_not_written" in result.missing

    def test_present_gate_a_passes(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        eid = _write_parsed_flux(ctx)
        _write_gate(ctx, eid, "modification", "A", True)
        result = check_sq1(ctx)
        assert result.ok is True
        assert result.expected == 1
        assert result.found == 1


class TestCheckSq2:
    def test_no_flux_edits_skips(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        _write_parsed_deletion_only(ctx)
        result = check_sq2(ctx)
        assert result.ok is True
        assert result.expected == 0
        assert result.skip is True

    def test_flux_edits_missing_gate_c_fails(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        _write_parsed_flux(ctx)
        result = check_sq2(ctx)
        assert result.ok is False
        assert "gate_C_not_written" in result.missing

    def test_flux_edits_present_gate_c_passes(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        eid = _write_parsed_flux(ctx)
        _write_gate(ctx, eid, "modification", "C", True)
        result = check_sq2(ctx)
        assert result.ok is True


class TestCheckSq3:
    def test_missing_gate_e_fails(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        eid = _write_parsed_flux(ctx)
        _write_gate(ctx, eid, "modification", "A", True)
        result = check_sq3(ctx)
        assert result.ok is False
        assert "gate_E_not_written" in result.missing

    def test_present_gate_e_passes(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        eid = _write_parsed_flux(ctx)
        _write_gate(ctx, eid, "modification", "E", True)
        result = check_sq3(ctx)
        assert result.ok is True


class TestApplyCheck:
    def test_apply_sq1_pass_updates_status_ok(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        eid = _write_parsed_flux(ctx)
        _write_gate(ctx, eid, "modification", "A", True)
        check = apply_check(ctx, "sq1")
        assert check.ok is True
        s = load_status(ctx)
        assert s["steps"]["sq1_qc_A"]["status"] == STATUS_OK

    def test_apply_sq1_fail_updates_status_fail(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        _write_parsed_flux(ctx)
        check = apply_check(ctx, "sq1")
        assert check.ok is False
        s = load_status(ctx)
        assert s["steps"]["sq1_qc_A"]["status"] == STATUS_FAIL

    def test_apply_sq3_pass(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        eid = _write_parsed_flux(ctx)
        _write_gate(ctx, eid, "modification", "E", True)
        check = apply_check(ctx, "sq3")
        assert check.ok is True

    def test_apply_sq2_skip_counts_as_ok(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        _write_parsed_deletion_only(ctx)
        check = apply_check(ctx, "sq2")
        assert check.ok is True
        assert check.skip is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
