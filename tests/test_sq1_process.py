"""Unit tests for sq1_qc_a — rule layer + async VLM integration.

All VLM calls are mocked; no GPU / network required.
Tests cover:
  - missing parsed.json → error recorded in status
  - corrupt parsed.json → error
  - rule failures → qc updated, VLM not called
  - rule passes + VLM pass → n_pass incremented
  - rule passes + VLM fail → n_fail incremented
  - rule passes, no overview.png → default pass (no_overview_skip)
  - step already done + force=False → skipped
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.pipeline_v2.paths import PipelineRoot
from partcraft.pipeline_v2.qc_io import load_qc, is_edit_qc_failed
from partcraft.pipeline_v2.status import STATUS_OK, STATUS_FAIL, load_status


# ─── helpers ─────────────────────────────────────────────────────────

def _make_ctx(tmp: Path, obj_id: str = "obj001"):
    root = PipelineRoot(tmp / "pipeline_out")
    ctx = root.context("00", obj_id)
    (ctx.dir / "phase1").mkdir(parents=True, exist_ok=True)
    return ctx


def _minimal_parsed(parts=None, edits=None) -> dict:
    parts = parts or [{"part_id": 0, "name": "leg"}, {"part_id": 1, "name": "seat"}]
    edits = edits or []
    return {"parsed": {"object": {"parts": parts}, "edits": edits}}


def _run(coro):
    return asyncio.run(coro)


# ─── test: missing parsed.json ────────────────────────────────────────

class TestMissingParsedJson:
    def test_missing_parsed_returns_error(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))
        assert result["error"] == "missing_parsed_json"

    def test_missing_parsed_marks_status_fail(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        _run(_process_one(ctx, "http://fake", "fake-model", force=True))
        s = load_status(ctx)
        assert s["steps"]["sq1_qc_A"]["status"] == STATUS_FAIL


# ─── test: corrupt parsed.json ────────────────────────────────────────

class TestCorruptParsedJson:
    def test_corrupt_json_returns_error(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        ctx.parsed_path.write_text("{NOT VALID JSON")
        result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))
        assert "corrupt_parsed_json" in result["error"]


# ─── test: rule layer failures ────────────────────────────────────────

class TestRuleLayerFailures:
    def test_prompt_too_short_no_vlm_call(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edit = {"edit_type": "deletion", "prompt": "hi", "selected_part_ids": [0]}
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=[edit])))

        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI") as mock_openai:
            result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_fail"] == 1
        assert result["n_pass"] == 0

    def test_rule_fail_marks_qc(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edit = {"edit_type": "deletion", "prompt": "hi", "selected_part_ids": [0]}
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=[edit])))
        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI"):
            _run(_process_one(ctx, "http://fake", "fake-model", force=True))
        assert is_edit_qc_failed(ctx, ctx.edit_id("deletion", 0))

    def test_rule_fail_parts_missing(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edit = {"edit_type": "deletion",
                "prompt": "Remove the leg from chair", "selected_part_ids": []}
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=[edit])))
        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI"):
            result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))
        assert result["n_fail"] == 1
        edit_id = ctx.edit_id("deletion", 0)
        qc = load_qc(ctx)
        assert qc["edits"][edit_id]["fail_reason"] == "parts_missing"


# ─── test: VLM pass ───────────────────────────────────────────────────

class TestVLMPass:
    def _mock_vlm_pass(self):
        resp = MagicMock()
        resp.choices[0].message.content = (
            '{"instruction_clear":true,"part_identifiable":true,'
            '"type_consistent":true,"reason":"clear"}'
        )
        return resp

    def test_vlm_pass_increments_n_pass(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edit = {"edit_type": "deletion",
                "prompt": "Remove the wooden leg", "selected_part_ids": [0],
                "target_part_desc": "wooden leg"}
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=[edit])))
        ctx.overview_path.write_bytes(b"\x89PNG")

        mock_resp = self._mock_vlm_pass()
        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create = AsyncMock(return_value=mock_resp)
            result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_pass"] == 1
        assert result["n_fail"] == 0
        assert not is_edit_qc_failed(ctx, ctx.edit_id("deletion", 0))

    def test_vlm_pass_marks_qc_pass(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edit = {"edit_type": "deletion",
                "prompt": "Remove the wooden leg", "selected_part_ids": [0]}
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=[edit])))
        ctx.overview_path.write_bytes(b"\x89PNG")

        mock_resp = self._mock_vlm_pass()
        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create = AsyncMock(return_value=mock_resp)
            _run(_process_one(ctx, "http://fake", "fake-model", force=True))

        edit_id = ctx.edit_id("deletion", 0)
        qc = load_qc(ctx)
        assert qc["edits"][edit_id]["final_pass"] is True


# ─── test: VLM fail ───────────────────────────────────────────────────

class TestVLMFail:
    def test_vlm_fail_increments_n_fail(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edit = {"edit_type": "deletion",
                "prompt": "Remove the wooden leg", "selected_part_ids": [0]}
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=[edit])))
        ctx.overview_path.write_bytes(b"\x89PNG")

        bad_resp = MagicMock()
        bad_resp.choices[0].message.content = (
            '{"instruction_clear":false,"part_identifiable":true,'
            '"type_consistent":true,"reason":"ambiguous"}'
        )
        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create = AsyncMock(return_value=bad_resp)
            result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_fail"] == 1
        assert is_edit_qc_failed(ctx, ctx.edit_id("deletion", 0))


# ─── test: no overview.png → default pass ────────────────────────────

class TestNoOverview:
    def test_no_overview_gives_default_pass(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edit = {"edit_type": "deletion",
                "prompt": "Remove the wooden leg", "selected_part_ids": [0]}
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=[edit])))

        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI"):
            result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_pass"] == 1
        assert result["n_fail"] == 0


# ─── test: step already done ─────────────────────────────────────────

class TestStepAlreadyDone:
    def test_skip_when_done_force_false(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        from partcraft.pipeline_v2.status import update_step
        ctx = _make_ctx(tmp_path)
        update_step(ctx, "sq1_qc_A", status=STATUS_OK, n_pass=5, n_fail=0)

        result = _run(_process_one(ctx, "http://fake", "fake-model", force=False))
        assert result.get("skipped") is True

    def test_no_skip_when_force_true(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        from partcraft.pipeline_v2.status import update_step
        ctx = _make_ctx(tmp_path)
        update_step(ctx, "sq1_qc_A", status=STATUS_OK, n_pass=5, n_fail=0)
        result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))
        assert result.get("skipped") is not True


# ─── test: multiple edits ─────────────────────────────────────────────

class TestMultipleEdits:
    def test_mixed_pass_fail(self, tmp_path):
        from partcraft.pipeline_v2.sq1_qc_a import _process_one
        ctx = _make_ctx(tmp_path)
        edits = [
            # will fail rule (prompt_too_short)
            {"edit_type": "deletion", "prompt": "del", "selected_part_ids": [0]},
            # will pass rule, VLM mocked pass
            {"edit_type": "deletion", "prompt": "Remove the old wooden leg", "selected_part_ids": [0]},
        ]
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed(edits=edits)))
        ctx.overview_path.write_bytes(b"\x89PNG")

        vlm_resp = MagicMock()
        vlm_resp.choices[0].message.content = (
            '{"instruction_clear":true,"part_identifiable":true,'
            '"type_consistent":true,"reason":"ok"}'
        )
        with patch("partcraft.pipeline_v2.sq1_qc_a.AsyncOpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create = AsyncMock(return_value=vlm_resp)
            result = _run(_process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_fail"] == 1
        assert result["n_pass"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
