"""Unit tests for sq2_qc_c — image stitching + VLM region alignment.

All VLM calls are mocked; no GPU / network required.
Uses synthetic in-memory images (opencv numpy arrays) to test _stitch().

Tests cover:
  - _stitch() returns valid PNG bytes from two images
  - _process_one: already-failed QC edit is skipped
  - _process_one: missing highlight file → fail recorded
  - _process_one: missing 2D edit output → fail recorded
  - _process_one: VLM says region_match=true → pass
  - _process_one: VLM says region_match=false → fail
  - step already done + force=False → skipped
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
_sq2 = importlib.import_module("partcraft.pipeline_v2.sq2_qc_c")
_stitch = _sq2._stitch

from partcraft.pipeline_v2.paths import PipelineRoot
from partcraft.pipeline_v2.qc_io import load_qc, update_edit_gate, is_edit_qc_failed
from partcraft.pipeline_v2.status import STATUS_OK, load_status, update_step


# ─── helpers ─────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_ctx(tmp: Path, obj_id: str = "obj001"):
    root = PipelineRoot(tmp / "pipeline_out")
    ctx = root.context("00", obj_id)
    (ctx.dir / "phase1").mkdir(parents=True, exist_ok=True)
    return ctx


def _save_png(path: Path, h: int = 64, w: int = 64) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = 200
    cv2.imwrite(str(path), img)


def _minimal_parsed_flux() -> dict:
    return {
        "parsed": {
            "object": {"parts": [{"part_id": 0, "name": "seat"}]},
            "edits": [{
                "edit_type": "modification",
                "prompt": "Change the seat color to red",
                "selected_part_ids": [0],
                "view_index": 0,
                "target_part_desc": "wooden seat",
                "new_parts_desc": "red seat",
                "new_parts_desc_stage1": "red seat geometry",
                "new_parts_desc_stage2": "",
            }],
        }
    }


# ─── test: _stitch() utility ─────────────────────────────────────────

class TestStitch:
    def test_returns_bytes(self):
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.zeros((100, 100, 3), dtype=np.uint8)
        data = _stitch(img_a, img_b)
        assert isinstance(data, bytes)
        assert len(data) > 100

    def test_different_sized_images(self):
        img_a = np.zeros((200, 300, 3), dtype=np.uint8)
        img_b = np.zeros((100, 150, 3), dtype=np.uint8)
        data = _stitch(img_a, img_b)
        assert isinstance(data, bytes)

    def test_output_is_valid_png(self):
        import cv2
        img_a = np.zeros((64, 64, 3), dtype=np.uint8)
        img_b = np.ones((64, 64, 3), dtype=np.uint8) * 128
        data = _stitch(img_a, img_b)
        arr = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[1] > 64


# ─── test: step already done ─────────────────────────────────────────

class TestAlreadyDone:
    def test_skip_force_false(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        update_step(ctx, "sq2_qc_C", status=STATUS_OK, n_pass=3, n_fail=0)
        result = _run(_sq2._process_one(ctx, "http://fake", "fake-model", force=False))
        assert result.get("skipped") is True


# ─── test: pre-failed edit is skipped ────────────────────────────────

class TestPreFailedEdit:
    def test_already_qc_failed_edit_is_skipped(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed_flux()))
        edit_id = ctx.edit_id("modification", 0)
        update_edit_gate(ctx, edit_id, "modification", "A",
                         rule_result={"pass": False, "checks": {"prompt_too_short": True}})

        with patch("partcraft.pipeline_v2.sq2_qc_c.AsyncOpenAI"):
            result = _run(_sq2._process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_pass"] == 0
        assert result["n_fail"] == 0


# ─── test: missing artifacts → fail ──────────────────────────────────

class TestMissingArtifacts:
    def test_missing_highlight_fails(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed_flux()))

        with patch("partcraft.pipeline_v2.sq2_qc_c.AsyncOpenAI"):
            result = _run(_sq2._process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_fail"] == 1
        assert result["n_pass"] == 0
        edit_id = ctx.edit_id("modification", 0)
        qc = load_qc(ctx)
        assert qc["edits"][edit_id]["gates"]["C"]["vlm"]["reason"] == "missing_artifact"

    def test_missing_edit_output_fails(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed_flux()))
        _save_png(ctx.highlight_path(0))

        with patch("partcraft.pipeline_v2.sq2_qc_c.AsyncOpenAI"):
            result = _run(_sq2._process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_fail"] == 1
        edit_id = ctx.edit_id("modification", 0)
        qc = load_qc(ctx)
        assert qc["edits"][edit_id]["gates"]["C"]["vlm"]["reason"] == "missing_artifact"


# ─── test: VLM region_match ───────────────────────────────────────────

class TestVLMRegionMatch:
    def test_region_match_true_passes(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed_flux()))
        _save_png(ctx.highlight_path(0))
        edit_id = ctx.edit_id("modification", 0)
        _save_png(ctx.edit_2d_output(edit_id))

        resp = MagicMock()
        resp.choices[0].message.content = '{"region_match":true,"reason":"edit in target area"}'
        with patch("partcraft.pipeline_v2.sq2_qc_c.AsyncOpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create = AsyncMock(return_value=resp)
            result = _run(_sq2._process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_pass"] == 1
        assert result["n_fail"] == 0
        qc = load_qc(ctx)
        assert qc["edits"][edit_id]["gates"]["C"]["vlm"]["pass"] is True

    def test_region_match_false_fails(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed_flux()))
        _save_png(ctx.highlight_path(0))
        edit_id = ctx.edit_id("modification", 0)
        _save_png(ctx.edit_2d_output(edit_id))

        resp = MagicMock()
        resp.choices[0].message.content = '{"region_match":false,"reason":"edit outside region"}'
        with patch("partcraft.pipeline_v2.sq2_qc_c.AsyncOpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create = AsyncMock(return_value=resp)
            result = _run(_sq2._process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_fail"] == 1
        assert result["n_pass"] == 0
        assert is_edit_qc_failed(ctx, edit_id)

    def test_vlm_error_fallback_fails(self, tmp_path):
        """VLM exception → fallback {"region_match": False, "reason": "vlm_error"}."""
        ctx = _make_ctx(tmp_path)
        ctx.parsed_path.write_text(json.dumps(_minimal_parsed_flux()))
        _save_png(ctx.highlight_path(0))
        edit_id = ctx.edit_id("modification", 0)
        _save_png(ctx.edit_2d_output(edit_id))

        with patch("partcraft.pipeline_v2.sq2_qc_c.AsyncOpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("connection refused")
            )
            result = _run(_sq2._process_one(ctx, "http://fake", "fake-model", force=True))

        assert result["n_fail"] == 1
        qc = load_qc(ctx)
        assert qc["edits"][edit_id]["gates"]["C"]["vlm"]["pass"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
