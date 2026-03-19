#!/usr/bin/env python3
"""Tests for the semantic enricher (orthogonal 4-view VLM pipeline).

Covers:
  - JSON extraction and helper functions
  - VLM call with images (mocked)
  - Fallback enrichment (no VLM)
  - Result conversion to phase0 record format
  - Integration with enrich_semantic_labels() entry point

Usage:
    # Run all tests
    pytest tests/test_visual_enricher.py -v

    # Run only unit tests (no dataset needed)
    pytest tests/test_visual_enricher.py -v -k "not integration"
"""

import io
import json
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.phase1_planning.enricher import (
    _enrich_one_object_visual,
    _extract_json,
    _fallback_enrichment,
    _is_core_part,
    _result_to_phase0_record,
    _vlm_call_with_images,
    enrich_semantic_labels,
)


# ---------------------------------------------------------------------------
# Test fixtures: mock ObjectRecord
# ---------------------------------------------------------------------------

def _make_rgba_png(w: int = 64, h: int = 64, color=(200, 100, 50)) -> bytes:
    """Create a simple RGBA PNG in memory."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = color
    arr[:, :, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="WEBP")
    return buf.getvalue()


@dataclass
class MockPartInfo:
    part_id: int
    cluster_name: str = ""
    mesh_node_names: list = field(default_factory=list)
    cluster_size: int = 100


class MockObjectRecord:
    """Minimal mock of ObjectRecord for unit tests.

    NOTE: The enricher now uses _select_orthogonal_views (needs
    get_transforms) and _render_plain_views (needs get_image_bytes).
    Tests that exercise _enrich_one_object_visual need a mock that
    provides get_transforms() returning a frames list with
    transform_matrix entries.  Tests here that call into that function
    use MagicMock for the dataset object instead.
    """

    def __init__(self, obj_id="test_obj", n_parts=4, n_views=6):
        self.obj_id = obj_id
        self.shard = "00"
        self.num_views = n_views
        self.parts = [MockPartInfo(part_id=i, cluster_name=f"part_{i}")
                      for i in range(n_parts)]
        self._n_parts = n_parts
        self._image_bytes = _make_rgba_png()

    def get_image_bytes(self, view_idx: int) -> bytes:
        return self._image_bytes

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Unit tests: VLM call with images (mocked)
# ---------------------------------------------------------------------------

class TestVlmCallWithImages:
    def _mock_client(self, response_text: str):
        client = MagicMock()
        msg = MagicMock()
        msg.content = response_text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        return client

    def test_parses_valid_json(self):
        data = {"object_desc": "A chair", "part_groups": []}
        client = self._mock_client(json.dumps(data))
        result = _vlm_call_with_images(
            client, "test-model", "prompt", [_make_rgba_png()])
        assert result == data

    def test_returns_none_on_empty_response(self):
        client = self._mock_client("")
        result = _vlm_call_with_images(
            client, "test-model", "prompt", [_make_rgba_png()],
            max_retries=0)
        assert result is None

    def test_extracts_json_from_fenced_block(self):
        data = {"parts": [{"part_id": 0}]}
        text = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
        client = self._mock_client(text)
        result = _vlm_call_with_images(
            client, "m", "p", [_make_rgba_png()])
        assert result == data

    def test_retries_on_failure(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("timeout"),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content='{"ok": true}'))]),
        ]
        result = _vlm_call_with_images(
            client, "m", "p", [_make_rgba_png()], max_retries=1)
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_sends_multiple_images(self):
        data = {"result": "ok"}
        client = self._mock_client(json.dumps(data))
        imgs = [_make_rgba_png(), _make_rgba_png(), _make_rgba_png()]
        _vlm_call_with_images(client, "m", "p", imgs)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        content = messages[0]["content"]
        # 3 images + 1 text
        image_items = [c for c in content if c["type"] == "image_url"]
        text_items = [c for c in content if c["type"] == "text"]
        assert len(image_items) == 3
        assert len(text_items) == 1


# ---------------------------------------------------------------------------
# Unit tests: result conversion
# ---------------------------------------------------------------------------

class TestResultToPhase0Record:
    def test_converts_visual_result(self):
        result = {
            "object_desc": "A sword",
            "parts": [
                {"part_id": 0, "label": "blade", "is_core": False,
                 "desc": "sharp blade", "desc_without": "hilt only",
                 "deletion": {"prompt": "Remove blade", "after_desc": "hilt"},
                 "addition": {"prompt": "Add blade", "after_desc": "sword"},
                 "swaps": [{"prompt": "Replace blade with axe",
                            "after_desc": "axe weapon",
                            "before_part_desc": "blade",
                            "after_part_desc": "axe head"}]},
                {"part_id": 1, "label": "hilt", "is_core": True,
                 "desc": "leather hilt", "desc_without": "",
                 "deletion": None, "addition": None, "swaps": []},
            ],
            "global_edits": [
                {"prompt": "Make it wooden", "after_desc": "wooden sword"},
            ],
        }
        record = _result_to_phase0_record(result, "uid123", "Weapon", "00")

        assert record["obj_id"] == "uid123"
        assert record["num_parts"] == 2
        assert record["object_desc"] == "A sword"

        # Non-core part should have deletion + addition + swap edits
        blade = record["parts"][0]
        assert blade["core"] is False
        edit_types = [e["type"] for e in blade["edits"]]
        assert "deletion" in edit_types
        assert "addition" in edit_types
        assert "modification" in edit_types

        # Core part should have no edits
        hilt = record["parts"][1]
        assert hilt["core"] is True
        assert len(hilt["edits"]) == 0

        # Global edits
        assert len(record["global_edits"]) == 1


# ---------------------------------------------------------------------------
# Unit tests: existing functions (regression)
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"b": 2}\n```') == {"b": 2}

    def test_json_with_surrounding_text(self):
        text = 'Here is the answer: {"c": 3} that is it'
        assert _extract_json(text) == {"c": 3}

    def test_invalid_json_returns_none(self):
        assert _extract_json("not json at all") is None


class TestIsCorePartFunction:
    def test_core_labels(self):
        assert _is_core_part("body") is True
        assert _is_core_part("main_frame") is True
        assert _is_core_part("Head") is True

    def test_non_core_labels(self):
        assert _is_core_part("wheel") is False
        assert _is_core_part("spoiler") is False
        assert _is_core_part("arm_left") is False


class TestFallbackEnrichment:
    def test_produces_valid_structure(self):
        result = _fallback_enrichment("Chair", ["seat", "back", "leg"])
        assert "object_desc" in result
        assert len(result["parts"]) == 3
        assert len(result["global_edits"]) == 2

        # Non-core part should have edits
        leg = result["parts"][2]
        assert leg["deletion"] is not None
        assert leg["addition"] is not None
        assert len(leg["swaps"]) == 1

    def test_core_parts_have_no_edits(self):
        result = _fallback_enrichment("Robot", ["body", "arm"])
        body = result["parts"][0]
        assert body["is_core"] is True
        assert body["deletion"] is None


# ---------------------------------------------------------------------------
# Integration tests: enrich_semantic_labels with mocked VLM
# ---------------------------------------------------------------------------

class TestEnrichSemanticLabelsIntegration:
    """Test the full enrich_semantic_labels() entry point with mocked VLM."""

    def _make_semantic_json(self, tmpdir: Path, uids_labels: dict) -> Path:
        """Create a minimal semantic.json."""
        sem = {"TestCategory": uids_labels}
        path = tmpdir / "semantic.json"
        path.write_text(json.dumps(sem))
        return path

    def _mock_vlm_response(self, labels):
        """Generate a valid VLM response for given labels."""
        parts = []
        for i, lbl in enumerate(labels):
            is_core = _is_core_part(lbl)
            p = {
                "part_id": i, "label": lbl,
                "is_core": is_core,
                "desc": lbl, "desc_without": f"without {lbl}",
            }
            if not is_core:
                p["deletion"] = {"prompt": f"Remove {lbl}", "after_desc": f"no {lbl}"}
                p["addition"] = {"prompt": f"Add {lbl}", "after_desc": f"with {lbl}"}
                p["swaps"] = [{"prompt": f"Replace {lbl} with X",
                               "after_desc": f"with X instead of {lbl}",
                               "before_part_desc": lbl,
                               "after_part_desc": "X"}]
            else:
                p["deletion"] = None
                p["addition"] = None
                p["swaps"] = []
            parts.append(p)
        return {
            "object_desc": "A test object",
            "parts": parts,
            "global_edits": [{"prompt": "Make it shiny", "after_desc": "shiny object"}],
        }

    def test_legacy_mode_no_dataset(self):
        """Legacy single-call mode when dataset=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            labels = ["body", "wing", "tail"]
            sem_path = self._make_semantic_json(
                tmpdir, {"uid_001": labels})
            output_path = tmpdir / "output.jsonl"

            response = self._mock_vlm_response(labels)
            with patch("partcraft.phase1_planning.enricher.OpenAI") as MockOAI:
                mock_client = MagicMock()
                MockOAI.return_value = mock_client
                msg = MagicMock()
                msg.content = json.dumps(response)
                choice = MagicMock()
                choice.message = msg
                resp = MagicMock()
                resp.choices = [choice]
                mock_client.chat.completions.create.return_value = resp

                cfg = {"phase0": {"vlm_model": "test", "vlm_base_url": "",
                                  "vlm_api_key": "fake-key"}}
                enrich_semantic_labels(
                    cfg, str(sem_path), str(output_path),
                    visual_grounding=False, dataset=None, limit=1)

            assert output_path.exists()
            with open(output_path) as f:
                records = [json.loads(line) for line in f]
            assert len(records) == 1
            assert records[0]["obj_id"] == "uid_001"
            assert records[0]["num_parts"] == 3

    def test_visual_fallback_to_legacy_on_load_failure(self):
        """If dataset.load_object fails, should fall back to legacy mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            labels = ["body", "arm"]
            sem_path = self._make_semantic_json(
                tmpdir, {"uid_003": labels})
            output_path = tmpdir / "output.jsonl"

            # Legacy response (single-call format with "parts" key)
            legacy_response = self._mock_vlm_response(labels)

            with patch("partcraft.phase1_planning.enricher.OpenAI") as MockOAI:
                mock_client = MagicMock()
                MockOAI.return_value = mock_client
                msg = MagicMock()
                msg.content = json.dumps(legacy_response)
                choice = MagicMock()
                choice.message = msg
                resp = MagicMock()
                resp.choices = [choice]
                mock_client.chat.completions.create.return_value = resp

                # Dataset that fails to load
                mock_dataset = MagicMock()
                mock_dataset.load_object.side_effect = FileNotFoundError("NPZ missing")

                cfg = {"phase0": {"vlm_model": "test", "vlm_base_url": "",
                                  "vlm_api_key": "fake-key"}}
                enrich_semantic_labels(
                    cfg, str(sem_path), str(output_path),
                    visual_grounding=True, dataset=mock_dataset, limit=1)

            assert output_path.exists()
            with open(output_path) as f:
                records = [json.loads(line) for line in f]
            assert len(records) == 1

    def test_resume_skips_done_objects(self):
        """Already-enriched objects should be skipped on resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sem_path = self._make_semantic_json(
                tmpdir, {"uid_a": ["x"], "uid_b": ["y"]})
            output_path = tmpdir / "output.jsonl"

            # Pre-populate with uid_a
            with open(output_path, "w") as f:
                f.write(json.dumps({"obj_id": "uid_a", "parts": []}) + "\n")

            with patch("partcraft.phase1_planning.enricher.OpenAI") as MockOAI:
                mock_client = MagicMock()
                MockOAI.return_value = mock_client
                msg = MagicMock()
                msg.content = json.dumps(self._mock_vlm_response(["y"]))
                choice = MagicMock()
                choice.message = msg
                resp = MagicMock()
                resp.choices = [choice]
                mock_client.chat.completions.create.return_value = resp

                cfg = {"phase0": {"vlm_model": "test", "vlm_base_url": "",
                                  "vlm_api_key": "fake-key"}}
                enrich_semantic_labels(
                    cfg, str(sem_path), str(output_path),
                    visual_grounding=False, dataset=None)

            with open(output_path) as f:
                records = [json.loads(line) for line in f]
            assert len(records) == 2
            assert records[0]["obj_id"] == "uid_a"  # old
            assert records[1]["obj_id"] == "uid_b"  # new


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
