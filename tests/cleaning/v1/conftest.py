"""Shared fixtures for cleaning/v1 tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _touch(p: Path, content: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


@pytest.fixture
def v2_obj_dir(tmp_path: Path) -> Path:
    obj = tmp_path / "pipeline_v2_shard05" / "objects" / "05" / "objA"
    _write_json(obj / "phase1" / "parsed.json", {
        "obj_id": "objA", "shard": "05", "validation": {"ok": True},
        "parsed": {
            "object": {
                "full_desc_stage1": "An object",
                "full_desc_stage2": "",
                "parts": [
                    {"part_id": 0, "name": "handle"},
                    {"part_id": 1, "name": "body"},
                ],
            },
            "edits": [
                {"edit_type": "deletion", "selected_part_ids": [0],
                 "prompt": "Remove the handle.", "target_part_desc": "handle on right",
                 "view_index": 0, "edit_params": {}},
                {"edit_type": "modification", "selected_part_ids": [1],
                 "prompt": "Make the body wooden.", "target_part_desc": "the body",
                 "view_index": 1, "edit_params": {"new_part_desc": "wooden body"}},
            ],
        },
    })
    _write_json(obj / "edit_status.json", {
        "obj_id": "objA", "shard": "05", "schema_version": 1,
        "edits": {
            "del_objA_000": {
                "edit_type": "deletion",
                "stages": {"gate_a": {"status": "pass"}},
                "gates": {
                    "A": {"rule": {"pass": True}, "vlm": {"pass": True, "score": 1.0, "reason": "ok"}},
                    "C": None,
                    "E": {"vlm": {"pass": True, "score": 0.92, "reason": "good"}},
                },
                "final_pass": True,
            },
            "mod_objA_001": {
                "edit_type": "modification",
                "stages": {"gate_a": {"status": "pass"}},
                "gates": {
                    "A": {"rule": {"pass": True}, "vlm": {"pass": True, "score": 1.0, "reason": "ok"}},
                    "C": None,
                    "E": {"vlm": {"pass": False, "score": 0.2, "reason": "blurry"}},
                },
                "final_pass": False,
            },
        },
    })
    _touch(obj / "edits_3d" / "del_objA_000" / "after_new.glb")
    for k in range(5):
        _touch(obj / "edits_3d" / "del_objA_000" / f"preview_{k}.png")
    _touch(obj / "edits_3d" / "mod_objA_001" / "after.npz")
    for k in range(5):
        _touch(obj / "edits_3d" / "mod_objA_001" / f"preview_{k}.png")
    return obj


@pytest.fixture
def v3_obj_dir(tmp_path: Path) -> Path:
    obj = tmp_path / "shard08_run" / "mode_e_text_align" / "objects" / "08" / "objB"
    _write_json(obj / "phase1" / "parsed.json", {
        "obj_id": "objB", "shard": "08", "validation": {"ok": True},
        "parsed": {
            "object": {
                "full_desc_stage1": "Another object",
                "parts": [{"part_id": 0, "name": "leg"}],
            },
            "edits": [
                {"edit_type": "deletion", "selected_part_ids": [0],
                 "prompt": "Remove the leg.", "target_part_desc": "the leg",
                 "view_index": 0, "edit_params": {}},
            ],
        },
    })
    _write_json(obj / "edit_status.json", {
        "obj_id": "objB", "shard": "08", "schema_version": 1,
        "edits": {
            "del_objB_000": {
                "edit_type": "deletion",
                "stages": {"sq3_qc_E": {"status": "ok"}},
                "gates": {
                    "A": {"rule": {"pass": True}, "vlm": {"pass": True, "score": 1.0, "reason": "ok"}},
                    "C": None,
                    "E": {"vlm": {"pass": True, "score": 0.88, "reason": "ok"}},
                },
                "final_pass": True,
            },
        },
    })
    _touch(obj / "edits_3d" / "del_objB_000" / "after_new.glb")
    for k in range(5):
        _touch(obj / "edits_3d" / "del_objB_000" / f"preview_{k}.png")
    return obj


@pytest.fixture
def v2_obj_dir_with_addition(tmp_path: Path) -> Path:
    """v2 object with one deletion (PASS Gate A) and its inverse addition.

    Mirrors what ``pipeline_v3.mesh_deletion._write_addition_meta`` produces:
    add_*/meta.json carries a ``source_del_id`` linking back to the deletion.
    """
    obj = tmp_path / "pipeline_v2_shard05" / "objects" / "05" / "objA"
    _write_json(obj / "phase1" / "parsed.json", {
        "obj_id": "objA", "shard": "05", "validation": {"ok": True},
        "parsed": {
            "object": {
                "full_desc_stage1": "An object",
                "parts": [{"part_id": 0, "name": "wheel"}],
            },
            "edits": [
                {"edit_type": "deletion", "selected_part_ids": [0],
                 "prompt": "Remove the wheel.", "view_index": 0,
                 "target_part_desc": "the wheel", "edit_params": {}},
            ],
        },
    })
    _write_json(obj / "edit_status.json", {
        "obj_id": "objA", "shard": "05", "schema_version": 1,
        "edits": {
            "del_objA_000": {
                "edit_type": "deletion",
                "gates": {
                    "A": {"rule": {"pass": True},
                          "vlm": {"pass": True, "score": 1.0,
                                  "reason": "auto_pass_deletion"}},
                    "C": None, "E": None,
                },
            },
            "add_objA_000": {
                "edit_type": "addition",
                # NB: no gates — v2 never runs Gate A on additions.
            },
        },
    })
    _touch(obj / "edits_3d" / "del_objA_000" / "after_new.glb")
    for k in range(5):
        _touch(obj / "edits_3d" / "del_objA_000" / f"preview_{k}.png")
    _write_json(obj / "edits_3d" / "add_objA_000" / "meta.json", {
        "edit_id": "add_objA_000", "edit_type": "addition",
        "obj_id": "objA", "shard": "05",
        "source_del_id": "del_objA_000",
        "selected_part_ids": [0],
        "prompt": "Add the wheel.",
        "target_part_desc": "the wheel",
    })
    _touch(obj / "edits_3d" / "add_objA_000" / "after_new.glb")
    for k in range(5):
        _touch(obj / "edits_3d" / "add_objA_000" / f"preview_{k}.png")
    return obj
