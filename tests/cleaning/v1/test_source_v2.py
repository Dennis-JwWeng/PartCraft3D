import json
from pathlib import Path

import pytest

from partcraft.cleaning.v1.canonical_record import PassResult
from partcraft.cleaning.v1.source_v2 import iter_records_from_v2_obj


def test_v2_adapter_yields_one_record_per_edit(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    assert len(recs) == 2
    by_id = {r.edit_id: r for r in recs}
    assert set(by_id.keys()) == {"del_objA_000", "mod_objA_001"}


def test_v2_adapter_extracts_canonical_passes(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    by_id = {r.edit_id: r for r in recs}
    rec = by_id["del_objA_000"]
    assert rec.passes["gate_text_align"].passed is True
    assert rec.passes["gate_quality"].passed is True
    assert rec.passes["gate_quality"].score == pytest.approx(0.92)


def test_v2_adapter_records_failing_gate_e(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    by_id = {r.edit_id: r for r in recs}
    rec = by_id["mod_objA_001"]
    assert rec.passes["gate_quality"].passed is False
    assert "blurry" in rec.passes["gate_quality"].reason


def test_v2_adapter_omits_pass_when_gate_e_is_null(v2_obj_dir: Path):
    es_path = v2_obj_dir / "edit_status.json"
    es = json.loads(es_path.read_text())
    es["edits"]["del_objA_000"]["gates"]["E"] = None
    es_path.write_text(json.dumps(es))
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    rec = next(r for r in recs if r.edit_id == "del_objA_000")
    assert "gate_quality" not in rec.passes


def test_v2_adapter_attaches_after_paths_per_branch(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    by_id = {r.edit_id: r for r in recs}
    del_rec = by_id["del_objA_000"]
    mod_rec = by_id["mod_objA_001"]
    assert del_rec.after_glb is not None and del_rec.after_glb.name == "after_new.glb"
    assert del_rec.after_npz is None
    assert mod_rec.after_glb is None
    assert mod_rec.after_npz is not None and mod_rec.after_npz.name == "after.npz"
    assert len(del_rec.preview_pngs) == 5


def test_v2_adapter_extracts_spec_subset(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    rec = next(r for r in recs if r.edit_id == "del_objA_000")
    assert rec.spec["prompt"] == "Remove the handle."
    assert rec.spec["edit_type"] == "deletion"
    assert rec.spec["selected_part_ids"] == [0]
    assert rec.spec["part_labels"] == ["handle"]


def test_v2_adapter_uses_run_tag_as_provenance(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    for r in recs:
        assert r.source_pipeline == "v2"
        assert r.source_run_tag == "pipeline_v2_shard05"
        assert r.source_run_dir == v2_obj_dir
        assert r.shard == "05"
        assert r.obj_id == "objA"



def test_addition_inherits_gate_text_align_from_source_del(
    v2_obj_dir_with_addition: Path,
):
    """v2 never wrote Gate A for additions; the v1 adapter inherits the
    verdict from the source deletion (linked via add_*/meta.json::source_del_id)."""
    recs = list(iter_records_from_v2_obj(v2_obj_dir_with_addition,
                                         run_tag="pipeline_v2_shard05"))
    by_id = {r.edit_id: r for r in recs}
    assert "add_objA_000" in by_id
    add_rec = by_id["add_objA_000"]
    assert "gate_text_align" in add_rec.passes
    p = add_rec.passes["gate_text_align"]
    assert p.passed is True
    assert "del_objA_000" in (p.extra or {}).get("inherited_from", "")
    assert "inherited" in p.producer


def test_addition_without_source_del_has_no_gate(tmp_path: Path):
    """Addition with no ``meta.json`` (no source_del link) does NOT auto-pass."""
    obj = tmp_path / "objects" / "05" / "objA"
    obj.mkdir(parents=True)
    (obj / "edit_status.json").write_text(json.dumps({
        "obj_id": "objA", "shard": "05",
        "edits": {"add_objA_000": {"edit_type": "addition"}},
    }))
    (obj / "phase1").mkdir()
    (obj / "phase1" / "parsed.json").write_text(json.dumps(
        {"parts": [], "edits": []}))
    recs = list(iter_records_from_v2_obj(obj, run_tag="t"))
    assert len(recs) == 1
    assert "gate_text_align" not in recs[0].passes


def test_addition_inheritance_preserves_fail_verdict(tmp_path: Path):
    """If source deletion FAILED Gate A, the addition inherits FAIL — not silent pass."""
    obj = tmp_path / "objects" / "05" / "objA"
    (obj / "phase1").mkdir(parents=True)
    (obj / "phase1" / "parsed.json").write_text(json.dumps(
        {"parts": [], "edits": [
            {"edit_type": "deletion", "selected_part_ids": [0],
             "prompt": "Remove X.", "view_index": 0,
             "target_part_desc": "x", "edit_params": {}}]}))
    (obj / "edit_status.json").write_text(json.dumps({
        "obj_id": "objA", "shard": "05",
        "edits": {
            "del_objA_000": {
                "edit_type": "deletion",
                "gates": {"A": {"rule": {"pass": False}, "vlm": {"pass": False, "score": 0.1, "reason": "bad"}}},
            },
            "add_objA_000": {"edit_type": "addition"},
        },
    }))
    add_dir = obj / "edits_3d" / "add_objA_000"
    add_dir.mkdir(parents=True)
    (add_dir / "meta.json").write_text(json.dumps({
        "edit_id": "add_objA_000", "edit_type": "addition",
        "source_del_id": "del_objA_000",
    }))
    recs = list(iter_records_from_v2_obj(obj, run_tag="t"))
    by_id = {r.edit_id: r for r in recs}
    p = by_id["add_objA_000"].passes["gate_text_align"]
    assert p.passed is False
    assert (p.extra or {}).get("inherited_from") == "del_objA_000"
