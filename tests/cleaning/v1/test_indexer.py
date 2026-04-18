import json
from pathlib import Path

from partcraft.cleaning.v1.indexer import rebuild_index
from partcraft.cleaning.v1.layout import V1Layout


def _fake_v1(tmp_path: Path) -> V1Layout:
    layout = V1Layout(root=tmp_path / "v1")
    obj = layout.object_dir("05", "objA")
    obj.mkdir(parents=True)
    layout.before_dir("05", "objA").mkdir(parents=True)
    layout.before_ss_npz("05", "objA").write_bytes(b"x")
    layout.before_slat_npz("05", "objA").write_bytes(b"x")
    for p in layout.before_view_paths("05", "objA"):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
    layout.meta_json("05", "objA").write_text(json.dumps({
        "obj_id": "objA", "shard": "05", "part_list": []
    }))
    for eid in ("del_objA_000", "mod_objA_001"):
        ed = layout.edit_dir("05", "objA", eid); ed.mkdir(parents=True)
        layout.spec_json("05", "objA", eid).write_text(json.dumps({
            "edit_id": eid,
            "edit_type": ("deletion" if eid.startswith("del_") else "modification"),
        }))
        layout.qc_json("05", "objA", eid).write_text(json.dumps({
            "edit_id": eid,
            "source": {"pipeline_version": "v2", "run_tag": "rt"},
            "passes": {"gate_text_align": {"pass": True}, "gate_quality": {"pass": True}},
        }))
        for p in layout.after_view_paths("05", "objA", eid):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
        if eid.startswith("mod_"):
            layout.after_npz_path("05", "objA", eid).write_bytes(b"x")
    return layout


def test_rebuild_writes_jsonl(tmp_path: Path):
    layout = _fake_v1(tmp_path)
    summary = rebuild_index(layout)
    assert summary["n_objects"] == 1
    assert summary["n_edits"] == 2
    edits_lines = layout.edits_jsonl().read_text().splitlines()
    assert len(edits_lines) == 2
    parsed = [json.loads(ln) for ln in edits_lines]
    by_id = {p["edit_id"]: p for p in parsed}
    assert by_id["mod_objA_001"]["after_npz"].endswith("after.npz")
    assert by_id["del_objA_000"]["after_npz"] is None
    assert by_id["del_objA_000"]["source_pipeline"] == "v2"


def test_rebuild_records_disambiguation_suffix(tmp_path: Path):
    layout = _fake_v1(tmp_path)
    eid = "del_objA_000"; suffix = "__r2"
    ed = layout.edit_dir("05", "objA", eid, suffix=suffix); ed.mkdir(parents=True)
    layout.spec_json("05", "objA", eid, suffix=suffix).write_text(json.dumps({
        "edit_id": eid, "edit_type": "deletion"
    }))
    layout.qc_json("05", "objA", eid, suffix=suffix).write_text(json.dumps({
        "edit_id": eid, "source": {"pipeline_version": "v3", "run_tag": "rt2"},
        "passes": {"gate_text_align": {"pass": True}, "gate_quality": {"pass": True}},
    }))
    rebuild_index(layout)
    rows = [json.loads(ln) for ln in layout.edits_jsonl().read_text().splitlines()]
    suffixed = [r for r in rows if r["edit_dir_suffix"] == "__r2"]
    assert len(suffixed) == 1
    assert suffixed[0]["source_pipeline"] == "v3"
