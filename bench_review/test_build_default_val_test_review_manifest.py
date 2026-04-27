from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_default_val_test_review_manifest as builder


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _touch_assets(repo: Path, edit_type: str, shard: str, obj_id: str, edit_id: str) -> None:
    h3d_dir = repo / "data/H3D_v1" / edit_type / shard / obj_id / edit_id
    h3d_dir.mkdir(parents=True, exist_ok=True)
    for name in ("before.png", "after.png", "before.npz", "after.npz"):
        (h3d_dir / name).write_bytes(b"asset")
    (h3d_dir / "meta.json").write_text(
        json.dumps({"instruction": {"prompt": f"edit {edit_id}"}}),
        encoding="utf-8",
    )


def _touch_pipeline(repo: Path, shard: str, obj_id: str, edit_id: str, *, with_2d: bool = False) -> None:
    obj_dir = repo / "outputs/partverse" / f"shard{shard}" / "mode_e_text_align" / "objects" / shard / obj_id
    (obj_dir / "edits_3d" / edit_id).mkdir(parents=True, exist_ok=True)
    if with_2d:
        e2d = obj_dir / "edits_2d"
        e2d.mkdir(parents=True, exist_ok=True)
        (e2d / f"{edit_id}_input.png").write_bytes(b"2d-in")
        (e2d / f"{edit_id}_edited.png").write_bytes(b"2d-out")


def test_default_val_test_manifest_requires_split_pipeline_and_flux_2d(tmp_path):
    repo = tmp_path
    split_dir = repo / "H3D_v1_hf/data/splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train.obj_ids.txt").write_text("obj_train\n", encoding="utf-8")
    (split_dir / "val.obj_ids.txt").write_text("obj_val\nobj_missing\n", encoding="utf-8")
    (split_dir / "test.obj_ids.txt").write_text("obj_test\n", encoding="utf-8")

    rows = [
        {"edit_id": "del_obj_val_000", "edit_type": "deletion", "obj_id": "obj_val", "shard": "02"},
        {"edit_id": "mat_obj_test_000", "edit_type": "material", "obj_id": "obj_test", "shard": "02"},
        {"edit_id": "clr_obj_val_000", "edit_type": "color", "obj_id": "obj_val", "shard": "02"},
        {"edit_id": "glb_obj_missing_000", "edit_type": "global", "obj_id": "obj_missing", "shard": "03"},
        {"edit_id": "add_obj_train_000", "edit_type": "addition", "obj_id": "obj_train", "shard": "02"},
    ]
    _write_jsonl(repo / "data/H3D_v1/manifests/all.jsonl", rows)
    for row in rows:
        _touch_assets(repo, row["edit_type"], row["shard"], row["obj_id"], row["edit_id"])
    _touch_pipeline(repo, "02", "obj_val", "del_obj_val_000")
    _touch_pipeline(repo, "02", "obj_test", "mat_obj_test_000", with_2d=True)
    _touch_pipeline(repo, "02", "obj_val", "clr_obj_val_000", with_2d=False)

    records, reject, missing = builder.iter_default_val_test_candidates(repo)

    assert {record["edit_id"] for record in records} == {"del_obj_val_000", "mat_obj_test_000"}
    assert reject["not_val_or_test"] == 1
    assert reject["flux_missing_2d_input"] == 1
    assert reject["no_pipeline_shard_dir"] == 1
    assert missing.by_shard_edit_count == {"03": 1}
    assert missing.by_shard_obj_ids == {"03": {"obj_missing"}}


def test_write_outputs_records_summary_and_missing_lists(tmp_path):
    record = {
        "bench_split": "default_val_test_available",
        "source_hf_split": "val",
        "edit_id": "del_obj_val_000",
        "edit_type": "deletion",
        "obj_id": "obj_val",
        "shard": "02",
        "h3d_before_png": "/tmp/before.png",
        "h3d_after_png": "/tmp/after.png",
        "h3d_before_npz": "/tmp/before.npz",
        "h3d_after_npz": "/tmp/after.npz",
        "h3d_meta_json": "/tmp/meta.json",
        "pipeline_edit_dir": "/tmp/edit",
        "two_d_input_png": "",
        "two_d_edited_png": "",
    }
    missing = builder.MissingRequirements()
    missing.add("03", "obj_missing", "glb_obj_missing_000", "global")

    outputs = builder.write_outputs(tmp_path, [record], {"no_pipeline_shard_dir": 1}, missing)

    assert outputs.manifest_path.read_text(encoding="utf-8").count("\n") == 1
    assert outputs.edit_ids_path.read_text(encoding="utf-8") == "del_obj_val_000\n"
    assert "default val+test" in outputs.summary_path.read_text(encoding="utf-8")
    assert (tmp_path / "missing/shard03_missing_obj_ids.txt").read_text(encoding="utf-8") == "obj_missing\n"


def test_default_val_test_manifest_can_filter_to_requested_shards(tmp_path):
    repo = tmp_path
    split_dir = repo / "H3D_v1_hf/data/splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train.obj_ids.txt").write_text("", encoding="utf-8")
    (split_dir / "val.obj_ids.txt").write_text("obj_02\nobj_03\n", encoding="utf-8")
    (split_dir / "test.obj_ids.txt").write_text("", encoding="utf-8")

    rows = [
        {"edit_id": "add_obj_02_000", "edit_type": "addition", "obj_id": "obj_02", "shard": "02"},
        {"edit_id": "add_obj_03_000", "edit_type": "addition", "obj_id": "obj_03", "shard": "03"},
    ]
    _write_jsonl(repo / "data/H3D_v1/manifests/all.jsonl", rows)
    for row in rows:
        _touch_assets(repo, row["edit_type"], row["shard"], row["obj_id"], row["edit_id"])
        _touch_pipeline(repo, row["shard"], row["obj_id"], row["edit_id"])

    records, reject, missing = builder.iter_default_val_test_candidates(repo, include_shards={"03"})

    assert [record["edit_id"] for record in records] == ["add_obj_03_000"]
    assert reject["not_requested_shard"] == 1
    assert missing.by_shard_edit_count == {}
