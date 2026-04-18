from pathlib import Path

import pytest

from partcraft.cleaning.v1.layout import V1Layout, parse_edit_id, EDIT_TYPE_BY_PREFIX


@pytest.fixture
def v1(tmp_path):
    return V1Layout(root=tmp_path / "v1")


def test_object_dir_uses_shard_subdir(v1):
    p = v1.object_dir("05", "abc123")
    assert p == v1.root / "objects" / "05" / "abc123"


def test_before_views_paths_use_view_k_naming(v1):
    paths = v1.before_view_paths("05", "abc123")
    assert [p.name for p in paths] == [f"view_{k}.png" for k in range(5)]
    assert all(p.parent.name == "views" for p in paths)


def test_edit_dir_with_no_collision(v1):
    p = v1.edit_dir("05", "abc123", "del_abc123_000", suffix="")
    assert p == v1.object_dir("05", "abc123") / "edits" / "del_abc123_000"


def test_edit_dir_with_disambiguation_suffix(v1):
    p = v1.edit_dir("05", "abc123", "del_abc123_000", suffix="__r2")
    assert p.name == "del_abc123_000__r2"


def test_after_views_naming(v1):
    paths = v1.after_view_paths("05", "abc123", "del_abc123_000", suffix="")
    assert [p.name for p in paths] == [f"view_{k}.png" for k in range(5)]


def test_after_npz_path(v1):
    p = v1.after_npz_path("05", "abc123", "del_abc123_000", suffix="")
    assert p.name == "after.npz"


def test_pending_del_latent_file(v1):
    p = v1.pending_del_latent_file()
    assert p == v1.root / "_pending" / "del_latent.txt"


def test_index_files(v1):
    assert v1.objects_jsonl() == v1.root / "index" / "objects.jsonl"
    assert v1.edits_jsonl() == v1.root / "index" / "edits.jsonl"


@pytest.mark.parametrize("edit_id, expected_type", [
    ("del_abc_000", "deletion"),
    ("add_abc_001", "addition"),
    ("mod_abc_002", "modification"),
    ("scl_abc_003", "scale"),
    ("mat_abc_004", "material"),
    ("clr_abc_005", "color"),
    ("glb_abc_006", "glb"),
])
def test_parse_edit_id_prefix(edit_id, expected_type):
    typ, _obj, _idx = parse_edit_id(edit_id)
    assert typ == expected_type


def test_parse_edit_id_returns_obj_id_and_index():
    typ, obj_id, idx = parse_edit_id("del_5dc4ca7d607c495bb82eca3d0153cc2c_007")
    assert typ == "deletion"
    assert obj_id == "5dc4ca7d607c495bb82eca3d0153cc2c"
    assert idx == 7


def test_parse_edit_id_rejects_unknown_prefix():
    with pytest.raises(ValueError, match="unknown edit type prefix"):
        parse_edit_id("xyz_abc_000")
