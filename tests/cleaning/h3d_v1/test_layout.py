"""Smoke tests for ``partcraft.cleaning.h3d_v1.layout``.

Path-only tests — no IO. Verifies the dataset path contract that every
other module depends on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from partcraft.cleaning.h3d_v1.layout import (
    EDIT_PREFIX_TO_TYPE,
    EDIT_TYPES_ALL,
    EDIT_TYPES_FLUX,
    H3DLayout,
    N_VIEWS,
    edit_type_from_id,
    paired_edit_id,
)


# ── prefix / type lookup ───────────────────────────────────────────────
@pytest.mark.parametrize(
    "edit_id, expected",
    [
        ("del_abc123_000", "deletion"),
        ("add_abc123_004", "addition"),
        ("mod_abc123_007", "modification"),
        ("scl_abc123_001", "scale"),
        ("mat_abc123_002", "material"),
        ("clr_abc123_003", "color"),
        ("glb_abc123_005", "global"),
    ],
)
def test_edit_type_from_id(edit_id: str, expected: str) -> None:
    assert edit_type_from_id(edit_id) == expected


def test_edit_type_from_id_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unrecognised"):
        edit_type_from_id("xyz_obj_000")


def test_prefix_table_covers_all_types() -> None:
    assert set(EDIT_PREFIX_TO_TYPE.values()) == set(EDIT_TYPES_ALL)
    # Flux subset is exactly EDIT_TYPES_ALL minus del/add.
    assert set(EDIT_TYPES_FLUX) == set(EDIT_TYPES_ALL) - {"deletion", "addition"}


# ── pairing ────────────────────────────────────────────────────────────
def test_paired_edit_id_del_add() -> None:
    assert paired_edit_id("del_abc_007") == "add_abc_007"
    assert paired_edit_id("add_abc_007") == "del_abc_007"


def test_paired_edit_id_returns_none_for_flux() -> None:
    for eid in ("mod_abc_000", "mat_abc_000", "scl_x_1", "clr_y_2", "glb_z_3"):
        assert paired_edit_id(eid) is None


# ── path resolution ────────────────────────────────────────────────────
@pytest.fixture()
def layout(tmp_path: Path) -> H3DLayout:
    return H3DLayout(root=tmp_path / "H3D_v1")


def test_assets_paths(layout: H3DLayout) -> None:
    assert layout.assets_obj_dir("08", "abc") == layout.root / "_assets" / "08" / "abc"
    assert layout.object_npz("08", "abc").name == "object.npz"
    assert layout.orig_view("08", "abc", 0).name == "view0.png"
    assert layout.orig_view("08", "abc", 4).parent.name == "orig_views"


def test_orig_view_index_bounds(layout: H3DLayout) -> None:
    with pytest.raises(ValueError):
        layout.orig_view("08", "abc", -1)
    with pytest.raises(ValueError):
        layout.orig_view("08", "abc", N_VIEWS)


def test_edit_paths(layout: H3DLayout) -> None:
    edit_dir = layout.edit_dir("deletion", "08", "abc", "del_abc_000")
    assert edit_dir == layout.root / "deletion" / "08" / "abc" / "del_abc_000"
    assert layout.before_npz("deletion", "08", "abc", "del_abc_000").name == "before.npz"
    assert layout.after_npz("deletion", "08", "abc", "del_abc_000").name == "after.npz"
    assert layout.before_view("deletion", "08", "abc", "del_abc_000", 2).name == "view2.png"
    assert layout.after_view("deletion", "08", "abc", "del_abc_000", 2).parent.name == "after_views"
    assert layout.meta_json("deletion", "08", "abc", "del_abc_000").name == "meta.json"


def test_edit_dir_rejects_unknown_type(layout: H3DLayout) -> None:
    with pytest.raises(ValueError, match="unknown edit_type"):
        layout.edit_dir("identity", "08", "abc", "id_abc_000")


def test_manifest_paths(layout: H3DLayout) -> None:
    assert layout.manifest_path("deletion", "08") == layout.root / "manifests" / "deletion" / "08.jsonl"
    assert layout.aggregated_manifest() == layout.root / "manifests" / "all.jsonl"
