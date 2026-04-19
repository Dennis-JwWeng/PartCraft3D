"""Unit tests for ``partcraft.cleaning.h3d_v1.promoter``.

Builds a synthetic mini-pipeline fixture (1 obj, 1 del + 1 add + 1 mod
with prebuilt edit_status.json + npz/png artefacts) and exercises all
three ``promote_*`` routines. Hardlink correctness is verified via
``os.stat(...).st_ino`` equality assertions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from partcraft.cleaning.h3d_v1.layout import H3DLayout, N_VIEWS
from partcraft.cleaning.h3d_v1.pipeline_io import PipelineEdit
from partcraft.cleaning.h3d_v1.promoter import (
    PromoteContext,
    PromoteResult,
    promote_addition,
    promote_deletion,
    promote_flux,
)


SHARD = "08"
OBJ = "obj42"
DEL_ID = f"del_{OBJ}_000"
ADD_ID = f"add_{OBJ}_000"
MOD_ID = f"mod_{OBJ}_007"


def _save_npz(path: Path, *, marker: int) -> None:
    """Write a small 3-key npz; ``marker`` differentiates content for inode checks."""
    feats = np.full((4, 8), marker, dtype=np.float32)
    coords = np.zeros((4, 4), dtype=np.int32)
    ss = np.zeros((8, 16, 16, 16), dtype=np.float32)
    np.savez(path, slat_feats=feats, slat_coords=coords, ss=ss)


def _save_pngs(dir_: Path, prefix: str, n: int = N_VIEWS) -> None:
    cv2 = pytest.importorskip("cv2")
    dir_.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        img = np.full((518, 518, 3), fill_value=(k + 1) * 30, dtype=np.uint8)
        cv2.imwrite(str(dir_ / f"{prefix}{k}.png"), img)


@pytest.fixture()
def fixture(tmp_path: Path) -> dict:
    """Build a synthetic pipeline obj dir with del + add + mod artefacts."""
    pipeline_root = tmp_path / "pipeline"
    obj_dir = pipeline_root / "objects" / SHARD / OBJ
    edits_root = obj_dir / "edits_3d"

    del_dir = edits_root / DEL_ID
    add_dir = edits_root / ADD_ID
    mod_dir = edits_root / MOD_ID
    del_dir.mkdir(parents=True)
    add_dir.mkdir(parents=True)
    mod_dir.mkdir(parents=True)

    _save_npz(del_dir / "after.npz", marker=11)
    _save_pngs(del_dir, "preview_")

    _save_pngs(add_dir, "preview_")

    _save_npz(mod_dir / "before.npz", marker=22)
    _save_npz(mod_dir / "after.npz", marker=33)
    _save_pngs(mod_dir, "preview_")

    edit_status = {
        "obj_id": OBJ, "shard": SHARD, "schema_version": 1,
        "edits": {
            DEL_ID: {"edit_type": "deletion",
                     "stages": {"gate_a": {"status": "pass"}, "gate_e": {"status": "pass"}},
                     "gates": {"A": {"vlm": {"pass": True, "score": 1.0}}, "C": None,
                                "E": {"vlm": {"pass": True, "score": 1.0}}},
                     "final_pass": True},
            ADD_ID: {"edit_type": "addition",
                     "stages": {"gate_e": {"status": "pass"}},
                     "gates": {"A": None, "C": None, "E": {"vlm": {"pass": True, "score": 1.0}}},
                     "final_pass": True},
            MOD_ID: {"edit_type": "modification",
                     "stages": {"gate_a": {"status": "pass"}, "gate_e": {"status": "pass"}},
                     "gates": {"A": {"vlm": {"pass": True}}, "C": None,
                                "E": {"vlm": {"pass": True}}},
                     "final_pass": True},
        },
    }
    (obj_dir / "edit_status.json").write_text(json.dumps(edit_status))

    layout = H3DLayout(root=tmp_path / "H3D_v1")
    ctx = PromoteContext(
        pipeline_obj_root=pipeline_root / "objects" / SHARD,
        slat_dir=tmp_path / "slat_unused",
        images_root=tmp_path / "images_unused",
    )

    def _edit(edit_id: str, edit_type: str) -> PipelineEdit:
        return PipelineEdit(
            obj_id=OBJ, shard=SHARD, edit_id=edit_id, edit_type=edit_type,
            obj_dir=obj_dir, edit_dir=edits_root / edit_id,
            edit_status_path=obj_dir / "edit_status.json",
        )

    return {
        "layout": layout, "ctx": ctx, "obj_dir": obj_dir,
        "del_edit": _edit(DEL_ID, "deletion"),
        "add_edit": _edit(ADD_ID, "addition"),
        "mod_edit": _edit(MOD_ID, "modification"),
    }


def _ino(p: Path) -> int:
    return p.stat().st_ino


# ── deletion ───────────────────────────────────────────────────────────
def test_promote_deletion_writes_files_and_links_to_assets(fixture: dict) -> None:
    layout: H3DLayout = fixture["layout"]
    edit: PipelineEdit = fixture["del_edit"]
    ctx: PromoteContext = fixture["ctx"]

    res = promote_deletion(edit, layout, ctx=ctx)
    assert res.ok, res.reason
    assert res.manifest_record is not None
    assert res.manifest_record["edit_type"] == "deletion"

    object_npz = layout.object_npz(SHARD, OBJ)
    before_npz = layout.before_npz("deletion", SHARD, OBJ, DEL_ID)
    after_npz = layout.after_npz("deletion", SHARD, OBJ, DEL_ID)
    assert object_npz.is_file() and before_npz.is_file() and after_npz.is_file()
    assert _ino(before_npz) == _ino(object_npz), "before.npz must hardlink to _assets/object.npz"
    assert _ino(after_npz) == _ino(edit.edit_dir / "after.npz"), \
        "after.npz must hardlink to pipeline edit_dir/after.npz"
    for k in range(N_VIEWS):
        bv = layout.before_view("deletion", SHARD, OBJ, DEL_ID, k)
        av = layout.after_view("deletion", SHARD, OBJ, DEL_ID, k)
        assert bv.is_file() and av.is_file()
        assert _ino(bv) == _ino(layout.orig_view(SHARD, OBJ, k))
        assert _ino(av) == _ino(edit.edit_dir / f"preview_{k}.png")

    meta = json.loads(layout.meta_json("deletion", SHARD, OBJ, DEL_ID).read_text())
    assert meta["edit_id"] == DEL_ID
    assert meta["quality"]["final_pass"] is True
    assert meta["quality"].get("gate_A_score") == 1.0
    assert meta["lineage"]["pipeline_version"] == "v3"
    assert meta["lineage"]["source_dataset"] == "partverse"
    assert "promoted_at" in meta["lineage"]
    assert meta["lineage"].get("paired_edit_id") == "add_" + DEL_ID[4:]


def test_promote_deletion_idempotent(fixture: dict) -> None:
    layout, edit, ctx = fixture["layout"], fixture["del_edit"], fixture["ctx"]
    res1 = promote_deletion(edit, layout, ctx=ctx)
    assert res1.ok
    ino_before = _ino(layout.before_npz("deletion", SHARD, OBJ, DEL_ID))
    res2 = promote_deletion(edit, layout, ctx=ctx)
    assert res2.ok
    assert _ino(layout.before_npz("deletion", SHARD, OBJ, DEL_ID)) == ino_before


def test_promote_deletion_missing_after(fixture: dict) -> None:
    layout, edit, ctx = fixture["layout"], fixture["del_edit"], fixture["ctx"]
    (edit.edit_dir / "after.npz").unlink()
    res = promote_deletion(edit, layout, ctx=ctx)
    assert res.ok is False
    assert "after.npz" in (res.reason or "")


def test_promote_deletion_rejects_wrong_type(fixture: dict) -> None:
    layout, ctx = fixture["layout"], fixture["ctx"]
    bad = fixture["mod_edit"]
    with pytest.raises(ValueError, match="promote_deletion"):
        promote_deletion(bad, layout, ctx=ctx)


# ── flux ───────────────────────────────────────────────────────────────
def test_promote_flux_links_correctly(fixture: dict) -> None:
    layout, edit, ctx = fixture["layout"], fixture["mod_edit"], fixture["ctx"]
    res = promote_flux(edit, layout, ctx=ctx)
    assert res.ok, res.reason

    object_npz = layout.object_npz(SHARD, OBJ)
    before = layout.before_npz("modification", SHARD, OBJ, MOD_ID)
    after = layout.after_npz("modification", SHARD, OBJ, MOD_ID)
    assert _ino(before) == _ino(object_npz), "flux before.npz must hardlink to _assets/object.npz"
    assert _ino(after) == _ino(edit.edit_dir / "after.npz")
    for k in range(N_VIEWS):
        assert _ino(layout.after_view("modification", SHARD, OBJ, MOD_ID, k)) == \
            _ino(edit.edit_dir / f"preview_{k}.png")

    res2 = promote_flux(edit, layout, ctx=ctx)
    assert res2.ok


def test_promote_flux_rejects_wrong_type(fixture: dict) -> None:
    with pytest.raises(ValueError, match="promote_flux"):
        promote_flux(fixture["del_edit"], fixture["layout"], ctx=fixture["ctx"])


# ── addition ───────────────────────────────────────────────────────────
def test_promote_addition_requires_paired_deletion(fixture: dict) -> None:
    layout, ctx = fixture["layout"], fixture["ctx"]
    res = promote_addition(fixture["add_edit"], layout, ctx=ctx)
    assert res.ok is False
    assert "paired deletion" in (res.reason or "")


def test_promote_addition_succeeds_after_deletion(fixture: dict) -> None:
    layout, ctx = fixture["layout"], fixture["ctx"]
    assert promote_deletion(fixture["del_edit"], layout, ctx=ctx).ok
    res = promote_addition(fixture["add_edit"], layout, ctx=ctx)
    assert res.ok, res.reason
    assert res.manifest_record["lineage"]["paired_edit_id"] == DEL_ID

    object_npz = layout.object_npz(SHARD, OBJ)
    add_before = layout.before_npz("addition", SHARD, OBJ, ADD_ID)
    add_after = layout.after_npz("addition", SHARD, OBJ, ADD_ID)
    paired_after = layout.after_npz("deletion", SHARD, OBJ, DEL_ID)

    # add.before == deletion.after (the deletion mesh)
    assert _ino(add_before) == _ino(paired_after)
    # add.after == _assets/object.npz (the original obj)
    assert _ino(add_after) == _ino(object_npz)

    for k in range(N_VIEWS):
        assert _ino(layout.before_view("addition", SHARD, OBJ, ADD_ID, k)) == \
            _ino(layout.after_view("deletion", SHARD, OBJ, DEL_ID, k))
        assert _ino(layout.after_view("addition", SHARD, OBJ, ADD_ID, k)) == \
            _ino(layout.orig_view(SHARD, OBJ, k))


def test_promote_addition_idempotent(fixture: dict) -> None:
    layout, ctx = fixture["layout"], fixture["ctx"]
    promote_deletion(fixture["del_edit"], layout, ctx=ctx)
    res1 = promote_addition(fixture["add_edit"], layout, ctx=ctx)
    assert res1.ok
    res2 = promote_addition(fixture["add_edit"], layout, ctx=ctx)
    assert res2.ok
