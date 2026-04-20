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


def _make_image_npz(path: Path) -> None:
    """Create a minimal valid ``images.npz`` covering VIEW_INDICES.

    ``asset_pool.ensure_object_views`` calls ``load_views_from_npz`` which
    requires each listed frame index as ``{idx:03d}.png`` plus a
    ``transforms.json`` blob with matching ``file_path`` entries.
    """
    import json as _json
    cv2 = pytest.importorskip("cv2")
    from partcraft.render.overview import VIEW_INDICES  # noqa: PLC0415
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, np.ndarray] = {}
    frames = []
    for k, idx in enumerate(VIEW_INDICES):
        img = np.full((64, 64, 3), fill_value=(k + 1) * 30, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        entries[f"{idx:03d}.png"] = np.frombuffer(buf.tobytes(), dtype=np.uint8)
        frames.append({
            "file_path": f"{idx:03d}.png",
            "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "camera_angle_x": 0.5,
        })
    tf_blob = _json.dumps({"frames": frames}).encode()
    entries["transforms.json"] = np.frombuffer(tf_blob, dtype=np.uint8)
    np.savez(path, **entries)


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

    # gates.A.vlm.best_view drives meta.json["views"].best_view_index.
    # Use distinct values per edit so the test catches mis-wiring.
    edit_status = {
        "obj_id": OBJ, "shard": SHARD, "schema_version": 1,
        "edits": {
            DEL_ID: {"edit_type": "deletion",
                     "stages": {"gate_a": {"status": "pass"}, "gate_e": {"status": "pass"}},
                     "gates": {"A": {"vlm": {"pass": True, "score": 1.0, "best_view": 2}},
                                "C": None,
                                "E": {"vlm": {"pass": True, "score": 0.8}}},
                     "final_pass": True},
            ADD_ID: {"edit_type": "addition",
                     "stages": {"gate_e": {"status": "pass"}},
                     "gates": {"A": None, "C": None,
                                "E": {"vlm": {"pass": True, "score": 0.9}}},
                     "final_pass": True},
            MOD_ID: {"edit_type": "modification",
                     "stages": {"gate_a": {"status": "pass"}, "gate_e": {"status": "pass"}},
                     "gates": {"A": {"vlm": {"pass": True, "score": 0.95, "best_view": 3}},
                                "C": None,
                                "E": {"vlm": {"pass": True, "score": 0.7}}},
                     "final_pass": True},
        },
    }
    (obj_dir / "edit_status.json").write_text(json.dumps(edit_status))

    layout = H3DLayout(root=tmp_path / "H3D_v1")
    images_root = tmp_path / "images"
    _make_image_npz(images_root / SHARD / f"{OBJ}.npz")
    ctx = PromoteContext(
        pipeline_obj_root=pipeline_root / "objects" / SHARD,
        slat_dir=tmp_path / "slat_unused",
        images_root=images_root,
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
    # Flat schema-v3: single before.png/after.png at best_view_index=2 (from fixture).
    k = 2  # gates.A.vlm.best_view set in fixture
    bv = layout.before_image("deletion", SHARD, OBJ, DEL_ID)
    av = layout.after_image("deletion", SHARD, OBJ, DEL_ID)
    assert bv.is_file() and av.is_file()
    assert _ino(bv) == _ino(layout.orig_view(SHARD, OBJ, k))
    assert _ino(av) == _ino(edit.edit_dir / f"preview_{k}.png")

    meta = json.loads(layout.meta_json("deletion", SHARD, OBJ, DEL_ID).read_text())
    assert meta["edit_id"] == DEL_ID
    # quality: final_pass + gate-sourced semantic scores (new names)
    assert meta["quality"]["final_pass"] is True
    assert meta["quality"]["alignment_score"] == 1.0
    assert meta["quality"]["quality_score"] == 0.8
    # views: populated from gates.A.vlm.best_view in edit_status
    assert meta["views"]["best_view_index"] == 2
    # lineage: slim (no promoted_at / pipeline_config / pipeline_git_sha / paired_edit_id)
    assert meta["lineage"] == {"pipeline_version": "v3", "source_dataset": "partverse"}
    # stats block dropped from schema v3
    assert "stats" not in meta
    # paired_edit_id no longer persisted in meta; downstream derives from edit_id convention.
    assert "paired_edit_id" not in meta
    assert "paired_edit_id" not in meta["lineage"]


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
    # Flat schema-v3: best_view_index=3 for mod_edit (from fixture).
    k = 3
    assert _ino(layout.before_image("modification", SHARD, OBJ, MOD_ID)) == \
        _ino(layout.orig_view(SHARD, OBJ, k))
    assert _ino(layout.after_image("modification", SHARD, OBJ, MOD_ID)) == \
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
    # addition should mirror paired deletion's best_view + alignment_score
    add_meta = json.loads(layout.meta_json("addition", SHARD, OBJ, ADD_ID).read_text())
    assert add_meta["views"]["best_view_index"] == 2  # mirrored from DEL_ID
    assert add_meta["quality"]["alignment_score"] == 1.0  # mirrored from DEL_ID gate A
    assert add_meta["quality"]["quality_score"] == 0.9
    # lineage still slim
    assert add_meta["lineage"] == {"pipeline_version": "v3", "source_dataset": "partverse"}

    object_npz = layout.object_npz(SHARD, OBJ)
    add_before = layout.before_npz("addition", SHARD, OBJ, ADD_ID)
    add_after = layout.after_npz("addition", SHARD, OBJ, ADD_ID)
    paired_after = layout.after_npz("deletion", SHARD, OBJ, DEL_ID)

    # add.before == deletion.after (the deletion mesh)
    assert _ino(add_before) == _ino(paired_after)
    # add.after == _assets/object.npz (the original obj)
    assert _ino(add_after) == _ino(object_npz)

    # Flat schema-v3: addition mirrors paired deletion's K (= 2 here).
    k = 2
    assert _ino(layout.before_image("addition", SHARD, OBJ, ADD_ID)) == \
        _ino(layout.after_image("deletion", SHARD, OBJ, DEL_ID))
    assert _ino(layout.after_image("addition", SHARD, OBJ, ADD_ID)) == \
        _ino(layout.orig_view(SHARD, OBJ, k))


def test_promote_addition_idempotent(fixture: dict) -> None:
    layout, ctx = fixture["layout"], fixture["ctx"]
    promote_deletion(fixture["del_edit"], layout, ctx=ctx)
    res1 = promote_addition(fixture["add_edit"], layout, ctx=ctx)
    assert res1.ok
    res2 = promote_addition(fixture["add_edit"], layout, ctx=ctx)
    assert res2.ok

# ── views block / promote_log ─────────────────────────────────────────
from partcraft.cleaning.h3d_v1.promoter import (  # noqa: E402
    DEFAULT_FRONT_VIEW_INDEX,
    _views_block,
)


def test_views_block_deletion_reads_best_view(fixture: dict) -> None:
    meta_views = _views_block(fixture["del_edit"])
    assert meta_views == {"best_view_index": 2}


def test_views_block_flux_reads_best_view(fixture: dict) -> None:
    meta_views = _views_block(fixture["mod_edit"])
    assert meta_views == {"best_view_index": 3}


def test_views_block_addition_mirrors_paired_deletion(fixture: dict) -> None:
    # paired del has best_view=2; addition's gate A is null, so _views_block
    # must fall back to the paired deletion's best_view in the same file.
    meta_views = _views_block(fixture["add_edit"])
    assert meta_views == {"best_view_index": 2}


def test_views_block_global_uses_default_front(tmp_path: Path, fixture: dict) -> None:
    from partcraft.cleaning.h3d_v1.pipeline_io import PipelineEdit
    glb_id = f"glb_{OBJ}_000"
    es_path = fixture["obj_dir"] / "edit_status.json"
    data = json.loads(es_path.read_text())
    # Add a global edit with a non-default best_view; _views_block should
    # ignore it and force DEFAULT_FRONT_VIEW_INDEX for edit_type=="global".
    data["edits"][glb_id] = {
        "edit_type": "global",
        "gates": {"A": {"vlm": {"pass": True, "score": 1.0, "best_view": 1}},
                   "C": None, "E": {"vlm": {"pass": True, "score": 1.0}}},
        "final_pass": True,
    }
    es_path.write_text(json.dumps(data))
    edit = PipelineEdit(
        obj_id=OBJ, shard=SHARD, edit_id=glb_id, edit_type="global",
        obj_dir=fixture["obj_dir"], edit_dir=fixture["obj_dir"] / "edits_3d" / glb_id,
        edit_status_path=es_path,
    )
    meta_views = _views_block(edit)
    assert meta_views == {"best_view_index": DEFAULT_FRONT_VIEW_INDEX}


def test_views_block_fallback_when_status_missing(tmp_path: Path) -> None:
    """Missing edit_status → DEFAULT_FRONT_VIEW_INDEX."""
    from partcraft.cleaning.h3d_v1.pipeline_io import PipelineEdit
    obj_dir = tmp_path / "no_status"
    obj_dir.mkdir()
    edit = PipelineEdit(
        obj_id=OBJ, shard=SHARD, edit_id=f"mod_{OBJ}_999", edit_type="modification",
        obj_dir=obj_dir, edit_dir=obj_dir / "edits_3d" / f"mod_{OBJ}_999",
        edit_status_path=obj_dir / "edit_status.json",
    )
    meta_views = _views_block(edit)
    assert meta_views == {"best_view_index": DEFAULT_FRONT_VIEW_INDEX}


def test_promote_log_is_appended_and_not_in_meta(fixture: dict) -> None:
    """Promote-time metadata goes to manifests/_internal/promote_log.jsonl,
    never into per-edit meta.json."""
    layout, edit, ctx = fixture["layout"], fixture["del_edit"], fixture["ctx"]
    ctx.pipeline_config = "configs/pipeline_v3_shard08.yaml"
    ctx.pipeline_git_sha = "abc1234"
    assert promote_deletion(edit, layout, ctx=ctx).ok

    log_path = layout.promote_log()
    assert log_path.is_file(), "promote_log.jsonl must exist after a promote"
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["edit_id"] == DEL_ID
    assert entry["pipeline_config"] == "configs/pipeline_v3_shard08.yaml"
    assert entry["pipeline_git_sha"] == "abc1234"
    assert entry["promoted_at"].endswith("Z")

    # And none of these leak into meta.json:
    meta = json.loads(layout.meta_json("deletion", SHARD, OBJ, DEL_ID).read_text())
    for leaked in ("promoted_at", "pipeline_config", "pipeline_git_sha"):
        assert leaked not in meta
        assert leaked not in meta.get("lineage", {})
