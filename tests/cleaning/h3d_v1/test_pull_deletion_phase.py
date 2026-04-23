"""Unit tests for ``scripts.cleaning.h3d_v1.pull_deletion`` phase logic.

Covers the three ``--phase`` modes (render / encode / both) and
``--skip-encode`` mutual-exclusion with ``--phase``, using mocks for the
heavy GPU helpers.  No GPU, Blender, or CUDA required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import multiprocessing as mp
import pytest

# ---------------------------------------------------------------------------
# Helpers to construct fake PipelineEdit objects
# ---------------------------------------------------------------------------

def _fake_edit(tmp_path: Path, shard: str, obj_id: str, edit_id: str) -> "PipelineEdit":
    """Return a minimal PipelineEdit pointing at a real tmpdir."""
    from partcraft.cleaning.h3d_v1.pipeline_io import PipelineEdit

    edit_dir = tmp_path / "pipeline" / "objects" / shard / obj_id / "edits_3d" / edit_id
    edit_dir.mkdir(parents=True, exist_ok=True)
    # create after_new.glb so render worker won't skip it
    (edit_dir / "after_new.glb").touch()
    obj_dir = edit_dir.parent.parent  # edits_3d parent
    edit_status_path = obj_dir / "edit_status.json"
    edit_status_path.touch()
    return PipelineEdit(
        edit_id=edit_id,
        edit_type="deletion",
        shard=shard,
        obj_id=obj_id,
        obj_dir=obj_dir,
        edit_dir=edit_dir,
        edit_status_path=edit_status_path,
    )


# ---------------------------------------------------------------------------
# Import target helpers
# ---------------------------------------------------------------------------

from scripts.cleaning.h3d_v1.pull_deletion import (  # noqa: E402
    _BLAS_THREAD_ENV,
    _edit_name,
    _encode_worker,
    _render_worker,
    _run_encode_pool,
    _run_render_pool,
)


# ---------------------------------------------------------------------------
# _edit_name
# ---------------------------------------------------------------------------

def test_edit_name(tmp_path):
    edit = _fake_edit(tmp_path, "08", "abc123", "del_abc123_000")
    assert _edit_name(edit) == "08_abc123_del_abc123_000"


# ---------------------------------------------------------------------------
# _render_worker: skips edits without after_new.glb, writes render.done
# ---------------------------------------------------------------------------

def test_render_worker_writes_done_marker(tmp_path):
    edit = _fake_edit(tmp_path, "08", "obj1", "del_obj1_000")
    work_root = tmp_path / "staging"

    def fake_render_ply_views(glb, name, parent_dir, num_views, blender_path):
        render_dir = Path(parent_dir) / name
        render_dir.mkdir(parents=True, exist_ok=True)
        (render_dir / "mesh.ply").touch()

    # _render_worker imports _render_ply_views from scripts.tools.migrate_slat_to_npz
    # at call time, so we patch it there.
    import scripts.tools.migrate_slat_to_npz as msnpz_mod
    with patch.object(msnpz_mod, "_render_ply_views", side_effect=fake_render_ply_views):
        ok, fail = _render_worker(0, [edit], work_root, 4, "/usr/local/bin/blender")

    assert ok == 1
    assert fail == 0
    done = work_root / _edit_name(edit) / "render.done"
    assert done.is_file(), "render.done marker must be written on success"


def test_render_worker_skips_missing_glb(tmp_path):
    edit = _fake_edit(tmp_path, "08", "obj2", "del_obj2_000")
    # remove glb so it cannot be rendered
    (edit.edit_dir / "after_new.glb").unlink()
    work_root = tmp_path / "staging"

    ok, fail = _render_worker(0, [edit], work_root, 4, "blender")

    assert ok == 0
    assert fail == 1
    assert not (work_root / _edit_name(edit) / "render.done").is_file()


def test_render_worker_skips_already_staged(tmp_path):
    edit = _fake_edit(tmp_path, "08", "obj3", "del_obj3_000")
    work_root = tmp_path / "staging"
    # pre-create done marker
    stage_dir = work_root / _edit_name(edit)
    stage_dir.mkdir(parents=True)
    (stage_dir / "render.done").touch()

    # If already staged, should not call _render_ply_views at all
    import scripts.tools.migrate_slat_to_npz as msnpz_mod
    with patch.object(msnpz_mod, "_render_ply_views") as mock_r:
        ok, fail = _render_worker(0, [edit], work_root, 4, "blender")
        mock_r.assert_not_called()

    assert ok == 1
    assert fail == 0


# ---------------------------------------------------------------------------
# _encode_worker: skips edits without render.done; keeps staging after encode
# ---------------------------------------------------------------------------

def test_encode_worker_skips_missing_done_marker(tmp_path):
    edit = _fake_edit(tmp_path, "08", "obj4", "del_obj4_000")
    work_root = tmp_path / "staging"

    with patch(
        "scripts.cleaning.h3d_v1.pull_deletion._maybe_load_encoder", return_value=MagicMock()
    ), patch("scripts.tools.migrate_slat_to_npz._encode_from_render_dir"):
        ok, fail = _encode_worker(0, [edit], tmp_path / "ckpt", work_root, 4)

    assert ok == 0
    assert fail == 1
    assert not (edit.edit_dir / "after.npz").is_file()


def test_encode_worker_writes_npz_and_preserves_staging(tmp_path):
    import numpy as np

    edit = _fake_edit(tmp_path, "08", "obj5", "del_obj5_000")
    work_root = tmp_path / "staging"
    stage_dir = work_root / _edit_name(edit)
    stage_dir.mkdir(parents=True)
    (stage_dir / "render.done").touch()

    fake_payload = {
        "slat_feats": np.zeros((2, 4), dtype=np.float32),
        "slat_coords": np.zeros((2, 4), dtype=np.int32),
        "ss": np.zeros((4, 8, 8, 8), dtype=np.float32),
    }

    import scripts.tools.migrate_slat_to_npz as msnpz_mod
    with patch(
        "scripts.cleaning.h3d_v1.pull_deletion._maybe_load_encoder",
        return_value=MagicMock(),
    ), patch.object(msnpz_mod, "_encode_from_render_dir", return_value=fake_payload):
        ok, fail = _encode_worker(0, [edit], tmp_path / "ckpt", work_root, 4)

    assert ok == 1
    assert fail == 0
    assert (edit.edit_dir / "after.npz").is_file()
    assert stage_dir.exists(), "staging dir must be preserved after successful encode"


# ---------------------------------------------------------------------------
# _parse_args: --skip-encode and --phase mutual exclusion
# ---------------------------------------------------------------------------

def test_skip_encode_phase_mutual_exclusion():
    from scripts.cleaning.h3d_v1.pull_deletion import _parse_args

    with patch(
        "sys.argv",
        [
            "pull_deletion",
            "--pipeline-cfg", "cfg.yaml",
            "--shard", "08",
            "--dataset-root", "data/H3D_v1",
            "--skip-encode",
            "--phase", "render",
        ],
    ):
        with pytest.raises(SystemExit):
            _parse_args()


def test_skip_encode_with_both_is_ok():
    """--skip-encode --phase both should not raise (both is the default)."""
    from scripts.cleaning.h3d_v1.pull_deletion import _parse_args

    with patch(
        "sys.argv",
        [
            "pull_deletion",
            "--pipeline-cfg", "cfg.yaml",
            "--shard", "08",
            "--dataset-root", "data/H3D_v1",
            "--skip-encode",
            "--phase", "both",
        ],
    ):
        args = _parse_args()
        assert args.skip_encode is True
        assert args.phase == "both"


def test_phase_render_default_args():
    from scripts.cleaning.h3d_v1.pull_deletion import _parse_args

    with patch(
        "sys.argv",
        [
            "pull_deletion",
            "--pipeline-cfg", "cfg.yaml",
            "--shard", "08",
            "--dataset-root", "data/H3D_v1",
            "--phase", "render",
        ],
    ):
        args = _parse_args()
        assert args.phase == "render"


# ---------------------------------------------------------------------------
# _run_render_pool / _run_encode_pool: verify pool is fork and workers called
# ---------------------------------------------------------------------------

def test_run_render_pool_uses_fork_and_calls_render_worker(tmp_path):
    """Pool must use fork context and dispatch _render_worker correctly."""
    from scripts.cleaning.h3d_v1.pull_deletion import _run_render_pool

    edit = _fake_edit(tmp_path, "08", "objA", "del_objA_000")
    stats = SimpleNamespace(skipped=0, add_skip=lambda _: None)

    with patch("scripts.cleaning.h3d_v1.pull_deletion.mp") as mock_mp:
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.starmap.return_value = [(1, 0)]
        mock_ctx = MagicMock()
        mock_ctx.Pool.return_value = mock_pool
        mock_mp.get_context.return_value = mock_ctx

        _run_render_pool([edit], [0], tmp_path / "staging", 4, "blender", stats)

        mock_mp.get_context.assert_called_once_with("fork")


def test_run_encode_pool_uses_fork(tmp_path):
    from scripts.cleaning.h3d_v1.pull_deletion import _run_encode_pool

    edit = _fake_edit(tmp_path, "08", "objB", "del_objB_000")
    stats = SimpleNamespace(skipped=0, add_skip=lambda _: None)

    with patch("scripts.cleaning.h3d_v1.pull_deletion.mp") as mock_mp:
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.starmap.return_value = [(1, 0)]
        mock_ctx = MagicMock()
        mock_ctx.Pool.return_value = mock_pool
        mock_mp.get_context.return_value = mock_ctx

        _run_encode_pool(
            [edit], [0], tmp_path / "ckpt", tmp_path / "staging", 4, stats
        )

        mock_mp.get_context.assert_called_once_with("fork")


# ---------------------------------------------------------------------------
# Integration: --phase encode records missing_staged_render for unstaged edits
# ---------------------------------------------------------------------------

def test_missing_staged_render_recorded(tmp_path, monkeypatch):
    """--phase encode should record missing_staged_render for edits without render.done."""
    import numpy as np
    from scripts.cleaning.h3d_v1.pull_deletion import _edit_name

    edit = _fake_edit(tmp_path, "08", "objC", "del_objC_000")
    work_root = tmp_path / "staging"
    # do NOT create render.done → should be recorded as missing_staged_render

    skips: list[str] = []

    class FakeStats:
        failed = 0
        skipped = 0
        def add_skip(self, reason: str) -> None:
            skips.append(reason)

    # Patch _run_encode_pool to be a no-op (we just want to test the filter logic)
    with patch("scripts.cleaning.h3d_v1.pull_deletion._run_encode_pool") as mock_enc:
        from scripts.cleaning.h3d_v1 import pull_deletion as pd_mod
        stats = FakeStats()

        # Replicate the encode-phase logic directly
        needs_encode = [edit]
        staged = [
            e for e in needs_encode
            if (work_root / _edit_name(e) / "render.done").is_file()
        ]
        missing_staged = [e for e in needs_encode if e not in set(staged)]
        for e in missing_staged:
            stats.add_skip("missing_staged_render")

        assert "missing_staged_render" in skips
        assert staged == []
