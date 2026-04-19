"""Unit tests for ``partcraft.cleaning.h3d_v1.asset_pool``.

Builds a synthetic mini "pipeline obj dir" with a fake flux ``before.npz``
and a fake addition ``preview_*.png`` set, exercises the happy path of
both ``ensure_*`` functions, and verifies idempotency + lock release.

GPU encode_ss fallback is *not* tested here (would require a real
``ss_encoder`` instance); covered by Stage E real-shard smoke if it
fires there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from partcraft.cleaning.h3d_v1.asset_pool import (
    PREVIEW_SIDE,
    ensure_object_npz,
    ensure_object_views,
)
from partcraft.cleaning.h3d_v1.layout import H3DLayout, N_VIEWS


@pytest.fixture()
def layout(tmp_path: Path) -> H3DLayout:
    return H3DLayout(root=tmp_path / "H3D_v1")


@pytest.fixture()
def pipeline_obj_dir(tmp_path: Path) -> Path:
    """Synthetic ``objects/<NN>/<obj_id>/`` with one flux + one add edit."""
    obj = tmp_path / "pipeline" / "objects" / "08" / "obj"
    flux = obj / "edits_3d" / "mat_obj_000"
    add = obj / "edits_3d" / "add_obj_000"
    flux.mkdir(parents=True)
    add.mkdir(parents=True)

    feats = np.zeros((4, 8), dtype=np.float32)
    coords = np.zeros((4, 4), dtype=np.int32)
    ss = np.zeros((8, 16, 16, 16), dtype=np.float32)
    np.savez(flux / "before.npz", slat_feats=feats, slat_coords=coords, ss=ss)

    cv2 = pytest.importorskip("cv2")
    for k in range(N_VIEWS):
        img = np.full((PREVIEW_SIDE, PREVIEW_SIDE, 3), fill_value=k * 50, dtype=np.uint8)
        cv2.imwrite(str(add / f"preview_{k}.png"), img)
    return obj


def test_ensure_object_npz_copies_from_flux_before(layout: H3DLayout, pipeline_obj_dir: Path,
                                                   tmp_path: Path) -> None:
    out = ensure_object_npz(
        layout, "08", "obj",
        pipeline_obj_dir=pipeline_obj_dir,
        slat_dir=tmp_path / "slat_does_not_exist",
    )
    assert out == layout.object_npz("08", "obj")
    assert out.is_file()
    npz = np.load(out)
    assert set(npz.files) == {"slat_feats", "slat_coords", "ss"}
    assert npz["ss"].shape == (8, 16, 16, 16)


def test_ensure_object_npz_idempotent(layout: H3DLayout, pipeline_obj_dir: Path,
                                      tmp_path: Path) -> None:
    p1 = ensure_object_npz(layout, "08", "obj",
                           pipeline_obj_dir=pipeline_obj_dir,
                           slat_dir=tmp_path / "slat")
    mtime1 = p1.stat().st_mtime
    p2 = ensure_object_npz(layout, "08", "obj",
                           pipeline_obj_dir=pipeline_obj_dir,
                           slat_dir=tmp_path / "slat")
    assert p2 == p1
    assert p1.stat().st_mtime == mtime1


def test_ensure_object_npz_raises_without_source(layout: H3DLayout, tmp_path: Path) -> None:
    empty_obj = tmp_path / "empty_obj"
    (empty_obj / "edits_3d").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no flux before.npz"):
        ensure_object_npz(layout, "08", "obj",
                         pipeline_obj_dir=empty_obj,
                         slat_dir=tmp_path / "slat")


def test_ensure_object_views_from_addition(layout: H3DLayout, pipeline_obj_dir: Path) -> None:
    views_dir = ensure_object_views(layout, "08", "obj",
                                    pipeline_obj_dir=pipeline_obj_dir,
                                    image_npz=None)
    assert views_dir == layout.orig_views_dir("08", "obj")
    for k in range(N_VIEWS):
        p = layout.orig_view("08", "obj", k)
        assert p.is_file()
    cv2 = pytest.importorskip("cv2")
    img = cv2.imread(str(layout.orig_view("08", "obj", 0)))
    assert img.shape == (PREVIEW_SIDE, PREVIEW_SIDE, 3)


def test_ensure_object_views_idempotent(layout: H3DLayout, pipeline_obj_dir: Path) -> None:
    p1 = ensure_object_views(layout, "08", "obj",
                             pipeline_obj_dir=pipeline_obj_dir, image_npz=None)
    mtimes_1 = [layout.orig_view("08", "obj", k).stat().st_mtime for k in range(N_VIEWS)]
    p2 = ensure_object_views(layout, "08", "obj",
                             pipeline_obj_dir=pipeline_obj_dir, image_npz=None)
    assert p2 == p1
    mtimes_2 = [layout.orig_view("08", "obj", k).stat().st_mtime for k in range(N_VIEWS)]
    assert mtimes_1 == mtimes_2


def test_ensure_object_views_raises_without_source(layout: H3DLayout, tmp_path: Path) -> None:
    empty_obj = tmp_path / "empty_obj"
    (empty_obj / "edits_3d").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no add preview"):
        ensure_object_views(layout, "08", "obj",
                            pipeline_obj_dir=empty_obj, image_npz=None)


def test_lock_file_created_and_releasable(layout: H3DLayout, pipeline_obj_dir: Path,
                                          tmp_path: Path) -> None:
    """After ensure_*, the lock file exists and is no longer flock-held.

    Verified by acquiring an exclusive flock on it from a fresh fd in
    a non-blocking call (would fail with BlockingIOError if still held).
    """
    import fcntl
    ensure_object_npz(layout, "08", "obj",
                     pipeline_obj_dir=pipeline_obj_dir,
                     slat_dir=tmp_path / "slat")
    lock = layout.asset_lock("08", "obj")
    assert lock.is_file()
    with open(lock, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
