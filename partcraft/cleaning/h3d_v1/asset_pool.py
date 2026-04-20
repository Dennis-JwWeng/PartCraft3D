"""Materialise the per-object ``_assets/`` pool for H3D_v1.

Each obj in the dataset has exactly one canonical bundle:

* ``_assets/<NN>/<obj_id>/object.npz``     — slat_feats + slat_coords + ss
* ``_assets/<NN>/<obj_id>/orig_views/view{0..4}.png``  — 5 fixed-view RGB

Every per-edit dir then **hardlinks** into this pool, so the pool is the
sole physical-storage point per obj. fcntl-locked at the obj_dir level
to make concurrent ``pull_deletion`` / ``pull_flux`` runs safe.

Source resolution (per spec §4):

``object.npz``:
  1. Copy from any flux edit's ``before.npz`` (already has ss).
  2. GPU fallback — load slat_dir tensor + run a user-supplied ``ss_encoder``.

``orig_views/view{i}.png``:
  Rendered from the original ``image_npz`` using ``VIEW_INDICES``,
  then resized to ``PREVIEW_SIDE`` × ``PREVIEW_SIDE`` so all
  per-edit ``before.png`` / ``after.png`` are dimensionally
  consistent across the dataset.

  Note: addition previews (``add_*/preview_*.png``) are NOT used here
  because pipeline_v3 s6p generates them as copies of the source
  deletion's after-image (part-removed state), NOT the original object.

Both ``ensure_*`` are idempotent: a second call on a fully-materialised
obj is a no-op (returns immediately after the inexpensive existence
check inside the lock).
"""
from __future__ import annotations

import contextlib
import fcntl
import logging
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from partcraft.cleaning.h3d_v1.layout import H3DLayout, N_VIEWS

LOGGER = logging.getLogger(__name__)

# Match s6p preview resolution so dataset images line up dimensionally.
PREVIEW_SIDE: int = 518


# ── locking ────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _obj_lock(layout: H3DLayout, shard: str, obj_id: str) -> Iterator[None]:
    """Hold an exclusive lock for ``_assets/<NN>/<obj_id>/.lock``."""
    layout.assets_obj_dir(shard, obj_id).mkdir(parents=True, exist_ok=True)
    lock_path = layout.asset_lock(shard, obj_id)
    with open(lock_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ── object.npz ─────────────────────────────────────────────────────────
def _find_flux_before_npz(pipeline_obj_dir: Path) -> Path | None:
    """Return any flux edit's ``before.npz`` under ``edits_3d/``, or None."""
    edits_root = pipeline_obj_dir / "edits_3d"
    if not edits_root.is_dir():
        return None
    for edit_dir in sorted(edits_root.iterdir()):
        if not edit_dir.is_dir():
            continue
        prefix = edit_dir.name.split("_", 1)[0]
        if prefix in ("del", "add"):
            continue
        candidate = edit_dir / "before.npz"
        if candidate.is_file():
            return candidate
    return None


def _find_slat_pt(slat_dir: Path, shard: str, obj_id: str) -> Path | None:
    """Locate the per-obj SLAT tensor — ``<slat_dir>/<NN>/<obj_id>_*.pt``."""
    shard_dir = slat_dir / shard
    if not shard_dir.is_dir():
        return None
    matches = sorted(shard_dir.glob(f"{obj_id}_*.pt"))
    return matches[0] if matches else None


def ensure_object_npz(
    layout: H3DLayout,
    shard: str,
    obj_id: str,
    *,
    pipeline_obj_dir: Path,
    slat_dir: Path,
    ss_encoder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Path:
    """Materialise ``_assets/<NN>/<obj_id>/object.npz``. Returns its path.

    Args:
        ss_encoder: Optional callable ``coords[N,4] -> ss[8,16,16,16]``.
            Only consulted if no flux ``before.npz`` is on disk for this
            obj. If both branches fail and ``ss_encoder`` is ``None``, a
            ``RuntimeError`` is raised — the caller should arrange to
            provide an encoder lazily.
    """
    out = layout.object_npz(shard, obj_id)
    with _obj_lock(layout, shard, obj_id):
        if out.is_file():
            return out

        out.parent.mkdir(parents=True, exist_ok=True)

        # Source 1: copy from any flux before.npz (has ss baked in).
        src = _find_flux_before_npz(pipeline_obj_dir)
        if src is not None:
            shutil.copy2(src, out)
            LOGGER.info("object.npz[%s/%s] copied from %s", shard, obj_id, src)
            return out

        # Source 2: GPU fallback via slat_dir + ss_encoder.
        slat_pt = _find_slat_pt(slat_dir, shard, obj_id)
        if slat_pt is None:
            raise RuntimeError(
                f"object.npz[{shard}/{obj_id}]: no flux before.npz and no slat tensor at "
                f"{slat_dir}/{shard}/{obj_id}_*.pt"
            )
        if ss_encoder is None:
            raise RuntimeError(
                f"object.npz[{shard}/{obj_id}]: ss_encoder required for fallback but none provided"
            )
        _encode_object_npz_via_slat(slat_pt, ss_encoder, out)
        LOGGER.info("object.npz[%s/%s] encoded from %s", shard, obj_id, slat_pt)
        return out


def _encode_object_npz_via_slat(
    slat_pt: Path,
    ss_encoder: Callable[[np.ndarray], np.ndarray],
    out: Path,
) -> None:
    """Load ``slat_pt``, encode ss, write ``object.npz`` with the 3 keys.

    Lazy-imports torch only on this path so CPU-only callers stay light.
    """
    import torch  # noqa: PLC0415

    blob: Any = torch.load(slat_pt, map_location="cpu", weights_only=False)
    feats = np.asarray(blob["feats"], dtype=np.float32) if isinstance(blob, dict) else None
    coords = np.asarray(blob["coords"], dtype=np.int32) if isinstance(blob, dict) else None
    if feats is None or coords is None:
        raise RuntimeError(f"slat tensor {slat_pt} missing 'feats'/'coords' keys")

    ss = np.asarray(ss_encoder(coords), dtype=np.float32)
    if ss.shape != (8, 16, 16, 16):
        raise RuntimeError(f"ss_encoder returned bad shape {ss.shape}; expected (8,16,16,16)")

    np.savez(out, slat_feats=feats, slat_coords=coords, ss=ss)


# ── orig_views/ ────────────────────────────────────────────────────────
def ensure_object_views(
    layout: H3DLayout,
    shard: str,
    obj_id: str,
    *,
    pipeline_obj_dir: Path,
    image_npz: Path | None,
) -> Path:
    """Materialise ``_assets/<NN>/<obj_id>/orig_views/view{0..4}.png``.

    Always rendered from the original ``image_npz`` (raw sensor input)
    using ``VIEW_INDICES``.  ``pipeline_obj_dir`` is accepted for API
    compatibility but is not used.

    Returns the orig_views directory.
    """
    views_dir = layout.orig_views_dir(shard, obj_id)
    with _obj_lock(layout, shard, obj_id):
        if all(layout.orig_view(shard, obj_id, k).is_file() for k in range(N_VIEWS)):
            return views_dir

        views_dir.mkdir(parents=True, exist_ok=True)

        if image_npz is None or not image_npz.is_file():
            raise RuntimeError(
                f"orig_views[{shard}/{obj_id}]: image_npz unavailable ({image_npz}); "
                f"cannot materialise original views"
            )
        _materialise_views_from_image_npz(image_npz, views_dir)
        LOGGER.info("orig_views[%s/%s] rendered from %s", shard, obj_id, image_npz)
        return views_dir


def _materialise_views_from_image_npz(image_npz: Path, views_dir: Path) -> None:
    """Load 5 raw views from image_npz, resize to PREVIEW_SIDE, write PNGs."""
    import cv2  # noqa: PLC0415

    from partcraft.render.overview import VIEW_INDICES, load_views_from_npz  # noqa: PLC0415

    if len(VIEW_INDICES) != N_VIEWS:
        raise RuntimeError(
            f"VIEW_INDICES has {len(VIEW_INDICES)} entries but N_VIEWS={N_VIEWS}"
        )
    imgs, _frames = load_views_from_npz(image_npz, VIEW_INDICES)
    for k, img in enumerate(imgs):
        if img.shape[0] != PREVIEW_SIDE or img.shape[1] != PREVIEW_SIDE:
            img = cv2.resize(img, (PREVIEW_SIDE, PREVIEW_SIDE), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(views_dir / f"view{k}.png"), img)


__all__ = [
    "PREVIEW_SIDE",
    "ensure_object_npz",
    "ensure_object_views",
]
