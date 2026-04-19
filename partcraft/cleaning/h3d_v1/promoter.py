"""Per-edit promote routines: hardlink the dataset bundle, write meta.json.

Three entrypoints, one per edit-type group, all returning a
``PromoteResult``. Each is idempotent — a second call on an already-
promoted edit is a no-op (existing hardlinks with matching inodes are
recognised and reused).

The promoter does **not** know about gates or pipeline configs; it
takes a fully-resolved ``PipelineEdit`` (from ``pipeline_io``) plus a
``PromoteContext`` (from the CLI) and physically writes the dataset.
Filtering happens upstream in the CLI via ``filter.accept_*``.

The promoter does **not** run s6b for deletion. Per spec §6.1 the
caller (``pull_deletion`` CLI) is responsible for materialising
``<pipeline_edit_dir>/after.npz`` before invoking ``promote_deletion``;
the promoter raises ``RuntimeError`` if the file is missing so the
caller can decide whether to encode lazily or skip.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from datetime import datetime, timezone

from partcraft.cleaning.h3d_v1 import asset_pool
from partcraft.cleaning.h3d_v1.instruction import load_instructions
from partcraft.cleaning.h3d_v1.layout import (
    EDIT_TYPES_FLUX,
    H3DLayout,
    N_VIEWS,
    paired_edit_id,
)
from partcraft.cleaning.h3d_v1.pipeline_io import PipelineEdit, load_edit_status

LOGGER = logging.getLogger(__name__)
META_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class PromoteResult:
    ok: bool
    reason: str | None = None
    manifest_record: dict[str, Any] | None = None


@dataclass
class PromoteContext:
    """Per-shard arguments shared across all promote_* calls."""

    pipeline_obj_root: Path  # outputs/.../objects/<NN>/
    slat_dir: Path
    images_root: Path  # data.images_root, used to derive image_npz per obj
    ss_encoder: Callable[[np.ndarray], np.ndarray] | None = None
    cross_fs_warned: set[str] = field(default_factory=set)
    instruction_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Lineage fields (schema v3). Defaults make the object usable from tests.
    pipeline_version: str = "v3"
    pipeline_config: str = ""           # repo-relative path, e.g. "configs/pipeline_v3_shard08.yaml"
    pipeline_git_sha: str = ""          # short SHA at promote time
    source_dataset: str = "partverse"   # upstream object source
    promoted_at: str = ""               # ISO-8601 UTC; auto-filled at first read if empty

    def image_npz_for(self, shard: str, obj_id: str) -> Path:
        return self.images_root / shard / f"{obj_id}.npz"

    def instructions_for(self, obj_dir: Path) -> dict[str, Any]:
        """Return cached ``{edit_id: instruction}`` map for an obj_dir."""
        key = str(obj_dir)
        cached = self.instruction_cache.get(key)
        if cached is None:
            cached = load_instructions(obj_dir)
            self.instruction_cache[key] = cached
        return cached

    def _now_iso(self) -> str:
        if not self.promoted_at:
            self.promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.promoted_at


# ── linking primitives ────────────────────────────────────────────────
def _same_inode(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_ino == b.stat().st_ino and a.stat().st_dev == b.stat().st_dev
    except OSError:
        return False


def _hardlink_or_copy(src: Path, dst: Path, *, ctx: PromoteContext | None = None) -> None:
    """Idempotently hardlink ``src`` → ``dst``; copy on cross-FS (EXDEV)."""
    if not src.is_file():
        raise FileNotFoundError(f"link source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if _same_inode(src, dst):
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        if ctx is not None and str(src.parent) not in ctx.cross_fs_warned:
            LOGGER.warning("cross-FS hardlink failed at %s; falling back to copy", src.parent)
            ctx.cross_fs_warned.add(str(src.parent))
        shutil.copy2(src, dst)


def _write_meta(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2))
    os.replace(tmp, path)


def _instruction_or_empty(edit: PipelineEdit, ctx: "PromoteContext") -> dict[str, Any]:
    """Look up the parsed-VLM instruction for this edit_id.

    Returns ``{}`` if parsed.json is missing or the edit_id has no
    matching parsed entry. This should not happen in practice but we
    tolerate it so promotion never fails for a metadata-only reason.
    """
    table = ctx.instructions_for(edit.obj_dir)
    instr = table.get(edit.edit_id)
    if instr is None:
        LOGGER.warning("no parsed instruction for %s", edit.edit_id)
        return {}
    return instr


def _quality_block(edit: PipelineEdit) -> dict[str, Any]:
    """Slim quality summary from edit_status.json (schema v3).

    Only carries: ``final_pass`` + per-gate VLM scores. Status fields
    are dropped (presence in dataset implies pass). Timestamps are
    captured at promote time in ``lineage.promoted_at``.
    """
    es = load_edit_status(edit.obj_dir)
    e = es.get("edits", {}).get(edit.edit_id, {}) or {}
    gates = e.get("gates", {}) or {}

    def _score(letter: str) -> float | None:
        g = gates.get(letter)
        vlm = g.get("vlm") if isinstance(g, dict) else None
        s = vlm.get("score") if isinstance(vlm, dict) else None
        return float(s) if isinstance(s, (int, float)) else None

    out: dict[str, Any] = {"final_pass": bool(e.get("final_pass"))}
    for letter in ("A", "E"):
        score = _score(letter)
        if score is not None:
            out[f"gate_{letter}_score"] = score
    return out


def _voxel_count(npz_path: Path) -> int | None:
    """Read ``slat_coords.shape[0]`` cheaply; tolerate IO errors."""
    try:
        with np.load(npz_path) as d:
            return int(d["slat_coords"].shape[0])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not read voxel count from %s: %s", npz_path, exc)
        return None


def _stats_block(before_npz: Path, after_npz: Path) -> dict[str, Any]:
    bn = _voxel_count(before_npz)
    an = _voxel_count(after_npz)
    out: dict[str, Any] = {}
    if bn is not None:
        out["before_n_voxels"] = bn
    if an is not None:
        out["after_n_voxels"] = an
    if bn is not None and an is not None:
        out["delta_voxels"] = an - bn
    return out


def _lineage_block(edit: PipelineEdit, ctx: "PromoteContext") -> dict[str, Any]:
    out: dict[str, Any] = {
        "pipeline_version": ctx.pipeline_version,
        "source_dataset": ctx.source_dataset,
        "promoted_at": ctx._now_iso(),
    }
    if ctx.pipeline_config:
        out["pipeline_config"] = ctx.pipeline_config
    if ctx.pipeline_git_sha:
        out["pipeline_git_sha"] = ctx.pipeline_git_sha
    pair = paired_edit_id(edit.edit_id)
    if pair is not None:
        out["paired_edit_id"] = pair
    return out


def _base_record(
    edit: PipelineEdit,
    ctx: "PromoteContext",
    *,
    before_npz: Path | None = None,
    after_npz: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "edit_id": edit.edit_id,
        "edit_type": edit.edit_type,
        "obj_id": edit.obj_id,
        "shard": edit.shard,
        "schema_version": META_SCHEMA_VERSION,
        "instruction": _instruction_or_empty(edit, ctx),
        "quality": _quality_block(edit),
    }
    if before_npz is not None and after_npz is not None:
        stats = _stats_block(before_npz, after_npz)
        if stats:
            record["stats"] = stats
    record["lineage"] = _lineage_block(edit, ctx)
    return record


# ── deletion ───────────────────────────────────────────────────────────
def promote_deletion(
    edit: PipelineEdit,
    layout: H3DLayout,
    *,
    ctx: PromoteContext,
) -> PromoteResult:
    """Promote one ``del_*`` edit. ``<edit.edit_dir>/after.npz`` must exist."""
    if edit.edit_type != "deletion":
        raise ValueError(f"promote_deletion called with edit_type={edit.edit_type!r}")

    pipeline_after = edit.edit_dir / "after.npz"
    if not pipeline_after.is_file():
        return PromoteResult(False, f"missing pipeline after.npz at {pipeline_after}")

    object_npz = asset_pool.ensure_object_npz(
        layout, edit.shard, edit.obj_id,
        pipeline_obj_dir=edit.obj_dir,
        slat_dir=ctx.slat_dir,
        ss_encoder=ctx.ss_encoder,
    )
    asset_pool.ensure_object_views(
        layout, edit.shard, edit.obj_id,
        pipeline_obj_dir=edit.obj_dir,
        image_npz=ctx.image_npz_for(edit.shard, edit.obj_id) if ctx.images_root else None,
    )

    before_dst = layout.before_npz("deletion", edit.shard, edit.obj_id, edit.edit_id)
    after_dst = layout.after_npz("deletion", edit.shard, edit.obj_id, edit.edit_id)
    _hardlink_or_copy(object_npz, before_dst, ctx=ctx)
    _hardlink_or_copy(pipeline_after, after_dst, ctx=ctx)
    for k in range(N_VIEWS):
        _hardlink_or_copy(
            layout.orig_view(edit.shard, edit.obj_id, k),
            layout.before_view("deletion", edit.shard, edit.obj_id, edit.edit_id, k),
            ctx=ctx,
        )
        _hardlink_or_copy(
            edit.edit_dir / f"preview_{k}.png",
            layout.after_view("deletion", edit.shard, edit.obj_id, edit.edit_id, k),
            ctx=ctx,
        )

    record = _base_record(edit, ctx, before_npz=object_npz, after_npz=pipeline_after)
    _write_meta(layout.meta_json("deletion", edit.shard, edit.obj_id, edit.edit_id), record)
    return PromoteResult(True, None, record)


# ── flux (modification | scale | material | color | global) ───────────
def promote_flux(
    edit: PipelineEdit,
    layout: H3DLayout,
    *,
    ctx: PromoteContext,
) -> PromoteResult:
    if edit.edit_type not in EDIT_TYPES_FLUX:
        raise ValueError(f"promote_flux called with edit_type={edit.edit_type!r}")

    pipeline_before = edit.edit_dir / "before.npz"
    pipeline_after = edit.edit_dir / "after.npz"
    if not pipeline_before.is_file():
        return PromoteResult(False, f"missing pipeline before.npz at {pipeline_before}")
    if not pipeline_after.is_file():
        return PromoteResult(False, f"missing pipeline after.npz at {pipeline_after}")

    object_npz = asset_pool.ensure_object_npz(
        layout, edit.shard, edit.obj_id,
        pipeline_obj_dir=edit.obj_dir,
        slat_dir=ctx.slat_dir,
        ss_encoder=ctx.ss_encoder,
    )
    asset_pool.ensure_object_views(
        layout, edit.shard, edit.obj_id,
        pipeline_obj_dir=edit.obj_dir,
        image_npz=ctx.image_npz_for(edit.shard, edit.obj_id) if ctx.images_root else None,
    )

    before_dst = layout.before_npz(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id)
    after_dst = layout.after_npz(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id)
    # Flux's before.npz is content-identical to object.npz; link from the pool
    # so all "before" copies in the dataset share one inode per obj.
    _hardlink_or_copy(object_npz, before_dst, ctx=ctx)
    _hardlink_or_copy(pipeline_after, after_dst, ctx=ctx)
    for k in range(N_VIEWS):
        _hardlink_or_copy(
            layout.orig_view(edit.shard, edit.obj_id, k),
            layout.before_view(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id, k),
            ctx=ctx,
        )
        _hardlink_or_copy(
            edit.edit_dir / f"preview_{k}.png",
            layout.after_view(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id, k),
            ctx=ctx,
        )

    record = _base_record(edit, ctx, before_npz=object_npz, after_npz=pipeline_after)
    _write_meta(layout.meta_json(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id), record)
    return PromoteResult(True, None, record)


# ── addition ───────────────────────────────────────────────────────────
def promote_addition(
    edit: PipelineEdit,
    layout: H3DLayout,
    *,
    ctx: PromoteContext,
) -> PromoteResult:
    """Promote one ``add_*`` edit. Paired deletion must already be in dataset."""
    if edit.edit_type != "addition":
        raise ValueError(f"promote_addition called with edit_type={edit.edit_type!r}")

    paired = paired_edit_id(edit.edit_id)
    if paired is None:
        return PromoteResult(False, "no paired deletion convention for this id")

    paired_after = layout.after_npz("deletion", edit.shard, edit.obj_id, paired)
    if not paired_after.is_file():
        return PromoteResult(False, f"paired deletion {paired} not promoted yet")
    object_npz = layout.object_npz(edit.shard, edit.obj_id)
    if not object_npz.is_file():
        return PromoteResult(False, f"_assets object.npz missing — promote a deletion or flux for {edit.obj_id} first")
    paired_after_views = [
        layout.after_view("deletion", edit.shard, edit.obj_id, paired, k) for k in range(N_VIEWS)
    ]
    if not all(p.is_file() for p in paired_after_views):
        return PromoteResult(False, f"paired deletion {paired} after_views incomplete")

    before_dst = layout.before_npz("addition", edit.shard, edit.obj_id, edit.edit_id)
    after_dst = layout.after_npz("addition", edit.shard, edit.obj_id, edit.edit_id)
    _hardlink_or_copy(paired_after, before_dst, ctx=ctx)
    _hardlink_or_copy(object_npz, after_dst, ctx=ctx)
    for k in range(N_VIEWS):
        _hardlink_or_copy(
            paired_after_views[k],
            layout.before_view("addition", edit.shard, edit.obj_id, edit.edit_id, k),
            ctx=ctx,
        )
        _hardlink_or_copy(
            layout.orig_view(edit.shard, edit.obj_id, k),
            layout.after_view("addition", edit.shard, edit.obj_id, edit.edit_id, k),
            ctx=ctx,
        )

    record = _base_record(edit, ctx, before_npz=paired_after, after_npz=object_npz)
    _write_meta(layout.meta_json("addition", edit.shard, edit.obj_id, edit.edit_id), record)
    return PromoteResult(True, None, record)


__all__ = [
    "META_SCHEMA_VERSION",
    "PromoteContext",
    "PromoteResult",
    "promote_addition",
    "promote_deletion",
    "promote_flux",
]
