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

# Fallback view index when ``edit_status.json`` has no usable
# ``gates.A.vlm.best_view`` (missing/stale) or for rare edit types that
# never record one.  ``view4`` = VIEW_INDICES[4] = 8 = front upward
# (yaw +22°, pitch +52°).  Prefer pipeline-recorded ``best_view`` when
# available — especially for ``global`` edits, where forcing this index
# historically picked an upward-looking camera users disliked.
# See ``partcraft/render/overview.py::VIEW_INDICES``.
DEFAULT_FRONT_VIEW_INDEX: int = 4


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


def _append_promote_log(
    layout: H3DLayout, edit: PipelineEdit, ctx: "PromoteContext",
) -> None:
    """Append one promote record to ``manifests/_internal/promote_log.jsonl``.

    Captures repro metadata that we intentionally keep out of the
    released ``meta.json`` (timestamp, git sha, pipeline config path).
    Non-fatal on error — this is an audit trail, not a correctness gate.
    ``pack_shard`` excludes ``manifests/_internal/`` so this file is never
    shipped.
    """
    try:
        log_path = layout.promote_log()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "edit_id":          edit.edit_id,
            "edit_type":        edit.edit_type,
            "obj_id":           edit.obj_id,
            "shard":            edit.shard,
            "promoted_at":      ctx._now_iso(),
            "pipeline_version": ctx.pipeline_version,
            "pipeline_config":  ctx.pipeline_config,
            "pipeline_git_sha": ctx.pipeline_git_sha,
            "source_dataset":   ctx.source_dataset,
        }
        # Single O_APPEND write ≤ 4 KiB is atomic on POSIX; safe across
        # concurrent promote_* callers on the same log file.
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("failed to append promote_log for %s: %s", edit.edit_id, exc)


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

    Only carries: ``final_pass`` + semantic scores:
    ``alignment_score`` (from gate A — text/image alignment) and
    ``quality_score`` (from gate E — 3D render quality). Status fields
    are dropped (presence in dataset implies pass).  Promote-time
    metadata (timestamp, git sha, config path) is kept out of per-edit
    meta.json and written to ``manifests/_internal/promote_log.jsonl``.
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
    # Gate A → alignment (text-image semantic alignment).
    # For addition edits, gate A is null in edit_status (synthesised from
    # paired deletion), so we mirror the paired del's gate A score so the
    # per-edit meta.json still carries a meaningful alignment value.
    a_score = _score("A")
    if a_score is None and edit.edit_type == "addition":
        paired = paired_edit_id(edit.edit_id)
        if paired is not None:
            pe = es.get("edits", {}).get(paired, {}) or {}
            pg = (pe.get("gates") or {}).get("A")
            pv = pg.get("vlm") if isinstance(pg, dict) else None
            ps = pv.get("score") if isinstance(pv, dict) else None
            if isinstance(ps, (int, float)):
                a_score = float(ps)
    if a_score is not None:
        out["alignment_score"] = a_score
    # Gate E → quality (3D render visual quality).
    e_score = _score("E")
    if e_score is not None:
        out["quality_score"] = e_score
    return out


def _lineage_block(edit: PipelineEdit, ctx: "PromoteContext") -> dict[str, Any]:
    """Minimal lineage for release: only ``source_dataset`` and
    ``pipeline_version``.  Everything else (config path, git sha,
    promote timestamp, pairing) lives in ``manifests/_internal/`` or
    can be derived (paired del↔add from the edit_id convention).
    """
    return {
        "pipeline_version": ctx.pipeline_version,
        "source_dataset":   ctx.source_dataset,
    }


# ── views block ────────────────────────────────────────────────────────
def _read_best_view_from_status(
    es: dict[str, Any], edit_id: str,
) -> int | None:
    e = es.get("edits", {}).get(edit_id, {}) or {}
    gate_a = (e.get("gates") or {}).get("A")
    vlm = gate_a.get("vlm") if isinstance(gate_a, dict) else None
    bv = vlm.get("best_view") if isinstance(vlm, dict) else None
    if isinstance(bv, int) and 0 <= bv < N_VIEWS:
        return int(bv)
    return None


def _views_block(edit: PipelineEdit) -> dict[str, Any]:
    """Return ``{"best_view_index": int}``.

    Index is in 0..N_VIEWS-1.  In the flat schema-v3 layout each edit
    only ships a single ``before.png`` / ``after.png`` picked at this
    index (hardlinked from ``_assets/<obj>/orig_views/view{k}.png`` and
    the pipeline's ``preview_{k}.png`` respectively).

    Resolution order:

    1. ``global`` edits: read ``gates.A.vlm.best_view`` from
       ``edit_status.json`` when present (same field other flux types
       use).  Whole-scene edits still get a VLM-picked canonical view;
       only fall back to ``DEFAULT_FRONT_VIEW_INDEX`` when it is absent.
    2. Other non-addition edits: read pipeline_v3 pixel-mask argmax from
       ``edit_status.json.edits[edit_id].gates.A.vlm.best_view``.
    3. ``addition`` edits: gate A is null in edit_status (synthesised
       from paired deletion), so mirror the paired deletion's best_view.
    4. If none of the above yield a valid index (stale status, missing
       edit entry, etc.), fall back to ``DEFAULT_FRONT_VIEW_INDEX``.
    """
    es = load_edit_status(edit.obj_dir)

    if edit.edit_type == "global":
        bv = _read_best_view_from_status(es, edit.edit_id)
        if bv is None:
            bv = DEFAULT_FRONT_VIEW_INDEX
        return {"best_view_index": int(bv)}

    bv = _read_best_view_from_status(es, edit.edit_id)
    if bv is None and edit.edit_type == "addition":
        paired = paired_edit_id(edit.edit_id)
        if paired is not None:
            bv = _read_best_view_from_status(es, paired)
    if bv is None:
        bv = DEFAULT_FRONT_VIEW_INDEX
    return {"best_view_index": int(bv)}


def _base_record(
    edit: PipelineEdit,
    ctx: "PromoteContext",
    *,
    before_npz: Path | None = None,
    after_npz: Path | None = None,
) -> dict[str, Any]:
    """Build the released ``meta.json`` record.

    Schema v3 blocks:

    * ``edit_id`` / ``edit_type`` / ``obj_id`` / ``shard`` / ``schema_version``
    * ``instruction`` — prompt + descs + edit_params (no part bookkeeping)
    * ``views.best_view_index`` — single int 0..4 matching ``view{k}.png``
    * ``quality``  — ``final_pass`` + ``alignment_score`` + ``quality_score``
    * ``lineage``  — ``source_dataset`` + ``pipeline_version``
    """
    # before_npz / after_npz kwargs retained for backwards compatibility
    # with existing callers; stats block has been removed from schema v3.
    del before_npz, after_npz
    return {
        "edit_id":        edit.edit_id,
        "edit_type":      edit.edit_type,
        "obj_id":         edit.obj_id,
        "shard":          edit.shard,
        "schema_version": META_SCHEMA_VERSION,
        "instruction":    _instruction_or_empty(edit, ctx),
        "views":          _views_block(edit),
        "quality":        _quality_block(edit),
        "lineage":        _lineage_block(edit, ctx),
    }


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

    # Flat schema-v3 layout: one before.png + one after.png per edit, picked
    # at best_view_index.  Compute the record first so the K used for linking
    # is the exact same K that downstream reads from meta.json.
    record = _base_record(edit, ctx, before_npz=object_npz, after_npz=pipeline_after)
    k = int(record["views"]["best_view_index"])
    _hardlink_or_copy(
        layout.orig_view(edit.shard, edit.obj_id, k),
        layout.before_image("deletion", edit.shard, edit.obj_id, edit.edit_id),
        ctx=ctx,
    )
    _hardlink_or_copy(
        edit.edit_dir / f"preview_{k}.png",
        layout.after_image("deletion", edit.shard, edit.obj_id, edit.edit_id),
        ctx=ctx,
    )
    _write_meta(layout.meta_json("deletion", edit.shard, edit.obj_id, edit.edit_id), record)
    _append_promote_log(layout, edit, ctx)
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

    record = _base_record(edit, ctx, before_npz=object_npz, after_npz=pipeline_after)
    k = int(record["views"]["best_view_index"])
    _hardlink_or_copy(
        layout.orig_view(edit.shard, edit.obj_id, k),
        layout.before_image(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id),
        ctx=ctx,
    )
    _hardlink_or_copy(
        edit.edit_dir / f"preview_{k}.png",
        layout.after_image(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id),
        ctx=ctx,
    )
    _write_meta(layout.meta_json(edit.edit_type, edit.shard, edit.obj_id, edit.edit_id), record)
    _append_promote_log(layout, edit, ctx)
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
    paired_after_image = layout.after_image("deletion", edit.shard, edit.obj_id, paired)
    if not paired_after_image.is_file():
        return PromoteResult(False, f"paired deletion {paired} after.png missing")

    before_dst = layout.before_npz("addition", edit.shard, edit.obj_id, edit.edit_id)
    after_dst = layout.after_npz("addition", edit.shard, edit.obj_id, edit.edit_id)
    _hardlink_or_copy(paired_after, before_dst, ctx=ctx)
    _hardlink_or_copy(object_npz, after_dst, ctx=ctx)

    record = _base_record(edit, ctx, before_npz=paired_after, after_npz=object_npz)
    # _views_block mirrors the paired deletion's best_view_index for additions,
    # so the same K is used for both sides and the inodes naturally line up.
    k = int(record["views"]["best_view_index"])
    _hardlink_or_copy(
        paired_after_image,
        layout.before_image("addition", edit.shard, edit.obj_id, edit.edit_id),
        ctx=ctx,
    )
    _hardlink_or_copy(
        layout.orig_view(edit.shard, edit.obj_id, k),
        layout.after_image("addition", edit.shard, edit.obj_id, edit.edit_id),
        ctx=ctx,
    )
    _write_meta(layout.meta_json("addition", edit.shard, edit.obj_id, edit.edit_id), record)
    _append_promote_log(layout, edit, ctx)
    return PromoteResult(True, None, record)


__all__ = [
    "DEFAULT_FRONT_VIEW_INDEX",
    "META_SCHEMA_VERSION",
    "PromoteContext",
    "PromoteResult",
    "promote_addition",
    "promote_deletion",
    "promote_flux",
]
