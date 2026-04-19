"""Iterate pipeline_v3 outputs to surface per-edit work units.

Reads a pipeline_v3 YAML config (any shard) and yields a
``PipelineEdit`` per edit on disk. Resolves paths via
``partcraft.utils.config.load_config`` so the same code adapts to any
host's ``data.output_dir`` / ``mode_*`` / env-override conventions.

Layout assumed (matches all v3 configs as of 2026-04-19):

::

    <output_dir>/
      objects/
        <NN>/
          <obj_id>/
            edit_status.json
            edits_3d/
              <edit_id>/
                ... (after_new.glb / before.npz / after.npz / preview_*.png)

This module is deliberately read-only — no parsing of edit_status.json
content (beyond loading it once per object). Filtering lives in
``filter.py``; promotion lives in ``promoter.py``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from partcraft.cleaning.h3d_v1.layout import EDIT_TYPES_ALL, edit_type_from_id

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineEdit:
    """One promotable edit on disk, plus its addressing context."""

    obj_id: str
    shard: str
    edit_id: str
    edit_type: str  # one of EDIT_TYPES_ALL
    obj_dir: Path  # outputs/.../objects/<NN>/<obj_id>/
    edit_dir: Path  # obj_dir/edits_3d/<edit_id>/
    edit_status_path: Path  # obj_dir/edit_status.json


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved roots from a pipeline_v3 config."""

    output_dir: Path  # data.output_dir, e.g. .../shard08/mode_e_text_align/
    images_root: Path  # data.images_root  (raw image npz dir, by-shard)
    mesh_root: Path  # data.mesh_root
    slat_dir: Path  # data.slat_dir

    @property
    def objects_root(self) -> Path:
        return self.output_dir / "objects"


def resolve_paths(pipeline_cfg_path: str | Path) -> PipelinePaths:
    """Load a pipeline_v3 YAML and return the resolved roots.

    Defers to ``partcraft.utils.config.load_config`` so env overrides
    (``PARTCRAFT_OUTPUT_ROOT`` etc.) are honoured.
    """
    from partcraft.utils.config import load_config

    cfg = load_config(pipeline_cfg_path)
    data = cfg.get("data", {})
    missing = [k for k in ("output_dir", "images_root", "mesh_root", "slat_dir") if not data.get(k)]
    if missing:
        raise ValueError(f"pipeline cfg {pipeline_cfg_path} missing data.{missing}")
    return PipelinePaths(
        output_dir=Path(data["output_dir"]),
        images_root=Path(data["images_root"]),
        mesh_root=Path(data["mesh_root"]),
        slat_dir=Path(data["slat_dir"]),
    )


def list_objects(paths: PipelinePaths, shard: str) -> list[Path]:
    """Return all per-object dirs under ``<output_dir>/objects/<shard>/`` (sorted)."""
    shard_root = paths.objects_root / shard
    if not shard_root.is_dir():
        return []
    return sorted(p for p in shard_root.iterdir() if p.is_dir())


def load_edit_status(obj_dir: Path) -> dict[str, Any]:
    """Read ``<obj_dir>/edit_status.json``; return ``{}`` if missing."""
    status_path = obj_dir / "edit_status.json"
    if not status_path.is_file():
        return {}
    try:
        return json.loads(status_path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("malformed edit_status.json at %s: %s", status_path, exc)
        return {}


def iter_edits(
    pipeline_cfg_path: str | Path,
    shard: str,
    *,
    types: Iterable[str] | None = None,
    obj_id_allowlist: Iterable[str] | None = None,
    require_status: bool = True,
) -> Iterator[PipelineEdit]:
    """Yield every promotable edit on disk for ``shard``.

    Args:
        pipeline_cfg_path: Path to the v3 YAML.
        shard: Two-digit shard string (e.g. ``"08"``).
        types: Optional filter — restrict to these dataset edit_types
            (e.g. ``("deletion",)`` or ``("modification","scale")``). If
            ``None``, yields all known types.
        obj_id_allowlist: Optional set of obj_ids to restrict to.
        require_status: If ``True`` (default), skip objects whose
            ``edit_status.json`` is missing or empty (these are not yet
            ready to promote). When ``False``, yields edits even when
            status is unknown — the consumer is expected to handle that.

    The yielded ``PipelineEdit`` has all addressing info for the edit
    but does *not* embed the edit_status dict (re-load via
    ``load_edit_status(obj_dir)`` if needed — typically once per object).
    """
    paths = resolve_paths(pipeline_cfg_path)
    type_filter: set[str] | None = set(types) if types is not None else None
    if type_filter is not None:
        unknown = type_filter - set(EDIT_TYPES_ALL)
        if unknown:
            raise ValueError(f"unknown edit_types in filter: {sorted(unknown)}")
    obj_filter: set[str] | None = set(obj_id_allowlist) if obj_id_allowlist is not None else None

    for obj_dir in list_objects(paths, shard):
        obj_id = obj_dir.name
        if obj_filter is not None and obj_id not in obj_filter:
            continue
        if require_status and not (obj_dir / "edit_status.json").is_file():
            continue
        edits_root = obj_dir / "edits_3d"
        if not edits_root.is_dir():
            continue
        for edit_dir in sorted(p for p in edits_root.iterdir() if p.is_dir()):
            edit_id = edit_dir.name
            try:
                etype = edit_type_from_id(edit_id)
            except ValueError:
                LOGGER.debug("skipping unrecognised edit_id %s", edit_id)
                continue
            if type_filter is not None and etype not in type_filter:
                continue
            yield PipelineEdit(
                obj_id=obj_id,
                shard=shard,
                edit_id=edit_id,
                edit_type=etype,
                obj_dir=obj_dir,
                edit_dir=edit_dir,
                edit_status_path=obj_dir / "edit_status.json",
            )


__all__ = [
    "PipelineEdit",
    "PipelinePaths",
    "resolve_paths",
    "list_objects",
    "load_edit_status",
    "iter_edits",
]
