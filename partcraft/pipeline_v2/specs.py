"""Edit-spec iteration for pipeline v2.

Replaces the middleware ``edit_specs.jsonl`` produced by
``scripts/tools/parsed_to_edit_specs.py``. Specs are derived in-memory
from a per-object ``parsed.json`` and consumed directly by step
runners. ``iter_flux_specs`` yields specs that need a FLUX 2D edit;
``iter_deletion_specs`` yields deletion specs (mesh-direct path);
``iter_all_specs`` yields everything in original parsed order plus
their assigned ``edit_id``s.

Numbering rules
---------------
* mod / scl / mat / glb share a single per-object sequence (``flux_seq``)
  — matches the legacy ``parsed_to_edit_specs.py`` and the existing
  on-disk artifacts in ``pipeline_v2_mirror5``.
* deletion has its own per-object sequence (``del_seq``).
* addition is *not* emitted by the VLM and is not produced here; it is
  backfilled from deletion in step s7 (see future ``backfill.py``).
* identity (``idt``) is intentionally skipped.

The ``EditSpec`` dataclass intentionally carries only fields that step
runners actually need; full edit metadata stays in ``meta.json`` and
``parsed.json``. ``to_legacy_dict`` produces the dict consumed by
``scripts.run_2d_edit.process_one`` so we can keep using the existing
FLUX worker untouched.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import EDIT_TYPE_PREFIX, FLUX_TYPES, ObjectContext
from .s1_vlm_core import VIEW_INDICES  # single source of truth; defined in s1_vlm_core.py


@dataclass
class EditSpec:
    """Per-edit descriptor used by every pipeline_v2 step."""
    edit_id: str
    edit_type: str          # canonical name (modification, deletion, ...)
    obj_id: str
    shard: str
    edit_idx: int           # index inside parsed.edits (stable)
    view_index: int         # 0..4 (phase1_v2 short index)
    npz_view: int           # absolute frame index into images npz
    selected_part_ids: list[int] = field(default_factory=list)
    part_labels: list[str] = field(default_factory=list)
    prompt: str = ""
    target_part_desc: str = ""
    new_parts_desc: str = ""
    edit_params: dict[str, Any] = field(default_factory=dict)
    object_desc: str = ""
    # VLM-generated S1/S2 decompositions — all default to "" for backward
    # compat with old parsed.json files.  build_prompts_from_spec falls back
    # to _decompose_local() when these are empty.
    object_desc_s1: str = ""     # object.full_desc_stage1 (geometry-only)
    object_desc_s2: str = ""     # object.full_desc_stage2 (appearance-only)
    after_desc_s1: str = ""      # edit.after_desc_stage1  (non-deletion)
    after_desc_s2: str = ""      # edit.after_desc_stage2  (non-deletion)
    new_parts_desc_s1: str = ""  # edit.new_parts_desc_stage1 (modification)
    new_parts_desc_s2: str = ""  # edit.new_parts_desc_stage2 (modification)

    # ─── factory ─────────────────────────────────────────────────────

    @classmethod
    def from_parsed_edit(
        cls,
        ctx: ObjectContext,
        edit_idx: int,
        edit: dict[str, Any],
        seq: int,
        parts_by_id: dict[int, dict],
        object_desc: str,
        object_desc_s1: str = "",
        object_desc_s2: str = "",
    ) -> "EditSpec":
        et = edit.get("edit_type", "?")
        vi = int(edit.get("view_index", 0))
        npz_view = (VIEW_INDICES[vi]
                    if 0 <= vi < len(VIEW_INDICES) else -1)
        pids = list(edit.get("selected_part_ids") or [])
        labels = [parts_by_id.get(p, {}).get("name", "") for p in pids]
        return cls(
            edit_id=ctx.edit_id(et, seq),
            edit_type=et,
            obj_id=ctx.obj_id,
            shard=ctx.shard,
            edit_idx=edit_idx,
            view_index=vi,
            npz_view=npz_view,
            selected_part_ids=pids,
            part_labels=labels,
            prompt=edit.get("prompt") or "",
            target_part_desc=edit.get("target_part_desc") or "",
            new_parts_desc=(edit.get("new_parts_desc")
                            or edit.get("target_part_desc") or ""),
            edit_params=dict(edit.get("edit_params") or {}),
            object_desc=object_desc,
            object_desc_s1=object_desc_s1,
            object_desc_s2=object_desc_s2,
            after_desc_s1=edit.get("after_desc_stage1") or "",
            after_desc_s2=edit.get("after_desc_stage2") or "",
            new_parts_desc_s1=edit.get("new_parts_desc_stage1") or "",
            new_parts_desc_s2=edit.get("new_parts_desc_stage2") or "",
        )

    # ─── interop ─────────────────────────────────────────────────────

    def to_legacy_dict(self) -> dict[str, Any]:
        """Produce the dict shape expected by ``scripts.run_2d_edit.process_one``.

        Field semantics are inherited from the legacy pipeline; we only
        populate what FLUX backends actually consume.
        """
        first_pid = self.selected_part_ids[0] if self.selected_part_ids else -1
        first_label = self.part_labels[0] if self.part_labels else ""
        return {
            "edit_id":          self.edit_id,
            "edit_type":        self.edit_type,
            "obj_id":           self.obj_id,
            "shard":            self.shard,
            "object_desc":      self.object_desc,
            "before_desc":      "",
            "remove_part_ids":  list(self.selected_part_ids),
            "remove_labels":    list(self.part_labels),
            "keep_part_ids":    [],
            "add_part_ids":     [],
            "add_labels":       [],
            "base_part_ids":    [],
            "old_part_id":      first_pid,
            "old_label":        first_label,
            "source_del_id":    "",
            "edit_prompt":      self.prompt,
            "after_desc":       self.new_parts_desc,
            "before_part_desc": self.target_part_desc,
            "after_part_desc":  self.new_parts_desc,
            "mod_type":         "",
            "best_view":        self.npz_view,
        }

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────── parsing helpers ──────────────────────────────────

def _load_parsed(
    ctx: ObjectContext,
) -> tuple[list[dict], dict[int, dict], str, str, str]:
    """Return ``(edits, parts_by_id, object_desc, object_desc_s1, object_desc_s2)``."""
    if not ctx.parsed_path.is_file():
        return [], {}, "", "", ""
    j = json.loads(ctx.parsed_path.read_text())
    parsed = j.get("parsed") or {}
    obj = parsed.get("object") or {}
    parts_by_id = {p["part_id"]: p for p in (obj.get("parts") or [])
                   if isinstance(p, dict) and "part_id" in p}
    return (
        parsed.get("edits") or [],
        parts_by_id,
        obj.get("full_desc", "") or "",
        obj.get("full_desc_stage1", "") or "",
        obj.get("full_desc_stage2", "") or "",
    )


# ─────────────────── public iterators ─────────────────────────────────

def iter_all_specs(ctx: ObjectContext) -> Iterator[EditSpec]:
    """Yield every spec for the object in parsed-edits order, with
    correct edit_id sequencing across types."""
    edits, parts_by_id, object_desc, object_desc_s1, object_desc_s2 = _load_parsed(ctx)
    flux_seq = 0
    del_seq = 0
    for idx, e in enumerate(edits):
        et = e.get("edit_type", "?")
        if et in FLUX_TYPES:
            seq = flux_seq
            flux_seq += 1
        elif et == "deletion":
            seq = del_seq
            del_seq += 1
        elif et == "identity":
            continue  # silently skip; not implemented
        elif et == "addition":
            continue  # backfilled from deletion in s7, not VLM-produced
        else:
            continue
        yield EditSpec.from_parsed_edit(
            ctx, idx, e, seq, parts_by_id,
            object_desc, object_desc_s1, object_desc_s2,
        )


def iter_flux_specs(ctx: ObjectContext) -> Iterator[EditSpec]:
    """Yield only specs that need a FLUX 2D edit (mod/scl/mat/glb)."""
    for s in iter_all_specs(ctx):
        if s.edit_type in FLUX_TYPES:
            yield s


def iter_deletion_specs(ctx: ObjectContext) -> Iterator[EditSpec]:
    for s in iter_all_specs(ctx):
        if s.edit_type == "deletion":
            yield s


def iter_specs_for_objects(
    ctxs: Iterable[ObjectContext],
    *,
    types: Iterable[str] | None = None,
) -> Iterator[EditSpec]:
    """Flatten specs across many objects, optionally filtered by type."""
    type_set = set(types) if types is not None else None
    for ctx in ctxs:
        for s in iter_all_specs(ctx):
            if type_set is None or s.edit_type in type_set:
                yield s


__all__ = [
    "VIEW_INDICES",
    "EditSpec",
    "iter_all_specs",
    "iter_flux_specs",
    "iter_deletion_specs",
    "iter_specs_for_objects",
]
