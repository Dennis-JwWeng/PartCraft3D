"""Resolve per-edit instruction (prompt + descs + params) from pipeline outputs.

The ground-truth source of edit prompts is the Phase-A VLM output saved at
``<obj_dir>/phase1/parsed.json``. The pipeline assigns ``edit_id`` to parsed
entries via two independent per-type counters (see
``partcraft/pipeline_v3/specs.iter_all_specs``):

* deletion entries → ``del_<obj>_NNN`` with ``del_seq`` (0-based, deletion-only)
* mod / scl / mat / clr / glb entries → ``<prefix>_<obj>_MMM`` with a
  shared ``flux_seq`` (0-based across all flux types)
* identity / addition entries do not consume a sequence slot

This module re-implements that mapping (read-only) so we can attach the
authoritative instruction back onto each promoted edit's ``meta.json`` —
including additions, whose prompt is the inverse of the paired deletion.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from partcraft.cleaning.h3d_v1.layout import paired_edit_id
from partcraft.edit_types import (
    DELETION,
    GLOBAL,
    ID_PREFIX,
    MATERIAL,
    MODIFICATION,
    SCALE,
    COLOR,
)

LOGGER = logging.getLogger(__name__)

# Flux-type set (matches partcraft.pipeline_v3.paths.FLUX_TYPES). Listed
# explicitly to avoid an import that pulls heavy pipeline_v3 deps.
_FLUX_TYPES = {MODIFICATION, SCALE, MATERIAL, COLOR, GLOBAL}


# Edit-type groups
_DELETION_LIKE = {"deletion", "addition"}
_FLUX_GROUP_WITH_AFTER_DESC = {"modification", "scale", "material", "color", "global"}


def _instruction_for(parsed_edit: dict[str, Any], object_desc: str) -> dict[str, Any]:
    """Build a slim, JSON-friendly instruction record from a parsed.edits entry.

    Schema v3: drop partverse-specific fields (``selected_part_ids``,
    ``view_index``, ``n_parts_selected``, ``part_labels``); only emit
    type-applicable fields (``new_parts_desc`` only for ``modification``;
    ``after_desc`` only when non-empty for the flux group; ``edit_params``
    only when non-empty). Downstream consumers see the edit effect
    (prompt + descs + params) without pipeline-internal part bookkeeping.
    """
    et = parsed_edit.get("edit_type") or ""

    instr: dict[str, Any] = {
        "prompt": parsed_edit.get("prompt") or "",
        "object_desc": object_desc,
        "target_part_desc": parsed_edit.get("target_part_desc") or "",
    }

    if et == "modification":
        instr["new_parts_desc"] = (
            parsed_edit.get("new_parts_desc")
            or parsed_edit.get("target_part_desc")
            or ""
        )

    after_desc = parsed_edit.get("after_desc") or ""
    if et in _FLUX_GROUP_WITH_AFTER_DESC and after_desc:
        instr["after_desc"] = after_desc

    params = dict(parsed_edit.get("edit_params") or {})
    if params:
        instr["edit_params"] = params

    return instr


def load_instructions(obj_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``{edit_id: instruction_dict}`` for every edit on this object.

    Includes synthesised entries for ``add_*`` edits (back-filled prompt
    derived from the paired ``del_*`` via ``invert_delete_prompt``).

    Returns an empty dict if ``parsed.json`` is missing or malformed.
    """
    parsed_path = obj_dir / "phase1" / "parsed.json"
    if not parsed_path.is_file():
        return {}
    try:
        j = json.loads(parsed_path.read_text())
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("failed to load %s: %s", parsed_path, e)
        return {}

    parsed = j.get("parsed") or {}
    obj = parsed.get("object") or {}
    object_desc = obj.get("full_desc") or ""
    obj_id = j.get("obj_id") or obj_dir.name
    edits = parsed.get("edits") or []

    result: dict[str, dict[str, Any]] = {}
    flux_seq = 0
    del_seq = 0
    for e in edits:
        et = e.get("edit_type", "?")
        if et in _FLUX_TYPES:
            seq = flux_seq
            flux_seq += 1
        elif et == DELETION:
            seq = del_seq
            del_seq += 1
        else:
            # identity / addition / unknown: do not consume a slot
            continue
        prefix = ID_PREFIX[et]
        edit_id = f"{prefix}_{obj_id}_{seq:03d}"
        instr = _instruction_for(e, object_desc)
        result[edit_id] = instr

    from partcraft.pipeline_v3.addition_utils import invert_delete_prompt
    for del_id, instr in list(result.items()):
        if not del_id.startswith(ID_PREFIX[DELETION] + "_"):
            continue
        add_id = paired_edit_id(del_id)
        if add_id is None or add_id in result:
            continue
        result[add_id] = {
            "prompt": invert_delete_prompt(instr["prompt"]),
            "object_desc": instr["object_desc"],
            "target_part_desc": instr["target_part_desc"],
            "synthesized": True,
        }

    return result


__all__ = ["load_instructions"]
