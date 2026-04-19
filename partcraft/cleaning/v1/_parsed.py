"""Shared helpers for reading ``phase1/parsed.json``.

Both pipeline_v2 and pipeline_v3 emit a wrapped schema::

    {
      "obj_id": ..., "shard": ..., "validation": {...},
      "parsed": {
        "object": {"parts": [{"part_id": 0, "name": "..."}, ...], ...},
        "edits":  [{"edit_type": ..., "prompt": ..., ...}, ...]
      }
    }

Older fixtures used the un-wrapped variant where ``edits`` and ``parts`` lived
at the top level (and ``parts[i]`` used key ``id`` instead of ``part_id``).
This helper accepts either form so adapters and tests stay aligned.
"""
from __future__ import annotations

from typing import Any


def extract_edits_and_parts(
    parsed_doc: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Return ``(edits_list, {part_id: part_name})`` from a parsed.json doc."""
    d = parsed_doc or {}
    inner = d.get("parsed") if isinstance(d.get("parsed"), dict) else d
    edits = list(inner.get("edits") or [])
    parts_src = inner.get("parts") or (inner.get("object") or {}).get("parts") or []
    parts_by_id: dict[int, str] = {}
    for p in parts_src:
        if not isinstance(p, dict):
            continue
        pid = p.get("part_id", p.get("id"))
        if pid is None:
            continue
        try:
            parts_by_id[int(pid)] = p.get("name", "")
        except (TypeError, ValueError):
            continue
    return edits, parts_by_id


__all__ = ["extract_edits_and_parts"]
