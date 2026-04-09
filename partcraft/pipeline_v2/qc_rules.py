from __future__ import annotations
from typing import Any

_PART_REQUIRED = frozenset({"deletion", "modification", "scale", "material"})
_ADD_VERBS = ("add ", "insert", "attach", "place", "put ")
_REMOVE_ONLY = ("remove", "delete", "erase", "strip", "eliminate")
_REPLACE_IND = ("replace", "swap", "change", "modify", "convert")


def check_rules(edit: dict[str, Any], parts_by_id: dict[int, Any]) -> dict[str, bool]:
    """Run 7 rule checks. Returns dict of failing codes (empty = all pass)."""
    et = edit.get("edit_type", "")
    prompt = (edit.get("prompt") or "").strip()
    pids = list(edit.get("selected_part_ids") or [])
    pl = prompt.lower()
    fails: dict[str, bool] = {}

    if len(prompt) < 8:
        fails["prompt_empty"] = True
    if et in _PART_REQUIRED:
        if not pids:
            fails["parts_missing"] = True
        elif any(p not in parts_by_id for p in pids):
            fails["parts_invalid"] = True
    if et == "modification" and not (edit.get("new_parts_desc") or "").strip():
        fails["new_desc_missing"] = True
    if et in ("modification", "scale", "material"):
        if not (edit.get("target_part_desc") or "").strip():
            fails["target_desc_missing"] = True
    if et == "modification":
        s1 = (edit.get("new_parts_desc_stage1") or "").strip()
        s2 = (edit.get("new_parts_desc_stage2") or "").strip()
        if not s1 and not s2:
            fails["stage_decomp_missing"] = True
    if et == "deletion" and any(v in pl for v in _ADD_VERBS):
        fails["verb_conflict"] = True
    elif et in ("modification", "scale", "material", "global"):
        if any(v in pl for v in _REMOVE_ONLY) and not any(v in pl for v in _REPLACE_IND):
            fails["verb_conflict"] = True
    return fails

__all__ = ["check_rules"]
