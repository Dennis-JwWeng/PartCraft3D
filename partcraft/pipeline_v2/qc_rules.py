from __future__ import annotations
from typing import Any

import numpy as np

from partcraft.render.overview import _PALETTE

# ── overview.png layout constants (must match stitch_two_rows in overview.py) ─
_N_VIEWS = 5        # columns
_COL_SEP = 4        # px separator between columns
_ROW_SEP = 6        # px separator between top/bottom rows

# Palette pre-converted to BGR (overview.py stores RGB; OpenCV loads BGR)
_PALETTE_BGR: list[list[int]] = [[c[2], c[1], c[0]] for c in _PALETTE]

_PART_REQUIRED = frozenset({"deletion", "modification", "scale", "material"})
_ADD_VERBS = ("add ", "insert ", "attach ", "put ")
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
        fails["prompt_too_short"] = True
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


def count_part_pixels_in_overview(
    overview_img: np.ndarray,
    view_index: int,
    selected_part_ids: list[int],
    color_tol: int = 60,
) -> int:
    """Count palette-colored pixels for *selected_part_ids* in the bottom-row
    cell of *view_index* inside an already-decoded overview BGR image.

    The overview is a 5-column × 2-row grid produced by ``stitch_two_rows``:
    - top row: original RGB photos
    - bottom row: same views re-rendered with each part in its palette color

    Returns the total matching pixel count across all selected parts.
    Zero means none of the selected parts are visible from this viewpoint.

    ``color_tol`` is the L2-distance threshold for palette-color matching.
    Blender Cycles rendering with 3-point lighting shifts pure palette emission
    colors by up to ~60 L2 units, so the default is 60. The minimum distance
    between any two palette colors is 80, so tol=60 cannot confuse adjacent
    palette entries.
    """
    if overview_img is None or not (0 <= view_index < _N_VIEWS):
        return -1   # sentinel: cannot check

    H_total, W_total = overview_img.shape[:2]
    W_img = (W_total - (_N_VIEWS - 1) * _COL_SEP) // _N_VIEWS
    H_img = (H_total - _ROW_SEP) // 2
    bot_y0 = H_img + _ROW_SEP
    x0 = view_index * (W_img + _COL_SEP)
    cell = overview_img[bot_y0: bot_y0 + H_img, x0: x0 + W_img]

    total = 0
    for pid in selected_part_ids:
        color = np.array(_PALETTE_BGR[pid % len(_PALETTE_BGR)], dtype=int)
        diff = np.linalg.norm(cell.astype(int) - color, axis=2)
        total += int(np.sum(diff < color_tol))
    return total


__all__ = ["check_rules", "count_part_pixels_in_overview"]
