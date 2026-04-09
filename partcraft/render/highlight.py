"""Part highlight rendering utilities.

Renders a single-view highlight image for one edit's selected parts.
Used by Phase 1 VLM scoring (run_phase1_v2_score.py) and s2_highlights.py.

Previously lived in ``scripts/tools/render_part_highlight.py`` as a
standalone script.  Moved here so pipeline_v2 modules can import cleanly
without ``sys.path`` manipulation.

``scripts/tools/render_part_highlight.py`` is kept as a thin shim + CLI
entry point that re-exports everything from this module.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from partcraft.render.overview import (
    VIEW_INDICES, extract_parts, load_views_from_npz, run_blender,
)

# Highlight color used for ALL selected parts (regardless of part_id).
# Single bright color -> unambiguous "this is the edit target".
HIGHLIGHT = [230, 40, 200]   # magenta
GRAY = [210, 210, 210]       # non-selected parts


def render_highlight(mesh_npz: Path, img_npz: Path, view_index: int,
                     selected_part_ids: list[int], blender: str,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (original_bgr, highlight_bgr) for the chosen view."""
    if not (0 <= view_index < len(VIEW_INDICES)):
        raise ValueError(f"view_index must be in [0,{len(VIEW_INDICES)})")
    top_imgs, frames = load_views_from_npz(img_npz, VIEW_INDICES)
    orig = top_imgs[view_index]
    frame = frames[view_index]
    H = orig.shape[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        part_ids = extract_parts(mesh_npz, tmp)
        if not part_ids:
            raise RuntimeError("no parts in mesh npz")
        max_pid = max(part_ids) + 1
        pid_palette = [list(GRAY)] * max_pid
        sel_set = set(selected_part_ids)
        for pid in part_ids:
            if pid in sel_set:
                pid_palette[pid] = list(HIGHLIGHT)
        rendered = run_blender(tmp, blender, H, pid_palette, [frame])
    return orig, rendered[0]


def make_header(width: int, edit_type: str, prompt: str,
                height: int = 56) -> np.ndarray:
    bar = np.full((height, width, 3), 245, dtype=np.uint8)
    line1 = f"[{edit_type}]"
    cv2.putText(bar, line1, (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (30, 30, 30), 2, cv2.LINE_AA)
    max_chars = max(20, width // 11)
    p = prompt if len(prompt) <= max_chars else prompt[:max_chars - 1] + "..."
    cv2.putText(bar, p, (12, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return bar


def stitch_pair(orig: np.ndarray, hl: np.ndarray,
                edit_type: str = "", prompt: str = "") -> np.ndarray:
    H, W = orig.shape[:2]
    if hl.shape[:2] != (H, W):
        hl = cv2.resize(hl, (W, H), interpolation=cv2.INTER_AREA)
    sep = np.full((H, 6, 3), 180, dtype=np.uint8)

    def label_strip(text):
        s = np.full((28, W, 3), 230, dtype=np.uint8)
        cv2.putText(s, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 30, 30), 1, cv2.LINE_AA)
        return s

    orig_lab = np.vstack([label_strip("ORIGINAL"), orig])
    hl_lab = np.vstack([label_strip("HIGHLIGHTED PARTS"), hl])
    sep2 = np.full((orig_lab.shape[0], 6, 3), 180, dtype=np.uint8)
    body = np.hstack([orig_lab, sep2, hl_lab])
    if edit_type or prompt:
        header = make_header(body.shape[1], edit_type, prompt)
        body = np.vstack([header, body])
    return body


__all__ = [
    "HIGHLIGHT", "GRAY",
    "render_highlight", "make_header", "stitch_pair",
]
