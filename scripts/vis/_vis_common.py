"""Shared visualization utilities for rendering and composing edit pairs.

Consolidates SLAT loading, Gaussian rendering, text/label bars, and
comparison composition that were duplicated across vis scripts.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


# ──────────────────────────── SLAT loading ─────────────────────────────────

def load_slat(path: Path | str, device: str = "cuda"):
    """Load SLAT from NPZ or legacy ``feats.pt + coords.pt`` directory."""
    from trellis.modules import sparse as sp

    path = Path(path)
    npz = path if path.suffix == ".npz" else path.with_suffix(".npz")
    if npz.exists():
        data = np.load(str(npz))
        feats = torch.from_numpy(data["slat_feats"]).to(device)
        coords = torch.from_numpy(data["slat_coords"]).to(device)
        return sp.SparseTensor(feats=feats, coords=coords)
    if path.is_dir():
        feats = torch.load(path / "feats.pt", weights_only=True).to(device)
        coords = torch.load(path / "coords.pt", weights_only=True).to(device)
        return sp.SparseTensor(feats=feats, coords=coords)
    raise FileNotFoundError(f"No SLAT found at {path}")


# ──────────────────────────── Gaussian rendering ───────────────────────────

def render_gaussian_views(
    gaussian,
    num_views: int = 8,
    pitch: float = 0.45,
) -> list[np.ndarray]:
    """Render orbiting multiview images from a Gaussian.

    Returns list of [H, W, 3] uint8 arrays.
    """
    from trellis.utils import render_utils

    yaws = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]
    pitches = torch.tensor([pitch] * num_views)
    imgs = render_utils.Trellis_render_multiview_images(
        gaussian, yaws.tolist(), pitches.tolist()
    )["color"]
    return imgs


def decode_and_render(
    pipeline,
    slat,
    num_views: int = 8,
    pitch: float = 0.45,
) -> list[np.ndarray]:
    """SLAT → Gaussian decode → multiview render (convenience wrapper)."""
    outputs = pipeline.decode_slat(slat, ["gaussian"])
    gaussian = outputs["gaussian"][0]
    return render_gaussian_views(gaussian, num_views, pitch)


# ──────────────────────────── Text composition ─────────────────────────────

def wrap_text(text: str, max_chars: int = 60) -> list[str]:
    """Word-wrap text into lines."""
    lines: list[str] = []
    for raw_line in text.split("\n"):
        remaining = raw_line
        while remaining:
            if len(remaining) <= max_chars:
                lines.append(remaining)
                break
            split = remaining[:max_chars].rfind(" ")
            if split <= 0:
                split = max_chars
            lines.append(remaining[:split])
            remaining = remaining[split:].strip()
    return lines


def make_text_bar(
    text: str,
    width: int,
    bar_height: int = 48,
    bg_color: tuple = (30, 30, 30),
    fg_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Create a prompt text bar (dark background, white text, word-wrapped)."""
    bar = np.full((bar_height, width, 3), bg_color, dtype=np.uint8)
    lines = wrap_text(text, max_chars=width // 8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 20
    for line in lines[:3]:
        cv2.putText(bar, line, (10, y), font, 0.55, fg_color, 1, cv2.LINE_AA)
        y += 18
    return bar


def make_label_bar(
    label: str,
    width: int,
    height: int = 30,
    bg_color: tuple = (240, 240, 240),
    fg_color: tuple = (40, 40, 40),
) -> np.ndarray:
    """Create a centered label bar (e.g. 'Before' / 'After')."""
    bar = np.full((height, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw = cv2.getTextSize(label, font, 0.65, 2)[0][0]
    cv2.putText(
        bar, label, ((width - tw) // 2, 22), font, 0.65, fg_color, 2, cv2.LINE_AA
    )
    return bar


# ──────────────────────────── Edit type inference ──────────────────────────

_PREFIX_TO_EDIT_TYPE: tuple[tuple[str, str], ...] = (
    ("gdel", "deletion"),
    ("gadd", "addition"),
    ("del", "deletion"),
    ("add", "addition"),
    ("mod", "modification"),
    ("scl", "scale"),
    ("mat", "material"),
    ("glb", "global"),
    ("idt", "identity"),
)


def infer_edit_type(edit_id: str) -> str:
    """Guess edit type from the edit_id prefix."""
    for prefix, etype in _PREFIX_TO_EDIT_TYPE:
        if edit_id.startswith(prefix + "_"):
            return etype
    return "unknown"
