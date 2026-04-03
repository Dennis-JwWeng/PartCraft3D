"""Layer 1-PLY: Mesh-level geometric checks for deletion/addition PLY pairs.

Reuses metrics from ``partcraft.phase3_filter.filter`` but returns
``MetricResult`` objects (same as npz_checks / pair_checks) for
unified scoring in the cleaning pipeline.

These checks are meant for edits that only have PLY output (no SLAT/NPZ),
typically deletion pairs produced by direct mesh manipulation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .npz_checks import MetricResult

logger = logging.getLogger(__name__)


def _load_mesh(ply_path: Path):
    """Load a PLY mesh via trimesh (lazy import)."""
    import trimesh
    return trimesh.load(str(ply_path), process=False)


def check_deletion_ply(
    before_ply: Path,
    after_ply: Path,
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Run geometric checks on a deletion PLY pair (before.ply, after.ply).

    Checks:
      - has_geometry: after mesh has enough vertices
      - not_degenerate: after mesh is not flat/line
      - volume_ratio: reasonable volume change
      - connected_components: after mesh not overly fragmented
      - center_drift: centroid didn't drift too far
      - watertight: after mesh is watertight (soft check)

    Returns list of MetricResult for unified scoring.
    """
    import trimesh

    c = cfg or {}
    results: list[MetricResult] = []

    try:
        before = _load_mesh(before_ply)
        after = _load_mesh(after_ply)
    except Exception as e:
        return [MetricResult("ply_load", 0.0, False, 3.0,
                             f"failed to load PLY: {e}")]

    # --- has_geometry ---
    min_verts = c.get("min_vertices", 50)
    n_before = len(before.vertices)
    n_after = len(after.vertices)
    passed = n_after >= min_verts
    reason = "" if passed else f"after has {n_after} vertices < {min_verts}"
    results.append(MetricResult("has_geometry", float(n_after), passed, 2.0, reason))

    # --- not_degenerate ---
    min_extent = c.get("min_extent", 0.0001)
    extents = after.bounding_box.extents
    min_ext_val = float(extents.min())
    passed = min_ext_val > min_extent
    reason = "" if passed else f"min extent {min_ext_val:.6f} <= {min_extent}"
    results.append(MetricResult("not_degenerate", min_ext_val, passed, 2.0, reason))

    # --- volume_ratio ---
    vol_before = max(float(before.bounding_box.volume), 1e-12)
    vol_after = float(after.bounding_box.volume)
    vol_ratio = vol_after / vol_before
    min_vr = c.get("min_volume_ratio", 0.05)
    max_vr = c.get("max_volume_ratio", 0.95)
    passed = min_vr <= vol_ratio <= max_vr
    reason = "" if passed else f"volume ratio {vol_ratio:.3f} outside [{min_vr},{max_vr}]"
    results.append(MetricResult("volume_ratio", vol_ratio, passed, 2.0, reason))

    # --- connected_components ---
    max_comp = c.get("max_components", 5)
    try:
        bodies = after.split(only_watertight=False)
        n_comp = len(bodies)
    except Exception:
        n_comp = 1
    passed = n_comp <= max_comp
    reason = "" if passed else f"{n_comp} components > {max_comp}"
    results.append(MetricResult("connected_components", float(n_comp), passed, 1.5, reason))

    # --- center_drift ---
    max_drift = c.get("max_center_drift", 0.3)
    center_b = before.centroid
    center_a = after.centroid
    diag = float(np.linalg.norm(before.bounding_box.extents))
    drift = float(np.linalg.norm(center_a - center_b))
    drift_ratio = drift / max(diag, 1e-8)
    passed = drift_ratio < max_drift
    reason = "" if passed else f"center drift {drift_ratio:.3f} >= {max_drift}"
    results.append(MetricResult("center_drift", drift_ratio, passed, 1.5, reason))

    # --- watertight (informational only, weight=0) ---
    # Deletion inherently breaks watertightness (open hole where part was
    # removed), so this must not block pass/fail.
    is_wt = bool(after.is_watertight)
    results.append(MetricResult(
        "watertight", 1.0 if is_wt else 0.0, True, 0.0,
        "" if is_wt else "after mesh is not watertight (expected for deletion)"))

    return results


def check_addition_ply(
    before_ply: Path,
    after_ply: Path,
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Run geometric checks on an addition PLY pair.

    Addition is the reverse of deletion: ``before`` is smaller (post-delete),
    ``after`` is the original (larger) object.
    """
    c = cfg or {}
    results: list[MetricResult] = []

    try:
        before = _load_mesh(before_ply)
        after = _load_mesh(after_ply)
    except Exception as e:
        return [MetricResult("ply_load", 0.0, False, 3.0,
                             f"failed to load PLY: {e}")]

    # --- has_geometry ---
    min_verts = c.get("min_vertices", 50)
    passed = len(before.vertices) >= min_verts and len(after.vertices) >= min_verts
    reason = "" if passed else f"vertices: before={len(before.vertices)}, after={len(after.vertices)}"
    results.append(MetricResult("has_geometry", float(min(len(before.vertices), len(after.vertices))),
                                passed, 2.0, reason))

    # --- volume increase ---
    vol_before = max(float(before.bounding_box.volume), 1e-12)
    vol_after = float(after.bounding_box.volume)
    vol_ratio = vol_after / vol_before
    min_vr = c.get("min_volume_ratio", 1.05)
    max_vr = c.get("max_volume_ratio", 20.0)
    passed = min_vr <= vol_ratio <= max_vr
    reason = "" if passed else f"volume ratio {vol_ratio:.3f} outside [{min_vr},{max_vr}]"
    results.append(MetricResult("volume_ratio", vol_ratio, passed, 2.0, reason))

    return results
