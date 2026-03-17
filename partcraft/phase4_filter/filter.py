"""Phase 4: Quality filtering for assembled mesh pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from partcraft.phase2_assembly.assembler import AssembledPair


@dataclass
class QualityResult:
    passed: bool
    score: float
    checks: dict[str, bool]
    details: dict[str, float]


def check_quality(pair: AssembledPair, cfg: dict) -> QualityResult:
    """Run all quality checks on an assembled pair."""
    p4 = cfg["phase4"]
    checks = {}
    details = {}

    before = pair.before_mesh
    after = pair.after_mesh

    # 1. Volume ratio check
    before_vol = before.bounding_box.volume if before.bounding_box is not None else 0
    after_vol = after.bounding_box.volume if after.bounding_box is not None else 0

    if before_vol > 0:
        vol_ratio = after_vol / before_vol
    else:
        vol_ratio = 0.0
    details["volume_ratio"] = vol_ratio
    checks["volume_reasonable"] = p4["min_volume_ratio"] <= vol_ratio <= p4["max_volume_ratio"]

    # 2. Connected components check
    try:
        components = after.split()
        n_components = len(components)
    except Exception:
        n_components = 1
    details["n_components"] = n_components
    checks["not_fragmented"] = n_components <= p4["max_components"]

    # 3. Vertex count sanity
    checks["has_geometry"] = len(after.vertices) > 50
    details["after_vertices"] = len(after.vertices)

    # 4. Edit ratio (geometric difference)
    before_verts = set(map(tuple, np.round(before.vertices, 4).tolist()))
    after_verts = set(map(tuple, np.round(after.vertices, 4).tolist()))
    if before_verts:
        sym_diff = len(before_verts.symmetric_difference(after_verts))
        edit_ratio = sym_diff / max(len(before_verts), len(after_verts))
    else:
        edit_ratio = 1.0
    details["edit_ratio"] = edit_ratio
    checks["edit_nontrivial"] = edit_ratio >= p4["min_edit_ratio"]
    checks["edit_not_total"] = edit_ratio <= p4["max_edit_ratio"]

    # 5. Bounding box sanity (no degenerate dimensions)
    extents = after.bounding_box.extents if after.bounding_box is not None else np.zeros(3)
    min_extent = extents.min()
    checks["not_degenerate"] = min_extent > 1e-4
    details["min_extent"] = float(min_extent)

    # 6. Bounding box containment: after should largely overlap with before
    #    (prevents swapped parts from flying off to random locations)
    if before.bounding_box is not None and after.bounding_box is not None:
        before_max = before.bounding_box.bounds[1]
        before_min = before.bounding_box.bounds[0]
        before_diag = np.linalg.norm(before_max - before_min)
        after_center = after.bounding_box.centroid
        before_center = before.bounding_box.centroid
        center_drift = np.linalg.norm(after_center - before_center)
        details["center_drift_ratio"] = float(center_drift / max(before_diag, 1e-8))
        checks["center_stable"] = details["center_drift_ratio"] < 0.3
    else:
        details["center_drift_ratio"] = 0.0
        checks["center_stable"] = True

    passed = all(checks.values())
    # Composite score: fraction of checks passed
    score = sum(checks.values()) / max(len(checks), 1)

    return QualityResult(passed=passed, score=score, checks=checks, details=details)
