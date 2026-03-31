"""Layer 2: Edit-pair comparison checks (before vs after).

Each edit type has a dedicated checker that exploits the type's invariants.
All checks operate on numpy arrays from NPZ files — no PLY / trimesh needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from .npz_checks import MetricResult


# =========================================================================
# Utility functions
# =========================================================================

def bbox_from_coords(coords: np.ndarray):
    """Compute bounding box from SLAT coords [N,4] (batch,x,y,z).

    Returns (min_xyz[3], max_xyz[3], center[3], diagonal).
    """
    xyz = coords[:, 1:].astype(np.float64)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) / 2.0
    diagonal = float(np.linalg.norm(hi - lo))
    return lo, hi, center, diagonal


def bbox_iou_voxel(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Axis-aligned bounding box IoU in voxel space."""
    lo_a, hi_a, _, _ = bbox_from_coords(coords_a)
    lo_b, hi_b, _, _ = bbox_from_coords(coords_b)
    inter_lo = np.maximum(lo_a, lo_b)
    inter_hi = np.minimum(hi_a, hi_b)
    inter_dims = np.maximum(inter_hi - inter_lo, 0.0)
    inter_vol = float(np.prod(inter_dims))
    vol_a = float(np.prod(np.maximum(hi_a - lo_a, 1.0)))
    vol_b = float(np.prod(np.maximum(hi_b - lo_b, 1.0)))
    union_vol = vol_a + vol_b - inter_vol
    if union_vol < 1e-12:
        return 0.0
    return inter_vol / union_vol


def ss_cosine_sim(ss_a: np.ndarray, ss_b: np.ndarray) -> float:
    """Cosine similarity between two SS latents (flattened)."""
    a = ss_a.flatten().astype(np.float64)
    b = ss_b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def ss_l2_distance(ss_a: np.ndarray, ss_b: np.ndarray) -> float:
    """Relative L2 distance between two SS latents."""
    a = ss_a.flatten().astype(np.float64)
    b = ss_b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    if norm_a < 1e-12:
        return float("inf")
    return float(np.linalg.norm(a - b) / norm_a)


def connected_components_voxel(coords: np.ndarray, spatial_range: int = 64) -> int:
    """Count 6-connected components in the voxel occupancy grid."""
    xyz = coords[:, 1:]
    grid = np.zeros((spatial_range, spatial_range, spatial_range), dtype=bool)
    # Clip to avoid out-of-bounds
    x = np.clip(xyz[:, 0], 0, spatial_range - 1).astype(int)
    y = np.clip(xyz[:, 1], 0, spatial_range - 1).astype(int)
    z = np.clip(xyz[:, 2], 0, spatial_range - 1).astype(int)
    grid[x, y, z] = True
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    _, n_components = ndimage.label(grid, structure=struct)
    return int(n_components)


def voxel_set_ops(coords_a: np.ndarray, coords_b: np.ndarray):
    """Compute set operations between two coord arrays (using xyz columns).

    Returns (only_a, only_b, common) counts.
    """
    def _to_set(c):
        return set(map(tuple, c[:, 1:].tolist()))
    sa, sb = _to_set(coords_a), _to_set(coords_b)
    return len(sa - sb), len(sb - sa), len(sa & sb)


def voxel_diff_ratio(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Fraction of voxels that differ (symmetric difference / union)."""
    only_a, only_b, common = voxel_set_ops(coords_a, coords_b)
    total = only_a + only_b + common
    if total == 0:
        return 0.0
    return (only_a + only_b) / total


def feat_change_ratio(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
) -> float:
    """Relative L2 change in feature vectors (requires same length)."""
    if len(feats_a) != len(feats_b):
        # Different voxel counts; use norms as proxy
        norm_a = float(np.linalg.norm(feats_a))
        norm_b = float(np.linalg.norm(feats_b))
        if max(norm_a, norm_b) < 1e-12:
            return 0.0
        return abs(norm_a - norm_b) / max(norm_a, norm_b)
    diff = feats_a.astype(np.float64) - feats_b.astype(np.float64)
    norm_a = float(np.linalg.norm(feats_a))
    if norm_a < 1e-12:
        return float("inf")
    return float(np.linalg.norm(diff) / norm_a)


def feat_change_coverage(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    eps: float = 0.01,
) -> float:
    """Fraction of voxels whose feature vector changed more than eps (L2 per row).

    Requires same voxel count (coords must match).
    """
    if len(feats_a) != len(feats_b):
        return 1.0  # incomparable → treat as fully changed
    per_row = np.linalg.norm(
        feats_a.astype(np.float64) - feats_b.astype(np.float64), axis=1
    )
    return float((per_row > eps).mean())


# =========================================================================
# Per-type checkers
# =========================================================================

def check_deletion(
    original: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Deletion: after should be a subset of original with parts removed."""
    c = cfg or {}
    results = []
    n_before = len(original["coords"])
    n_after = len(after["coords"])

    # Voxel ratio
    ratio = n_after / max(n_before, 1)
    min_r, max_r = c.get("min_voxel_ratio", 0.05), c.get("max_voxel_ratio", 0.95)
    passed = min_r <= ratio <= max_r
    reason = "" if passed else f"voxel ratio {ratio:.3f} outside [{min_r},{max_r}]"
    results.append(MetricResult("voxel_ratio", ratio, passed, 2.0, reason))

    # Delete ratio
    del_ratio = 1.0 - ratio
    min_d, max_d = c.get("min_delete_ratio", 0.02), c.get("max_delete_ratio", 0.80)
    passed = min_d <= del_ratio <= max_d
    reason = "" if passed else f"delete ratio {del_ratio:.3f} outside [{min_d},{max_d}]"
    results.append(MetricResult("delete_ratio", del_ratio, passed, 2.0, reason))

    # Bbox IoU
    iou = bbox_iou_voxel(original["coords"], after["coords"])
    min_iou = c.get("min_bbox_iou", 0.15)
    passed = iou >= min_iou
    reason = "" if passed else f"bbox iou {iou:.3f} < {min_iou}"
    results.append(MetricResult("bbox_iou", iou, passed, 1.5, reason))

    # SS relative change
    ss_dist = ss_l2_distance(original["ss"], after["ss"])
    min_ss, max_ss = c.get("min_ss_change", 0.01), c.get("max_ss_change", 0.90)
    passed = min_ss <= ss_dist <= max_ss
    reason = "" if passed else f"ss change {ss_dist:.3f} outside [{min_ss},{max_ss}]"
    results.append(MetricResult("ss_change", ss_dist, passed, 1.0, reason))

    # Connected components
    n_comp = connected_components_voxel(after["coords"])
    max_comp = c.get("max_components", 3)
    passed = n_comp <= max_comp
    reason = "" if passed else f"{n_comp} components > {max_comp}"
    results.append(MetricResult("connected_components", float(n_comp), passed, 1.5, reason))

    # Degeneracy
    _, hi, _, diag = bbox_from_coords(after["coords"])
    lo, _, _, _ = bbox_from_coords(after["coords"])
    extents = hi - lo
    min_ext = float(extents.min())
    passed = min_ext > 1.0
    reason = "" if passed else f"min extent {min_ext:.1f} <= 1 voxel"
    results.append(MetricResult("not_degenerate", min_ext, passed, 2.0, reason))

    return results


def check_addition(
    before: dict[str, np.ndarray],
    original: dict[str, np.ndarray],
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Addition: reverse of deletion. before = deletion's after, after = original."""
    c = cfg or {}
    results = []
    n_before = len(before["coords"])
    n_after = len(original["coords"])

    # Voxel increase ratio
    ratio = n_after / max(n_before, 1)
    min_r, max_r = c.get("min_voxel_ratio", 1.05), c.get("max_voxel_ratio", 20.0)
    passed = min_r <= ratio <= max_r
    reason = "" if passed else f"voxel ratio {ratio:.3f} outside [{min_r},{max_r}]"
    results.append(MetricResult("voxel_ratio", ratio, passed, 2.0, reason))

    # Add ratio
    add_ratio = 1.0 - n_before / max(n_after, 1)
    min_a, max_a = c.get("min_add_ratio", 0.02), c.get("max_add_ratio", 0.80)
    passed = min_a <= add_ratio <= max_a
    reason = "" if passed else f"add ratio {add_ratio:.3f} outside [{min_a},{max_a}]"
    results.append(MetricResult("add_ratio", add_ratio, passed, 2.0, reason))

    # Bbox IoU
    iou = bbox_iou_voxel(before["coords"], original["coords"])
    min_iou = c.get("min_bbox_iou", 0.15)
    passed = iou >= min_iou
    reason = "" if passed else f"bbox iou {iou:.3f} < {min_iou}"
    results.append(MetricResult("bbox_iou", iou, passed, 1.5, reason))

    return results


def check_modification(
    original: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Modification (swap): TRELLIS S1+S2, geometry changes locally."""
    c = cfg or {}
    results = []
    n_before = len(original["coords"])
    n_after = len(after["coords"])

    # Voxel count ratio
    ratio = n_after / max(n_before, 1)
    min_r, max_r = c.get("min_voxel_ratio", 0.3), c.get("max_voxel_ratio", 3.0)
    passed = min_r <= ratio <= max_r
    reason = "" if passed else f"voxel ratio {ratio:.3f} outside [{min_r},{max_r}]"
    results.append(MetricResult("voxel_ratio", ratio, passed, 1.5, reason))

    # SS cosine similarity (structural preservation)
    sim = ss_cosine_sim(original["ss"], after["ss"])
    min_sim = c.get("min_ss_cosine", 0.3)
    passed = sim >= min_sim
    reason = "" if passed else f"ss cosine sim {sim:.3f} < {min_sim}"
    results.append(MetricResult("ss_cosine_sim", sim, passed, 2.0, reason))

    # Edit locality (voxel diff ratio)
    diff_r = voxel_diff_ratio(original["coords"], after["coords"])
    min_loc = c.get("min_edit_locality", 0.02)
    max_loc = c.get("max_edit_locality", 0.70)
    passed = min_loc <= diff_r <= max_loc
    reason = "" if passed else f"edit locality {diff_r:.3f} outside [{min_loc},{max_loc}]"
    results.append(MetricResult("edit_locality", diff_r, passed, 2.5, reason))

    # Connected components
    n_comp = connected_components_voxel(after["coords"])
    max_comp = c.get("max_components", 5)
    passed = n_comp <= max_comp
    reason = "" if passed else f"{n_comp} components > {max_comp}"
    results.append(MetricResult("connected_components", float(n_comp), passed, 1.0, reason))

    # Center drift
    _, _, center_b, diag_b = bbox_from_coords(original["coords"])
    _, _, center_a, _ = bbox_from_coords(after["coords"])
    drift = float(np.linalg.norm(center_a - center_b))
    drift_ratio = drift / max(diag_b, 1e-8)
    max_drift = c.get("max_center_drift", 0.3)
    passed = drift_ratio < max_drift
    reason = "" if passed else f"center drift {drift_ratio:.3f} >= {max_drift}"
    results.append(MetricResult("center_drift", drift_ratio, passed, 1.5, reason))

    # Feature distribution change
    fr = feat_change_ratio(original["feats"], after["feats"])
    max_kl = c.get("max_feat_kl", 5.0)
    passed = fr < max_kl
    reason = "" if passed else f"feat change {fr:.3f} >= {max_kl}"
    results.append(MetricResult("feat_change", fr, passed, 1.0, reason))

    return results


def check_scale(
    original: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Scale: TRELLIS S1+S2, size change concentrated on one part."""
    c = cfg or {}
    results = []
    n_before = len(original["coords"])
    n_after = len(after["coords"])

    # Voxel count ratio (tighter than modification)
    ratio = n_after / max(n_before, 1)
    min_r, max_r = c.get("min_voxel_ratio", 0.5), c.get("max_voxel_ratio", 2.0)
    passed = min_r <= ratio <= max_r
    reason = "" if passed else f"voxel ratio {ratio:.3f} outside [{min_r},{max_r}]"
    results.append(MetricResult("voxel_ratio", ratio, passed, 2.0, reason))

    # SS cosine similarity (higher bar than modification)
    sim = ss_cosine_sim(original["ss"], after["ss"])
    min_sim = c.get("min_ss_cosine", 0.5)
    passed = sim >= min_sim
    reason = "" if passed else f"ss cosine sim {sim:.3f} < {min_sim}"
    results.append(MetricResult("ss_cosine_sim", sim, passed, 2.5, reason))

    # Edit locality
    diff_r = voxel_diff_ratio(original["coords"], after["coords"])
    min_loc = c.get("min_edit_locality", 0.01)
    max_loc = c.get("max_edit_locality", 0.50)
    passed = min_loc <= diff_r <= max_loc
    reason = "" if passed else f"edit locality {diff_r:.3f} outside [{min_loc},{max_loc}]"
    results.append(MetricResult("edit_locality", diff_r, passed, 2.0, reason))

    # Center drift (tighter)
    _, _, center_b, diag_b = bbox_from_coords(original["coords"])
    _, _, center_a, _ = bbox_from_coords(after["coords"])
    drift = float(np.linalg.norm(center_a - center_b))
    drift_ratio = drift / max(diag_b, 1e-8)
    max_drift = c.get("max_center_drift", 0.2)
    passed = drift_ratio < max_drift
    reason = "" if passed else f"center drift {drift_ratio:.3f} >= {max_drift}"
    results.append(MetricResult("center_drift", drift_ratio, passed, 1.5, reason))

    # Per-axis bbox ratio
    lo_b, hi_b, _, _ = bbox_from_coords(original["coords"])
    lo_a, hi_a, _, _ = bbox_from_coords(after["coords"])
    ext_b = np.maximum(hi_b - lo_b, 1.0)
    ext_a = np.maximum(hi_a - lo_a, 1.0)
    axis_ratios = ext_a / ext_b
    max_axis = float(axis_ratios.max())
    min_axis = float(axis_ratios.min())
    min_ar = c.get("min_bbox_axis_ratio", 0.7)
    max_ar = c.get("max_bbox_axis_ratio", 1.8)
    passed = min_ar <= min_axis and max_axis <= max_ar
    reason = ""
    if not passed:
        reason = f"axis ratios [{min_axis:.2f},{max_axis:.2f}] outside [{min_ar},{max_ar}]"
    results.append(MetricResult("bbox_axis_ratio", max_axis, passed, 1.5, reason))

    return results


def check_material(
    original: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Material: S2-only, coords and SS must be identical, only feats change."""
    c = cfg or {}
    results = []

    # Coords must match exactly
    if c.get("require_coords_match", True):
        coords_match = (
            original["coords"].shape == after["coords"].shape
            and bool(np.array_equal(original["coords"], after["coords"]))
        )
        results.append(MetricResult(
            "coords_match", 1.0 if coords_match else 0.0, coords_match, 3.0,
            "" if coords_match else "coords differ (S2-only should preserve geometry)"
        ))

    # Voxel count must be identical
    n_before = len(original["coords"])
    n_after = len(after["coords"])
    count_match = n_before == n_after
    results.append(MetricResult(
        "voxel_count_match", 1.0 if count_match else 0.0, count_match, 3.0,
        "" if count_match else f"voxel count {n_before} vs {n_after}"
    ))

    # SS must match (within tolerance)
    if c.get("require_ss_match", True):
        tol = c.get("ss_match_tol", 1e-4)
        ss_diff = float(np.abs(original["ss"].astype(np.float64) -
                                after["ss"].astype(np.float64)).max())
        ss_match = ss_diff < tol
        results.append(MetricResult(
            "ss_match", ss_diff, ss_match, 3.0,
            "" if ss_match else f"ss max diff {ss_diff:.6f} >= {tol}"
        ))

    # Feature change should exist
    if count_match:
        fr = feat_change_ratio(original["feats"], after["feats"])
        min_fc = c.get("min_feat_change", 0.01)
        max_fc = c.get("max_feat_change", 2.0)
        passed = min_fc <= fr <= max_fc
        reason = "" if passed else f"feat change {fr:.4f} outside [{min_fc},{max_fc}]"
        results.append(MetricResult("feat_change", fr, passed, 1.5, reason))

    return results


def check_global(
    original: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Global: S2-only full mask, coords/SS identical, broad feat change."""
    c = cfg or {}
    # Start with material checks (same geometry constraints)
    results = check_material(original, after, cfg)

    # Additionally: change should be widespread (global style, not local)
    n_before = len(original["coords"])
    n_after = len(after["coords"])
    if n_before == n_after:
        coverage = feat_change_coverage(original["feats"], after["feats"])
        min_cov = c.get("min_change_coverage", 0.3)
        passed = coverage >= min_cov
        reason = "" if passed else f"change coverage {coverage:.3f} < {min_cov}"
        results.append(MetricResult("change_coverage", coverage, passed, 1.5, reason))

    return results


def check_identity(
    original: dict[str, np.ndarray],
) -> list[MetricResult]:
    """Identity: no-op, data itself should be sane (no pair comparison needed).

    In the object-centric format, identity edits reference original.npz for
    both before and after, so they're identical by construction.
    """
    return [MetricResult("identity_valid", 1.0, True, 1.0)]


# =========================================================================
# Unified dispatcher
# =========================================================================

_CHECKERS = {
    "deletion": lambda orig, after, cfg: check_deletion(orig, after, cfg),
    "addition": lambda orig, after, cfg: check_addition(after, orig, cfg),
    "modification": lambda orig, after, cfg: check_modification(orig, after, cfg),
    "scale": lambda orig, after, cfg: check_scale(orig, after, cfg),
    "material": lambda orig, after, cfg: check_material(orig, after, cfg),
    "global": lambda orig, after, cfg: check_global(orig, after, cfg),
    "identity": lambda orig, after, cfg: check_identity(orig),
}


def check_pair(
    edit_type: str,
    original_data: dict[str, np.ndarray],
    after_data: Optional[dict[str, np.ndarray]],
    cfg: Optional[dict] = None,
) -> list[MetricResult]:
    """Dispatch to the appropriate type-specific checker.

    Args:
        edit_type: One of deletion/addition/modification/scale/material/global/identity.
        original_data: Arrays from original.npz (coords, feats, ss).
        after_data: Arrays from the type-specific NPZ (or deletion's NPZ for addition).
                    None for identity.
        cfg: Type-specific config dict (e.g. cfg["cleaning"]["deletion"]).
    """
    checker = _CHECKERS.get(edit_type)
    if checker is None:
        return [MetricResult("unknown_type", 0.0, False, 1.0,
                             f"unknown edit type: {edit_type}")]
    return checker(original_data, after_data, cfg)
