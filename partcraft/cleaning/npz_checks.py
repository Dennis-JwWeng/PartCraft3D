"""Layer 1: NPZ sanity checks for individual SLAT/SS files.

Each check returns a MetricResult. All checks operate purely on numpy
arrays loaded from .npz files (keys: slat_coords, slat_feats, ss).

Supports ``require_ss=False`` mode for legacy data that only has
``feats.pt`` + ``coords.pt`` without SS latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class MetricResult:
    """Result of a single quality metric."""
    name: str
    value: float
    passed: bool
    weight: float = 1.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_voxel_count(
    coords: np.ndarray,
    min_voxels: int = 100,
    max_voxels: int = 40000,
) -> MetricResult:
    """Reject degenerate (too few) or abnormally inflated (too many) voxel grids."""
    n = len(coords)
    passed = min_voxels <= n <= max_voxels
    reason = ""
    if n < min_voxels:
        reason = f"too few voxels ({n} < {min_voxels})"
    elif n > max_voxels:
        reason = f"too many voxels ({n} > {max_voxels})"
    return MetricResult("voxel_count", float(n), passed, weight=2.0, reason=reason)


def check_feat_range(
    feats: np.ndarray,
    max_abs: float = 50.0,
    min_std: float = 0.01,
) -> MetricResult:
    """Detect NaN/Inf, exploded values, or constant (dead) features."""
    if np.any(~np.isfinite(feats)):
        return MetricResult("feat_range", 0.0, False, weight=3.0,
                            reason="NaN or Inf in slat_feats")
    abs_max = float(np.abs(feats).max())
    std = float(feats.std())
    if abs_max > max_abs:
        return MetricResult("feat_range", abs_max, False, weight=3.0,
                            reason=f"feat abs max {abs_max:.2f} > {max_abs}")
    if std < min_std:
        return MetricResult("feat_range", std, False, weight=2.0,
                            reason=f"feat std {std:.4f} < {min_std} (constant)")
    return MetricResult("feat_range", abs_max, True, weight=2.0)


def check_ss_range(
    ss: np.ndarray,
    max_abs: float = 100.0,
    min_std: float = 0.001,
) -> MetricResult:
    """Detect NaN/Inf, exploded, or all-zero SS latents."""
    if np.any(~np.isfinite(ss)):
        return MetricResult("ss_range", 0.0, False, weight=3.0,
                            reason="NaN or Inf in ss")
    abs_max = float(np.abs(ss).max())
    std = float(ss.std())
    if abs_max > max_abs:
        return MetricResult("ss_range", abs_max, False, weight=3.0,
                            reason=f"ss abs max {abs_max:.2f} > {max_abs}")
    if std < min_std:
        return MetricResult("ss_range", std, False, weight=2.0,
                            reason=f"ss std {std:.6f} < {min_std} (dead)")
    return MetricResult("ss_range", abs_max, True, weight=2.0)


def check_coords_valid(
    coords: np.ndarray,
    spatial_range: int = 64,
) -> MetricResult:
    """Ensure coords are in valid [0, spatial_range) and batch_idx >= 0."""
    if coords.shape[1] != 4:
        return MetricResult("coords_valid", 0.0, False, weight=3.0,
                            reason=f"coords shape[1]={coords.shape[1]}, expected 4")
    batch_ok = bool(np.all(coords[:, 0] >= 0))
    xyz = coords[:, 1:]
    xyz_ok = bool(np.all(xyz >= 0) and np.all(xyz < spatial_range))
    passed = batch_ok and xyz_ok
    reason = ""
    if not batch_ok:
        reason = "negative batch index"
    elif not xyz_ok:
        lo, hi = int(xyz.min()), int(xyz.max())
        reason = f"xyz out of [0,{spatial_range}): range [{lo},{hi}]"
    return MetricResult("coords_valid", 1.0 if passed else 0.0, passed,
                        weight=3.0, reason=reason)


def check_coords_unique(coords: np.ndarray) -> MetricResult:
    """All voxel coordinates must be unique."""
    n = len(coords)
    n_unique = len(np.unique(coords, axis=0))
    passed = n_unique == n
    reason = "" if passed else f"{n - n_unique} duplicate coords out of {n}"
    return MetricResult("coords_unique", float(n_unique) / max(n, 1), passed,
                        weight=2.0, reason=reason)


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

def check_npz_sanity(
    npz_path: str,
    *,
    min_voxels: int = 100,
    max_voxels: int = 40000,
    max_feat_abs: float = 50.0,
    min_feat_std: float = 0.01,
    max_ss_abs: float = 100.0,
    min_ss_std: float = 0.001,
    spatial_range: int = 64,
    require_ss: bool = True,
) -> list[MetricResult]:
    """Run all Layer-1 sanity checks on a single NPZ file.

    Args:
        require_ss: If False, skip SS checks (for legacy data without SS).

    Returns a list of MetricResult (one per check).
    Raises FileNotFoundError / ValueError on missing keys.
    """
    data = np.load(npz_path)
    required = {"slat_coords", "slat_feats"}
    if require_ss:
        required.add("ss")
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"NPZ {npz_path} missing keys: {missing}")

    coords = data["slat_coords"]
    feats = data["slat_feats"]

    results = [
        check_voxel_count(coords, min_voxels, max_voxels),
        check_feat_range(feats, max_feat_abs, min_feat_std),
    ]
    if require_ss:
        results.append(check_ss_range(data["ss"], max_ss_abs, min_ss_std))
    results.extend([
        check_coords_valid(coords, spatial_range),
        check_coords_unique(coords),
    ])
    return results


def load_npz_arrays(npz_path: str, *, require_ss: bool = True) -> dict[str, np.ndarray]:
    """Load and validate arrays from an NPZ file.

    Args:
        require_ss: If False, ``ss`` key is optional; missing SS yields None.
    """
    data = np.load(npz_path)
    required = {"slat_coords", "slat_feats"}
    if require_ss:
        required.add("ss")
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"NPZ {npz_path} missing keys: {missing}")
    result = {
        "coords": data["slat_coords"],
        "feats": data["slat_feats"],
    }
    if "ss" in data:
        result["ss"] = data["ss"]
    else:
        result["ss"] = None
    return result


def load_slat_dir_arrays(slat_dir: str | Path) -> dict[str, np.ndarray]:
    """Load arrays from legacy ``*_slat/feats.pt`` + ``coords.pt`` directory.

    Returns dict with keys ``coords``, ``feats``, ``ss`` (always None).
    """
    slat_dir = Path(slat_dir)
    feats_path = slat_dir / "feats.pt"
    coords_path = slat_dir / "coords.pt"
    if not feats_path.exists() or not coords_path.exists():
        raise FileNotFoundError(f"Missing feats.pt or coords.pt in {slat_dir}")
    feats = torch.load(feats_path, weights_only=True)
    coords = torch.load(coords_path, weights_only=True)
    return {
        "coords": coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else coords,
        "feats": feats.detach().cpu().numpy() if isinstance(feats, torch.Tensor) else feats,
        "ss": None,
    }
