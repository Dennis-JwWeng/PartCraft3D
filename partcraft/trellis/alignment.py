"""Geometric alignment utilities for part swapping."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _find_contact_region(part: trimesh.Trimesh, body: trimesh.Trimesh,
                         distance_threshold: float = 0.02,
                         n_samples: int = 2000) -> np.ndarray | None:
    """Find the region of `part` that is closest to `body` (the connection surface).

    Returns the centroid of the contact region, or None if no contact found.
    """
    try:
        part_points = part.sample(n_samples)
        body_tree = cKDTree(body.vertices)
        dists, _ = body_tree.query(part_points)

        # Contact points: those within threshold distance of the body
        # Use adaptive threshold based on part size
        part_extent = part.bounding_box.extents.max()
        threshold = max(distance_threshold, part_extent * 0.05)

        contact_mask = dists < threshold
        if contact_mask.sum() < 10:
            # Fallback: use the closest 10% of points
            k = max(10, n_samples // 10)
            contact_mask = np.zeros(n_samples, dtype=bool)
            contact_mask[np.argsort(dists)[:k]] = True

        return part_points[contact_mask].mean(axis=0)
    except Exception:
        return None


def align_part_to_target(
    new_part: trimesh.Trimesh,
    old_part: trimesh.Trimesh,
    body: trimesh.Trimesh | None = None,
    strategy: str = "bbox",
) -> tuple[trimesh.Trimesh, float]:
    """Align new_part to match old_part's position and approximate size.

    Args:
        new_part: The replacement part mesh (will be copied, not modified in-place).
        old_part: The original part mesh (reference).
        body: The remaining body mesh (all parts except old_part). If provided,
              used for contact-aware alignment.
        strategy: "bbox" for bounding-box alignment, "centroid" for center-only,
                  "contact" for contact-surface-aware alignment (requires body).

    Returns:
        (aligned_mesh, scale_ratio) where scale_ratio = old_extent / new_extent.
    """
    new_part = new_part.copy()

    old_center = old_part.bounding_box.centroid
    old_extents = old_part.bounding_box.extents
    new_center = new_part.bounding_box.centroid
    new_extents = new_part.bounding_box.extents + 1e-8

    # Step 1: Scale to match old part's size (uniform scaling)
    scale = float(np.min(old_extents / new_extents))
    scale_ratio = float(np.max(old_extents / new_extents))

    if strategy in ("bbox", "contact"):
        new_part.apply_scale(scale)
        new_center = new_part.bounding_box.centroid

    if strategy == "contact" and body is not None:
        # Contact-aware: align the connection surfaces instead of centers
        old_contact = _find_contact_region(old_part, body)
        if old_contact is not None:
            new_contact = _find_contact_region(new_part, new_part)  # self-contact = bottom/top
            if new_contact is None:
                new_contact = new_part.bounding_box.centroid
            # Translate so new_part's contact region aligns with old_part's contact region
            new_part.apply_translation(old_contact - new_contact)
        else:
            # Fallback to center alignment
            new_part.apply_translation(old_center - new_center)

    elif strategy == "bbox":
        new_part.apply_translation(old_center - new_center)

    elif strategy == "centroid":
        new_part.apply_translation(old_center - new_center)

    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")

    return new_part, scale_ratio


def compute_penetration_ratio(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                               n_samples: int = 1000) -> float:
    """Estimate how much mesh_a penetrates into mesh_b.

    Returns fraction of mesh_a surface points that are inside mesh_b.
    """
    if not mesh_b.is_watertight:
        return 0.0

    try:
        points = mesh_a.sample(n_samples)
        inside = mesh_b.contains(points)
        return float(inside.sum()) / n_samples
    except Exception:
        return 0.0


def compute_gap_distance(part: trimesh.Trimesh, body: trimesh.Trimesh,
                          n_samples: int = 1000) -> float:
    """Compute the minimum average distance between part and body surfaces.

    A small value means the part sits close to the body (good).
    A large value means the part is floating (bad).
    """
    try:
        part_points = part.sample(n_samples)
        body_tree = cKDTree(body.vertices)
        dists, _ = body_tree.query(part_points)
        # Use 10th percentile as "contact distance" (robust to outliers)
        return float(np.percentile(dists, 10))
    except Exception:
        return float("inf")
