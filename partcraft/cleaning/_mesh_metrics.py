"""Phase 3: Quality scoring & filtering for assembled mesh pairs.

Evaluates geometric, topological, and semantic quality of before/after mesh pairs.
Designed for deletion/addition pairs, with extension points for modification/swap.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from tqdm import tqdm


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Result of a single quality metric."""
    name: str
    value: float
    passed: bool
    weight: float = 1.0       # importance weight for composite score
    reason: str = ""          # human-readable explanation if failed


@dataclass
class QualityReport:
    """Full quality report for one edit pair."""
    edit_id: str
    edit_type: str
    passed: bool
    score: float                            # weighted composite score [0, 1]
    metrics: list[MetricResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "edit_id": self.edit_id,
            "edit_type": self.edit_type,
            "passed": self.passed,
            "score": round(self.score, 4),
            "metrics": {m.name: {"value": round(m.value, 6), "passed": m.passed,
                                  "reason": m.reason} for m in self.metrics},
        }


# ---------------------------------------------------------------------------
# Individual metrics — each returns a MetricResult
# ---------------------------------------------------------------------------

def metric_volume_ratio(before: trimesh.Trimesh, after: trimesh.Trimesh,
                        cfg: dict) -> MetricResult:
    """Bounding-box volume ratio: after/before should be in a reasonable range.

    Too small → most of the object was removed.
    Too large → something was added that dwarfs the original.
    """
    p = cfg["phase3"]
    bv = before.bounding_box.volume if before.bounding_box is not None else 0
    av = after.bounding_box.volume if after.bounding_box is not None else 0
    ratio = av / bv if bv > 0 else 0.0

    lo, hi = p["min_volume_ratio"], p["max_volume_ratio"]
    passed = lo <= ratio <= hi
    reason = "" if passed else f"volume ratio {ratio:.3f} outside [{lo}, {hi}]"
    return MetricResult("volume_ratio", ratio, passed, weight=1.5, reason=reason)


def metric_surface_area_ratio(before: trimesh.Trimesh, after: trimesh.Trimesh,
                              cfg: dict) -> MetricResult:
    """Surface area ratio: complementary to volume — catches thin/flat edits."""
    p = cfg["phase3"]
    ba = before.area if before.area > 0 else 1e-8
    aa = after.area
    ratio = aa / ba

    lo = p.get("min_area_ratio", 0.1)
    hi = p.get("max_area_ratio", 8.0)
    passed = lo <= ratio <= hi
    reason = "" if passed else f"area ratio {ratio:.3f} outside [{lo}, {hi}]"
    return MetricResult("surface_area_ratio", ratio, passed, weight=1.0, reason=reason)


def metric_edit_ratio(before: trimesh.Trimesh, after: trimesh.Trimesh,
                      cfg: dict) -> MetricResult:
    """Geometric edit ratio: fraction of vertices that changed.

    Uses symmetric difference of rounded vertex sets.
    Too low = trivial/no-op edit.  Too high = near-total replacement.
    """
    p = cfg["phase3"]
    precision = p.get("edit_ratio_precision", 4)
    bv = set(map(tuple, np.round(before.vertices, precision).tolist()))
    av = set(map(tuple, np.round(after.vertices, precision).tolist()))

    if not bv and not av:
        return MetricResult("edit_ratio", 0.0, False, weight=2.0,
                            reason="both meshes empty")

    sym_diff = len(bv.symmetric_difference(av))
    ratio = sym_diff / max(len(bv), len(av))

    lo, hi = p["min_edit_ratio"], p["max_edit_ratio"]
    passed = lo <= ratio <= hi
    reason = "" if passed else f"edit ratio {ratio:.3f} outside [{lo}, {hi}]"
    return MetricResult("edit_ratio", ratio, passed, weight=2.0, reason=reason)


def metric_part_proportion(before: trimesh.Trimesh, after: trimesh.Trimesh,
                           edit_type: str, cfg: dict) -> MetricResult:
    """Edited part size relative to the full object.

    For deletion: removed part = before - after (face count).
    For addition: added part = after - before (face count).
    Part shouldn't be too tiny (noise) or too dominant (removing the whole object).
    """
    p = cfg["phase3"]
    bf, af = len(before.faces), len(after.faces)
    full_faces = max(bf, af)

    if full_faces == 0:
        return MetricResult("part_proportion", 0.0, False, weight=1.5,
                            reason="zero faces")

    if edit_type == "deletion":
        # before = full, after = reduced → diff is the deleted part
        # Note: full.ply vs concatenated parts may have different face counts,
        # so use absolute difference
        part_faces = abs(bf - af)
    elif edit_type == "addition":
        part_faces = abs(af - bf)
    else:
        # For modification: compare face counts as a rough proxy
        part_faces = abs(bf - af)

    proportion = part_faces / full_faces

    lo = p.get("min_part_proportion", 0.005)
    hi = p.get("max_part_proportion", 0.80)
    passed = lo <= proportion <= hi
    reason = "" if passed else f"part proportion {proportion:.3f} outside [{lo}, {hi}]"
    return MetricResult("part_proportion", proportion, passed, weight=1.5, reason=reason)


def metric_connected_components(before: trimesh.Trimesh, after: trimesh.Trimesh,
                                cfg: dict) -> MetricResult:
    """After mesh shouldn't be overly fragmented."""
    p = cfg["phase3"]
    max_comp = p.get("max_components", 5)

    try:
        n = len(after.split())
    except Exception:
        n = 1

    passed = n <= max_comp
    reason = "" if passed else f"{n} components > limit {max_comp}"
    return MetricResult("connected_components", float(n), passed, weight=1.0,
                        reason=reason)


def metric_has_geometry(before: trimesh.Trimesh, after: trimesh.Trimesh,
                        cfg: dict) -> MetricResult:
    """Both meshes must have meaningful geometry."""
    min_verts = cfg["phase3"].get("min_vertices", 50)
    bv, av = len(before.vertices), len(after.vertices)
    passed = bv >= min_verts and av >= min_verts
    reason = "" if passed else f"before={bv}v, after={av}v (min={min_verts})"
    return MetricResult("has_geometry", float(min(bv, av)), passed, weight=2.0,
                        reason=reason)


def metric_not_degenerate(before: trimesh.Trimesh, after: trimesh.Trimesh,
                          cfg: dict) -> MetricResult:
    """No degenerate bounding box dimensions (flat/line meshes)."""
    threshold = cfg["phase3"].get("min_extent", 1e-4)

    for label, mesh in [("before", before), ("after", after)]:
        if mesh.bounding_box is not None:
            ext = mesh.bounding_box.extents.min()
            if ext < threshold:
                return MetricResult("not_degenerate", float(ext), False, weight=1.0,
                                    reason=f"{label} min extent {ext:.6f} < {threshold}")

    return MetricResult("not_degenerate", 1.0, True, weight=1.0)


def metric_center_drift(before: trimesh.Trimesh, after: trimesh.Trimesh,
                        cfg: dict) -> MetricResult:
    """After mesh center shouldn't drift far from before mesh center.

    Normalized by before mesh diagonal. Catches misaligned swaps/grafts.
    """
    max_drift = cfg["phase3"].get("max_center_drift", 0.3)

    if before.bounding_box is None or after.bounding_box is None:
        return MetricResult("center_drift", 0.0, True, weight=1.0)

    diag = np.linalg.norm(before.bounding_box.bounds[1] - before.bounding_box.bounds[0])
    drift = np.linalg.norm(after.bounding_box.centroid - before.bounding_box.centroid)
    ratio = drift / max(diag, 1e-8)

    passed = ratio < max_drift
    reason = "" if passed else f"center drift {ratio:.3f} > {max_drift}"
    return MetricResult("center_drift", ratio, passed, weight=1.0, reason=reason)


def metric_vertex_color_preserved(before: trimesh.Trimesh, after: trimesh.Trimesh,
                                  cfg: dict) -> MetricResult:
    """Check that vertex colors exist in both meshes (not lost during assembly)."""
    if not cfg["phase3"].get("check_vertex_color", True):
        return MetricResult("vertex_color", 1.0, True, weight=0.5)

    def _has_color(m: trimesh.Trimesh) -> bool:
        if m.visual is None:
            return False
        vc = m.visual.vertex_colors
        if vc is None or len(vc) == 0:
            return False
        # Check not all same color (gray fallback)
        if vc.shape[1] >= 3:
            rgb = vc[:, :3]
            return rgb.std() > 1.0  # some variation
        return True

    bc = _has_color(before)
    ac = _has_color(after)
    passed = bc and ac
    reason = "" if passed else f"color: before={bc}, after={ac}"
    return MetricResult("vertex_color", float(bc and ac), passed, weight=0.5,
                        reason=reason)


def metric_bbox_overlap(before: trimesh.Trimesh, after: trimesh.Trimesh,
                        cfg: dict) -> MetricResult:
    """Bounding box IoU between before and after.

    For deletion/addition the after should largely overlap with before.
    """
    min_iou = cfg["phase3"].get("min_bbox_iou", 0.1)

    if before.bounding_box is None or after.bounding_box is None:
        return MetricResult("bbox_overlap", 0.0, False, weight=1.0,
                            reason="missing bounding box")

    b_min, b_max = before.bounding_box.bounds
    a_min, a_max = after.bounding_box.bounds

    inter_min = np.maximum(b_min, a_min)
    inter_max = np.minimum(b_max, a_max)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = inter_dims.prod()

    bv = np.prod(b_max - b_min)
    av = np.prod(a_max - a_min)
    union_vol = bv + av - inter_vol
    iou = inter_vol / max(union_vol, 1e-12)

    passed = iou >= min_iou
    reason = "" if passed else f"bbox IoU {iou:.3f} < {min_iou}"
    return MetricResult("bbox_overlap", iou, passed, weight=1.0, reason=reason)


# ---------------------------------------------------------------------------
# Modification/swap specific metrics (extension point)
# ---------------------------------------------------------------------------

def metric_penetration(new_part: trimesh.Trimesh, body: trimesh.Trimesh,
                       cfg: dict) -> MetricResult:
    """New part shouldn't be mostly inside the body mesh. For swap/graft."""
    max_pen = cfg["phase3"].get("max_penetration", 0.3)
    try:
        from partcraft.trellis.alignment import compute_penetration_ratio
        pen = compute_penetration_ratio(new_part, body)
    except Exception:
        pen = 0.0
    passed = pen <= max_pen
    reason = "" if passed else f"penetration {pen:.1%} > {max_pen:.0%}"
    return MetricResult("penetration", pen, passed, weight=2.0, reason=reason)


def metric_gap_distance(new_part: trimesh.Trimesh, body: trimesh.Trimesh,
                        cfg: dict) -> MetricResult:
    """New part shouldn't float far from the body. For swap/graft."""
    max_gap = cfg["phase3"].get("max_gap_ratio", 0.15)
    if max_gap <= 0:
        return MetricResult("gap_distance", 0.0, True, weight=1.0)
    try:
        from partcraft.trellis.alignment import compute_gap_distance
        gap = compute_gap_distance(new_part, body)
        extent = body.bounding_box.extents.max()
        ratio = gap / max(extent, 1e-8)
    except Exception:
        ratio = 0.0
    passed = ratio <= max_gap
    reason = "" if passed else f"gap {ratio:.3f} > {max_gap}"
    return MetricResult("gap_distance", ratio, passed, weight=1.5, reason=reason)


def metric_scale_match(new_part: trimesh.Trimesh, old_part: trimesh.Trimesh,
                       cfg: dict) -> MetricResult:
    """Scale ratio between swapped parts should be reasonable."""
    max_scale = cfg["phase3"].get("max_scale_ratio", 3.0)
    new_ext = new_part.bounding_box.extents.max() if new_part.bounding_box else 0
    old_ext = old_part.bounding_box.extents.max() if old_part.bounding_box else 0
    if old_ext > 0 and new_ext > 0:
        ratio = max(new_ext / old_ext, old_ext / new_ext)
    else:
        ratio = 999.0
    passed = ratio <= max_scale
    reason = "" if passed else f"scale ratio {ratio:.2f}x > {max_scale}x"
    return MetricResult("scale_match", ratio, passed, weight=1.5, reason=reason)


# ---------------------------------------------------------------------------
# Metric registry — easy to extend
# ---------------------------------------------------------------------------

# Metrics for deletion / addition pairs
DELETION_ADDITION_METRICS: list[Callable] = [
    metric_volume_ratio,
    metric_surface_area_ratio,
    metric_edit_ratio,
    metric_part_proportion,
    metric_connected_components,
    metric_has_geometry,
    metric_not_degenerate,
    metric_center_drift,
    metric_vertex_color_preserved,
    metric_bbox_overlap,
]

# Additional metrics for modification/swap (used on top of common metrics)
MODIFICATION_METRICS: list[Callable] = [
    # These require new_part / old_part / body, invoked separately
    # metric_penetration, metric_gap_distance, metric_scale_match
]


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def weighted_score(results: list[MetricResult]) -> tuple[float, bool]:
    """Compute weighted composite score and all-pass flag.

    Returns (score, all_passed) where score is in [0, 1].
    Shared by mesh-based filter and NPZ-based cleaning.
    """
    if not results:
        return 0.0, True
    total_weight = sum(r.weight for r in results)
    if total_weight <= 0:
        return 0.0, True
    score = sum(r.weight * (1.0 if r.passed else 0.0) for r in results) / total_weight
    passed = all(r.passed for r in results)
    return round(score, 4), passed


def evaluate_pair(edit_id: str, edit_type: str,
                  before: trimesh.Trimesh, after: trimesh.Trimesh,
                  cfg: dict) -> QualityReport:
    """Run all applicable metrics on a before/after mesh pair."""
    results: list[MetricResult] = []

    # Common metrics (work for all edit types)
    for metric_fn in DELETION_ADDITION_METRICS:
        try:
            if metric_fn == metric_part_proportion:
                r = metric_fn(before, after, edit_type, cfg)
            else:
                r = metric_fn(before, after, cfg)
            results.append(r)
        except Exception as e:
            results.append(MetricResult(
                metric_fn.__name__.replace("metric_", ""),
                0.0, False, weight=1.0, reason=f"error: {e}"))

    score, passed = weighted_score(results)

    return QualityReport(
        edit_id=edit_id,
        edit_type=edit_type,
        passed=passed,
        score=score,
        metrics=results,
    )


def evaluate_modification_pair(edit_id: str,
                               before: trimesh.Trimesh, after: trimesh.Trimesh,
                               new_part: trimesh.Trimesh, old_part: trimesh.Trimesh,
                               body: trimesh.Trimesh,
                               cfg: dict) -> QualityReport:
    """Run all metrics including modification-specific ones.

    Extension point for future swap/graft evaluation.
    """
    # Start with common metrics
    report = evaluate_pair(edit_id, "modification", before, after, cfg)

    # Add modification-specific metrics
    extra: list[MetricResult] = []
    try:
        extra.append(metric_penetration(new_part, body, cfg))
    except Exception as e:
        extra.append(MetricResult("penetration", 0.0, False, reason=str(e)))
    try:
        extra.append(metric_gap_distance(new_part, body, cfg))
    except Exception as e:
        extra.append(MetricResult("gap_distance", 0.0, False, reason=str(e)))
    try:
        extra.append(metric_scale_match(new_part, old_part, cfg))
    except Exception as e:
        extra.append(MetricResult("scale_match", 0.0, False, reason=str(e)))

    all_metrics = report.metrics + extra

    # Recompute score with extra metrics
    score, passed = weighted_score(all_metrics)

    return QualityReport(
        edit_id=edit_id,
        edit_type="modification",
        passed=passed,
        score=score,
        metrics=all_metrics,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _evaluate_one(entry: dict, cfg: dict, mesh_dir: Path) -> dict:
    """Evaluate a single manifest entry. Runs in worker process."""
    eid = entry["edit_id"]
    before_path = entry["before_mesh"]
    after_path = entry["after_mesh"]

    try:
        before = trimesh.load(before_path, process=False)
        after = trimesh.load(after_path, process=False)
    except Exception as e:
        return {"edit_id": eid, "error": f"load failed: {e}",
                "passed": False, "score": 0.0}

    report = evaluate_pair(eid, entry["edit_type"], before, after, cfg)
    return report.to_dict()


def run_phase3(cfg: dict, manifest_path: str | None = None,
               limit: int | None = None,
               max_workers: int | None = None) -> tuple[list[dict], list[dict]]:
    """Run Phase 3: quality scoring & filtering on assembled pairs.

    Reads the Phase 2 manifest, evaluates each pair, outputs:
      - Passed pairs manifest (with scores)
      - Failed pairs log (with failure reasons)
      - Per-pair score details
      - Summary statistics

    Returns (passed_entries, failed_entries).
    """
    p3_cfg = cfg.get("phase3", {})
    cache_dir = Path(p3_cfg.get("cache_dir", "cache/phase3"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = str(Path(cfg["phase2"]["cache_dir"]) / "assembled_pairs.jsonl")

    mesh_dir = Path(cfg["data"]["output_dir"]) / "mesh_pairs"

    # Load manifest
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if limit:
        entries = entries[:limit]

    n_workers = max_workers or min(len(entries), os.cpu_count() or 4, 8)
    print(f"Phase 3: Evaluating {len(entries)} pairs ({n_workers} workers)...")

    # Run evaluation
    results: list[dict] = []

    if n_workers <= 1:
        for entry in tqdm(entries, desc="Phase 3: Filter"):
            results.append(_evaluate_one(entry, cfg, mesh_dir))
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for entry in entries:
                fut = pool.submit(_evaluate_one, entry, cfg, mesh_dir)
                futures[fut] = entry["edit_id"]

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Phase 3: Filter"):
                try:
                    results.append(fut.result())
                except Exception as e:
                    eid = futures[fut]
                    results.append({"edit_id": eid, "passed": False, "score": 0.0,
                                    "error": str(e)})

    # Split passed / failed
    passed, failed = [], []
    for r in results:
        if r.get("passed", False):
            passed.append(r)
        else:
            failed.append(r)

    # Sort passed by score descending
    passed.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Merge back manifest info into passed entries
    entry_map = {e["edit_id"]: e for e in entries}
    passed_full = []
    for r in passed:
        eid = r["edit_id"]
        if eid in entry_map:
            merged = {**entry_map[eid], "quality_score": r["score"],
                      "quality_metrics": r.get("metrics", {})}
            passed_full.append(merged)

    # Write outputs
    passed_path = cache_dir / "passed_pairs.jsonl"
    failed_path = cache_dir / "failed_pairs.jsonl"
    scores_path = cache_dir / "all_scores.jsonl"
    summary_path = cache_dir / "summary.json"

    with open(passed_path, "w") as f:
        for entry in passed_full:
            f.write(_json_dumps(entry) + "\n")

    with open(failed_path, "w") as f:
        for r in failed:
            f.write(_json_dumps(r) + "\n")

    with open(scores_path, "w") as f:
        for r in results:
            f.write(_json_dumps(r) + "\n")

    # Summary statistics
    scores = [r["score"] for r in results if "score" in r]
    by_type = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        et = r.get("edit_type", "unknown")
        by_type[et]["total"] += 1
        if r.get("passed"):
            by_type[et]["passed"] += 1

    # Per-metric pass rates
    metric_stats: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        for mname, minfo in r.get("metrics", {}).items():
            metric_stats[mname]["total"] += 1
            if minfo.get("passed"):
                metric_stats[mname]["passed"] += 1

    summary = {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / max(len(results), 1),
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "by_type": dict(by_type),
        "metric_pass_rates": {
            k: {"pass_rate": v["passed"] / max(v["total"], 1),
                "passed": v["passed"], "total": v["total"]}
            for k, v in metric_stats.items()
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Phase 3 Quality Filter Summary")
    print(f"{'='*60}")
    print(f"  Total:  {summary['total']}")
    print(f"  Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"  Failed: {summary['failed']}")
    print(f"  Avg score:    {summary['avg_score']:.3f}")
    print(f"  Median score: {summary['median_score']:.3f}")
    print(f"\n  By edit type:")
    for et, counts in by_type.items():
        pr = counts['passed'] / max(counts['total'], 1)
        print(f"    {et:15s}: {counts['passed']}/{counts['total']} ({pr:.1%})")
    print(f"\n  Per-metric pass rates:")
    for mname, stats in sorted(metric_stats.items()):
        pr = stats['passed'] / max(stats['total'], 1)
        print(f"    {mname:25s}: {stats['passed']}/{stats['total']} ({pr:.1%})")
    print(f"\n  Output: {passed_path}")
    print(f"{'='*60}")

    return passed_full, failed
