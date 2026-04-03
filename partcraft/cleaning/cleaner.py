"""Main cleaning entry point for object-centric training data.

Traverses ``partverse_pairs/shard_XX/{obj_id}/`` directories,
runs Layer 1 (NPZ sanity) + Layer 2 (pair checks) on every edit,
writes per-object ``quality.json`` and global outputs.

Usage (programmatic)::

    from partcraft.cleaning.cleaner import run_cleaning
    run_cleaning(input_dir="partverse_pairs", shards=["00","01"], cfg=cfg)

See also ``scripts/tools/run_cleaning.py`` for CLI.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .npz_checks import (
    MetricResult, check_npz_sanity, load_npz_arrays, load_slat_dir_arrays,
)
from .pair_checks import check_pair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def weighted_score(results: list[MetricResult]) -> tuple[float, bool]:
    """Compute weighted composite score and all-pass flag."""
    if not results:
        return 0.0, True
    total_w = sum(r.weight for r in results)
    if total_w < 1e-12:
        return 0.0, True
    score = sum(r.weight * (1.0 if r.passed else 0.0) for r in results) / total_w
    passed = all(r.passed for r in results)
    return round(score, 4), passed


def classify_tier(score: float, passed: bool, cfg: dict | None = None) -> str:
    """Map score + pass status to quality tier."""
    if not passed:
        return "negative"
    c = cfg or {}
    thresholds = c.get("tier_thresholds", {})
    high = thresholds.get("high", 0.8)
    medium = thresholds.get("medium", 0.6)
    low = thresholds.get("low", 0.4)
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    if score >= low:
        return "low"
    return "negative"


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class CleaningResult:
    edit_id: str
    edit_type: str
    tier: str = "rejected"
    score: float = 0.0
    layer1_passed: bool = False
    layer2_passed: bool = False
    layer1: dict = field(default_factory=dict)
    layer2: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-edit cleaning
# ---------------------------------------------------------------------------

def _metrics_to_dict(results: list[MetricResult]) -> dict:
    """Serialize list of MetricResult to {name: {value, passed, reason}}."""
    return {
        r.name: {"value": r.value, "passed": r.passed, "reason": r.reason}
        for r in results
    }


def _get_type_cfg(cfg: dict, edit_type: str) -> dict:
    """Extract type-specific cleaning config."""
    cleaning = cfg.get("cleaning", {})
    return cleaning.get(edit_type, {})


def _load_after_data(
    obj_dir: Path,
    edit_entry: dict,
    edit_type: str,
    *,
    require_ss: bool,
    mesh_pairs_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray] | None, str]:
    """Resolve and load after-edit data from various formats.

    Returns (data_dict_or_None, error_reason).
    """
    # --- Object-centric layout (shard_XX/{obj_id}/mod_000.npz) ---
    npz_file = edit_entry.get("file")
    if edit_type == "addition":
        source_del_seq = edit_entry.get("source_del_seq", -1)
        if source_del_seq < 0:
            return None, "missing source_del_seq for addition"
        npz_file = f"del_{source_del_seq:03d}.npz"

    if npz_file is not None:
        npz_path = obj_dir / npz_file
        if npz_path.exists():
            return load_npz_arrays(str(npz_path), require_ss=require_ss), ""

    # --- Flat mesh_pairs layout ({edit_id}/after.npz or after_slat/) ---
    edit_id = edit_entry.get("edit_id", "")
    if mesh_pairs_dir is not None and edit_id:
        pair_dir = mesh_pairs_dir / edit_id
        # Try NPZ first
        after_npz = pair_dir / "after.npz"
        if after_npz.exists():
            return load_npz_arrays(str(after_npz), require_ss=require_ss), ""
        # Try legacy *_slat/ directory
        after_slat = pair_dir / "after_slat"
        if after_slat.is_dir():
            return load_slat_dir_arrays(after_slat), ""

    if npz_file is not None:
        return None, f"after data not found: {npz_file}"
    return None, f"no file for edit type {edit_type}"


def _load_original_data(
    obj_dir: Path,
    edit_entry: dict,
    *,
    require_ss: bool,
    mesh_pairs_dir: Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load original (before) data from various formats."""
    # Object-centric layout
    original_path = obj_dir / "original.npz"
    if original_path.exists():
        return load_npz_arrays(str(original_path), require_ss=require_ss)
    # Flat mesh_pairs layout
    edit_id = edit_entry.get("edit_id", "")
    if mesh_pairs_dir is not None and edit_id:
        pair_dir = mesh_pairs_dir / edit_id
        before_npz = pair_dir / "before.npz"
        if before_npz.exists():
            return load_npz_arrays(str(before_npz), require_ss=require_ss)
        before_slat = pair_dir / "before_slat"
        if before_slat.is_dir():
            return load_slat_dir_arrays(before_slat)
    return None


def clean_edit(
    obj_dir: Path,
    edit_entry: dict,
    original_data: dict[str, np.ndarray] | None,
    cfg: dict,
    *,
    require_ss: bool = True,
    mesh_pairs_dir: Path | None = None,
) -> CleaningResult:
    """Run Layer 1 + Layer 2 on a single edit.

    Args:
        require_ss: If False, skip SS-dependent checks (for legacy data).
        mesh_pairs_dir: If set, also search flat ``mesh_pairs/{edit_id}/``
            layout for data files (before_slat/, after_slat/, *.npz).
    """
    edit_id = edit_entry["edit_id"]
    edit_type = edit_entry["type"]
    result = CleaningResult(edit_id=edit_id, edit_type=edit_type)
    cleaning_cfg = cfg.get("cleaning", {})
    type_cfg = _get_type_cfg(cfg, edit_type)

    # --- Identity: trivially valid ---
    if edit_type == "identity":
        if original_data is not None:
            l1 = _run_l1_on_arrays(original_data, cleaning_cfg,
                                   require_ss=require_ss)
            l1_score, l1_passed = weighted_score(l1)
            result.layer1 = _metrics_to_dict(l1)
            result.layer1_passed = l1_passed
        else:
            result.layer1_passed = True
        result.layer2_passed = True
        l2 = check_pair("identity", original_data, None, type_cfg,
                        require_ss=require_ss)
        result.layer2 = _metrics_to_dict(l2)
        score = 1.0 if result.layer1_passed else 0.5
        result.score = score
        result.tier = classify_tier(score, result.layer1_passed, cleaning_cfg)
        return result

    # --- Load after data ---
    try:
        after_data, err = _load_after_data(
            obj_dir, edit_entry, edit_type,
            require_ss=require_ss, mesh_pairs_dir=mesh_pairs_dir,
        )
    except Exception as e:
        result.reason = f"Load after error: {e}"
        return result

    if after_data is None:
        result.reason = err or "after data not found"
        return result

    # --- Layer 1: sanity on the after data ---
    l1_after = _run_l1_on_arrays(after_data, cleaning_cfg, require_ss=require_ss)
    l1_score, l1_passed = weighted_score(l1_after)
    result.layer1 = _metrics_to_dict(l1_after)
    result.layer1_passed = l1_passed

    if not l1_passed:
        failed = [r for r in l1_after if not r.passed]
        result.reason = f"Layer1 failed: {failed[0].name} — {failed[0].reason}"
        result.score = l1_score * 0.5
        result.tier = "rejected"
        return result

    # --- Layer 2: Pair comparison ---
    if original_data is None:
        result.reason = "original data not loaded"
        return result

    try:
        l2 = check_pair(edit_type, original_data, after_data, type_cfg,
                        require_ss=require_ss)
    except Exception as e:
        result.reason = f"Layer2 error: {e}"
        return result

    l2_score, l2_passed = weighted_score(l2)
    result.layer2 = _metrics_to_dict(l2)
    result.layer2_passed = l2_passed

    # Composite score: average of L1 and L2
    combined_score = (l1_score + l2_score) / 2.0
    result.score = round(combined_score, 4)
    result.tier = classify_tier(combined_score, l1_passed and l2_passed, cleaning_cfg)
    if not l2_passed:
        failed = [r for r in l2 if not r.passed]
        result.reason = f"Layer2 failed: {failed[0].name} — {failed[0].reason}"

    return result


def _run_l1_on_arrays(
    data: dict[str, np.ndarray],
    cfg: dict,
    *,
    require_ss: bool = True,
) -> list[MetricResult]:
    """Run Layer 1 checks directly on loaded arrays (no file I/O)."""
    from .npz_checks import (
        check_voxel_count, check_feat_range, check_ss_range,
        check_coords_valid, check_coords_unique,
    )
    results = [
        check_voxel_count(data["coords"], cfg.get("min_voxels", 100),
                          cfg.get("max_voxels", 40000)),
        check_feat_range(data["feats"], cfg.get("max_feat_abs", 50.0),
                         cfg.get("min_feat_std", 0.01)),
    ]
    if require_ss and data.get("ss") is not None:
        results.append(
            check_ss_range(data["ss"], cfg.get("max_ss_abs", 100.0),
                           cfg.get("min_ss_std", 0.001)))
    results.extend([
        check_coords_valid(data["coords"]),
        check_coords_unique(data["coords"]),
    ])
    return results


# ---------------------------------------------------------------------------
# Per-object cleaning
# ---------------------------------------------------------------------------

def clean_object(
    obj_dir: Path,
    cfg: dict,
    *,
    require_ss: bool = True,
    mesh_pairs_dir: Path | None = None,
) -> list[CleaningResult]:
    """Clean all edits for one object.

    Args:
        require_ss: If False, skip SS-dependent checks.
        mesh_pairs_dir: If set, also search flat mesh_pairs layout.
    """
    meta_path = obj_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning("No metadata.json in %s, skipping", obj_dir)
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    # Load original data once (shared by all edits)
    original_data = None
    original_path = obj_dir / "original.npz"
    if original_path.exists():
        try:
            original_data = load_npz_arrays(
                str(original_path), require_ss=require_ss)
        except Exception as e:
            logger.warning("Failed to load original.npz in %s: %s", obj_dir, e)

    # Fallback: try loading from first edit's before_slat/ in mesh_pairs
    if original_data is None and mesh_pairs_dir is not None:
        edits = meta.get("edits", [])
        for edit in edits:
            try:
                original_data = _load_original_data(
                    obj_dir, edit, require_ss=require_ss,
                    mesh_pairs_dir=mesh_pairs_dir)
                if original_data is not None:
                    break
            except Exception:
                continue

    results = []
    for edit in meta.get("edits", []):
        r = clean_edit(obj_dir, edit, original_data, cfg,
                       require_ss=require_ss, mesh_pairs_dir=mesh_pairs_dir)
        results.append(r)

    # Write quality.json
    quality = {
        "obj_id": meta["obj_id"],
        "shard": meta.get("shard", ""),
        "num_edits": len(results),
        "num_passed": sum(1 for r in results if r.tier in ("high", "medium")),
        "edits": [r.to_dict() for r in results],
    }
    quality_path = obj_dir / "quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)

    return results


def _clean_object_worker(args: tuple) -> tuple[str, list[dict]]:
    """Top-level pickleable worker for ProcessPoolExecutor."""
    obj_dir_str, cfg, require_ss, mesh_pairs_str = args
    obj_dir = Path(obj_dir_str)
    mesh_pairs_dir = Path(mesh_pairs_str) if mesh_pairs_str else None
    results = clean_object(obj_dir, cfg, require_ss=require_ss,
                           mesh_pairs_dir=mesh_pairs_dir)
    obj_id = obj_dir.name
    return obj_id, [r.to_dict() for r in results]


# ---------------------------------------------------------------------------
# Per-shard / full cleaning
# ---------------------------------------------------------------------------

def clean_shard(
    shard_dir: Path,
    cfg: dict,
    workers: int = 1,
    *,
    require_ss: bool = True,
    mesh_pairs_dir: Path | None = None,
) -> dict:
    """Clean all objects in a shard directory.

    Returns summary dict with counts by tier and edit type.
    """
    obj_dirs = sorted([
        d for d in shard_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])
    if not obj_dirs:
        logger.warning("No objects found in %s", shard_dir)
        return {"total": 0}

    shard = shard_dir.name.replace("shard_", "")
    logger.info("Cleaning shard %s: %d objects, %d workers, require_ss=%s",
                shard, len(obj_dirs), workers, require_ss)

    all_results: list[dict] = []
    mp_str = str(mesh_pairs_dir) if mesh_pairs_dir else ""

    if workers <= 1:
        for obj_dir in tqdm(obj_dirs, desc=f"shard_{shard}"):
            results = clean_object(obj_dir, cfg, require_ss=require_ss,
                                   mesh_pairs_dir=mesh_pairs_dir)
            all_results.extend(r.to_dict() for r in results)
    else:
        tasks = [(str(d), cfg, require_ss, mp_str) for d in obj_dirs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_clean_object_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"shard_{shard}"):
                try:
                    _, results_dicts = fut.result()
                    all_results.extend(results_dicts)
                except Exception as e:
                    obj_path = futures[fut]
                    logger.error("Worker failed for %s: %s", obj_path, e)

    # Build summary
    summary = _build_summary(all_results, shard)
    return summary


def _build_summary(results: list[dict], shard: str = "") -> dict:
    """Aggregate cleaning results into a summary."""
    tier_counts: dict[str, int] = defaultdict(int)
    type_tier_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    type_counts: dict[str, int] = defaultdict(int)
    fail_reasons: dict[str, int] = defaultdict(int)

    for r in results:
        tier = r.get("tier", "rejected")
        etype = r.get("edit_type", "unknown")
        tier_counts[tier] += 1
        type_tier_counts[etype][tier] += 1
        type_counts[etype] += 1
        reason = r.get("reason", "")
        if reason:
            # Truncate reason to first 80 chars for grouping
            key = reason[:80]
            fail_reasons[key] += 1

    return {
        "shard": shard,
        "total": len(results),
        "by_tier": dict(tier_counts),
        "by_type": dict(type_counts),
        "by_type_tier": {k: dict(v) for k, v in type_tier_counts.items()},
        "top_fail_reasons": dict(
            sorted(fail_reasons.items(), key=lambda x: -x[1])[:20]
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cleaning(
    input_dir: str,
    cfg: dict,
    shards: list[str] | None = None,
    edit_types: set[str] | None = None,
    workers: int = 1,
    min_tier: str = "medium",
    dry_run: bool = False,
    *,
    require_ss: bool = True,
    mesh_pairs_dir: str | Path | None = None,
) -> Path:
    """Run cleaning on object-centric training data.

    Args:
        input_dir: Root of repacked data (contains shard_XX dirs).
        cfg: Full pipeline config dict (uses cfg["cleaning"]).
        shards: Which shards to process (default: all found).
        edit_types: If set, only clean these edit types.
        workers: Parallel workers per shard.
        min_tier: Minimum tier for manifest_clean.jsonl.
        dry_run: If True, only scan and report, don't write quality.json.
        require_ss: If False, skip SS-dependent checks (for legacy data).
        mesh_pairs_dir: If set, also search flat mesh_pairs layout for data.

    Returns:
        Path to the generated cleaning_summary.json.
    """
    root = Path(input_dir)
    mp_dir = Path(mesh_pairs_dir) if mesh_pairs_dir else None
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    # Discover shard directories
    shard_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("shard_")
    ])
    if shards:
        shard_set = set(shards)
        shard_dirs = [d for d in shard_dirs
                      if d.name.replace("shard_", "") in shard_set]

    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found in {root}")

    logger.info("Cleaning %d shards in %s", len(shard_dirs), root)

    # Process each shard — write per-shard manifests independently
    tier_order = {"high": 0, "medium": 1, "low": 2, "negative": 3, "rejected": 4}
    min_tier_val = tier_order.get(min_tier, 1)
    all_summaries = []
    all_results: list[dict] = []

    for shard_dir in shard_dirs:
        summary = clean_shard(shard_dir, cfg, workers=workers,
                              require_ss=require_ss, mesh_pairs_dir=mp_dir)
        all_summaries.append(summary)

        # Collect results for this shard
        shard_results: list[dict] = []
        for obj_dir in sorted(shard_dir.iterdir()):
            quality_path = obj_dir / "quality.json"
            if not quality_path.exists():
                continue
            with open(quality_path) as f:
                q = json.load(f)
            for edit in q.get("edits", []):
                if edit_types and edit.get("edit_type") not in edit_types:
                    continue
                edit["shard"] = q.get("shard", "")
                edit["obj_id"] = q.get("obj_id", "")
                shard_results.append(edit)

        all_results.extend(shard_results)

        # Write per-shard manifests (never overwrite other shards)
        shard_id = shard_dir.name.replace("shard_", "")
        shard_clean = shard_dir / f"manifest_clean_{shard_id}.jsonl"
        shard_tiered = shard_dir / f"manifest_tiered_{shard_id}.jsonl"
        shard_summary = shard_dir / f"cleaning_summary_{shard_id}.json"
        n_shard_clean = 0
        with open(shard_clean, "w") as fc, open(shard_tiered, "w") as ft:
            for r in shard_results:
                line = json.dumps(r, ensure_ascii=False)
                ft.write(line + "\n")
                tier = r.get("tier", "rejected")
                if tier_order.get(tier, 4) <= min_tier_val:
                    fc.write(line + "\n")
                    n_shard_clean += 1
        with open(shard_summary, "w") as f:
            json.dump({
                "shard": shard_id,
                "total_edits": len(shard_results),
                "clean_edits": n_shard_clean,
                "min_tier": min_tier,
                **summary,
            }, f, ensure_ascii=False, indent=2)
        logger.info("  Shard %s: %d/%d passed → %s",
                     shard_id, n_shard_clean, len(shard_results), shard_clean)

    # Write merged global manifests (union of all shards processed in this run)
    clean_path = root / "manifest_clean.jsonl"
    tiered_path = root / "manifest_tiered.jsonl"
    n_clean = 0
    with open(clean_path, "w") as f_clean, open(tiered_path, "w") as f_all:
        for r in all_results:
            line = json.dumps(r, ensure_ascii=False)
            f_all.write(line + "\n")
            tier = r.get("tier", "rejected")
            if tier_order.get(tier, 4) <= min_tier_val:
                f_clean.write(line + "\n")
                n_clean += 1

    # Write global summary
    global_summary = {
        "input_dir": str(root),
        "num_shards": len(shard_dirs),
        "total_edits": len(all_results),
        "clean_edits": n_clean,
        "min_tier": min_tier,
        "per_shard": all_summaries,
        "global": _build_summary(all_results),
    }
    summary_path = root / "cleaning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    logger.info("Cleaning done: %d/%d edits passed (%s tier+)",
                n_clean, len(all_results), min_tier)
    logger.info("  Per-shard manifests: shard_XX/manifest_clean_XX.jsonl")
    logger.info("  Merged manifest_clean.jsonl: %s", clean_path)
    logger.info("  cleaning_summary.json: %s", summary_path)

    return summary_path
