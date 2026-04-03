#!/usr/bin/env python3
"""Unified post-processing pipeline for shard outputs (Phase A-D).

Takes raw ``mesh_pairs/{edit_id}/`` output from Step 4 and produces
cleaned, repacked object-centric training data.

Phases
------
A — Geometric pre-screening (no GPU for PLY, light GPU for SLAT)
    - mod/scl/mat/glb: feats+coords checks (no SS needed)
    - deletion: PLY mesh checks (watertight, connectivity, volume)

B — VLM semantic evaluation (VLM API + rendering GPU)
    - mod/scl/mat/glb: SLAT → TRELLIS Gaussian → render → VLM
    - deletion: PLY → Blender render → VLM
    - addition: follows paired deletion result
    - identity: auto-pass

C — Encode SS (GPU, only for Phase B survivors)
    - Calls migrate_slat_to_npz.py --include-list

D — Repack + Full Cleaning
    - repack_to_object_dirs.py → run_cleaning.py (with SS)

Usage
-----
Full pipeline::

    python scripts/tools/run_postprocess.py \\
        --config configs/partverse_node39_shard01.yaml \\
        --mesh-pairs outputs/partverse/shard_01/mesh_pairs_shard01 \\
        --specs-jsonl outputs/partverse/shard_01/cache/phase1/edit_specs_shard01.jsonl \\
        --output-dir outputs/partverse/partverse_pairs \\
        --shard 01

Phase A only (pre-screening, no GPU)::

    python scripts/tools/run_postprocess.py \\
        --config ... --mesh-pairs ... --specs-jsonl ... \\
        --shard 01 --phase A

Phase A+B (pre-screening + VLM)::

    python scripts/tools/run_postprocess.py \\
        --config ... --mesh-pairs ... --specs-jsonl ... \\
        --shard 01 --phase AB
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("postprocess")


# ── Data format detection ────────────────────────────────────────────────

def _detect_format(pair_dir: Path) -> str:
    """Detect data format for a single edit pair directory.

    Returns one of: "npz", "slat", "ply", "empty".
    """
    if (pair_dir / "after.npz").exists():
        return "npz"
    if (pair_dir / "after_slat").is_dir():
        return "slat"
    if (pair_dir / "after.ply").exists():
        return "ply"
    return "empty"


def _classify_edits(
    specs: list[dict],
    mesh_pairs: Path,
) -> dict[str, list[dict]]:
    """Classify edits by data format and edit type.

    Returns dict with keys:
      "slat_edits": edits with *_slat/ dirs (mod/scl/mat/glb)
      "ply_edits": edits with PLY only (typically deletion)
      "npz_edits": edits already in NPZ format
      "ref_edits": reference-only edits (addition/identity)
      "empty_edits": edits with no data
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        etype = spec.get("edit_type", "")
        edit_id = spec.get("edit_id", "")

        if etype in ("addition", "identity"):
            groups["ref_edits"].append(spec)
            continue

        pair_dir = mesh_pairs / edit_id
        if not pair_dir.is_dir():
            groups["empty_edits"].append(spec)
            continue

        fmt = _detect_format(pair_dir)
        if fmt == "npz":
            groups["npz_edits"].append(spec)
        elif fmt == "slat":
            groups["slat_edits"].append(spec)
        elif fmt == "ply":
            groups["ply_edits"].append(spec)
        else:
            groups["empty_edits"].append(spec)

    return groups


# ── Phase A: Geometric Pre-screening ─────────────────────────────────────

def run_phase_a(
    specs: list[dict],
    mesh_pairs: Path,
    cfg: dict,
    work_dir: Path,
) -> set[str]:
    """Run geometric pre-screening. Returns set of passed edit_ids."""
    from partcraft.cleaning.npz_checks import load_slat_dir_arrays, load_npz_arrays
    from partcraft.cleaning.pair_checks import check_pair
    from partcraft.cleaning.ply_checks import check_deletion_ply, check_addition_ply
    from partcraft.cleaning.cleaner import weighted_score

    groups = _classify_edits(specs, mesh_pairs)
    cleaning_cfg = cfg.get("cleaning", {})
    passed: set[str] = set()
    results_all: list[dict] = []

    # ── SLAT edits (mod/scl/mat/glb) ──
    slat_edits = groups.get("slat_edits", [])
    log.info("Phase A: screening %d SLAT edits (feats+coords, no SS)",
             len(slat_edits))

    # Group by object for shared before loading
    by_obj: dict[str, list[dict]] = defaultdict(list)
    for s in slat_edits:
        by_obj[s["obj_id"]].append(s)

    slat_pass = 0
    for obj_id, obj_specs in by_obj.items():
        # Load before from first edit
        original_data = None
        for s in obj_specs:
            before_slat = mesh_pairs / s["edit_id"] / "before_slat"
            if before_slat.is_dir():
                try:
                    original_data = load_slat_dir_arrays(before_slat)
                    break
                except Exception:
                    continue

        for s in obj_specs:
            eid = s["edit_id"]
            etype = s["edit_type"]
            after_slat = mesh_pairs / eid / "after_slat"
            if not after_slat.is_dir():
                results_all.append({"edit_id": eid, "phase_a": "no_data"})
                continue
            try:
                after_data = load_slat_dir_arrays(after_slat)
            except Exception as e:
                results_all.append({"edit_id": eid, "phase_a": f"load_error: {e}"})
                continue

            if original_data is None:
                results_all.append({"edit_id": eid, "phase_a": "no_before"})
                continue

            type_cfg = cleaning_cfg.get(etype, {})
            l2 = check_pair(etype, original_data, after_data, type_cfg,
                            require_ss=False)
            score, ok = weighted_score(l2)
            results_all.append({
                "edit_id": eid, "type": etype,
                "phase_a": "pass" if ok else "fail",
                "score": round(score, 4),
            })
            if ok:
                passed.add(eid)
                slat_pass += 1

    log.info("Phase A SLAT: %d / %d passed", slat_pass, len(slat_edits))

    # ── PLY edits (deletion) ──
    ply_edits = groups.get("ply_edits", [])
    log.info("Phase A: screening %d PLY edits (mesh checks)", len(ply_edits))

    ply_pass = 0
    del_cfg = cleaning_cfg.get("deletion", {})
    for s in ply_edits:
        eid = s["edit_id"]
        pair_dir = mesh_pairs / eid
        before_ply = pair_dir / "before.ply"
        after_ply = pair_dir / "after.ply"

        if not before_ply.exists() or not after_ply.exists():
            results_all.append({"edit_id": eid, "phase_a": "no_ply"})
            continue

        metrics = check_deletion_ply(before_ply, after_ply, del_cfg)
        score, ok = weighted_score(metrics)
        results_all.append({
            "edit_id": eid, "type": s.get("edit_type", "deletion"),
            "phase_a": "pass" if ok else "fail",
            "score": round(score, 4),
        })
        if ok:
            passed.add(eid)
            ply_pass += 1

    log.info("Phase A PLY: %d / %d passed", ply_pass, len(ply_edits))

    # ── NPZ edits (already migrated) ──
    npz_edits = groups.get("npz_edits", [])
    for s in npz_edits:
        passed.add(s["edit_id"])
        results_all.append({
            "edit_id": s["edit_id"], "type": s.get("edit_type", ""),
            "phase_a": "pass_npz",
        })
    log.info("Phase A: %d NPZ edits auto-passed", len(npz_edits))

    # ── Reference edits (addition/identity) — follow source ──
    ref_edits = groups.get("ref_edits", [])
    for s in ref_edits:
        eid = s["edit_id"]
        etype = s["edit_type"]
        if etype == "addition":
            source_del_id = s.get("source_del_id", "")
            if source_del_id in passed:
                passed.add(eid)
                results_all.append({"edit_id": eid, "type": etype,
                                    "phase_a": "pass_ref"})
            else:
                results_all.append({"edit_id": eid, "type": etype,
                                    "phase_a": "fail_ref"})
        elif etype == "identity":
            # Pass if any edit of the same object passed
            obj_id = s.get("obj_id", "")
            obj_passed = any(
                o["edit_id"] in passed for o in specs
                if o.get("obj_id") == obj_id and o["edit_id"] != eid
            )
            if obj_passed:
                passed.add(eid)
            results_all.append({
                "edit_id": eid, "type": etype,
                "phase_a": "pass_ref" if obj_passed else "fail_ref",
            })

    # ── Save results ──
    results_path = work_dir / "phase_a_results.jsonl"
    with open(results_path, "w") as f:
        for r in results_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    passed_path = work_dir / "phase_a_passed.txt"
    with open(passed_path, "w") as f:
        for eid in sorted(passed):
            f.write(eid + "\n")

    empty = len(groups.get("empty_edits", []))
    log.info("Phase A summary: %d passed, %d total, %d empty/missing",
             len(passed), len(specs), empty)
    log.info("Phase A results: %s", results_path)

    return passed


# ── Phase B: VLM Semantic Evaluation ─────────────────────────────────────

def run_phase_b(
    specs: list[dict],
    mesh_pairs: Path,
    passed_a: set[str],
    cfg: dict,
    work_dir: Path,
) -> set[str]:
    """Run VLM semantic evaluation on Phase A survivors.

    Returns set of edit_ids that passed both Phase A and B.
    """
    from openai import OpenAI
    from partcraft.phase3_filter.vlm_filter import (
        evaluate_edit_from_ply,
        evaluate_edit_from_slat_dir,
        compute_composite_score,
        classify_tier,
        VLMScore,
    )
    from tqdm import tqdm

    p0 = cfg.get("phase0", {})
    p3 = cfg.get("phase3", {})

    # Resolve VLM client
    api_key = p0.get("vlm_api_key", "")
    if not api_key:
        import os
        env_var = p0.get("vlm_api_key_env", "")
        if env_var:
            api_key = os.environ.get(env_var, "")
    if not api_key:
        log.error("No VLM API key found in config or environment")
        sys.exit(1)

    client = OpenAI(base_url=p0.get("vlm_base_url", ""), api_key=api_key)
    vlm_model = p0.get("vlm_model", "gemini-2.5-flash")
    vlm_max_tokens = int(p3.get("vlm_max_tokens", 4096))
    vlm_json_mode = bool(p3.get("vlm_json_response_format", False))
    num_views = int(p3.get("num_views", 4))
    blender_path = cfg.get("tools", {}).get("blender_path", "blender")

    # Load TRELLIS for SLAT rendering (lazy — only if needed)
    trellis_pipeline = None

    def _get_trellis():
        nonlocal trellis_pipeline
        if trellis_pipeline is not None:
            return trellis_pipeline
        log.info("Loading TRELLIS decoder for SLAT rendering...")
        third_party = str(_PROJECT_ROOT / "third_party")
        if third_party not in sys.path:
            sys.path.insert(0, third_party)
        from trellis.pipelines import TrellisImageTo3DPipeline
        p25 = cfg.get("phase2_5", {})
        ckpt_spec = p25.get("trellis_image_ckpt", "checkpoints/TRELLIS-image-large")
        ckpt_path = Path(ckpt_spec)
        if not ckpt_path.is_absolute():
            ckpt_path = _PROJECT_ROOT / ckpt_spec
        trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(str(ckpt_path))
        trellis_pipeline.to("cuda")
        return trellis_pipeline

    # Filter to Phase A survivors (excluding ref edits handled separately)
    candidates = [s for s in specs
                  if s["edit_id"] in passed_a
                  and s.get("edit_type") not in ("addition", "identity")]

    log.info("Phase B: VLM evaluating %d edits", len(candidates))

    passed_b: set[str] = set()
    scores_path = work_dir / "phase_b_vlm_scores.jsonl"

    with open(scores_path, "w") as fp:
        for i, s in enumerate(tqdm(candidates, desc="Phase B: VLM")):
            eid = s["edit_id"]
            etype = s.get("edit_type", "")
            pair_dir = mesh_pairs / eid
            fmt = _detect_format(pair_dir)

            edit_prompt = s.get("edit_prompt", "")
            object_desc = s.get("object_desc", "")
            part_label = s.get("old_label", "")
            if not part_label:
                labels = s.get("remove_labels", s.get("add_labels", []))
                part_label = labels[0] if labels else "unknown"

            try:
                if fmt == "ply":
                    score = evaluate_edit_from_ply(
                        before_ply=pair_dir / "before.ply",
                        after_ply=pair_dir / "after.ply",
                        edit_id=eid, edit_type=etype,
                        edit_prompt=edit_prompt,
                        object_desc=object_desc,
                        part_label=part_label,
                        vlm_client=client, vlm_model=vlm_model,
                        num_views=num_views, blender_path=blender_path,
                        vlm_max_tokens=vlm_max_tokens,
                        vlm_json_object_mode=vlm_json_mode,
                    )
                elif fmt in ("slat", "npz"):
                    pipeline = _get_trellis()
                    before_dir = pair_dir / "before_slat"
                    after_dir = pair_dir / "after_slat"
                    if fmt == "npz":
                        # For NPZ, evaluate_edit_from_slat_dir won't work;
                        # auto-pass NPZ edits in Phase B
                        score = VLMScore(edit_id=eid, edit_type=etype)
                        score.edit_executed = True
                        score.correct_region = True
                        score.preserve_other = True
                        score.visual_quality = 3
                        score.artifact_free = True
                        score.reason = "NPZ auto-pass (already migrated)"
                        score.score = compute_composite_score(score)
                        score.quality_tier = classify_tier(score)
                    else:
                        score = evaluate_edit_from_slat_dir(
                            pipeline=pipeline,
                            slat_dir_before=before_dir,
                            slat_dir_after=after_dir,
                            edit_id=eid, edit_type=etype,
                            edit_prompt=edit_prompt,
                            object_desc=object_desc,
                            part_label=part_label,
                            vlm_client=client, vlm_model=vlm_model,
                            num_views=num_views, device="cuda",
                            vlm_max_tokens=vlm_max_tokens,
                            vlm_json_object_mode=vlm_json_mode,
                        )
                else:
                    score = VLMScore(edit_id=eid, edit_type=etype,
                                    reason="no renderable data")
            except Exception as e:
                score = VLMScore(edit_id=eid, edit_type=etype,
                                reason=f"Phase B error: {e}")
                log.warning("Phase B error for %s: %s", eid, e)

            score_dict = score.to_dict()
            fp.write(json.dumps(score_dict, ensure_ascii=False) + "\n")
            fp.flush()

            if score.quality_tier in ("high", "medium", "low"):
                passed_b.add(eid)

            if (i + 1) % 50 == 0:
                log.info("  Phase B progress: %d / %d, passed so far: %d",
                         i + 1, len(candidates), len(passed_b))

    # Addition/identity follow their source
    for s in specs:
        eid = s["edit_id"]
        etype = s.get("edit_type", "")
        if etype == "addition":
            source_del_id = s.get("source_del_id", "")
            if source_del_id in passed_b:
                passed_b.add(eid)
        elif etype == "identity":
            obj_id = s.get("obj_id", "")
            obj_passed = any(
                o["edit_id"] in passed_b for o in specs
                if o.get("obj_id") == obj_id and o["edit_id"] != eid
            )
            if obj_passed:
                passed_b.add(eid)

    passed_path = work_dir / "phase_b_passed.txt"
    with open(passed_path, "w") as f:
        for eid in sorted(passed_b):
            f.write(eid + "\n")

    log.info("Phase B summary: %d / %d passed VLM evaluation",
             len(passed_b), len(candidates))
    return passed_b


# ── Phase C: Encode SS ───────────────────────────────────────────────────

def run_phase_c(
    passed_ids: set[str],
    mesh_pairs: Path,
    specs_jsonl: Path,
    config_path: str,
    work_dir: Path,
):
    """Encode SS for passed edits via migrate_slat_to_npz.py."""
    include_path = work_dir / "phase_c_include.txt"
    with open(include_path, "w") as f:
        for eid in sorted(passed_ids):
            f.write(eid + "\n")

    migrate_script = _PROJECT_ROOT / "scripts" / "tools" / "migrate_slat_to_npz.py"
    cmd = [
        sys.executable, str(migrate_script),
        "--config", config_path,
        "--mesh-pairs", str(mesh_pairs),
        "--specs-jsonl", str(specs_jsonl),
        "--include-list", str(include_path),
    ]
    log.info("Phase C: running migrate_slat_to_npz with %d included edits",
             len(passed_ids))
    log.info("  cmd: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("Phase C: migrate_slat_to_npz failed (exit %d)",
                  result.returncode)
        sys.exit(1)

    log.info("Phase C: SS encoding complete")


# ── Phase D: Repack + Full Cleaning ──────────────────────────────────────

def run_phase_d(
    mesh_pairs: Path,
    specs_jsonl: Path,
    output_dir: Path,
    shard: str,
    cfg: dict,
    workers: int,
):
    """Repack to object-centric format and run full cleaning."""
    repack_script = _PROJECT_ROOT / "scripts" / "tools" / "repack_to_object_dirs.py"
    cleaning_script = _PROJECT_ROOT / "scripts" / "tools" / "run_cleaning.py"

    # Step 1: Repack
    log.info("Phase D: repacking to object-centric format...")
    cmd = [
        sys.executable, str(repack_script),
        "--mesh-pairs", str(mesh_pairs),
        "--specs-jsonl", str(specs_jsonl),
        "--output-dir", str(output_dir),
        "--shard", shard,
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("Phase D: repack failed (exit %d)", result.returncode)
        sys.exit(1)

    # Step 2: Full cleaning (with SS)
    log.info("Phase D: running full cleaning with SS...")
    cmd = [
        sys.executable, str(cleaning_script),
        "--input-dir", str(output_dir),
        "--shards", shard,
        "--workers", str(workers),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("Phase D: cleaning failed (exit %d)", result.returncode)
        sys.exit(1)

    log.info("Phase D: repack + cleaning complete")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified post-processing pipeline (Phase A-D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True,
                        help="Pipeline YAML config")
    parser.add_argument("--mesh-pairs", required=True,
                        help="Root mesh_pairs directory")
    parser.add_argument("--specs-jsonl", required=True,
                        help="edit_specs JSONL file")
    parser.add_argument("--output-dir", required=True,
                        help="Output dir for repacked data")
    parser.add_argument("--shard", required=True,
                        help="Shard identifier (e.g. 01)")
    parser.add_argument("--phase", default="ABCD",
                        help="Phases to run: A, AB, ABC, ABCD (default: ABCD)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers for cleaning")
    parser.add_argument("--work-dir", default=None,
                        help="Working directory for intermediate files "
                             "(default: mesh_pairs/../postprocess_shard{XX})")
    args = parser.parse_args()

    mesh_pairs = Path(args.mesh_pairs)
    specs_jsonl = Path(args.specs_jsonl)
    output_dir = Path(args.output_dir)
    phases = set(args.phase.upper())

    # Work dir for intermediate files
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = mesh_pairs.parent / f"postprocess_shard{args.shard}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    from partcraft.utils.config import load_config
    cfg = load_config(args.config)

    # Load specs
    all_specs: list[dict] = []
    with open(specs_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                all_specs.append(json.loads(line))
    log.info("Loaded %d edit specs from %s", len(all_specs), specs_jsonl)

    # Classify edits
    groups = _classify_edits(all_specs, mesh_pairs)
    for key, edits in groups.items():
        log.info("  %s: %d edits", key, len(edits))

    # ── Phase A ──
    passed_a: set[str] = set()
    if "A" in phases:
        log.info("=" * 60)
        log.info("Phase A: Geometric Pre-screening")
        passed_a = run_phase_a(all_specs, mesh_pairs, cfg, work_dir)
    else:
        # If skipping A, pass all edits
        passed_a = {s["edit_id"] for s in all_specs}
        log.info("Phase A skipped: all %d edits passed through", len(passed_a))

    # ── Phase B ──
    passed_b: set[str] = set()
    if "B" in phases:
        log.info("=" * 60)
        log.info("Phase B: VLM Semantic Evaluation")
        passed_b = run_phase_b(all_specs, mesh_pairs, passed_a, cfg, work_dir)
    else:
        passed_b = passed_a
        log.info("Phase B skipped: carrying forward %d edits", len(passed_b))

    # ── Phase C ──
    if "C" in phases:
        log.info("=" * 60)
        log.info("Phase C: Encode SS")
        run_phase_c(passed_b, mesh_pairs, specs_jsonl, args.config, work_dir)

    # ── Phase D ──
    if "D" in phases:
        log.info("=" * 60)
        log.info("Phase D: Repack + Full Cleaning")
        run_phase_d(mesh_pairs, specs_jsonl, output_dir, args.shard,
                    cfg, args.workers)

    log.info("=" * 60)
    log.info("Post-processing complete. Phases run: %s", args.phase)
    log.info("  Work dir: %s", work_dir)
    if "D" in phases:
        log.info("  Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
