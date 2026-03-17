#!/usr/bin/env python3
"""PartCraft3D Unified Pipeline — closed-loop 3D editing data generation.

Refactored pipeline with TRELLIS as the primary 3D editing engine for ALL
edit types (deletion, modification, global). Mesh assembly (old Phase 2) is
demoted to a fallback for simple del/add pairs when TRELLIS is unavailable.

Pipeline Steps:
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 0: PREPROCESS (one-time, GPU)                         │
  │   prerender.py → Blender 150 views + SLAT encoding         │
  │   Cost: GPU only, 0 API tokens                             │
  └────────────────────────┬────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 1: SEMANTIC (VLM API)                                 │
  │   Phase 0 labeling + VLM enrichment → semantic_labels.jsonl│
  │   Cost: ~4K tokens/object                                  │
  └────────────────────────┬────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 2: PLANNING (CPU)                                     │
  │   Phase 1 → edit_specs.jsonl (del/add/mod/global)          │
  │   Cost: 0 tokens                                           │
  └────────────────────────┬────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 3: 2D EDIT (VLM API, parallelizable)                  │
  │   Pre-generate edited reference images for all specs        │
  │   Cost: ~1K tokens + 1 image output per spec               │
  └────────────────────────┬────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 4: 3D EDIT — TRELLIS (GPU, main workload)             │
  │   Flow Inversion + Repaint for del/mod/global              │
  │   Large parts (>40%) auto-promote to Global                │
  │   Cost: GPU only, 0 API tokens                             │
  └────────────────────────┬────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 5: QUALITY (VLM API)                                  │
  │   Render before/after → VLM scoring → tier classification  │
  │   Cost: ~3K tokens per spec                                │
  └────────────────────────┬────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 6: EXPORT (CPU)                                       │
  │   Instruction generation + final dataset assembly          │
  │   Cost: 0 tokens                                           │
  └─────────────────────────────────────────────────────────────┘

Usage:
    # Full pipeline (all steps)
    ATTN_BACKEND=xformers python scripts/run_pipeline.py

    # Specific steps
    ATTN_BACKEND=xformers python scripts/run_pipeline.py --steps 3 4 5

    # With experiment tag
    ATTN_BACKEND=xformers python scripts/run_pipeline.py --tag v1 --limit 50

    # Skip 2D editing (text-only TRELLIS)
    ATTN_BACKEND=xformers python scripts/run_pipeline.py --no-2d-edit

    # Cost estimation only (dry run)
    python scripts/run_pipeline.py --dry-run
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.hy3d_loader import HY3DPartDataset
from partcraft.phase1_planning.planner import EditSpec


# =========================================================================
# Cost constants (Gemini 2.5 Flash via OpenAI-compatible API)
# =========================================================================
# Approximate token counts per call type
COST = {
    # Step 1: Semantic — per object
    "phase0_input_tokens": 1532,    # 4 images (4×258) + ~500 text
    "phase0_output_tokens": 2000,   # object desc + part labels
    "enrich_input_tokens": 1258,    # 1 image (258) + ~1000 text
    "enrich_output_tokens": 3000,   # enriched descriptions + prompts
    # Step 3: 2D Edit — per edit spec
    "2d_edit_input_tokens": 458,    # 1 annotated image (258) + ~200 text
    "2d_edit_output_images": 1,     # 1 edited image
    # Step 5: Quality — per edit spec
    "quality_input_tokens": 2564,   # 8 images (8×258) + ~500 text
    "quality_output_tokens": 500,   # scores + rationale
    # Pricing ($/M tokens, Gemini 2.5 Flash)
    "input_price_per_m": 0.15,
    "output_price_per_m": 0.60,
    "image_output_price": 0.02,     # per image generated
}


def estimate_cost(n_objects: int, n_edits: int, steps: set) -> dict:
    """Estimate API token cost for the pipeline.

    Returns dict with token counts and USD cost breakdown.
    """
    c = COST
    total_input = 0
    total_output = 0
    total_images = 0
    breakdown = {}

    if 1 in steps:
        inp = n_objects * (c["phase0_input_tokens"] + c["enrich_input_tokens"])
        out = n_objects * (c["phase0_output_tokens"] + c["enrich_output_tokens"])
        total_input += inp
        total_output += out
        breakdown["step1_semantic"] = {
            "input_tokens": inp, "output_tokens": out,
            "usd": inp / 1e6 * c["input_price_per_m"] + out / 1e6 * c["output_price_per_m"],
        }

    if 3 in steps:
        inp = n_edits * c["2d_edit_input_tokens"]
        imgs = n_edits * c["2d_edit_output_images"]
        total_input += inp
        total_images += imgs
        breakdown["step3_2d_edit"] = {
            "input_tokens": inp, "output_images": imgs,
            "usd": inp / 1e6 * c["input_price_per_m"] + imgs * c["image_output_price"],
        }

    if 5 in steps:
        inp = n_edits * c["quality_input_tokens"]
        out = n_edits * c["quality_output_tokens"]
        total_input += inp
        total_output += out
        breakdown["step5_quality"] = {
            "input_tokens": inp, "output_tokens": out,
            "usd": inp / 1e6 * c["input_price_per_m"] + out / 1e6 * c["output_price_per_m"],
        }

    total_usd = (total_input / 1e6 * c["input_price_per_m"]
                 + total_output / 1e6 * c["output_price_per_m"]
                 + total_images * c["image_output_price"])

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_output_images": total_images,
        "total_usd": total_usd,
        "per_object_usd": total_usd / max(n_objects, 1),
        "per_edit_usd": total_usd / max(n_edits, 1),
        "breakdown": breakdown,
    }


# =========================================================================
# Step 1: Semantic Labeling + Enrichment
# =========================================================================

def run_step_semantic(cfg, dataset, logger, limit=None, force=False):
    """Step 1: VLM semantic labeling (Phase 0) + enrichment."""
    from partcraft.phase0_semantic.labeler import run_phase0

    logger.info("=" * 60)
    logger.info("STEP 1: Semantic Labeling + Enrichment")
    logger.info("=" * 60)

    labels_path = run_phase0(cfg, dataset, limit=limit, force=force)
    logger.info(f"Labels: {labels_path}")

    # Enrichment (if run_enrich module available)
    try:
        from partcraft.phase1_planning.enricher import enrich_semantic_labels
        enrich_cfg = cfg.get("enrich", {})
        if enrich_cfg.get("enabled", True):
            enrich_semantic_labels(cfg, limit=limit or 0)
            logger.info("Enrichment complete")
    except ImportError:
        logger.info("Enrichment module not available, skipping")

    return labels_path


# =========================================================================
# Step 2: Edit Planning
# =========================================================================

def run_step_planning(cfg, labels_path, logger, suffix=""):
    """Step 2: Generate edit specs from Part Catalog."""
    from partcraft.phase0_semantic.catalog import PartCatalog
    from partcraft.phase1_planning.planner import run_phase1

    logger.info("=" * 60)
    logger.info("STEP 2: Edit Planning")
    logger.info("=" * 60)

    catalog = PartCatalog.from_phase0_output(labels_path)
    catalog_path = Path(cfg["phase1"]["cache_dir"]) / f"part_catalog{suffix}.json"
    catalog.save(catalog_path)
    logger.info(f"Catalog: {catalog_path}")

    specs = run_phase1(cfg, catalog, suffix=suffix)
    specs_path = Path(cfg["phase1"]["cache_dir"]) / f"edit_specs{suffix}.jsonl"
    logger.info(f"Specs: {specs_path}")
    return specs_path


# =========================================================================
# Step 3: 2D Image Editing (API, parallelizable)
# =========================================================================

def run_step_2d_edit(cfg, specs_path, dataset, logger,
                     tag=None, workers=4, limit=None, edit_types=None):
    """Step 3: Pre-generate 2D edited images for all specs.

    Runs independently of GPU — can execute in parallel with other work.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: 2D Image Editing (VLM API)")
    logger.info("=" * 60)

    p0 = cfg["phase0"]
    p25 = cfg.get("phase2_5", {})

    from openai import OpenAI
    api_key = _resolve_api_key(cfg)
    if not api_key:
        logger.warning("No API key — skipping 2D editing")
        return None

    client = OpenAI(
        base_url=p0.get("vlm_base_url", ""),
        api_key=api_key,
    )
    model = p25.get("image_edit_model", "gemini-2.5-flash-image")

    # Load specs
    edit_types = edit_types or ["modification", "deletion"]
    specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if spec.edit_type in edit_types:
                specs.append(spec)

    if limit:
        specs = specs[:limit]

    # Output directory
    cache_dir = Path(p25.get("cache_dir", "outputs/cache/phase2_5"))
    edit_subdir = f"2d_edits_{tag}" if tag else "2d_edits"
    output_dir = cache_dir / edit_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delegate to the 2D edit module
    from scripts.run_2d_edit import process_one, select_best_view, prepare_input_image

    manifest_path = output_dir / "manifest.jsonl"

    # Resume
    done_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [s for s in specs if s.edit_id not in done_ids]
    logger.info(f"2D edits: {len(pending)} pending ({len(done_ids)} done), "
                f"model={model}, workers={workers}")

    if not pending:
        logger.info("All 2D edits already done")
        return output_dir

    from concurrent.futures import ThreadPoolExecutor, as_completed
    success, fail = 0, 0
    with open(manifest_path, "a") as fp:
        if workers <= 1:
            for i, spec in enumerate(pending):
                logger.info(f"[{i+1}/{len(pending)}] {spec.edit_id}")
                result = process_one(spec, dataset, client, output_dir,
                                     model, logger)
                fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                fp.flush()
                if result["status"] == "success":
                    success += 1
                else:
                    fail += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                for spec in pending:
                    fut = pool.submit(process_one, spec, dataset, client,
                                      output_dir, model, logger)
                    futures[fut] = spec

                for i, fut in enumerate(as_completed(futures)):
                    spec = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {"edit_id": spec.edit_id, "status": "failed",
                                  "reason": str(e)}
                    fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fp.flush()
                    if result.get("status") == "success":
                        success += 1
                    else:
                        fail += 1
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Progress: {i+1}/{len(pending)} "
                                    f"({success} ok, {fail} fail)")

    logger.info(f"2D edits: {success} ok, {fail} fail → {output_dir}")
    return output_dir


# =========================================================================
# Step 4: 3D Editing — TRELLIS (main workload)
# =========================================================================

def run_step_3d_edit(cfg, specs_path, dataset, logger,
                     tag=None, seed=1, limit=None, use_2d=True,
                     edit_types=None, edit_ids=None, combinations=None,
                     edit_dir=None):
    """Step 4: TRELLIS 3D editing for all edit types.

    This is the primary editing step. Handles:
      - Deletion → guided Modification (closes holes)
      - Modification → Flow Inversion + Repaint
      - Global → full mask + Inverse Flow (auto-promoted from large parts)
      - Addition → swaps before/after from corresponding deletion
    """
    logger.info("=" * 60)
    logger.info("STEP 4: 3D Editing — TRELLIS")
    logger.info("=" * 60)

    p25_cfg = cfg.get("phase2_5", {})
    vinedresser_path = p25_cfg.get(
        "vinedresser_path", "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main")
    sys.path.insert(0, vinedresser_path)

    from partcraft.phase2_assembly.trellis_refine import (
        TrellisRefiner, build_prompts_from_spec)

    # ---- Paths ----
    data_dir = Path(cfg["data"].get("data_dir", "data/partobjaverse_tiny"))
    mesh_zip = data_dir / "source" / "mesh.zip"
    if not mesh_zip.exists():
        img_dir = Path(cfg["data"]["image_npz_dir"])
        data_dir = img_dir.parent
        mesh_zip = data_dir / "source" / "mesh.zip"

    if not mesh_zip.exists():
        logger.error(f"source/mesh.zip not found at {mesh_zip}")
        return None

    output_dir = Path(cfg["data"]["output_dir"])
    cache_dir = Path(p25_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # ---- Load specs ----
    edit_types = edit_types or ["modification", "deletion", "global"]
    all_specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if edit_ids and spec.edit_id not in edit_ids:
                continue
            if spec.edit_type in edit_types or spec.edit_type == "addition":
                all_specs.append(spec)

    if not edit_ids and limit:
        all_specs = all_specs[:limit]

    if not all_specs:
        logger.info("No edit specs to process")
        return None

    # ---- Resume ----
    output_path = cache_dir / f"edit_results{tag_suffix}.jsonl"
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [s for s in all_specs if s.edit_id not in done_ids]
    logger.info(f"3D edits: {len(pending)} pending ({len(done_ids)} done)")
    if not pending:
        logger.info("All 3D edits already done")
        return output_path

    # ---- VLM client for inline 2D editing ----
    vlm_client = None
    if use_2d:
        from openai import OpenAI
        api_key = _resolve_api_key(cfg)
        if api_key:
            p0 = cfg["phase0"]
            vlm_client = OpenAI(
                base_url=p0.get("vlm_base_url", ""),
                api_key=api_key,
            )

    # ---- Init refiner ----
    refiner = TrellisRefiner(
        vinedresser_path=vinedresser_path,
        cache_dir=str(cache_dir),
        device="cuda",
        image_edit_model=p25_cfg.get("image_edit_model", "gemini-2.5-flash-image"),
    )
    refiner.load_models()

    # ---- Group by object ----
    obj_groups: OrderedDict[str, list] = OrderedDict()
    for spec in pending:
        obj_groups.setdefault(spec.obj_id, []).append(spec)

    success, fail = 0, 0
    glb_tmp_dir = tempfile.mkdtemp(prefix="partcraft_glb_")

    with open(output_path, "a") as out_fp:
        for obj_id, specs in obj_groups.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Object: {obj_id} ({len(specs)} edits)")

            # Sort: deletion → addition → modification
            type_order = {"deletion": 0, "addition": 1, "modification": 2,
                          "global": 3}
            specs.sort(key=lambda s: type_order.get(s.edit_type, 9))

            # ---- Handle additions (swap from deletion) ----
            add_specs = [s for s in specs if s.edit_type == "addition"]
            run_specs = [s for s in specs if s.edit_type != "addition"]

            for spec in add_specs:
                del_pair_dir = mesh_pairs_dir / spec.source_del_id
                add_pair_dir = mesh_pairs_dir / spec.edit_id
                if (del_pair_dir / "before_slat").exists():
                    add_pair_dir.mkdir(parents=True, exist_ok=True)
                    for src, dst in [
                        (del_pair_dir / "before_slat", add_pair_dir / "after_slat"),
                        (del_pair_dir / "after_slat", add_pair_dir / "before_slat"),
                        (del_pair_dir / "before.ply", add_pair_dir / "after.ply"),
                        (del_pair_dir / "after.ply", add_pair_dir / "before.ply"),
                    ]:
                        if src.exists() and not dst.exists():
                            if src.is_dir():
                                shutil.copytree(str(src), str(dst))
                            else:
                                shutil.copy2(str(src), str(dst))
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "edit_type": "addition",
                        "obj_id": obj_id, "status": "success",
                        "source_del_id": spec.source_del_id,
                    }, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                else:
                    run_specs.append(spec)

            if not run_specs:
                continue

            # ---- Prepare object ----
            try:
                glb_path = refiner.extract_glb(str(mesh_zip), obj_id, glb_tmp_dir)
                ori_slat = refiner.encode_object(glb_path, obj_id)
                ori_gaussian = refiner.decode_to_gaussian(ori_slat)
                shard = run_specs[0].shard
                obj_record = dataset.load_object(shard, obj_id)
            except Exception as e:
                logger.error(f"Failed to prepare {obj_id}: {e}")
                for spec in run_specs:
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": f"Preparation failed: {e}",
                    }) + "\n")
                    fail += len(run_specs)
                out_fp.flush()
                continue

            for spec in run_specs:
                logger.info(f"\n[{success+fail+1}/{len(pending)}] "
                            f"{spec.edit_id} ({spec.edit_type}): "
                            f"\"{spec.edit_prompt[:80]}\"")

                try:
                    edit_type = spec.edit_type.capitalize()

                    if edit_type == "Deletion":
                        edit_part_ids = spec.remove_part_ids
                    elif edit_type == "Modification":
                        edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Global":
                        edit_part_ids = []
                    elif edit_type == "Addition":
                        # Deferred addition
                        del_pair_dir = mesh_pairs_dir / spec.source_del_id
                        add_pair_dir = mesh_pairs_dir / spec.edit_id
                        if (del_pair_dir / "before_slat").exists():
                            add_pair_dir.mkdir(parents=True, exist_ok=True)
                            for src, dst in [
                                (del_pair_dir / "before_slat",
                                 add_pair_dir / "after_slat"),
                                (del_pair_dir / "after_slat",
                                 add_pair_dir / "before_slat"),
                                (del_pair_dir / "before.ply",
                                 add_pair_dir / "after.ply"),
                                (del_pair_dir / "after.ply",
                                 add_pair_dir / "before.ply"),
                            ]:
                                if src.exists() and not dst.exists():
                                    if src.is_dir():
                                        shutil.copytree(str(src), str(dst))
                                    else:
                                        shutil.copy2(str(src), str(dst))
                            out_fp.write(json.dumps({
                                "edit_id": spec.edit_id, "edit_type": "addition",
                                "obj_id": obj_id, "status": "success",
                            }, ensure_ascii=False) + "\n")
                            out_fp.flush()
                            success += 1
                        else:
                            out_fp.write(json.dumps({
                                "edit_id": spec.edit_id, "status": "failed",
                                "reason": f"Source deletion {spec.source_del_id} not found",
                            }) + "\n")
                            out_fp.flush()
                            fail += 1
                        continue
                    else:
                        logger.warning(f"Unknown edit type: {spec.edit_type}")
                        continue

                    # Build mask (may auto-promote to Global)
                    mask, effective_type = refiner.build_part_mask(
                        obj_id, obj_record, edit_part_ids,
                        ori_slat, edit_type)

                    if effective_type != edit_type:
                        logger.info(f"  Auto-promoted {edit_type} → "
                                    f"{effective_type} (large part)")
                        edit_type = effective_type

                    if mask.sum() == 0:
                        logger.warning(f"Empty mask for {spec.edit_id}")
                        out_fp.write(json.dumps({
                            "edit_id": spec.edit_id, "status": "failed",
                            "reason": "Empty mask",
                        }) + "\n")
                        out_fp.flush()
                        fail += 1
                        continue

                    # Build prompts
                    prompts = build_prompts_from_spec(spec)
                    if prompts["edit_type"] != edit_type:
                        prompts["edit_type"] = edit_type

                    # 2D image conditioning
                    img_cond = None
                    if edit_type in ("Modification", "Deletion", "Global") and use_2d:
                        num_edit_views = p25_cfg.get("num_edit_views", 4)
                        edit_strength = p25_cfg.get("edit_strength", 1.0)
                        original_images, edited_images = \
                            refiner.obtain_edited_images(
                                ori_gaussian, prompts, vlm_client,
                                obj_id, spec.edit_id,
                                num_views=num_edit_views,
                                edit_dir=edit_dir)
                        if edited_images:
                            img_cond = refiner.encode_multiview_cond(
                                edited_images, original_images,
                                edit_strength=edit_strength)

                    # Run TRELLIS editing
                    slats_edited = refiner.edit(
                        ori_slat, mask, prompts,
                        img_cond=img_cond,
                        seed=seed,
                        combinations=combinations,
                    )

                    if not slats_edited:
                        raise RuntimeError("No edited SLATs produced")

                    best_slat = slats_edited[0]

                    # Export before/after pair
                    pair_dir = mesh_pairs_dir / spec.edit_id
                    export_paths = refiner.export_pair(
                        ori_slat, best_slat, pair_dir)

                    record = {
                        "edit_id": spec.edit_id,
                        "edit_type": spec.edit_type,
                        "effective_edit_type": edit_type,
                        "obj_id": obj_id,
                        "shard": spec.shard,
                        "object_desc": spec.object_desc,
                        "edit_prompt": spec.edit_prompt,
                        "after_desc": spec.after_desc,
                        **export_paths,
                        "status": "success",
                    }
                    out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                    logger.info(f"  → saved {pair_dir}")

                except Exception as e:
                    import traceback
                    logger.error(f"Failed {spec.edit_id}: {e}")
                    traceback.print_exc()
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": str(e),
                    }) + "\n")
                    out_fp.flush()
                    fail += 1

            obj_record.close()

    shutil.rmtree(glb_tmp_dir, ignore_errors=True)
    logger.info(f"\n3D edits: {success} ok, {fail} fail → {output_path}")
    return output_path


# =========================================================================
# Step 5: Quality Scoring
# =========================================================================

def run_step_quality(cfg, results_path, logger, tag=None, limit=None):
    """Step 5: VLM quality scoring + tier classification."""
    logger.info("=" * 60)
    logger.info("STEP 5: Quality Scoring (VLM)")
    logger.info("=" * 60)

    from partcraft.phase3_filter.vlm_filter import run_vlm_filter
    p25_cfg = cfg.get("phase2_5", {})
    cache_dir = Path(p25_cfg["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    tag_suffix = f"_{tag}" if tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    scores_path = run_vlm_filter(
        cfg, str(results_path), str(mesh_pairs_dir),
        str(cache_dir / f"phase3{tag_suffix}"),
        limit=limit)

    logger.info(f"Quality scores: {scores_path}")
    return scores_path


# =========================================================================
# Step 6: Export
# =========================================================================

def run_step_export(cfg, specs_path, scores_path, logger, tag=None):
    """Step 6: Instruction generation + final dataset export."""
    logger.info("=" * 60)
    logger.info("STEP 6: Export")
    logger.info("=" * 60)

    from partcraft.io.export import EditPairWriter, EditPairRecord
    from partcraft.phase4_filter.instruction import generate_instructions

    output_dir = Path(cfg["data"]["output_dir"])
    tag_suffix = f"_{tag}" if tag else ""
    n_variants = cfg.get("phase3", {}).get("instructions_per_edit", 3)

    # Load passed entries from quality scoring
    passed_entries = []
    if scores_path and Path(scores_path).exists():
        with open(scores_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("quality_tier") in ("high", "medium"):
                        passed_entries.append(entry)

    if not passed_entries:
        logger.warning("No passed entries for export")
        return None

    # Load edit specs
    specs_map: dict[str, EditSpec] = {}
    with open(specs_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                specs_map[d["edit_id"]] = EditSpec(**d)

    exported = 0
    export_path = output_dir / f"edit_pairs{tag_suffix}.jsonl"
    with EditPairWriter(output_dir, filename=f"edit_pairs{tag_suffix}.jsonl") as writer:
        for entry in passed_entries:
            eid = entry["edit_id"]
            spec = specs_map.get(eid)
            if spec is None:
                continue

            instructions = generate_instructions(spec, n_variants)
            if not instructions:
                continue

            record = EditPairRecord(
                edit_id=eid,
                edit_type=entry.get("effective_edit_type", spec.edit_type),
                instruction=instructions[0],
                instruction_variants=instructions[1:],
                source_obj_id=spec.obj_id,
                source_shard=spec.shard,
                object_desc=spec.object_desc,
                quality_score=entry.get("quality_score", 0.0),
            )
            writer.write_pair(record)
            exported += 1

    logger.info(f"Exported {exported} pairs → {export_path}")
    return export_path


# =========================================================================
# Utility
# =========================================================================

def _resolve_api_key(cfg: dict) -> str:
    """Resolve VLM API key from config, default.yaml, or env."""
    import yaml
    p0 = cfg.get("phase0", {})
    api_key = p0.get("vlm_api_key", "")

    if not api_key:
        default_path = (Path(__file__).resolve().parents[1]
                        / "configs" / "default.yaml")
        if default_path.exists():
            with open(default_path) as f:
                dcfg = yaml.safe_load(f)
            api_key = dcfg.get("phase0", {}).get("vlm_api_key", "")

    if not api_key:
        env_var = p0.get("vlm_api_key_env", "")
        if env_var:
            api_key = os.environ.get(env_var, "")

    return api_key


def print_cost_report(cost: dict, n_objects: int, n_edits: int):
    """Print a formatted cost estimation report."""
    print("\n" + "=" * 60)
    print("TOKEN COST ESTIMATION (per pipeline run)")
    print("=" * 60)
    print(f"Objects: {n_objects}  |  Edit specs: {n_edits}")
    print(f"Avg edits/object: {n_edits/max(n_objects,1):.1f}")
    print("-" * 60)

    for step, detail in cost["breakdown"].items():
        label = step.replace("_", " ").title()
        inp = detail.get("input_tokens", 0)
        out = detail.get("output_tokens", 0)
        imgs = detail.get("output_images", 0)
        usd = detail["usd"]
        parts = []
        if inp:
            parts.append(f"in={inp:,}")
        if out:
            parts.append(f"out={out:,}")
        if imgs:
            parts.append(f"imgs={imgs}")
        print(f"  {label:30s}  {' | '.join(parts):30s}  ${usd:.4f}")

    print("-" * 60)
    print(f"  {'TOTAL':30s}  "
          f"in={cost['total_input_tokens']:,} | "
          f"out={cost['total_output_tokens']:,} | "
          f"imgs={cost['total_output_images']}")
    print(f"  {'':30s}  ${cost['total_usd']:.4f}")
    print(f"\n  Per object: ${cost['per_object_usd']:.4f}")
    print(f"  Per edit:   ${cost['per_edit_usd']:.4f}")
    print("=" * 60)

    # Single item breakdown
    print("\n" + "=" * 60)
    print("SINGLE DATA ITEM (1 edit spec) TOKEN BREAKDOWN")
    print("=" * 60)
    c = COST
    avg_edits = n_edits / max(n_objects, 1)
    print(f"""
  ┌─ Step 1: Semantic (amortized over {avg_edits:.0f} edits/object) ──────┐
  │  Phase 0 labeling:  {c['phase0_input_tokens']:>6} in + {c['phase0_output_tokens']:>5} out tokens  │
  │  VLM enrichment:    {c['enrich_input_tokens']:>6} in + {c['enrich_output_tokens']:>5} out tokens  │
  │  Subtotal/object:   {c['phase0_input_tokens']+c['enrich_input_tokens']:>6} in + {c['phase0_output_tokens']+c['enrich_output_tokens']:>5} out tokens  │
  │  Amortized/edit:    {int((c['phase0_input_tokens']+c['enrich_input_tokens'])/avg_edits):>6} in + {int((c['phase0_output_tokens']+c['enrich_output_tokens'])/avg_edits):>5} out tokens  │
  └────────────────────────────────────────────────────────┘

  ┌─ Step 2: Planning (CPU) ───────────────────────────────┐
  │  0 tokens                                              │
  └────────────────────────────────────────────────────────┘

  ┌─ Step 3: 2D Image Editing ─────────────────────────────┐
  │  Input:   {c['2d_edit_input_tokens']:>6} tokens (1 annotated image + prompt)   │
  │  Output:  {c['2d_edit_output_images']:>6} generated image (~$0.02)             │
  └────────────────────────────────────────────────────────┘

  ┌─ Step 4: 3D Editing — TRELLIS (GPU) ───────────────────┐
  │  0 API tokens (GPU compute ~60-120s per edit)          │
  └────────────────────────────────────────────────────────┘

  ┌─ Step 5: Quality Scoring ──────────────────────────────┐
  │  Input:   {c['quality_input_tokens']:>6} tokens (8 rendered views + prompt)    │
  │  Output:  {c['quality_output_tokens']:>6} tokens (scores + rationale)          │
  └────────────────────────────────────────────────────────┘

  ┌─ Step 6: Export (CPU) ─────────────────────────────────┐
  │  0 tokens                                              │
  └────────────────────────────────────────────────────────┘
""")
    per_edit_in = (int((c['phase0_input_tokens'] + c['enrich_input_tokens']) / avg_edits)
                   + c['2d_edit_input_tokens'] + c['quality_input_tokens'])
    per_edit_out = (int((c['phase0_output_tokens'] + c['enrich_output_tokens']) / avg_edits)
                    + c['quality_output_tokens'])
    per_edit_imgs = c['2d_edit_output_images']
    per_edit_usd = (per_edit_in / 1e6 * c['input_price_per_m']
                    + per_edit_out / 1e6 * c['output_price_per_m']
                    + per_edit_imgs * c['image_output_price'])
    print(f"  TOTAL per edit: {per_edit_in:,} in + {per_edit_out:,} out + "
          f"{per_edit_imgs} img = ${per_edit_usd:.4f}")
    print(f"  At 1000 edits: ${per_edit_usd * 1000:.2f}")
    print(f"  At 10000 edits: ${per_edit_usd * 10000:.2f}")
    print("=" * 60)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PartCraft3D Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Semantic labeling + enrichment (VLM API)
  2  Edit planning (CPU)
  3  2D image editing (VLM API, parallelizable)
  4  3D editing — TRELLIS (GPU, main workload)
  5  Quality scoring (VLM API)
  6  Export (CPU)

Examples:
  # Full pipeline
  ATTN_BACKEND=xformers python scripts/run_pipeline.py

  # Only 3D editing + quality
  ATTN_BACKEND=xformers python scripts/run_pipeline.py --steps 4 5

  # Dry run (cost estimation only)
  python scripts/run_pipeline.py --dry-run
""")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="Steps to run (default: all). E.g. --steps 3 4 5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag for output isolation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for 2D editing API calls")
    parser.add_argument("--no-2d-edit", dest="use_2d", action="store_false",
                        default=True)
    parser.add_argument("--edit-dir", type=str, default=None,
                        help="Pre-generated 2D edits subdir")
    parser.add_argument("--edit-ids", nargs="*", default=None,
                        help="Specific edit IDs to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Cost estimation only, no actual processing")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for spec files (e.g. '_action')")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "pipeline")

    steps = set(args.steps) if args.steps else {1, 2, 3, 4, 5, 6}

    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )
    logger.info(f"Dataset: {len(dataset)} objects")

    # ---- Resolve paths ----
    suffix = args.suffix
    tag_suffix = f"_{args.tag}" if args.tag else ""
    labels_path = Path(cfg["phase0"]["cache_dir"]) / f"semantic_labels{suffix}.jsonl"
    specs_path = Path(cfg["phase1"]["cache_dir"]) / f"edit_specs{suffix}.jsonl"

    # ---- Dry run: cost estimation ----
    if args.dry_run:
        n_objects = len(dataset)
        n_edits = 0
        if specs_path.exists():
            with open(specs_path) as f:
                n_edits = sum(1 for line in f if line.strip())
        else:
            # Estimate: ~27 edits per object (from PartObjaverse-Tiny stats)
            n_edits = n_objects * 27

        cost = estimate_cost(n_objects, n_edits, steps)
        print_cost_report(cost, n_objects, n_edits)
        return

    # ---- Step 1: Semantic ----
    if 1 in steps:
        labels_path = run_step_semantic(
            cfg, dataset, logger, limit=args.limit)

    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path}")
        logger.error("Run step 1 first, or provide cached labels")
        sys.exit(1)

    # ---- Step 2: Planning ----
    if 2 in steps:
        specs_path = run_step_planning(cfg, labels_path, logger, suffix=suffix)

    if not specs_path.exists():
        logger.error(f"Specs not found: {specs_path}")
        logger.error("Run step 2 first, or provide cached specs")
        sys.exit(1)

    # ---- Step 3: 2D Edit ----
    edit_2d_dir = None
    if 3 in steps:
        edit_2d_dir = run_step_2d_edit(
            cfg, specs_path, dataset, logger,
            tag=args.tag, workers=args.workers, limit=args.limit)

    # ---- Step 4: 3D Edit ----
    results_path = None
    if 4 in steps:
        results_path = run_step_3d_edit(
            cfg, specs_path, dataset, logger,
            tag=args.tag, seed=args.seed, limit=args.limit,
            use_2d=args.use_2d, edit_ids=args.edit_ids,
            edit_dir=args.edit_dir or (
                str(edit_2d_dir) if edit_2d_dir else None))

    if results_path is None:
        p25_cfg = cfg.get("phase2_5", {})
        results_path = (Path(p25_cfg["cache_dir"])
                        / f"edit_results{tag_suffix}.jsonl")

    # ---- Step 5: Quality ----
    scores_path = None
    if 5 in steps and results_path and Path(results_path).exists():
        scores_path = run_step_quality(
            cfg, results_path, logger, tag=args.tag, limit=args.limit)

    # ---- Step 6: Export ----
    if 6 in steps:
        run_step_export(cfg, specs_path, scores_path, logger, tag=args.tag)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
