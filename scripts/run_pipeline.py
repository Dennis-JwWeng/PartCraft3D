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
  │   Deletion → S1 fills hole + S2 texture (dynamic mask/cfg)│
  │   Modification → Flow Inversion + Repaint                  │
  │   Global → TextureOnly (S1 unchanged, S2 repaint only)     │
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

from PIL import Image

# Use $PWD to preserve symlinks (os.getcwd() resolves them on Linux)
def _get_project_root() -> Path:
    script = Path(__file__)
    if not script.is_absolute():
        script = Path(os.environ.get('PWD', os.getcwd())) / script
    return script.parents[1]

PROJECT_ROOT = _get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.partcraft_loader import PartCraftDataset
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

# =========================================================================
# Step 1: Semantic Labeling + Enrichment
# =========================================================================

def run_step_semantic(cfg, dataset, logger, limit=None, force=False, tag=None,
                      debug=False):
    """Step 1: VLM semantic labeling via enricher.

    Uses dataset NPZ files to discover objects and part labels.
    No dependency on source/semantic.json.
    """
    from pathlib import Path as _Path
    from partcraft.phase1_planning.enricher import enrich_semantic_labels

    logger.info("=" * 60)
    logger.info("STEP 1: Semantic Labeling + Enrichment")
    logger.info("=" * 60)

    cache_dir = _Path(cfg["phase0"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    labels_path = cache_dir / f"semantic_labels{tag_suffix}.jsonl"

    image_npz_dir = cfg["data"].get("image_npz_dir")
    shards = cfg["data"].get("shards", ["00"])
    max_workers = cfg["phase0"].get("max_workers", 4)

    if force and labels_path.exists():
        backup = labels_path.with_suffix(".jsonl.bak")
        labels_path.rename(backup)
        logger.info(f"--force: backed up old labels to {backup}")

    enrich_semantic_labels(
        cfg,
        semantic_json_path=None,
        output_path=str(labels_path),
        image_npz_dir=image_npz_dir,
        shard=shards[0] if shards else "00",
        limit=limit or 0,
        max_workers=max_workers,
        visual_grounding=cfg.get("phase0", {}).get("visual_grounding", True),
        dataset=dataset,
        debug=debug,
    )

    logger.info(f"Labels: {labels_path}")
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

    specs = run_phase1(cfg, catalog, output_suffix=suffix)
    specs_path = Path(cfg["phase1"]["cache_dir"]) / f"edit_specs{suffix}.jsonl"
    logger.info(f"Specs: {specs_path}")
    return specs_path


# =========================================================================
# Step 3: 2D Image Editing (API, parallelizable)
# =========================================================================

def run_step_2d_edit(cfg, specs_path, dataset, logger,
                     tag=None, workers=4, limit=None, edit_types=None,
                     edit_ids=None):
    """Step 3: Pre-generate 2D edited images for all specs.

    Runs independently of GPU — can execute in parallel with other work.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: 2D Image Editing")
    logger.info("=" * 60)

    p0 = cfg["phase0"]
    p25 = cfg.get("phase2_5", {})

    image_edit_backend = p25.get("image_edit_backend", "api")

    # Local edit server or API client
    client = None  # OpenAI client (for API backend)
    edit_server_url = None  # local HTTP edit server URL
    model = p25.get("image_edit_model", "gemini-2.5-flash-image")

    if image_edit_backend == "local_diffusers":
        from scripts.run_2d_edit import check_edit_server
        edit_server_url = p25.get("image_edit_base_url", "http://localhost:8001")
        if not check_edit_server(edit_server_url):
            logger.warning(f"Image edit server not reachable at {edit_server_url}, "
                           "skipping 2D editing")
            return None
        logger.info(f"Image edit server OK at {edit_server_url}")
        # Use configured workers (default 1 for single-GPU sequential serving)
        cfg_workers = p25.get("image_edit_workers", 1)
        if workers != cfg_workers:
            logger.info(f"local_diffusers backend: workers={cfg_workers} "
                         f"(from config)")
            workers = cfg_workers
    else:
        from openai import OpenAI
        api_key = _resolve_api_key(cfg)
        if not api_key:
            logger.warning("No API key — skipping 2D editing")
            return None

        image_edit_url = p25.get("image_edit_base_url") or p0.get("vlm_base_url", "")
        client = OpenAI(
            base_url=image_edit_url,
            api_key=api_key,
        )

    # Load specs
    edit_types = edit_types or ["modification", "deletion", "global"]
    specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if edit_ids and spec.edit_id not in edit_ids:
                continue
            if spec.edit_type in edit_types:
                specs.append(spec)

    if not edit_ids and limit:
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
    backend_label = edit_server_url or model
    logger.info(f"2D edits: {len(pending)} pending ({len(done_ids)} done), "
                f"backend={backend_label}, workers={workers}")

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
                                     model, logger,
                                     edit_server_url=edit_server_url)
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
                                      output_dir, model, logger,
                                      edit_server_url=edit_server_url)
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
                     edit_dir=None, debug=False):
    """Step 4: TRELLIS 3D editing for all edit types.

    This is the primary editing step. Handles:
      - Deletion → Modification (S1 regenerates closing geometry, dynamic mask/cfg)
      - Modification → Flow Inversion + Repaint (contact-aware soft mask)
      - Global → TextureOnly (skip S1, only S2 repaint changes texture)
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
    image_edit_backend = p25_cfg.get("image_edit_backend", "api")
    vlm_client = None
    if use_2d and image_edit_backend != "local_diffusers":
        from openai import OpenAI
        api_key = _resolve_api_key(cfg)
        if api_key:
            p0 = cfg["phase0"]
            image_edit_url = p25_cfg.get("image_edit_base_url") or p0.get("vlm_base_url", "")
            vlm_client = OpenAI(
                base_url=image_edit_url,
                api_key=api_key,
            )

    # ---- Init refiner ----
    refiner = TrellisRefiner(
        vinedresser_path=vinedresser_path,
        cache_dir=str(cache_dir),
        device="cuda",
        image_edit_model=p25_cfg.get("image_edit_model", "gemini-2.5-flash-image"),
        image_edit_backend=image_edit_backend,
        image_edit_base_url=p25_cfg.get("image_edit_base_url", "http://localhost:8001"),
        debug=debug,
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

            # ---- Prepare object (pre-encoded SLAT, no mesh.zip) ----
            try:
                ori_slat = refiner.encode_object(None, obj_id)
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
                        # Group modifications: mask ALL parts in the group
                        if spec.remove_part_ids:
                            edit_part_ids = spec.remove_part_ids
                        else:
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

                        # Use prerendered orthogonal view if best_view set
                        prerender_img = None
                        if hasattr(spec, 'best_view') and spec.best_view >= 0:
                            try:
                                from scripts.run_2d_edit import (
                                    prepare_input_image, call_local_edit,
                                    call_vlm_edit)
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view)
                                after_desc = spec.after_desc or spec.after_part_desc or ""
                                before_desc = getattr(spec, 'before_part_desc', '') or ''

                                # Build part label: all labels for groups
                                remove_labels = getattr(spec, 'remove_labels', [])
                                old_label = getattr(spec, 'old_label', '') or ''
                                if remove_labels and len(remove_labels) > 1:
                                    part_label = ", ".join(remove_labels)
                                elif remove_labels:
                                    part_label = remove_labels[0]
                                else:
                                    part_label = old_label

                                if image_edit_backend == "local_diffusers":
                                    edit_url = p25_cfg.get("image_edit_base_url",
                                                           "http://localhost:8001")
                                    edited = call_local_edit(
                                        edit_url, img_bytes, spec.edit_prompt,
                                        after_desc, old_part_label=part_label,
                                        before_part_desc=before_desc,
                                        edit_type=spec.edit_type)
                                elif vlm_client is not None:
                                    edited = call_vlm_edit(
                                        vlm_client, img_bytes, spec.edit_prompt,
                                        after_desc,
                                        p25_cfg.get("image_edit_model", ""),
                                        old_part_label=part_label,
                                        before_part_desc=before_desc,
                                        edit_type=spec.edit_type)
                                else:
                                    edited = None
                                if edited is not None:
                                    edited = edited.resize((518, 518))
                                    prerender_img = (pil_img.resize((518, 518)),
                                                     edited)
                                    logger.info(f"  2D edit from prerender "
                                                f"view {spec.best_view}")
                            except Exception as e:
                                logger.warning(f"  Prerender 2D edit failed: {e}")

                        if prerender_img is not None:
                            original_images = [prerender_img[0]]
                            edited_images = [prerender_img[1]]
                        else:
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
        default_path = PROJECT_ROOT / "configs" / "default.yaml"
        if default_path.exists():
            with open(default_path) as f:
                dcfg = yaml.safe_load(f)
            api_key = dcfg.get("phase0", {}).get("vlm_api_key", "")

    if not api_key:
        env_var = p0.get("vlm_api_key_env", "")
        if env_var:
            api_key = os.environ.get(env_var, "")

    return api_key

# =========================================================================
# Streaming Mode: per-object pipeline (Step 1 → 2 → 4 per object)
# =========================================================================

def run_streaming(cfg, dataset, logger, args):
    """Streaming pipeline: each object goes through the full chain before
    moving to the next.

    Per object:
      1. VLM enrichment → semantic record
      2. Plan edits from record → EditSpecs
      4. For each spec: 2D edit → 3D TRELLIS edit (inline)

    Advantages:
      - First results appear fast (no waiting for all objects)
      - Lower peak memory (one object's SLAT at a time)
      - Immediate feedback on failures
    """
    from openai import OpenAI
    from partcraft.phase1_planning.enricher import (
        _enrich_one_object_visual, _call_vlm, _fallback_enrichment,
        _result_to_phase0_record, load_thumbnail_from_npz,
    )
    from partcraft.phase1_planning.planner import plan_edits_for_record

    logger.info("=" * 60)
    logger.info("STREAMING MODE: per-object pipeline")
    logger.info("=" * 60)

    p0 = cfg.get("phase0", {})
    p25 = cfg.get("phase2_5", {})

    # ---- Paths (resolve relative paths against project root) ----
    project_root = PROJECT_ROOT

    def _resolve(p: str | Path) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else project_root / pp

    tag_suffix = f"_{args.tag}" if args.tag else ""
    num_workers = getattr(args, 'num_workers_stream', 1)
    worker_id = getattr(args, 'worker_id', 0)
    # Per-worker output files to avoid write conflicts
    worker_suffix = f"_w{worker_id}" if num_workers > 1 else ""
    labels_path = _resolve(p0["cache_dir"]) / f"semantic_labels{tag_suffix}{worker_suffix}.jsonl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    specs_path = _resolve(cfg["phase1"]["cache_dir"]) / f"edit_specs{tag_suffix}{worker_suffix}.jsonl"
    specs_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = _resolve(p25["cache_dir"]) / f"edit_results{tag_suffix}{worker_suffix}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # 2D edit cache directory (shared across workers for same tag)
    edit_2d_subdir = f"2d_edits{tag_suffix}"
    edit_2d_dir = _resolve(p25["cache_dir"]) / edit_2d_subdir
    edit_2d_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _resolve(cfg["data"]["output_dir"])
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # ---- Resolve object list from dataset (NPZ files) ----
    # No dependency on source/semantic.json — labels come from NPZ split_mesh.
    uid_info: dict[str, tuple[str, list[str], list[int]]] = {}
    if dataset._index is None:
        dataset._build_index()
    for shard_id, obj_id in dataset._index:
        obj_rec = dataset.load_object(shard_id, obj_id)
        # Extract part labels from mesh_node_names (e.g. "Body_0" → "Body")
        labels = []
        actual_pids = []
        for p in obj_rec.parts:
            if p.mesh_node_names:
                raw = p.mesh_node_names[0]  # e.g. "Monkey Body_6"
                # Strip trailing "_N" suffix
                label = raw.rsplit("_", 1)[0] if "_" in raw else raw
            else:
                label = p.cluster_name
            labels.append(label)
            actual_pids.append(p.part_id)
        uid_info[obj_id] = ("object", labels, actual_pids)
        obj_rec.close()
    logger.info(f"Discovered {len(uid_info)} objects from dataset")

    all_uids = sorted(uid_info.keys())
    if args.limit:
        all_uids = all_uids[:args.limit]

    # ---- Worker partitioning for multi-GPU parallel ----
    num_workers = getattr(args, 'num_workers_stream', 1)
    worker_id = getattr(args, 'worker_id', 0)
    if num_workers > 1:
        all_uids = [uid for i, uid in enumerate(all_uids)
                     if i % num_workers == worker_id]
        logger.info(f"Worker {worker_id}/{num_workers}: "
                    f"processing {len(all_uids)} objects")

    # ---- VLM client for enrichment ----
    api_key = _resolve_api_key(cfg)
    if not api_key:
        logger.error("No API key for VLM enrichment")
        return
    vlm_model = p0.get("vlm_model", "gemini-2.5-flash")
    vlm_base_url = p0.get("vlm_base_url", "")
    vlm_client = OpenAI(base_url=vlm_base_url, api_key=api_key)

    # ---- Image edit setup ----
    image_edit_backend = p25.get("image_edit_backend", "api")
    edit_vlm_client = None
    if image_edit_backend != "local_diffusers":
        image_edit_url = p25.get("image_edit_base_url") or vlm_base_url
        edit_vlm_client = OpenAI(base_url=image_edit_url, api_key=api_key)

    # ---- TRELLIS setup ----
    vinedresser_path = p25.get(
        "vinedresser_path", "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main")
    sys.path.insert(0, vinedresser_path)
    from partcraft.phase2_assembly.trellis_refine import (
        TrellisRefiner, build_prompts_from_spec)

    refiner = TrellisRefiner(
        vinedresser_path=vinedresser_path,
        cache_dir=str(_resolve(p25["cache_dir"])),
        device="cuda",
        image_edit_model=p25.get("image_edit_model", "gemini-2.5-flash-image"),
        image_edit_backend=image_edit_backend,
        image_edit_base_url=p25.get("image_edit_base_url", "http://localhost:8001"),
        debug=args.debug,
    )
    refiner.load_models()

    # ---- Resume tracking ----
    done_labels: set[str] = set()
    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                try:
                    done_labels.add(json.loads(line)["obj_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    done_edits: set[str] = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_edits.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    logger.info(f"Streaming: {len(all_uids)} total objects, "
                f"{len(done_labels)} already enriched (resume)")

    # ---- Image source ----
    npz_dir = _resolve(cfg["data"]["image_npz_dir"])
    shard = cfg["data"].get("shards", ["00"])[0]

    # Import 2D edit helpers
    from scripts.run_2d_edit import (
        prepare_input_image, call_local_edit, call_vlm_edit)

    # ---- Main loop ----
    counters = {"del": 0, "add": 0, "mod": 0, "glb": 0}
    glb_tmp_dir = tempfile.mkdtemp(prefix="partcraft_stream_")
    total_specs = 0
    total_success = 0
    total_fail = 0

    # Load existing specs for resume: reuse them instead of re-planning
    existing_specs_by_obj: dict[str, list] = {}
    if specs_path.exists():
        with open(specs_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                existing_specs_by_obj.setdefault(d["obj_id"], []).append(d)
                # Update counters to continue from existing
                eid = d["edit_id"]
                if eid.startswith("del_") or eid.startswith("gdel_"):
                    counters["del"] = max(counters["del"],
                                          int(eid.split("_")[-1]) + 1)
                elif eid.startswith("add_") or eid.startswith("gadd_"):
                    counters["add"] = max(counters["add"],
                                          int(eid.split("_")[-1]) + 1)
                elif eid.startswith("mod_"):
                    counters["mod"] = max(counters["mod"],
                                          int(eid.split("_")[-1]) + 1)
                elif eid.startswith("glb_"):
                    counters["glb"] = max(counters["glb"],
                                          int(eid.split("_")[-1]) + 1)

    with open(labels_path, "a") as lbl_fp, \
         open(specs_path, "a") as spec_fp, \
         open(results_path, "a") as res_fp:

        for obj_idx, uid in enumerate(all_uids):
            category, labels, actual_pids = uid_info[uid]
            logger.info(f"\n{'='*60}")
            logger.info(f"[{obj_idx+1}/{len(all_uids)}] Object: {uid}")
            logger.info(f"  Category: {category}, Parts: {len(labels)}")

            # ---- Step 1: Enrich ----
            record = None
            if uid in done_labels:
                # Resume: already enriched in this tag's run, load record
                logger.info("  Resuming (already enriched), loading...")
                with open(labels_path) as _lf:
                    for _line in _lf:
                        if not _line.strip():
                            continue
                        try:
                            _rec = json.loads(_line)
                            if _rec["obj_id"] == uid:
                                record = _rec
                                break
                        except (json.JSONDecodeError, KeyError):
                            pass
            else:
                # Fresh VLM enrichment
                try:
                    obj = dataset.load_object(shard, uid)
                    result = _enrich_one_object_visual(
                        vlm_client, vlm_model, obj, category, labels)
                    obj.close()
                except Exception as e:
                    logger.warning(f"  Visual enrichment failed: {e}")
                    result = None

                if result is None:
                    npz_path = npz_dir / shard / f"{uid}.npz"
                    thumb = None
                    if npz_path.exists():
                        thumb = load_thumbnail_from_npz(str(npz_path), view_id=0)
                    result = _call_vlm(vlm_client, vlm_model, category, labels, thumb)

                if result is None:
                    result = _fallback_enrichment(category, labels)
                    logger.warning(f"  Using fallback enrichment for {uid}")

                record = _result_to_phase0_record(
                    result, uid, category, shard,
                    actual_part_ids=actual_pids)
                lbl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                lbl_fp.flush()

            if record is None:
                logger.warning(f"  No record for {uid}, skipping")
                continue

            n_grp = len(record.get("group_edits", []))
            n_glb = len(record.get("global_edits", []))
            logger.info(f"  Enriched: {n_grp} groups, {n_glb} global edits")

            # ---- Step 2: Plan edits for this object ----
            # Reuse existing specs on resume (same IDs → correct done_edits match)
            if uid in existing_specs_by_obj:
                obj_specs = [EditSpec(**d) for d in existing_specs_by_obj[uid]]
                logger.info(f"  Reusing {len(obj_specs)} existing specs (resume)")
            else:
                obj_specs = plan_edits_for_record(record, cfg, counters)
                for spec in obj_specs:
                    spec_fp.write(json.dumps(spec.to_dict(), ensure_ascii=False)
                                  + "\n")
                spec_fp.flush()

            n_del = sum(1 for s in obj_specs if s.edit_type == "deletion")
            n_add = sum(1 for s in obj_specs if s.edit_type == "addition")
            n_mod = sum(1 for s in obj_specs if s.edit_type == "modification")
            n_g = sum(1 for s in obj_specs if s.edit_type == "global")
            logger.info(f"  Planned: {len(obj_specs)} specs "
                        f"(del={n_del} add={n_add} mod={n_mod} glb={n_g})")
            total_specs += len(obj_specs)

            if not obj_specs:
                continue

            # ---- Step 4: 3D edit (with inline 2D) ----
            # Filter out already-done and addition specs
            run_specs = [s for s in obj_specs
                         if s.edit_id not in done_edits
                         and s.edit_type != "addition"]
            add_specs = [s for s in obj_specs if s.edit_type == "addition"]

            if not run_specs and not add_specs:
                logger.info("  All edits already done, skipping")
                continue

            # Prepare object for TRELLIS (pre-encoded SLAT, no mesh.zip)
            try:
                ori_slat = refiner.encode_object(None, uid)
                ori_gaussian = refiner.decode_to_gaussian(ori_slat)
                obj_record = dataset.load_object(shard, uid)
            except Exception as e:
                logger.error(f"  Failed to prepare for TRELLIS: {e}")
                for spec in run_specs:
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": f"Preparation failed: {e}",
                    }) + "\n")
                    total_fail += 1
                res_fp.flush()
                continue

            # Sort: deletion → modification → global
            type_order = {"deletion": 0, "modification": 1, "global": 2}
            run_specs.sort(key=lambda s: type_order.get(s.edit_type, 9))

            # Track first pair dir for shared before export
            first_pair_dir = None

            for spec in run_specs:
                if spec.edit_id in done_edits:
                    continue

                logger.info(f"\n  [{spec.edit_id}] {spec.edit_type}: "
                            f"\"{spec.edit_prompt[:60]}\"")

                try:
                    edit_type = spec.edit_type.capitalize()

                    if edit_type == "Deletion":
                        edit_part_ids = spec.remove_part_ids
                    elif edit_type == "Modification":
                        if spec.remove_part_ids:
                            edit_part_ids = spec.remove_part_ids
                        else:
                            edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Global":
                        edit_part_ids = []
                    else:
                        continue

                    # Build mask
                    mask, effective_type = refiner.build_part_mask(
                        uid, obj_record, edit_part_ids, ori_slat, edit_type)
                    if effective_type != edit_type:
                        logger.info(f"    Auto-promoted → {effective_type}")
                        edit_type = effective_type
                    if mask.sum() == 0:
                        raise RuntimeError("Empty mask")

                    # Build prompts
                    prompts = build_prompts_from_spec(spec)
                    if prompts["edit_type"] != edit_type:
                        prompts["edit_type"] = edit_type

                    # Inline 2D edit (with cache read/write)
                    img_cond = None
                    if args.use_2d:
                        num_edit_views = p25.get("num_edit_views", 4)
                        edit_strength = p25.get("edit_strength", 1.0)
                        prerender_img = None

                        # Check 2D edit cache first
                        cached_2d_path = edit_2d_dir / f"{spec.edit_id}_edited.png"
                        if cached_2d_path.exists():
                            try:
                                cached_edited = Image.open(
                                    str(cached_2d_path)).convert("RGB")
                                cached_edited = cached_edited.resize(
                                    (518, 518))
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view
                                    if hasattr(spec, 'best_view')
                                    and spec.best_view >= 0 else 0)
                                prerender_img = (
                                    pil_img.resize((518, 518)),
                                    cached_edited)
                                logger.info(
                                    f"    2D edit loaded from cache: "
                                    f"{cached_2d_path.name}")
                            except Exception as e:
                                logger.warning(
                                    f"    Failed to load cached 2D "
                                    f"edit: {e}")
                                prerender_img = None

                        if (prerender_img is None
                                and hasattr(spec, 'best_view')
                                and spec.best_view >= 0):
                            try:
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view)
                                after_desc = (spec.after_desc
                                              or spec.after_part_desc or "")
                                before_desc = getattr(
                                    spec, 'before_part_desc', '') or ''
                                remove_labels = getattr(
                                    spec, 'remove_labels', [])
                                old_label = getattr(
                                    spec, 'old_label', '') or ''
                                if (remove_labels
                                        and len(remove_labels) > 1):
                                    part_label = ", ".join(remove_labels)
                                elif remove_labels:
                                    part_label = remove_labels[0]
                                else:
                                    part_label = old_label

                                if image_edit_backend == "local_diffusers":
                                    edit_url = p25.get(
                                        "image_edit_base_url",
                                        "http://localhost:8001")
                                    edited = call_local_edit(
                                        edit_url, img_bytes,
                                        spec.edit_prompt, after_desc,
                                        old_part_label=part_label,
                                        before_part_desc=before_desc,
                                        edit_type=spec.edit_type)
                                elif edit_vlm_client is not None:
                                    edited = call_vlm_edit(
                                        edit_vlm_client, img_bytes,
                                        spec.edit_prompt, after_desc,
                                        p25.get("image_edit_model", ""),
                                        old_part_label=part_label,
                                        before_part_desc=before_desc,
                                        edit_type=spec.edit_type)
                                else:
                                    edited = None
                                if edited is not None:
                                    edited = edited.resize((518, 518))
                                    # Save to cache
                                    edited.save(str(cached_2d_path))
                                    prerender_img = (
                                        pil_img.resize((518, 518)),
                                        edited)
                                    logger.info(
                                        f"    2D edit from view "
                                        f"{spec.best_view} "
                                        f"(saved to cache)")
                            except Exception as e:
                                logger.warning(
                                    f"    Prerender 2D edit failed: {e}")

                        if prerender_img is not None:
                            original_images = [prerender_img[0]]
                            edited_images = [prerender_img[1]]
                        else:
                            original_images, edited_images = \
                                refiner.obtain_edited_images(
                                    ori_gaussian, prompts, edit_vlm_client,
                                    uid, spec.edit_id,
                                    num_views=num_edit_views)
                        if edited_images:
                            img_cond = refiner.encode_multiview_cond(
                                edited_images, original_images,
                                edit_strength=edit_strength)

                    # Run TRELLIS
                    slats_edited = refiner.edit(
                        ori_slat, mask, prompts,
                        img_cond=img_cond, seed=args.seed)
                    if not slats_edited:
                        raise RuntimeError("No edited SLATs produced")

                    # Export (reuse shared before for same object)
                    pair_dir = mesh_pairs_dir / spec.edit_id
                    export_paths = refiner.export_pair_shared_before(
                        ori_slat, slats_edited[0], pair_dir,
                        shared_before_dir=first_pair_dir)

                    record = {
                        "edit_id": spec.edit_id,
                        "edit_type": spec.edit_type,
                        "effective_edit_type": edit_type,
                        "obj_id": uid,
                        "edit_prompt": spec.edit_prompt,
                        **export_paths,
                        "status": "success",
                    }
                    res_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    total_success += 1
                    done_edits.add(spec.edit_id)
                    # First successful export → reuse its before for rest
                    if first_pair_dir is None:
                        first_pair_dir = pair_dir
                    logger.info(f"    OK → {pair_dir}")

                except Exception as e:
                    import traceback
                    logger.error(f"    Failed: {e}")
                    traceback.print_exc()
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": str(e),
                    }) + "\n")
                    res_fp.flush()
                    total_fail += 1

            # Handle additions (swap from deletion)
            for spec in add_specs:
                if spec.edit_id in done_edits:
                    continue
                del_pair = mesh_pairs_dir / spec.source_del_id
                add_pair = mesh_pairs_dir / spec.edit_id
                if (del_pair / "before_slat").exists():
                    add_pair.mkdir(parents=True, exist_ok=True)
                    for src, dst in [
                        (del_pair / "before_slat", add_pair / "after_slat"),
                        (del_pair / "after_slat", add_pair / "before_slat"),
                        (del_pair / "before.ply", add_pair / "after.ply"),
                        (del_pair / "after.ply", add_pair / "before.ply"),
                    ]:
                        if src.exists() and not dst.exists():
                            if src.is_dir():
                                shutil.copytree(str(src), str(dst))
                            else:
                                shutil.copy2(str(src), str(dst))
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "edit_type": "addition",
                        "obj_id": uid, "status": "success",
                        "source_del_id": spec.source_del_id,
                    }, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    total_success += 1
                else:
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": f"Source deletion {spec.source_del_id} "
                                  f"not found",
                    }) + "\n")
                    res_fp.flush()
                    total_fail += 1

            obj_record.close()

    shutil.rmtree(glb_tmp_dir, ignore_errors=True)
    logger.info(f"\n{'='*60}")
    logger.info(f"Streaming complete: {total_specs} specs planned, "
                f"{total_success} ok, {total_fail} fail")
    logger.info(f"  Labels:  {labels_path}")
    logger.info(f"  Specs:   {specs_path}")
    logger.info(f"  Results: {results_path}")
    logger.info("=" * 60)


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
    parser.add_argument("--debug", action="store_true",
                        help="Save debug output files (masks, views, enricher "
                             "ortho images). Off by default.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run (delete and regenerate cached results)")
    parser.add_argument("--streaming", action="store_true",
                        help="Streaming mode: process each object through the "
                             "full chain (enrich → plan → 2D+3D edit) before "
                             "moving to the next. Faster first results, no "
                             "cross-object swap modifications.")
    parser.add_argument("--num-workers", type=int, default=1,
                        dest="num_workers_stream",
                        help="Number of parallel streaming workers (for "
                             "multi-GPU). Each worker processes a different "
                             "subset of objects.")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="This worker's ID (0-indexed). Use with "
                             "--num-workers for multi-GPU parallel.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Normalize cache_dir paths ----
    # Ensure all cache_dir values resolve under output_dir so that
    # streaming, step-by-step, and multi-worker modes share the same
    # cache location.  Relative paths like "cache/phase0" are rewritten
    # to "{output_dir}/cache/phase0".  Absolute paths are kept as-is.
    _out = cfg["data"].get("output_dir", "outputs")
    for _phase_key in ("phase0", "phase1", "phase2", "phase2_5", "phase3", "phase4"):
        _pcfg = cfg.get(_phase_key, {})
        _cd = _pcfg.get("cache_dir", "")
        if _cd and not os.path.isabs(_cd) and not _cd.startswith(_out):
            # Relative cache_dir not under output_dir — rewrite it
            # "cache/phase0" → "outputs/partobjaverse_tiny/cache/phase0"
            _pcfg["cache_dir"] = os.path.join(_out, _cd)

    # Set ATTN_BACKEND from config (unless already set via env)
    attn_backend = cfg.get("pipeline", {}).get("attn_backend", "")
    if attn_backend and "ATTN_BACKEND" not in os.environ:
        os.environ["ATTN_BACKEND"] = attn_backend

    logger = setup_logging(cfg, "pipeline")

    steps = set(args.steps) if args.steps else {1, 2, 3, 4, 5, 6}

    dataset = PartCraftDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )
    logger.info(f"Dataset: {len(dataset)} objects")

    # ---- Streaming mode: per-object full pipeline ----
    if args.streaming:
        run_streaming(cfg, dataset, logger, args)
        return

    # ---- Resolve paths ----
    # Labels are per-dataset (no suffix); specs may vary by semantic style.
    suffix = args.suffix
    tag_suffix = f"_{args.tag}" if args.tag else ""
    labels_path = Path(cfg["phase0"]["cache_dir"]) / f"semantic_labels{tag_suffix}.jsonl"
    specs_path = Path(cfg["phase1"]["cache_dir"]) / f"edit_specs{suffix}{tag_suffix}.jsonl"

    # ---- Step 1: Semantic ----
    if 1 in steps:
        labels_path = run_step_semantic(
            cfg, dataset, logger, limit=args.limit,
            force=args.force, tag=args.tag, debug=args.debug)

    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path}")
        logger.error("Run step 1 first, or provide cached labels")
        sys.exit(1)

    # ---- Step 2: Planning ----
    if 2 in steps:
        specs_path = run_step_planning(cfg, labels_path, logger,
                                       suffix=f"{suffix}{tag_suffix}")

    if not specs_path.exists():
        logger.error(f"Specs not found: {specs_path}")
        logger.error("Run step 2 first, or provide cached specs")
        sys.exit(1)

    # ---- Step 3: 2D Edit ----
    edit_2d_dir = None
    if 3 in steps:
        edit_2d_dir = run_step_2d_edit(
            cfg, specs_path, dataset, logger,
            tag=args.tag, workers=args.workers, limit=args.limit,
            edit_ids=set(args.edit_ids) if args.edit_ids else None)

    # ---- Step 4: 3D Edit ----
    # Resolve 2D edit subdir name from tag (e.g. "2d_edits_v1")
    edit_subdir = args.edit_dir
    if not edit_subdir and args.tag:
        edit_subdir = f"2d_edits_{args.tag}"
    elif not edit_subdir:
        edit_subdir = "2d_edits"

    results_path = None
    if 4 in steps:
        results_path = run_step_3d_edit(
            cfg, specs_path, dataset, logger,
            tag=args.tag, seed=args.seed, limit=args.limit,
            use_2d=args.use_2d, edit_ids=args.edit_ids,
            edit_dir=edit_subdir, debug=args.debug)

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
