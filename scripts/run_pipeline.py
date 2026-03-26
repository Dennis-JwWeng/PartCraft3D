#!/usr/bin/env python3
"""Batch pipeline: phases 1–6 (semantic → export), one phase at a time, resumable.

Steps: 1 semantic 2 plan 3 2D 4 TRELLIS 3D 5 quality 6 export.
Examples: ``python scripts/run_pipeline.py`` | ``--steps 3 4 5`` | ``--tag v1``
| ``--no-2d-edit`` | ``--steps 4 --2d-cache-only --edit-dir 2d_edits_TAG``
| ``--dry-run``. Per-object mode: ``scripts/run_streaming.py``."""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

# Ensure project root is on sys.path before any project imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image

from scripts.pipeline_common import (
    PROJECT_ROOT, COST,
    load_config, setup_logging,
    PartCraftDataset, EditSpec,
    resolve_api_key, normalize_cache_dirs, set_attn_backend, create_dataset,
    resolve_data_dirs,
)
from partcraft.edit_types import IDENTITY, TYPE_ORDER


# =========================================================================
# Step 1: Semantic Labeling + Enrichment
# =========================================================================

def run_step_semantic(cfg, dataset, logger, limit=None, force=False, tag=None,
                      debug=False):
    """Step 1: VLM semantic labeling via enricher.

    Uses dataset NPZ files to discover objects and part labels.
    No dependency on source/semantic.json.
    """
    from partcraft.phase1_planning.enricher import enrich_semantic_labels

    logger.info("=" * 60)
    logger.info("STEP 1: Semantic Labeling + Enrichment")
    logger.info("=" * 60)

    cache_dir = Path(cfg["phase0"]["cache_dir"])
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
    """Step 3: Pre-generate 2D edited images for all specs."""
    logger.info("=" * 60)
    logger.info("STEP 3: 2D Image Editing")
    logger.info("=" * 60)

    p0 = cfg["phase0"]
    p25 = cfg.get("phase2_5", {})

    image_edit_backend = p25.get("image_edit_backend", "api")

    # Local edit server or API client
    client = None
    edit_server_url = None
    model = p25.get("image_edit_model", "gemini-2.5-flash-image")

    if image_edit_backend == "local_diffusers":
        from scripts.run_2d_edit import check_edit_server
        edit_server_url = p25.get("image_edit_base_url", "http://localhost:8001")
        if not check_edit_server(edit_server_url):
            logger.warning(f"Image edit server not reachable at {edit_server_url}, "
                           "skipping 2D editing")
            return None
        logger.info(f"Image edit server OK at {edit_server_url}")
        cfg_workers = p25.get("image_edit_workers", 1)
        if workers != cfg_workers:
            logger.info(f"local_diffusers backend: workers={cfg_workers} "
                         f"(from config)")
            workers = cfg_workers
    else:
        from openai import OpenAI
        api_key = resolve_api_key(cfg)
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
                     edit_dir=None, debug=False, cache_only_2d: bool = False):
    """TRELLIS Step 4. With ``use_2d``: load ``{edit_dir}/{edit_id}_edited.png``
    first; if ``cache_only_2d``, skip image HTTP/VLM (missing PNG → no cond)."""
    logger.info("=" * 60)
    logger.info("STEP 4: 3D Editing — TRELLIS")
    logger.info("=" * 60)

    p25_cfg = cfg.get("phase2_5", {})

    from partcraft.phase2_assembly.trellis_refine import (
        TrellisRefiner, build_prompts_from_spec)

    # ---- Paths ----
    output_dir = Path(cfg["data"]["output_dir"])
    cache_dir = Path(p25_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # ---- Load specs (match streaming / PartVerse) ----
    edit_types = edit_types or [
        "deletion", "addition", "modification", "scale", "material",
        "global", "identity",
    ]
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
        api_key = resolve_api_key(cfg)
        if api_key:
            p0 = cfg["phase0"]
            image_edit_url = (p25_cfg.get("image_edit_base_url")
                              or p0.get("vlm_base_url", ""))
            vlm_client = OpenAI(
                base_url=image_edit_url,
                api_key=api_key,
            )

    # ---- Init refiner ----
    slat_dir, img_enc_dir = resolve_data_dirs(cfg)
    refiner = TrellisRefiner(
        cache_dir=str(cache_dir),
        device="cuda",
        image_edit_model=p25_cfg.get("image_edit_model", "gemini-2.5-flash-image"),
        ckpt_dir=cfg.get("ckpt_root"),
        image_edit_backend=image_edit_backend,
        image_edit_base_url=p25_cfg.get("image_edit_base_url", "http://localhost:8001"),
        debug=debug,
        slat_dir=slat_dir,
        img_enc_dir=img_enc_dir,
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

            def _spec_order(s: EditSpec):
                t = s.edit_type
                if t == "deletion":
                    return (0, 0)
                if t == "addition":
                    return (1, 0)
                return (2, TYPE_ORDER.get(t, 99))

            specs.sort(key=_spec_order)

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

            idt_specs = [s for s in run_specs if s.edit_type == IDENTITY]
            run_specs = [s for s in run_specs if s.edit_type != IDENTITY]

            if not run_specs:
                # Identity-only object: still need first_pair_dir from prior
                # edits on same object (not supported here).
                for spec in idt_specs:
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": "identity needs prior edits on object",
                    }, ensure_ascii=False) + "\n")
                    fail += 1
                out_fp.flush()
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

            first_pair_dir = None

            for spec in run_specs:
                logger.info(f"\n[{success+fail+1}/{len(pending)}] "
                            f"{spec.edit_id} ({spec.edit_type}): "
                            f"\"{spec.edit_prompt[:80]}\"")

                try:
                    edit_type = spec.edit_type.capitalize()

                    if edit_type == "Deletion":
                        # Direct GT mesh deletion — no SLAT/generation needed
                        from partcraft.phase2_assembly.trellis_refine import TrellisRefiner
                        pair_dir = mesh_pairs_dir / spec.edit_id
                        export_paths = TrellisRefiner.direct_delete_mesh(
                            obj_record, spec.remove_part_ids, pair_dir)
                        record = {
                            "edit_id": spec.edit_id,
                            "edit_type": spec.edit_type,
                            "effective_edit_type": "DirectDeletion",
                            "obj_id": obj_id,
                            "shard": spec.shard,
                            "object_desc": spec.object_desc,
                            "edit_prompt": spec.edit_prompt,
                            "after_desc": spec.after_desc,
                            **export_paths,
                            "status": "success",
                        }
                        out_fp.write(json.dumps(record,
                                                ensure_ascii=False) + "\n")
                        out_fp.flush()
                        success += 1
                        if first_pair_dir is None:
                            first_pair_dir = pair_dir
                        logger.info(f"  → direct deletion (GT mesh) "
                                    f"saved {pair_dir}")
                        continue
                    elif edit_type in ("Modification", "Scale"):
                        if spec.remove_part_ids:
                            edit_part_ids = spec.remove_part_ids
                        else:
                            edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Material":
                        edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Global":
                        edit_part_ids = []
                    elif edit_type == "Addition":
                        del_pair_dir = mesh_pairs_dir / spec.source_del_id
                        add_pair_dir = mesh_pairs_dir / spec.edit_id
                        del_has_output = ((del_pair_dir / "before_slat").exists()
                                          or (del_pair_dir / "before.ply").exists())
                        if del_has_output:
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

                    # 2D image conditioning (disk first — same paths as Step 3 / streaming)
                    img_cond = None
                    if edit_type in ("Modification", "Scale", "Material",
                                     "Global") and use_2d:
                        num_edit_views = p25_cfg.get("num_edit_views", 4)
                        edit_strength = p25_cfg.get("edit_strength", 1.0)

                        prerender_img = None
                        _2d_base = cache_dir / (edit_dir or "2d_edits")
                        _ced = _2d_base / f"{spec.edit_id}_edited.png"
                        if _ced.exists():
                            try:
                                from scripts.run_2d_edit import (
                                    prepare_input_image)
                                edited = Image.open(_ced).convert("RGB").resize(
                                    (518, 518))
                                _cin = _2d_base / f"{spec.edit_id}_input.png"
                                if _cin.exists():
                                    pil_in = Image.open(_cin).convert(
                                        "RGB").resize((518, 518))
                                elif (hasattr(spec, "best_view")
                                      and spec.best_view >= 0):
                                    _, pil_img = prepare_input_image(
                                        obj_record, spec.best_view)
                                    pil_in = pil_img.resize((518, 518))
                                else:
                                    pil_in = edited
                                prerender_img = (pil_in, edited)
                                logger.info(
                                    f"  2D from disk ({_2d_base.name}/"
                                    f"{spec.edit_id}_edited.png)")
                            except Exception as e:
                                logger.warning(f"  Cached 2D load failed: {e}")

                        if (prerender_img is None and not cache_only_2d
                                and hasattr(spec, 'best_view')
                                and spec.best_view >= 0):
                            try:
                                from scripts.run_2d_edit import (
                                    prepare_input_image, call_local_edit,
                                    call_vlm_edit)
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view)
                                after_desc = (spec.after_desc
                                              or spec.after_part_desc or "")
                                before_desc = (getattr(spec, 'before_part_desc', '')
                                               or '')
                                remove_labels = getattr(spec, 'remove_labels', [])
                                old_label = getattr(spec, 'old_label', '') or ''
                                if remove_labels and len(remove_labels) > 1:
                                    part_label = ", ".join(remove_labels)
                                elif remove_labels:
                                    part_label = remove_labels[0]
                                else:
                                    part_label = old_label

                                if image_edit_backend == "local_diffusers":
                                    edit_url = p25_cfg.get(
                                        "image_edit_base_url",
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
                                logger.warning(
                                    f"  Prerender 2D edit failed: {e}")

                        if prerender_img is not None:
                            original_images = [prerender_img[0]]
                            edited_images = [prerender_img[1]]
                        elif not cache_only_2d:
                            original_images, edited_images = \
                                refiner.obtain_edited_images(
                                    ori_gaussian, prompts, vlm_client,
                                    obj_id, spec.edit_id,
                                    num_views=num_edit_views,
                                    edit_dir=edit_dir)
                        else:
                            original_images, edited_images = [], []
                            logger.warning(
                                "  --2d-cache-only: missing %s_edited.png → no img cond",
                                spec.edit_id)

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
                    if first_pair_dir is None:
                        first_pair_dir = pair_dir
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

            for spec in idt_specs:
                idt_pair = mesh_pairs_dir / spec.edit_id
                if first_pair_dir and (first_pair_dir / "before.ply").exists():
                    idt_pair.mkdir(parents=True, exist_ok=True)
                    before_ply = first_pair_dir / "before.ply"
                    shutil.copy2(str(before_ply),
                                 str(idt_pair / "before.ply"))
                    shutil.copy2(str(before_ply),
                                 str(idt_pair / "after.ply"))
                    before_slat = first_pair_dir / "before_slat"
                    if before_slat.exists():
                        for _slat_name in ("before_slat", "after_slat"):
                            _dst = idt_pair / _slat_name
                            if _dst.exists():
                                shutil.rmtree(_dst)
                            shutil.copytree(
                                str(before_slat), str(_dst))
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id,
                        "edit_type": IDENTITY,
                        "effective_edit_type": "Identity",
                        "obj_id": obj_id, "status": "success",
                        "edit_prompt": spec.edit_prompt,
                    }, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                    logger.info(f"  [{spec.edit_id}] identity: before=after (no-op)")
                else:
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": "No before mesh available for identity",
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

    phase3_dir = cache_dir / f"phase3{tag_suffix}"
    run_vlm_filter(
        cfg, str(results_path), str(mesh_pairs_dir),
        str(phase3_dir),
        limit=limit)

    scores_file = phase3_dir / "vlm_scores.jsonl"
    out = str(scores_file) if scores_file.is_file() else None
    logger.info(f"Quality scores: {out}")
    return out


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
        with open(scores_path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip().replace("\x00", "")
                if not line or not line.startswith("{"):
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid JSON in scores file line %s", lineno
                    )
                    continue
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
    with EditPairWriter(output_dir,
                        filename=f"edit_pairs{tag_suffix}.jsonl") as writer:
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
                edit_prompt=spec.edit_prompt,
                after_desc=spec.after_desc,
                quality_tier=entry.get("quality_tier", ""),
                quality_score=float(
                    entry.get("quality_score", entry.get("score", 0.0))
                ),
            )
            writer.write_pair(record)
            exported += 1

    logger.info(f"Exported {exported} pairs → {export_path}")
    return export_path


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PartCraft3D Batch Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps 1–6: semantic, plan, 2D, TRELLIS, quality, export. "
               "Examples: run_pipeline.py | --steps 4 5 | --dry-run")
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
    parser.add_argument("--2d-cache-only", dest="cache_only_2d",
                        action="store_true", default=False,
                        help="Step 4: 2D from cache only (no image API)")
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    normalize_cache_dirs(cfg)
    set_attn_backend(cfg)
    if cfg.get("ckpt_root"):
        os.environ.setdefault("PARTCRAFT_CKPT_ROOT", cfg["ckpt_root"])

    logger = setup_logging(cfg, "pipeline")

    steps = set(args.steps) if args.steps else {1, 2, 3, 4, 5, 6}

    dataset = create_dataset(cfg)
    logger.info(f"Dataset: {len(dataset)} objects")

    # ---- Resolve paths ----
    suffix = args.suffix
    tag_suffix = f"_{args.tag}" if args.tag else ""
    labels_path = (Path(cfg["phase0"]["cache_dir"])
                   / f"semantic_labels{tag_suffix}.jsonl")
    specs_path = (Path(cfg["phase1"]["cache_dir"])
                  / f"edit_specs{suffix}{tag_suffix}.jsonl")

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
            edit_dir=edit_subdir, debug=args.debug,
            cache_only_2d=args.cache_only_2d)

    if results_path is None:
        p25_cfg = cfg.get("phase2_5", {})
        results_path = (Path(p25_cfg["cache_dir"])
                        / f"edit_results{tag_suffix}.jsonl")

    # ---- Step 5: Quality ----
    scores_path = None
    if 5 in steps and results_path and Path(results_path).exists():
        scores_path = run_step_quality(
            cfg, results_path, logger, tag=args.tag, limit=args.limit)

    if scores_path is None and args.tag:
        p25_cfg = cfg.get("phase2_5", {})
        cand = Path(p25_cfg["cache_dir"]) / f"phase3_{args.tag}" / "vlm_scores.jsonl"
        if cand.is_file():
            scores_path = str(cand)
            logger.info(f"Using existing quality scores: {scores_path}")

    # ---- Step 6: Export ----
    if 6 in steps:
        run_step_export(cfg, specs_path, scores_path, logger, tag=args.tag)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
