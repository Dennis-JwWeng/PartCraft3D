#!/usr/bin/env python3
"""PartCraft3D Streaming Pipeline — per-object full pipeline.

Each object goes through the full chain (enrich → plan → 2D+3D edit)
before moving to the next. First results appear fast, lower peak memory,
and immediate feedback on failures.

Per object:
  1. VLM enrichment → semantic record
  2. Plan edits from record → EditSpecs
  3. For each spec: 2D edit → 3D TRELLIS edit (inline)

Note: No cross-object swap modifications (addition from other objects).
For cross-object swaps, use the batch pipeline (run_pipeline.py).

After streaming completes, run quality scoring + export with:
    python scripts/run_pipeline.py --config <same> --steps 5 6 --tag <same>

Usage:
    # Basic streaming
    ATTN_BACKEND=xformers python scripts/run_streaming.py \\
        --config configs/hybrid_streaming.yaml --tag v1

    # Multi-GPU parallel (2 workers)
    ATTN_BACKEND=xformers python scripts/run_streaming.py \\
        --config configs/hybrid_streaming.yaml --tag v1 \\
        --num-workers 2 --worker-id 0   # GPU 0
    ATTN_BACKEND=xformers python scripts/run_streaming.py \\
        --config configs/hybrid_streaming.yaml --tag v1 \\
        --num-workers 2 --worker-id 1   # GPU 1
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path before any project imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image

from scripts.pipeline_common import (
    PROJECT_ROOT,
    load_config, setup_logging,
    PartCraftDataset, EditSpec,
    resolve_api_key, normalize_cache_dirs, set_attn_backend, create_dataset,
    resolve_data_dirs,
)


def run_streaming(cfg, dataset, logger, args):
    """Streaming pipeline: each object goes through the full chain."""
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

    def _resolve(p) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else project_root / pp

    tag_suffix = f"_{args.tag}" if args.tag else ""
    num_workers = args.num_workers
    worker_id = args.worker_id
    # Per-worker output files to avoid write conflicts
    worker_suffix = f"_w{worker_id}" if num_workers > 1 else ""
    labels_path = (_resolve(p0["cache_dir"])
                   / f"semantic_labels{tag_suffix}{worker_suffix}.jsonl")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    specs_path = (_resolve(cfg["phase1"]["cache_dir"])
                  / f"edit_specs{tag_suffix}{worker_suffix}.jsonl")
    specs_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = (_resolve(p25["cache_dir"])
                    / f"edit_results{tag_suffix}{worker_suffix}.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # 2D edit cache directory (shared across workers for same tag)
    edit_2d_subdir = f"2d_edits{tag_suffix}"
    edit_2d_dir = _resolve(p25["cache_dir"]) / edit_2d_subdir
    edit_2d_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _resolve(cfg["data"]["output_dir"])
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # ---- Resolve object list from dataset (NPZ files) ----
    # uid_info: obj_id -> (category, labels, actual_pids, shard_id)
    uid_info: dict[str, tuple[str, list[str], list[int], str]] = {}
    if dataset._index is None:
        dataset._build_index()
    for shard_id, obj_id in dataset._index:
        obj_rec = dataset.load_object(shard_id, obj_id)
        labels = []
        actual_pids = []
        for p in obj_rec.parts:
            if p.mesh_node_names:
                raw = p.mesh_node_names[0]
                label = raw.rsplit("_", 1)[0] if "_" in raw else raw
            else:
                label = p.cluster_name
            labels.append(label)
            actual_pids.append(p.part_id)
        uid_info[obj_id] = ("object", labels, actual_pids, shard_id)
        obj_rec.close()
    logger.info(f"Discovered {len(uid_info)} objects from dataset")

    all_uids = sorted(uid_info.keys())
    if args.limit:
        all_uids = all_uids[:args.limit]

    # ---- Worker partitioning for multi-GPU parallel ----
    if num_workers > 1:
        all_uids = [uid for i, uid in enumerate(all_uids)
                     if i % num_workers == worker_id]
        logger.info(f"Worker {worker_id}/{num_workers}: "
                    f"processing {len(all_uids)} objects")

    # ---- VLM client for enrichment ----
    api_key = resolve_api_key(cfg)
    if not api_key:
        logger.error("No API key for VLM enrichment")
        return
    vlm_model = p0.get("vlm_model", "gemini-3.1-flash-lite-preview")
    vlm_base_url = p0.get("vlm_base_url", "")
    vlm_client = OpenAI(base_url=vlm_base_url, api_key=api_key)

    # ---- Image edit setup ----
    image_edit_backend = p25.get("image_edit_backend", "api")
    edit_vlm_client = None
    if image_edit_backend != "local_diffusers":
        image_edit_url = p25.get("image_edit_base_url") or vlm_base_url
        edit_vlm_client = OpenAI(base_url=image_edit_url, api_key=api_key)

    # ---- TRELLIS setup ----
    from partcraft.phase2_assembly.trellis_refine import (
        TrellisRefiner, build_prompts_from_spec)

    slat_dir, img_enc_dir = resolve_data_dirs(cfg)
    refiner = TrellisRefiner(
        cache_dir=str(_resolve(p25["cache_dir"])),
        device="cuda",
        image_edit_model=p25.get("image_edit_model", "gemini-2.5-flash-image"),
        image_edit_backend=image_edit_backend,
        image_edit_base_url=p25.get("image_edit_base_url", "http://localhost:8001"),
        debug=args.debug,
        slat_dir=slat_dir,
        img_enc_dir=img_enc_dir,
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

    # Import 2D edit helpers
    from scripts.run_2d_edit import (
        prepare_input_image, call_local_edit, call_vlm_edit)

    # ---- Main loop ----
    # Edit IDs are now per-object unique (obj_id hash embedded), no global
    # counters needed. Same object always produces identical edit IDs, making
    # resume deterministic and multi-worker safe.
    glb_tmp_dir = tempfile.mkdtemp(prefix="partcraft_stream_")
    total_specs = 0
    total_success = 0
    total_fail = 0

    # Load existing specs for resume (reuse instead of re-planning)
    existing_specs_by_obj: dict[str, list] = {}
    if specs_path.exists():
        with open(specs_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                existing_specs_by_obj.setdefault(d["obj_id"], []).append(d)

    with open(labels_path, "a") as lbl_fp, \
         open(specs_path, "a") as spec_fp, \
         open(results_path, "a") as res_fp:

        for obj_idx, uid in enumerate(all_uids):
            category, labels, actual_pids, obj_shard = uid_info[uid]
            logger.info(f"\n{'='*60}")
            logger.info(f"[{obj_idx+1}/{len(all_uids)}] Object: {uid}")
            logger.info(f"  Category: {category}, Parts: {len(labels)}")

            # ---- Step 1: Enrich ----
            record = None
            if uid in done_labels:
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
                    obj = dataset.load_object(obj_shard, uid)
                    result = _enrich_one_object_visual(
                        vlm_client, vlm_model, obj, category, labels)
                    obj.close()
                except Exception as e:
                    logger.warning(f"  Visual enrichment failed: {e}")
                    result = None

                if result is None:
                    npz_path = npz_dir / obj_shard / f"{uid}.npz"
                    thumb = None
                    if npz_path.exists():
                        thumb = load_thumbnail_from_npz(str(npz_path),
                                                        view_id=0)
                    result = _call_vlm(vlm_client, vlm_model,
                                       category, labels, thumb)

                if result is None:
                    result = _fallback_enrichment(category, labels)
                    logger.warning(f"  Using fallback enrichment for {uid}")

                record = _result_to_phase0_record(
                    result, uid, category, obj_shard,
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
            if uid in existing_specs_by_obj:
                obj_specs = [EditSpec(**d)
                             for d in existing_specs_by_obj[uid]]
                logger.info(f"  Reusing {len(obj_specs)} existing specs "
                            f"(resume)")
            else:
                obj_specs = plan_edits_for_record(record, cfg)
                for spec in obj_specs:
                    spec_fp.write(json.dumps(spec.to_dict(),
                                             ensure_ascii=False) + "\n")
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

            # ---- Step 3+4: 3D edit (with inline 2D) ----
            run_specs = [s for s in obj_specs
                         if s.edit_id not in done_edits
                         and s.edit_type != "addition"]
            add_specs = [s for s in obj_specs if s.edit_type == "addition"]

            if not run_specs and not add_specs:
                logger.info("  All edits already done, skipping")
                continue

            # Prepare object for TRELLIS
            try:
                ori_slat = refiner.encode_object(None, uid)
                ori_gaussian = refiner.decode_to_gaussian(ori_slat)
                obj_record = dataset.load_object(obj_shard, uid)
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

            first_pair_dir = None

            for spec in run_specs:
                if spec.edit_id in done_edits:
                    continue

                logger.info(f"\n  [{spec.edit_id}] {spec.edit_type}: "
                            f"\"{spec.edit_prompt[:60]}\"")

                try:
                    edit_type = spec.edit_type.capitalize()

                    if edit_type == "Deletion":
                        # Direct GT mesh deletion — no SLAT/generation needed
                        from partcraft.phase2_assembly.trellis_refine import TrellisRefiner
                        pair_dir = mesh_pairs_dir / spec.edit_id
                        export_paths = TrellisRefiner.direct_delete_mesh(
                            obj_record, spec.remove_part_ids, pair_dir)
                        rec = {
                            "edit_id": spec.edit_id,
                            "edit_type": spec.edit_type,
                            "effective_edit_type": "DirectDeletion",
                            "obj_id": uid,
                            "edit_prompt": spec.edit_prompt,
                            **export_paths,
                            "status": "success",
                        }
                        res_fp.write(json.dumps(rec,
                                                ensure_ascii=False) + "\n")
                        res_fp.flush()
                        total_success += 1
                        done_edits.add(spec.edit_id)
                        if first_pair_dir is None:
                            first_pair_dir = pair_dir
                        logger.info(f"    OK (direct deletion, GT mesh) "
                                    f"→ {pair_dir}")
                        continue
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
                        cached_2d_path = (edit_2d_dir
                                          / f"{spec.edit_id}_edited.png")
                        if cached_2d_path.exists():
                            try:
                                cached_edited = Image.open(
                                    str(cached_2d_path))
                                cached_edited.load()  # force full decode
                                cached_edited = cached_edited.convert(
                                    "RGB").resize((518, 518))
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
                                    f"    Corrupt cached 2D edit "
                                    f"(deleting): {e}")
                                cached_2d_path.unlink(missing_ok=True)
                                prerender_img = None

                        if (prerender_img is None
                                and hasattr(spec, 'best_view')
                                and spec.best_view >= 0):
                            try:
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view)
                                after_desc = (spec.after_desc
                                              or spec.after_part_desc or "")
                                before_desc = (
                                    getattr(spec, 'before_part_desc', '')
                                    or '')
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
                                    # Atomic save: write to temp then rename
                                    tmp_path = cached_2d_path.with_suffix(
                                        ".tmp.png")
                                    edited.save(str(tmp_path))
                                    tmp_path.rename(cached_2d_path)
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

                    rec = {
                        "edit_id": spec.edit_id,
                        "edit_type": spec.edit_type,
                        "effective_edit_type": edit_type,
                        "obj_id": uid,
                        "edit_prompt": spec.edit_prompt,
                        **export_paths,
                        "status": "success",
                    }
                    res_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    total_success += 1
                    done_edits.add(spec.edit_id)
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
                del_has_output = ((del_pair / "before_slat").exists()
                                  or (del_pair / "before.ply").exists())
                if del_has_output:
                    # Clean up any partial previous copy
                    if add_pair.exists():
                        shutil.rmtree(str(add_pair))
                    add_pair.mkdir(parents=True, exist_ok=True)
                    for src, dst in [
                        (del_pair / "before_slat", add_pair / "after_slat"),
                        (del_pair / "after_slat", add_pair / "before_slat"),
                        (del_pair / "before.ply", add_pair / "after.ply"),
                        (del_pair / "after.ply", add_pair / "before.ply"),
                    ]:
                        if src.exists():
                            if src.is_dir():
                                shutil.copytree(str(src), str(dst))
                            else:
                                shutil.copy2(str(src), str(dst))
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "edit_type": "addition",
                        "obj_id": uid, "status": "success",
                        "edit_prompt": spec.edit_prompt,
                        "source_del_id": spec.source_del_id,
                    }, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    done_edits.add(spec.edit_id)
                    total_success += 1
                else:
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": f"Source deletion {spec.source_del_id} "
                                  f"output not found",
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
        description="PartCraft3D Streaming Pipeline (per-object)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each object goes through: enrich → plan → 2D+3D edit before moving on.

After streaming completes, run quality scoring + export separately:
    python scripts/run_pipeline.py --config <same> --steps 5 6 --tag <same>

Examples:
  # Basic streaming
  ATTN_BACKEND=xformers python scripts/run_streaming.py \\
      --config configs/hybrid_streaming.yaml --tag v1

  # Multi-GPU parallel
  ATTN_BACKEND=xformers python scripts/run_streaming.py \\
      --config configs/hybrid_streaming.yaml --tag v1 \\
      --num-workers 2 --worker-id 0
""")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag for output isolation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-2d-edit", dest="use_2d", action="store_false",
                        default=True)
    parser.add_argument("--debug", action="store_true",
                        help="Save debug output files")
    parser.add_argument("--num-workers", type=int, default=1,
                        dest="num_workers",
                        help="Number of parallel streaming workers "
                             "(for multi-GPU)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="This worker's ID (0-indexed)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    normalize_cache_dirs(cfg)
    set_attn_backend(cfg)

    logger = setup_logging(cfg, "streaming")

    dataset = create_dataset(cfg)
    logger.info(f"Dataset: {len(dataset)} objects")

    run_streaming(cfg, dataset, logger, args)


if __name__ == "__main__":
    main()
