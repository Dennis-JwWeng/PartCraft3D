#!/usr/bin/env python3
"""Run Phase 2.5: TRELLIS-based 3D editing following Vinedresser3D's pipeline.

Processes edit specs from Phase 1 (modification, deletion, addition) using:
  1. Vinedresser3D's exact encoding pipeline (Blender + DINOv2 + SLAT)
  2. HY3D-Part ground-truth part segmentation (replaces PartField)
  3. Phase 0/1 pre-computed prompts (replaces VLM prompt generation)
  4. Gemini 2D image editing (same as Vinedresser3D)
  5. TRELLIS Flow Inversion + Repaint (same as Vinedresser3D)

Usage:
    # Run all edit specs (modification + deletion + addition)
    ATTN_BACKEND=xformers python scripts/run_phase2_5.py

    # Modification only, first 10
    ATTN_BACKEND=xformers python scripts/run_phase2_5.py --type modification --limit 10

    # Resume from checkpoint
    ATTN_BACKEND=xformers python scripts/run_phase2_5.py --resume

    # Specific combinations (default: all 5 like Vinedresser3D)
    ATTN_BACKEND=xformers python scripts/run_phase2_5.py --combs 0 1 2
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.hy3d_loader import HY3DPartDataset
from partcraft.phase1_planning.planner import EditSpec


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.5: TRELLIS 3D editing (Vinedresser3D pipeline)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max edit specs to process")
    parser.add_argument("--type", type=str, default=None,
                        choices=["modification", "deletion", "addition", "global"],
                        help="Filter by edit type (default: all)")
    parser.add_argument("--combs", type=int, nargs="*", default=None,
                        help="Combination indices to run (default: all 5)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-2d-edit", dest="use_2d", action="store_false",
                        default=True,
                        help="Skip 2D image editing (text-only guidance)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        default=True,
                        help="Disable resume (re-process all edit_ids)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag — outputs go to mesh_pairs_{tag}/ "
                             "and edit_results_{tag}.jsonl (default: no tag)")
    parser.add_argument("--edit-ids", nargs="*", default=None,
                        help="Specific edit IDs to process (e.g. mod_000024 del_000020)")
    parser.add_argument("--specs", type=str, default=None,
                        help="Path to edit_specs JSONL "
                             "(default: {phase1.cache_dir}/edit_specs.jsonl)")
    parser.add_argument("--edit-dir", type=str, default=None,
                        help="Pre-generated 2D edits subdir name "
                             "(e.g. '2d_edits_action'). Looked up under "
                             "{phase2_5.cache_dir}/{edit-dir}/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "phase2_5")
    p25_cfg = cfg.get("phase2_5", {})

    # ---- Paths ----
    vinedresser_path = p25_cfg.get(
        "vinedresser_path", "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main")
    sys.path.insert(0, vinedresser_path)

    data_dir = Path(cfg["data"].get("data_dir",
                                     "data/partobjaverse_tiny"))
    mesh_zip = data_dir / "source" / "mesh.zip"
    if not mesh_zip.exists():
        # Fallback: construct from image_npz_dir
        img_dir = Path(cfg["data"]["image_npz_dir"])
        data_dir = img_dir.parent
        mesh_zip = data_dir / "source" / "mesh.zip"

    if not mesh_zip.exists():
        logger.error(f"source/mesh.zip not found at {mesh_zip}")
        sys.exit(1)

    output_dir = Path(cfg["data"]["output_dir"])
    # NOTE: load_config() already resolves relative cache_dir against
    # output_dir.  Do NOT join again — that caused double-nesting:
    #   outputs/X/outputs/X/cache/phase2_5
    cache_dir = Path(p25_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Tag-based output isolation for A/B experiments
    tag_suffix = f"_{args.tag}" if args.tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # ---- Load dataset ----
    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )

    # ---- Load edit specs ----
    specs_path = Path(args.specs) if args.specs else (
        Path(cfg["phase1"]["cache_dir"]) / "edit_specs.jsonl")
    all_specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            # Filter by type if requested
            if args.type and spec.edit_type != args.type:
                continue
            # Filter by edit IDs if requested
            if args.edit_ids and spec.edit_id not in args.edit_ids:
                continue
            all_specs.append(spec)

    if not args.edit_ids:
        limit = args.limit or p25_cfg.get("max_refine", 1000)
        all_specs = all_specs[:limit]

    if not all_specs:
        logger.info("No edit specs found. Run Phase 0 + Phase 1 first.")
        return

    # ---- Resume ----
    output_path = cache_dir / f"edit_results{tag_suffix}.jsonl"
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [s for s in all_specs if s.edit_id not in done_ids]
    logger.info(f"Phase 2.5: {len(pending)} edits to process "
                f"({len(done_ids)} already done)")
    if not pending:
        logger.info("All edits already processed")
        return

    # ---- VLM client for 2D image editing ----
    vlm_client = None
    if args.use_2d:
        from openai import OpenAI
        api_key = _resolve_api_key(cfg)
        if api_key:
            p0 = cfg["phase0"]
            image_edit_url = p25_cfg.get("image_edit_base_url") or p0.get("vlm_base_url", "")
            vlm_client = OpenAI(
                base_url=image_edit_url,
                api_key=api_key,
            )
            logger.info(f"2D image editing enabled "
                        f"(model: {p25_cfg.get('image_edit_model', 'gemini-2.5-flash-image')})")
        else:
            logger.warning("No API key found — 2D image editing disabled")

    # ---- Initialize refiner ----
    from partcraft.phase2_assembly.trellis_refine import (
        TrellisRefiner, build_prompts_from_spec)

    refiner = TrellisRefiner(
        vinedresser_path=vinedresser_path,
        cache_dir=str(cache_dir),
        device="cuda",
        image_edit_model=p25_cfg.get("image_edit_model",
                                      "gemini-2.5-flash-image"),
    )
    refiner.load_models()

    # ---- Build combinations list ----
    all_combinations = [
        {"s1_pos_cond": "new_s1_cpl", "s1_neg_cond": "ori_s1_cpl",
         "s2_pos_cond": "new_s2_cpl", "s2_neg_cond": "ori_s2_cpl",
         "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "new_s1_cpl", "s1_neg_cond": "null",
         "s2_pos_cond": "new_s2_cpl", "s2_neg_cond": "null",
         "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "new_s1_part", "s1_neg_cond": "ori_s1_part",
         "s2_pos_cond": "new_s2_part", "s2_neg_cond": "ori_s2_part",
         "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "new_s1_part", "s1_neg_cond": "null",
         "s2_pos_cond": "new_s2_part", "s2_neg_cond": "null",
         "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "null", "s1_neg_cond": "null",
         "s2_pos_cond": "null", "s2_neg_cond": "null",
         "cnt": 1, "cfg_strength": 0},
    ]

    if args.combs is not None:
        combinations = [all_combinations[i] for i in args.combs
                        if i < len(all_combinations)]
    else:
        combinations = None  # let refiner.edit() use defaults per edit_type

    # ---- Process each edit spec ----
    # Group by obj_id to reuse SLAT encoding.
    # Within each object, process deletion → addition → modification
    # (specs are already sorted this way by planner).
    from collections import OrderedDict
    obj_groups: OrderedDict[str, list] = OrderedDict()
    for spec in pending:
        obj_groups.setdefault(spec.obj_id, []).append(spec)

    success, fail = 0, 0
    glb_tmp_dir = tempfile.mkdtemp(prefix="partcraft_glb_")

    with open(output_path, "a") as out_fp:
        for obj_id, specs in obj_groups.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Object: {obj_id} ({len(specs)} edits)")
            logger.info(f"{'='*60}")

            # Sort: deletion first, then addition, then modification
            type_order = {"deletion": 0, "addition": 1, "modification": 2}
            specs.sort(key=lambda s: type_order.get(s.edit_type, 9))

            # ---- Addition = deletion with before/after swapped ----
            # Handle additions first (no VD inference needed).
            add_specs = [s for s in specs if s.edit_type == "addition"]
            run_specs = [s for s in specs if s.edit_type != "addition"]

            for spec in add_specs:
                del_id = spec.source_del_id
                del_pair_dir = mesh_pairs_dir / del_id
                add_pair_dir = mesh_pairs_dir / spec.edit_id

                # Check if the source deletion has already been exported
                # (could be from a previous run or will be created below)
                del_before_slat = del_pair_dir / "before_slat"
                del_after_slat = del_pair_dir / "after_slat"

                if del_before_slat.exists() and del_after_slat.exists():
                    # Swap: deletion's before→addition's after,
                    #        deletion's after→addition's before
                    add_pair_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
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

                    record = {
                        "edit_id": spec.edit_id,
                        "edit_type": "addition",
                        "obj_id": spec.obj_id,
                        "shard": spec.shard,
                        "object_desc": spec.object_desc,
                        "edit_prompt": spec.edit_prompt,
                        "after_desc": spec.after_desc,
                        "source_del_id": del_id,
                        "before_ply": str(add_pair_dir / "before.ply"),
                        "after_ply": str(add_pair_dir / "after.ply"),
                        "before_slat": str(add_pair_dir / "before_slat"),
                        "after_slat": str(add_pair_dir / "after_slat"),
                        "status": "success",
                    }
                    out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                    logger.info(f"  {spec.edit_id}: swapped from {del_id} "
                                f"(no VD inference)")
                else:
                    # Source deletion not yet exported — defer to after
                    # deletion processing
                    run_specs.append(spec)

            if not run_specs:
                continue

            # ---- Prepare object (only if we have deletion/modification) ----
            needs_vd = any(s.edit_type in ("deletion", "modification", "global")
                           for s in run_specs)
            if not needs_vd:
                # Only deferred additions left — process them after
                # we check again below
                pass

            try:
                glb_path = refiner.extract_glb(
                    str(mesh_zip), obj_id, glb_tmp_dir)
                logger.info(f"Extracted GLB: {glb_path}")

                ori_slat = refiner.encode_object(glb_path, obj_id)
                logger.info(f"SLAT: {ori_slat.feats.shape[0]} voxels, "
                            f"feats={ori_slat.feats.shape}")

                ori_gaussian = refiner.decode_to_gaussian(ori_slat)

                shard = run_specs[0].shard
                obj_record = dataset.load_object(shard, obj_id)

            except Exception as e:
                logger.error(f"Failed to prepare {obj_id}: {e}")
                for spec in run_specs:
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": f"Preparation failed: {e}",
                    }) + "\n")
                    fail += 1
                out_fp.flush()
                continue

            for spec in run_specs:
                prompt_preview = spec.edit_prompt[:80]
                if len(spec.edit_prompt) > 80:
                    prompt_preview += "..."
                logger.info(
                    f"\n[{success+fail+1}/{len(pending)}] "
                    f"{spec.edit_id} ({spec.edit_type}): "
                    f"\"{prompt_preview}\"")

                try:
                    edit_type = spec.edit_type.capitalize()

                    if edit_type == "Deletion":
                        edit_part_ids = spec.remove_part_ids
                    elif edit_type == "Modification":
                        edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Global":
                        edit_part_ids = []  # no part mask needed
                    elif edit_type == "Addition":
                        # Deferred addition — source deletion just ran above
                        del_id = spec.source_del_id
                        del_pair_dir = mesh_pairs_dir / del_id
                        add_pair_dir = mesh_pairs_dir / spec.edit_id
                        if (del_pair_dir / "before_slat").exists():
                            add_pair_dir.mkdir(parents=True, exist_ok=True)
                            import shutil
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
                            record = {
                                "edit_id": spec.edit_id,
                                "edit_type": "addition",
                                "obj_id": spec.obj_id,
                                "shard": spec.shard,
                                "object_desc": spec.object_desc,
                                "edit_prompt": spec.edit_prompt,
                                "after_desc": spec.after_desc,
                                "source_del_id": del_id,
                                "before_ply": str(add_pair_dir / "before.ply"),
                                "after_ply": str(add_pair_dir / "after.ply"),
                                "before_slat": str(add_pair_dir / "before_slat"),
                                "after_slat": str(add_pair_dir / "after_slat"),
                                "status": "success",
                            }
                            out_fp.write(
                                json.dumps(record, ensure_ascii=False) + "\n")
                            out_fp.flush()
                            success += 1
                            logger.info(f"  {spec.edit_id}: swapped from "
                                        f"{del_id} (no VD inference)")
                        else:
                            out_fp.write(json.dumps({
                                "edit_id": spec.edit_id,
                                "status": "failed",
                                "reason": f"Source deletion {del_id} not found",
                            }) + "\n")
                            out_fp.flush()
                            fail += 1
                        continue
                    else:
                        logger.warning(f"Unknown edit type: {spec.edit_type}")
                        continue

                    # Build mask (may promote large parts to Global)
                    mask, effective_type = refiner.build_part_mask(
                        obj_id, obj_record, edit_part_ids,
                        ori_slat, edit_type)

                    if effective_type != edit_type:
                        logger.info(
                            f"  {spec.edit_id}: promoted {edit_type} → "
                            f"{effective_type} (large part)")
                        edit_type = effective_type

                    if mask.sum() == 0:
                        logger.warning(f"Empty mask for {spec.edit_id}")
                        out_fp.write(json.dumps({
                            "edit_id": spec.edit_id,
                            "status": "failed",
                            "reason": "Empty mask",
                        }) + "\n")
                        out_fp.flush()
                        fail += 1
                        continue

                    # Build prompts
                    prompts = build_prompts_from_spec(spec)
                    # Override edit_type in prompts if promoted
                    if prompts["edit_type"] != edit_type:
                        prompts["edit_type"] = edit_type

                    # 2D image conditioning
                    # - Modification: always if enabled
                    # - Deletion: all deletions now use guided Modification
                    #   path, so image conditioning helps show the closed
                    #   surface after part removal
                    img_cond = None
                    use_img_cond = (
                        edit_type in ("Modification", "Deletion", "Global")
                        and args.use_2d)
                    if use_img_cond:
                        num_edit_views = p25_cfg.get("num_edit_views", 4)
                        edit_strength = p25_cfg.get("edit_strength", 1.0)
                        original_images, edited_images = \
                            refiner.obtain_edited_images(
                                ori_gaussian, prompts, vlm_client,
                                obj_id, spec.edit_id,
                                num_views=num_edit_views,
                                edit_dir=args.edit_dir)
                        if edited_images:
                            img_cond = refiner.encode_multiview_cond(
                                edited_images, original_images,
                                edit_strength=edit_strength)

                    # Run TRELLIS editing
                    slats_edited = refiner.edit(
                        ori_slat, mask, prompts,
                        img_cond=img_cond,
                        seed=args.seed,
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
                        "obj_id": spec.obj_id,
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
                    logger.info(f"  -> saved {pair_dir}")

                except Exception as e:
                    import traceback
                    logger.error(f"Failed {spec.edit_id}: {e}")
                    traceback.print_exc()
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": str(e),
                    }) + "\n")
                    out_fp.flush()
                    fail += 1

            obj_record.close()

    # Cleanup temp GLB directory
    import shutil
    shutil.rmtree(glb_tmp_dir, ignore_errors=True)

    logger.info(f"\nPhase 2.5 complete: {success} succeeded, {fail} failed")
    logger.info(f"Results: {output_path}")
    logger.info(f"Mesh pairs: {mesh_pairs_dir}")


def _resolve_api_key(cfg: dict) -> str:
    """Resolve VLM API key from config, default.yaml, or env."""
    p0 = cfg.get("phase0", {})
    api_key = p0.get("vlm_api_key", "")

    if not api_key:
        # Fallback: read from default.yaml
        import yaml
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


if __name__ == "__main__":
    main()
