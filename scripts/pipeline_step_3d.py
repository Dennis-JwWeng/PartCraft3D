from __future__ import annotations

import json
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

from PIL import Image

from partcraft.edit_types import ADDITION, DELETION, IDENTITY, S1_S2_TYPES, S2_ONLY_TYPES, TYPE_ORDER
from scripts.pipeline_common import EditSpec, resolve_api_key, resolve_data_dirs
from scripts.pipeline_jsonl import collect_success_ids, dedupe_specs_by_edit_id


STEP4_GPU_TYPES = set(S1_S2_TYPES) | set(S2_ONLY_TYPES)
STEP4_NON_GPU_TYPES = {ADDITION, DELETION, IDENTITY}


def write_step4_record(out_fp, record: dict):
    out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_fp.flush()


def _copy_addition_from_deletion(mesh_pairs_dir: Path, spec: EditSpec, obj_id: str) -> bool:
    del_pair_dir = mesh_pairs_dir / spec.source_del_id
    add_pair_dir = mesh_pairs_dir / spec.edit_id
    has_source_pair = (
        (del_pair_dir / "before.ply").exists() or (del_pair_dir / "before_slat").exists()
    )
    if not has_source_pair:
        return False
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
    return True


def handle_addition_promotion(out_fp, mesh_pairs_dir: Path, obj_id: str, spec: EditSpec) -> bool:
    ok = _copy_addition_from_deletion(mesh_pairs_dir, spec, obj_id)
    if ok:
        write_step4_record(out_fp, {
            "edit_id": spec.edit_id,
            "edit_type": "addition",
            "obj_id": obj_id,
            "status": "success",
            "source_del_id": spec.source_del_id,
        })
    return ok


def prepare_step4_context(
    cfg,
    specs_path,
    logger,
    *,
    tag=None,
    limit=None,
    edit_types=None,
    edit_ids=None,
    results_name: str | None = None,
):
    p25_cfg = cfg.get("phase2_5", {})
    output_dir = Path(cfg["data"]["output_dir"])
    cache_dir = Path(p25_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"
    output_path = (cache_dir / results_name) if results_name else (
        cache_dir / f"edit_results{tag_suffix}.jsonl"
    )

    accepted_types = edit_types or [
        "deletion", "addition", "modification", "scale", "material", "global", "identity"
    ]
    all_specs = []
    with open(specs_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if edit_ids and spec.edit_id not in edit_ids:
                continue
            if spec.edit_type in accepted_types or spec.edit_type == "addition":
                all_specs.append(spec)
    if not edit_ids and limit:
        all_specs = all_specs[:limit]
    all_specs = dedupe_specs_by_edit_id(all_specs, logger, "Step4(single)")
    done_ids = collect_success_ids(output_path, id_key="edit_id")
    pending = [s for s in all_specs if s.edit_id not in done_ids]
    return {
        "p25_cfg": p25_cfg,
        "cache_dir": cache_dir,
        "mesh_pairs_dir": mesh_pairs_dir,
        "output_path": output_path,
        "all_specs": all_specs,
        "pending": pending,
        "done_ids": done_ids,
    }


def resolve_2d_conditioning(
    *,
    spec: EditSpec,
    obj_id: str,
    obj_record,
    ori_gaussian,
    refiner,
    vlm_client,
    p25_cfg: dict,
    cache_dir: Path,
    edit_dir: str | None,
    cache_only_2d: bool,
    use_2d: bool,
    image_edit_backend: str,
    logger,
    prompts: dict,
):
    if not use_2d:
        return None
    edit_type = prompts.get("edit_type")
    if edit_type not in ("Modification", "Scale", "Material", "Global"):
        return None

    num_edit_views = p25_cfg.get("num_edit_views", 4)
    edit_strength = p25_cfg.get("edit_strength", 1.0)
    prerender_img = None
    base_dir = cache_dir / (edit_dir or "2d_edits")
    edited_path = base_dir / f"{spec.edit_id}_edited.png"

    if edited_path.exists():
        try:
            from scripts.run_2d_edit import prepare_input_image

            edited = Image.open(edited_path).convert("RGB").resize((518, 518))
            input_path = base_dir / f"{spec.edit_id}_input.png"
            if input_path.exists():
                pil_in = Image.open(input_path).convert("RGB").resize((518, 518))
            elif hasattr(spec, "best_view") and spec.best_view >= 0:
                _, pil_img = prepare_input_image(obj_record, spec.best_view)
                pil_in = pil_img.resize((518, 518))
            else:
                pil_in = edited
            prerender_img = (pil_in, edited)
            logger.info("  2D from disk (%s/%s_edited.png)", base_dir.name, spec.edit_id)
        except Exception as e:
            logger.warning("  Cached 2D load failed: %s", e)

    if prerender_img is None and not cache_only_2d and hasattr(spec, "best_view") and spec.best_view >= 0:
        try:
            from scripts.run_2d_edit import call_local_edit, call_vlm_edit, prepare_input_image

            img_bytes, pil_img = prepare_input_image(obj_record, spec.best_view)
            after_desc = spec.after_desc or spec.after_part_desc or ""
            before_desc = getattr(spec, "before_part_desc", "") or ""
            remove_labels = getattr(spec, "remove_labels", [])
            old_label = getattr(spec, "old_label", "") or ""
            if remove_labels and len(remove_labels) > 1:
                part_label = ", ".join(remove_labels)
            elif remove_labels:
                part_label = remove_labels[0]
            else:
                part_label = old_label

            if image_edit_backend == "local_diffusers":
                edit_url = p25_cfg.get("image_edit_base_url", "http://localhost:8001")
                edited = call_local_edit(
                    edit_url,
                    img_bytes,
                    spec.edit_prompt,
                    after_desc,
                    old_part_label=part_label,
                    before_part_desc=before_desc,
                    edit_type=spec.edit_type,
                )
            elif vlm_client is not None:
                edited = call_vlm_edit(
                    vlm_client,
                    img_bytes,
                    spec.edit_prompt,
                    after_desc,
                    p25_cfg.get("image_edit_model", ""),
                    old_part_label=part_label,
                    before_part_desc=before_desc,
                    edit_type=spec.edit_type,
                )
            else:
                edited = None
            if edited is not None:
                edited = edited.resize((518, 518))
                prerender_img = (pil_img.resize((518, 518)), edited)
                logger.info("  2D edit from prerender view %s", spec.best_view)
        except Exception as e:
            logger.warning("  Prerender 2D edit failed: %s", e)

    if prerender_img is not None:
        original_images, edited_images = [prerender_img[0]], [prerender_img[1]]
    elif not cache_only_2d:
        original_images, edited_images = refiner.obtain_edited_images(
            ori_gaussian, prompts, vlm_client, obj_id, spec.edit_id, num_views=num_edit_views, edit_dir=edit_dir
        )
    else:
        original_images, edited_images = [], []
        logger.warning("  --2d-cache-only: missing %s_edited.png -> no img cond", spec.edit_id)

    if edited_images:
        return refiner.encode_multiview_cond(edited_images, original_images, edit_strength=edit_strength)
    return None


def process_object_edits(
    *,
    cfg,
    dataset,
    logger,
    pending: list,
    output_path: Path,
    mesh_pairs_dir: Path,
    cache_dir: Path,
    p25_cfg: dict,
    use_2d: bool,
    seed: int,
    combinations,
    edit_dir,
    debug: bool,
    cache_only_2d: bool,
):
    from partcraft.phase2_assembly.trellis_refine import TrellisRefiner, build_prompts_from_spec

    image_edit_backend = p25_cfg.get("image_edit_backend", "api")
    vlm_client = None
    if use_2d and image_edit_backend != "local_diffusers":
        from openai import OpenAI

        api_key = resolve_api_key(cfg)
        if api_key:
            p0 = cfg["phase0"]
            image_edit_url = p25_cfg.get("image_edit_base_url") or p0.get("vlm_base_url", "")
            vlm_client = OpenAI(base_url=image_edit_url, api_key=api_key)

    refiner = None

    def ensure_refiner():
        nonlocal refiner
        if refiner is not None:
            return refiner
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
        logger.info("TRELLIS models initialized for GPU edit types")
        return refiner

    obj_groups: OrderedDict[str, list] = OrderedDict()
    for spec in pending:
        obj_groups.setdefault(spec.obj_id, []).append(spec)

    success, fail = 0, 0
    glb_tmp_dir = tempfile.mkdtemp(prefix="partcraft_glb_")
    with open(output_path, "a", encoding="utf-8") as out_fp:
        for obj_id, specs in obj_groups.items():
            logger.info("\n%s", "=" * 60)
            logger.info("Object: %s (%d edits)", obj_id, len(specs))

            specs.sort(key=lambda s: (0, 0) if s.edit_type == DELETION else (1, 0) if s.edit_type == ADDITION else (2, TYPE_ORDER.get(s.edit_type, 99)))

            add_specs = [s for s in specs if s.edit_type == ADDITION]
            run_specs = [s for s in specs if s.edit_type != ADDITION]
            deferred_add_specs = []
            for spec in add_specs:
                if handle_addition_promotion(out_fp, mesh_pairs_dir, obj_id, spec):
                    success += 1
                else:
                    deferred_add_specs.append(spec)

            idt_specs = [s for s in run_specs if s.edit_type == IDENTITY]
            run_specs = [s for s in run_specs if s.edit_type != IDENTITY]
            first_pair_dir = None
            cpu_specs = [s for s in run_specs if s.edit_type == DELETION]
            gpu_specs = [s for s in run_specs if s.edit_type in STEP4_GPU_TYPES]
            unknown_specs = [s for s in run_specs if s.edit_type not in (STEP4_NON_GPU_TYPES | STEP4_GPU_TYPES)]
            for spec in unknown_specs:
                logger.warning("Unknown edit type: %s", spec.edit_type)
                write_step4_record(out_fp, {
                    "edit_id": spec.edit_id,
                    "status": "failed",
                    "reason": f"Unknown edit type: {spec.edit_type}",
                })
                fail += 1
            if not cpu_specs and not gpu_specs:
                for spec in idt_specs:
                    write_step4_record(out_fp, {
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": "identity needs prior edits on object",
                    })
                    fail += 1
                continue

            obj_record = None
            ori_slat = None
            ori_gaussian = None
            try:
                shard = (cpu_specs[0].shard if cpu_specs else gpu_specs[0].shard)
                obj_record = dataset.load_object(shard, obj_id)
            except Exception as e:
                logger.error("Failed to load object %s: %s", obj_id, e)
                for spec in cpu_specs + gpu_specs:
                    write_step4_record(out_fp, {
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": f"Object load failed: {e}",
                    })
                fail += len(cpu_specs) + len(gpu_specs)
                continue

            for spec in cpu_specs:
                logger.info('\n[%d/%d] %s (%s): "%s"', success + fail + 1, len(pending), spec.edit_id, spec.edit_type, spec.edit_prompt[:80])
                try:
                    pair_dir = mesh_pairs_dir / spec.edit_id
                    export_paths = TrellisRefiner.direct_delete_mesh(obj_record, spec.remove_part_ids, pair_dir)
                    write_step4_record(out_fp, {
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
                    })
                    success += 1
                    if first_pair_dir is None:
                        first_pair_dir = pair_dir
                except Exception as e:
                    logger.error("Failed %s: %s", spec.edit_id, e)
                    write_step4_record(out_fp, {
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": str(e),
                    })
                    fail += 1

            for spec in deferred_add_specs:
                if handle_addition_promotion(out_fp, mesh_pairs_dir, obj_id, spec):
                    success += 1
                else:
                    write_step4_record(out_fp, {
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": f"Missing source deletion outputs: {spec.source_del_id}",
                    })
                    fail += 1

            if gpu_specs:
                try:
                    step_refiner = ensure_refiner()
                    ori_slat = step_refiner.encode_object(None, obj_id)
                    ori_gaussian = step_refiner.decode_to_gaussian(ori_slat)
                except Exception as e:
                    logger.error("Failed to prepare TRELLIS state for %s: %s", obj_id, e)
                    for spec in gpu_specs:
                        write_step4_record(out_fp, {
                            "edit_id": spec.edit_id,
                            "status": "failed",
                            "reason": f"Preparation failed: {e}",
                        })
                    fail += len(gpu_specs)
                    gpu_specs = []

            for spec in gpu_specs:
                logger.info('\n[%d/%d] %s (%s): "%s"', success + fail + 1, len(pending), spec.edit_id, spec.edit_type, spec.edit_prompt[:80])
                try:
                    edit_type = spec.edit_type.capitalize()
                    if edit_type in ("Modification", "Scale"):
                        edit_part_ids = spec.remove_part_ids if spec.remove_part_ids else [spec.old_part_id]
                    elif edit_type == "Material":
                        edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Global":
                        edit_part_ids = []
                    else:
                        logger.warning("Skipped non-GPU type in GPU queue: %s", spec.edit_type)
                        continue

                    step_refiner = ensure_refiner()
                    mask, effective_type = step_refiner.build_part_mask(obj_id, obj_record, edit_part_ids, ori_slat, edit_type)
                    if effective_type != edit_type:
                        logger.info("  Auto-promoted %s -> %s (large part)", edit_type, effective_type)
                        edit_type = effective_type
                    if mask.sum() == 0:
                        write_step4_record(out_fp, {
                            "edit_id": spec.edit_id,
                            "status": "failed",
                            "reason": "Empty mask",
                        })
                        fail += 1
                        continue

                    prompts = build_prompts_from_spec(spec)
                    if prompts["edit_type"] != edit_type:
                        prompts["edit_type"] = edit_type
                    img_cond = resolve_2d_conditioning(
                        spec=spec,
                        obj_id=obj_id,
                        obj_record=obj_record,
                        ori_gaussian=ori_gaussian,
                        refiner=step_refiner,
                        vlm_client=vlm_client,
                        p25_cfg=p25_cfg,
                        cache_dir=cache_dir,
                        edit_dir=edit_dir,
                        cache_only_2d=cache_only_2d,
                        use_2d=use_2d,
                        image_edit_backend=image_edit_backend,
                        logger=logger,
                        prompts=prompts,
                    )
                    slats_edited = step_refiner.edit(
                        ori_slat,
                        mask,
                        prompts,
                        img_cond=img_cond,
                        seed=seed,
                        combinations=combinations,
                    )
                    if not slats_edited:
                        raise RuntimeError("No edited SLATs produced")
                    best_slat = slats_edited[0]
                    pair_dir = mesh_pairs_dir / spec.edit_id
                    export_paths = step_refiner.export_pair(ori_slat, best_slat, pair_dir)
                    write_step4_record(out_fp, {
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
                    })
                    success += 1
                    if first_pair_dir is None:
                        first_pair_dir = pair_dir
                except Exception as e:
                    logger.error("Failed %s: %s", spec.edit_id, e)
                    write_step4_record(out_fp, {
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": str(e),
                    })
                    fail += 1

            for spec in idt_specs:
                idt_pair = mesh_pairs_dir / spec.edit_id
                if first_pair_dir and (first_pair_dir / "before.ply").exists():
                    idt_pair.mkdir(parents=True, exist_ok=True)
                    before_ply = first_pair_dir / "before.ply"
                    shutil.copy2(str(before_ply), str(idt_pair / "before.ply"))
                    shutil.copy2(str(before_ply), str(idt_pair / "after.ply"))
                    before_slat = first_pair_dir / "before_slat"
                    if before_slat.exists():
                        for slat_name in ("before_slat", "after_slat"):
                            dst = idt_pair / slat_name
                            if dst.exists():
                                shutil.rmtree(dst)
                            shutil.copytree(str(before_slat), str(dst))
                    write_step4_record(out_fp, {
                        "edit_id": spec.edit_id,
                        "edit_type": IDENTITY,
                        "effective_edit_type": "Identity",
                        "obj_id": obj_id,
                        "status": "success",
                        "edit_prompt": spec.edit_prompt,
                    })
                    success += 1
                else:
                    write_step4_record(out_fp, {
                        "edit_id": spec.edit_id,
                        "status": "failed",
                        "reason": "No before mesh available for identity",
                    })
                    fail += 1
            if obj_record is not None:
                obj_record.close()
    shutil.rmtree(glb_tmp_dir, ignore_errors=True)
    return success, fail


def run_step_3d_edit(
    cfg,
    specs_path,
    dataset,
    logger,
    *,
    tag=None,
    seed=1,
    limit=None,
    use_2d=True,
    edit_types=None,
    edit_ids=None,
    combinations=None,
    edit_dir=None,
    debug=False,
    cache_only_2d: bool = False,
    results_name: str | None = None,
):
    logger.info("=" * 60)
    logger.info("STEP 4: 3D Editing — TRELLIS")
    logger.info("=" * 60)

    ctx = prepare_step4_context(
        cfg,
        specs_path,
        logger,
        tag=tag,
        limit=limit,
        edit_types=edit_types,
        edit_ids=edit_ids,
        results_name=results_name,
    )
    all_specs = ctx["all_specs"]
    pending = ctx["pending"]
    output_path = ctx["output_path"]
    if not all_specs:
        logger.info("No edit specs to process")
        return None
    logger.info("3D edits: %d pending (%d done)", len(pending), len(ctx["done_ids"]))
    if not pending:
        logger.info("All 3D edits already done")
        return output_path

    success, fail = process_object_edits(
        cfg=cfg,
        dataset=dataset,
        logger=logger,
        pending=pending,
        output_path=output_path,
        mesh_pairs_dir=ctx["mesh_pairs_dir"],
        cache_dir=ctx["cache_dir"],
        p25_cfg=ctx["p25_cfg"],
        use_2d=use_2d,
        seed=seed,
        combinations=combinations,
        edit_dir=edit_dir,
        debug=debug,
        cache_only_2d=cache_only_2d,
    )
    logger.info("\n3D edits: %d ok, %d fail -> %s", success, fail, output_path)
    return output_path
