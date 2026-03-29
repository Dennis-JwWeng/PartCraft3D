"""Lookahead prefetch for run_streaming: VLM/plan/2D overlap with TRELLIS (no CUDA here)."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from partcraft.edit_types import (
    ADDITION,
    DELETION,
    GLOBAL,
    IDENTITY,
    MATERIAL,
    MODIFICATION,
    SCALE,
    TYPE_ORDER,
)


def run_streaming_with_lookahead(
    *,
    all_uids: list[str],
    uid_info: dict[str, tuple[str, list[str], list[int], str]],
    lookahead: int,
    p0: dict,
    p25: dict,
    labels_path: Path,
    specs_path: Path,
    results_path: Path,
    edit_2d_dir: Path,
    mesh_pairs_dir: Path,
    npz_dir: Path,
    dataset: Any,
    cfg: dict,
    args: Any,
    logger: logging.Logger,
    refiner: Any,
    build_prompts_from_spec: Callable,
    vlm_client: Any,
    vlm_model: str,
    image_edit_backend: str,
    edit_vlm_client: Any,
    done_labels: set[str],
    done_edits: set[str],
    existing_specs_by_obj: dict[str, list],
    prepare_input_image: Callable,
    call_local_edit: Callable,
    call_vlm_edit: Callable,
    EditSpec: Any,
    plan_edits_for_record: Callable,
    _enrich_one_object_visual: Callable,
    _call_vlm: Callable,
    _fallback_enrichment: Callable,
    _result_to_phase0_record: Callable,
    load_thumbnail_from_npz: Callable,
) -> None:
    from collections import Counter
    import tempfile

    local_edit_url = ""
    if args.use_2d and image_edit_backend == "local_diffusers":
        local_edit_url = str(p25.get("image_edit_base_url", "")).strip()
        if not local_edit_url:
            raise ValueError(
                "[CONFIG_ERROR] phase2_5.image_edit_base_url <missing> config "
                "local_diffusers backend requires explicit URL"
            )

    glb_tmp_dir = tempfile.mkdtemp(prefix="partcraft_stream_")
    total_specs = 0
    total_success = 0
    total_fail = 0

    lbl_lock = threading.Lock() if lookahead > 0 else None
    spec_lock = threading.Lock() if lookahead > 0 else None

    with open(labels_path, "a", encoding="utf-8") as lbl_fp, \
         open(specs_path, "a", encoding="utf-8") as spec_fp, \
         open(results_path, "a", encoding="utf-8") as res_fp:

        def _append_label(record: dict) -> None:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            if lbl_lock:
                with lbl_lock:
                    lbl_fp.write(line)
                    lbl_fp.flush()
            else:
                lbl_fp.write(line)
                lbl_fp.flush()
            done_labels.add(record["obj_id"])

        def _append_specs(uid: str, specs: list) -> None:
            if spec_lock:
                with spec_lock:
                    for spec in specs:
                        spec_fp.write(json.dumps(
                            spec.to_dict(), ensure_ascii=False) + "\n")
                    spec_fp.flush()
            else:
                for spec in specs:
                    spec_fp.write(json.dumps(
                        spec.to_dict(), ensure_ascii=False) + "\n")
                spec_fp.flush()
            existing_specs_by_obj[uid] = [s.to_dict() for s in specs]

        def _load_record_for_uid(uid: str):
            record = None
            with open(labels_path, encoding="utf-8") as _lf:
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
            return record

        def do_prepare(uid: str):
            category, labels, actual_pids, obj_shard = uid_info[uid]
            record = None
            if uid in done_labels:
                if lookahead > 0:
                    logger.info(
                        f"  [{uid[:12]}...] prefetch: load enriched record")
                else:
                    logger.info("  Resuming (already enriched), loading...")
                record = _load_record_for_uid(uid)
            else:
                try:
                    obj = dataset.load_object(obj_shard, uid)
                    result = _enrich_one_object_visual(
                        vlm_client, vlm_model, obj, category, labels)
                    obj.close()
                except Exception as e:
                    logger.warning(f"  Visual enrichment failed: {e}")
                    result = None

                if result is None:
                    npz_p = npz_dir / obj_shard / f"{uid}.npz"
                    thumb = None
                    if npz_p.exists():
                        thumb = load_thumbnail_from_npz(str(npz_p), view_id=0)
                    result = _call_vlm(
                        vlm_client, vlm_model, category, labels, thumb)

                if result is None:
                    result = _fallback_enrichment(category, labels)
                    logger.warning(
                        f"  Using fallback enrichment for {uid}")

                record = _result_to_phase0_record(
                    result, uid, category, obj_shard,
                    actual_part_ids=actual_pids)
                _append_label(record)

            if record is None:
                return None, []

            if uid in existing_specs_by_obj:
                obj_specs = [EditSpec(**d)
                             for d in existing_specs_by_obj[uid]]
            else:
                obj_specs = plan_edits_for_record(record, cfg)
                _append_specs(uid, obj_specs)

            if args.use_2d:
                prefetch_specs = [
                    s for s in obj_specs
                    if s.edit_id not in done_edits
                    and s.edit_type in (
                        MODIFICATION, SCALE, MATERIAL, GLOBAL)
                    and hasattr(s, "best_view")
                    and s.best_view >= 0]
                if prefetch_specs:
                    obj_record = dataset.load_object(obj_shard, uid)
                    try:
                        for spec in prefetch_specs:
                            cached_2d_path = (
                                edit_2d_dir / f"{spec.edit_id}_edited.png")
                            if cached_2d_path.exists():
                                continue
                            try:
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view)
                                after_desc = (
                                    spec.after_desc or spec.after_part_desc
                                    or "")
                                before_desc = (
                                    getattr(spec, "before_part_desc", "")
                                    or "")
                                remove_labels = getattr(
                                    spec, "remove_labels", [])
                                old_label = getattr(spec, "old_label", "") or ""
                                if remove_labels and len(remove_labels) > 1:
                                    part_label = ", ".join(remove_labels)
                                elif remove_labels:
                                    part_label = remove_labels[0]
                                else:
                                    part_label = old_label

                                if image_edit_backend == "local_diffusers":
                                    edited = call_local_edit(
                                        local_edit_url, img_bytes,
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
                                    tmp_path = cached_2d_path.with_suffix(
                                        ".tmp.png")
                                    edited.save(str(tmp_path))
                                    tmp_path.rename(cached_2d_path)
                                    if lookahead > 0:
                                        logger.info(
                                            f"    prefetch 2D {spec.edit_id}")
                            except Exception as e:
                                logger.warning(
                                    f"    Prefetch 2D failed {spec.edit_id}: {e}")
                    finally:
                        obj_record.close()

            return record, obj_specs
        def run_object_3d(uid: str, record: dict, obj_specs: list) -> None:
            nonlocal total_specs, total_success, total_fail
            category, labels, actual_pids, obj_shard = uid_info[uid]
            _tcounts = Counter(s.edit_type for s in obj_specs)
            _tparts = [
                f"{t}={_tcounts[t]}" for t in
                [DELETION, ADDITION, MODIFICATION, SCALE, MATERIAL,
                 GLOBAL, IDENTITY] if _tcounts.get(t)]
            logger.info(
                f"  Enriched: {len(record.get('group_edits', []))} "
                f"groups, {len(record.get('global_edits', []))} global edits")
            logger.info(f"  Planned: {len(obj_specs)} specs "
                        f"({' '.join(_tparts)})")
            total_specs += len(obj_specs)

            if not obj_specs:
                return

            run_specs = [s for s in obj_specs
                         if s.edit_id not in done_edits
                         and s.edit_type not in (ADDITION, IDENTITY)]
            add_specs = [s for s in obj_specs
                         if s.edit_type == ADDITION]
            idt_specs = [s for s in obj_specs
                         if s.edit_type == IDENTITY
                         and s.edit_id not in done_edits]

            if not run_specs and not add_specs and not idt_specs:
                logger.info("  All edits already done, skipping")
                return

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
                    }, ensure_ascii=False) + "\n")
                    total_fail += 1
                res_fp.flush()
                return

            run_specs.sort(key=lambda s: TYPE_ORDER.get(s.edit_type, 9))
            first_pair_dir = None

            for spec in run_specs:
                if spec.edit_id in done_edits:
                    continue

                logger.info(f"\n  [{spec.edit_id}] {spec.edit_type}: "
                            f"\"{spec.edit_prompt[:60]}\"")

                try:
                    edit_type = spec.edit_type

                    if edit_type == DELETION:
                        from partcraft.phase2_assembly.trellis_refine import (
                            TrellisRefiner)
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
                        res_fp.write(json.dumps(
                            rec, ensure_ascii=False) + "\n")
                        res_fp.flush()
                        total_success += 1
                        done_edits.add(spec.edit_id)
                        if first_pair_dir is None:
                            first_pair_dir = pair_dir
                        logger.info(f"    OK (direct deletion, GT mesh) "
                                    f"→ {pair_dir}")
                        continue
                    elif edit_type in (MODIFICATION, SCALE):
                        if spec.remove_part_ids:
                            edit_part_ids = spec.remove_part_ids
                        else:
                            edit_part_ids = [spec.old_part_id]
                    elif edit_type == MATERIAL:
                        edit_part_ids = [spec.old_part_id]
                    elif edit_type == GLOBAL:
                        edit_part_ids = []
                    else:
                        continue

                    mask_type = edit_type.capitalize()
                    mask, effective_type = refiner.build_part_mask(
                        uid, obj_record, edit_part_ids, ori_slat,
                        mask_type)
                    if effective_type != mask_type:
                        logger.info(f"    Auto-promoted → {effective_type}")
                    if mask.sum() == 0:
                        raise RuntimeError("Empty mask")

                    prompts = build_prompts_from_spec(spec)

                    img_cond = None
                    if args.use_2d:
                        num_edit_views = p25.get("num_edit_views", 4)
                        edit_strength = p25.get("edit_strength", 1.0)
                        prerender_img = None

                        cached_2d_path = (
                            edit_2d_dir / f"{spec.edit_id}_edited.png")
                        if cached_2d_path.exists():
                            try:
                                cached_edited = Image.open(
                                    str(cached_2d_path))
                                cached_edited.load()
                                cached_edited = cached_edited.convert(
                                    "RGB").resize((518, 518))
                                _, pil_img = prepare_input_image(
                                    obj_record, spec.best_view
                                    if hasattr(spec, "best_view")
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
                                and hasattr(spec, "best_view")
                                and spec.best_view >= 0):
                            try:
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view)
                                after_desc = (
                                    spec.after_desc or spec.after_part_desc
                                    or "")
                                before_desc = (
                                    getattr(spec, "before_part_desc", "")
                                    or "")
                                remove_labels = getattr(
                                    spec, "remove_labels", [])
                                old_label = getattr(spec, "old_label", "") or ""
                                if (remove_labels
                                        and len(remove_labels) > 1):
                                    part_label = ", ".join(remove_labels)
                                elif remove_labels:
                                    part_label = remove_labels[0]
                                else:
                                    part_label = old_label

                                if image_edit_backend == "local_diffusers":
                                    edited = call_local_edit(
                                        local_edit_url, img_bytes,
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

                    slats_edited = refiner.edit(
                        ori_slat, mask, prompts,
                        img_cond=img_cond, seed=args.seed)
                    if not slats_edited:
                        raise RuntimeError("No edited SLATs produced")

                    pair_dir = mesh_pairs_dir / spec.edit_id
                    export_paths = refiner.export_pair_shared_before(
                        ori_slat, slats_edited[0], pair_dir,
                        shared_before_dir=first_pair_dir)

                    rec = {
                        "edit_id": spec.edit_id,
                        "edit_type": spec.edit_type,
                        "effective_edit_type": effective_type,
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
                    }, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    total_fail += 1

            for spec in add_specs:
                if spec.edit_id in done_edits:
                    continue
                del_pair = mesh_pairs_dir / spec.source_del_id
                add_pair = mesh_pairs_dir / spec.edit_id
                del_has_output = ((del_pair / "before_slat").exists()
                                  or (del_pair / "before.ply").exists())
                if del_has_output:
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
                        "edit_id": spec.edit_id, "edit_type": ADDITION,
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
                    }, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    total_fail += 1

            for spec in idt_specs:
                if spec.edit_id in done_edits:
                    continue
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
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id,
                        "edit_type": IDENTITY,
                        "effective_edit_type": "Identity",
                        "obj_id": uid, "status": "success",
                        "edit_prompt": spec.edit_prompt,
                    }, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    done_edits.add(spec.edit_id)
                    total_success += 1
                    logger.info(f"  [{spec.edit_id}] identity: "
                                f"before=after (no-op)")
                else:
                    res_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": "No before mesh available for identity",
                    }, ensure_ascii=False) + "\n")
                    res_fp.flush()
                    total_fail += 1

            obj_record.close()

        if lookahead == 0:
            for obj_idx, uid in enumerate(all_uids):
                category, labels, actual_pids, obj_shard = uid_info[uid]
                logger.info(f"\n{'='*60}")
                logger.info(f"[{obj_idx+1}/{len(all_uids)}] Object: {uid}")
                logger.info(f"  Category: {category}, Parts: {len(labels)}")
                record, obj_specs = do_prepare(uid)
                if record is None:
                    logger.warning(f"  No record for {uid}, skipping")
                    continue
                run_object_3d(uid, record, obj_specs)
        else:
            prepared: dict = {}
            fut_map: dict = {}
            max_pe = max(1, min(
                int(p0.get("max_workers", 4)), lookahead + 1))

            def submit_prep(u: str, ex: ThreadPoolExecutor) -> None:
                if u in fut_map:
                    return

                def task():
                    try:
                        prepared[u] = do_prepare(u)
                    except Exception as e:
                        logger.exception(
                            "prepare failed for %s: %s", u, e)
                        prepared[u] = (None, [])

                fut_map[u] = ex.submit(task)

            with ThreadPoolExecutor(max_workers=max_pe) as ex:
                for obj_idx, uid in enumerate(all_uids):
                    category, labels, actual_pids, obj_shard = uid_info[uid]
                    logger.info(f"\n{'='*60}")
                    logger.info(f"[{obj_idx+1}/{len(all_uids)}] Object: {uid}")
                    logger.info(
                        f"  Category: {category}, Parts: {len(labels)}")
                    for k in range(1, lookahead + 1):
                        j = obj_idx + k
                        if j < len(all_uids):
                            submit_prep(all_uids[j], ex)
                    submit_prep(uid, ex)
                    fut_map[uid].result()
                    del fut_map[uid]
                    record, obj_specs = prepared.pop(uid)
                    if record is None:
                        logger.warning(f"  No record for {uid}, skipping")
                        continue
                    run_object_3d(uid, record, obj_specs)

    shutil.rmtree(glb_tmp_dir, ignore_errors=True)
    logger.info(f"\n{'='*60}")
    logger.info(f"Streaming complete: {total_specs} specs planned, "
                f"{total_success} ok, {total_fail} fail")
    logger.info(f"  Labels:  {labels_path}")
    logger.info(f"  Specs:   {specs_path}")
    logger.info(f"  Results: {results_path}")
    logger.info("=" * 60)
