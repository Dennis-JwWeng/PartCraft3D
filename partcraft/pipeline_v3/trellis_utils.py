"""3D step utilities for pipeline_v3.

Contains ``resolve_2d_conditioning``, migrated from pipeline_v2.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def resolve_2d_conditioning(
    *,
    spec,
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
    """Resolve a 2D conditioning image for TRELLIS repaint.

    Returns an encoded multiview conditioning tensor, or ``None`` if 2D
    conditioning is disabled / unavailable for this edit type.
    """
    if not use_2d:
        return None
    edit_type = prompts.get("edit_type")
    # NOTE: ``prompts["edit_type"]`` holds the capitalised PartCraft-level
    # edit_type (e.g. "Color", "Material"), not TRELLIS's internal
    # effective_type ("TextureOnly" / "Modification" / ...).  Color was
    # missing from this whitelist, which meant resolve_2d_conditioning
    # returned None for color edits and ``refiner.edit`` silently fell
    # back to a blank white image in ``repaint_mode='image'`` -- the 2D
    # FLUX recolour never reached TRELLIS.  See 2026-04-20 debugging
    # notes on shard08 clr_be1691a3..._011.
    if edit_type not in ("Modification", "Scale", "Material", "Color", "Global"):
        return None

    num_edit_views = p25_cfg.get("num_edit_views", 4)
    edit_strength = p25_cfg.get("edit_strength", 1.0)
    prerender_img = None
    if not edit_dir:
        raise ValueError(
            "[CONFIG_ERROR] pipeline.edit_dir <missing> runtime "
            "Step4 requires explicit 2D edit subdir"
        )
    base_dir = cache_dir / edit_dir
    edited_path = base_dir / f"{spec.edit_id}_edited.png"

    if edited_path.exists():
        try:
            from scripts.run_2d_edit import prepare_input_image

            edited = Image.open(edited_path).convert("RGB").resize((518, 518))
            input_path = base_dir / f"{spec.edit_id}_input.png"
            if input_path.exists():
                pil_in = Image.open(input_path).convert("RGB").resize((518, 518))
            elif hasattr(spec, "best_view") and spec.npz_view >= 0:
                _, pil_img = prepare_input_image(obj_record, spec.npz_view)
                pil_in = pil_img.resize((518, 518))
            else:
                pil_in = edited
            prerender_img = (pil_in, edited)
            logger.info("  2D from disk (%s/%s_edited.png)", base_dir.name, spec.edit_id)
        except Exception as e:
            logger.warning("  Cached 2D load failed: %s", e)

    if prerender_img is None and not cache_only_2d and hasattr(spec, "best_view") and spec.npz_view >= 0:
        try:
            from scripts.run_2d_edit import call_local_edit, call_vlm_edit, prepare_input_image

            img_bytes, pil_img = prepare_input_image(obj_record, spec.npz_view)
            after_desc = spec.new_parts_desc or ""
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
                edit_url = str(p25_cfg.get("image_edit_base_url", "")).strip()
                if not edit_url:
                    raise ValueError(
                        "[CONFIG_ERROR] services.image_edit.base_urls (or base_urls) <missing> config "
                        "local_diffusers backend requires explicit URL"
                    )
                edited = call_local_edit(
                    edit_url,
                    img_bytes,
                    spec.prompt,
                    after_desc,
                    old_part_label=part_label,
                    before_part_desc=before_desc,
                    edit_type=spec.edit_type,
                )
            elif vlm_client is not None:
                edited = call_vlm_edit(
                    vlm_client,
                    img_bytes,
                    spec.prompt,
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
                logger.info("  2D edit from prerender view %s", spec.npz_view)
        except Exception as e:
            logger.warning("  Prerender 2D edit failed: %s", e)

    if prerender_img is not None:
        original_images, edited_images = [prerender_img[0]], [prerender_img[1]]
    elif not cache_only_2d:
        original_images, edited_images = refiner.obtain_edited_images(
            ori_gaussian, prompts, vlm_client, obj_id, spec.edit_id,
            num_views=num_edit_views, edit_dir=edit_dir,
        )
    else:
        original_images, edited_images = [], []
        logger.warning("  --2d-cache-only: missing %s_edited.png -> no img cond", spec.edit_id)

    if edited_images:
        return refiner.encode_multiview_cond(
            edited_images, original_images, edit_strength=edit_strength,
        )
    return None


__all__ = ["resolve_2d_conditioning"]
