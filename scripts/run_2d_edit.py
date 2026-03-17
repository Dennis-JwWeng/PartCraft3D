#!/usr/bin/env python3
"""Phase 2D: Batch 2D image editing for modification specs.

Pre-generates edited reference images for Phase 2.5 TRELLIS,
so GPU-heavy 3D editing doesn't block on API calls.

For each modification spec:
  1. Select best view showing the target part
  2. Composite RGBA → RGB on white background
  3. Call VLM image editor (e.g. Gemini) with edit prompt
  4. Save edited image as PNG

Phase 2.5 reads these pre-edited images automatically when found.

Usage:
    # Edit all modification specs (parallel API calls)
    python scripts/run_2d_edit.py --config configs/partobjaverse.yaml --workers 8

    # Limit to first 10
    python scripts/run_2d_edit.py --limit 10

    # Resume (skip already-done edits)
    python scripts/run_2d_edit.py --resume
"""

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.hy3d_loader import HY3DPartDataset
from partcraft.phase1_planning.planner import EditSpec


def select_best_view(obj_record, edit_part_ids: list[int]) -> int:
    """Pick the perspective view where edit parts have most visible pixels."""
    import numpy as np
    transforms_data = obj_record.get_transforms()
    frames = transforms_data["frames"]

    best_view, best_count = 0, 0
    for v in range(obj_record.num_views):
        if v < len(frames) and frames[v].get("proj_type") == "ortho":
            continue
        mask = obj_record.get_mask(v)
        count = sum(int(np.sum(mask == pid)) for pid in edit_part_ids)
        if count > best_count:
            best_count = count
            best_view = v
    return best_view


def prepare_input_image(obj_record, view_id: int,
                        edit_part_ids: list[int] | None = None) -> bytes:
    """Load view from NPZ, composite RGBA onto white, return PNG bytes.

    If edit_part_ids is given, also draw a semi-transparent green overlay
    and a red contour on the target part so the VLM knows exactly which
    region to edit.  Returns (plain_png_bytes, annotated_pil, plain_pil).
    """
    import numpy as np
    from PIL import Image

    view_bytes = obj_record.get_image_bytes(view_id)
    pil_img = Image.open(io.BytesIO(view_bytes)).convert("RGBA")
    pil_img = pil_img.resize((518, 518))
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    pil_img = Image.alpha_composite(bg, pil_img).convert("RGB")

    # Plain image (for saving as _input.png and for phase2.5 conditioning)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    plain_bytes = buf.getvalue()

    if not edit_part_ids:
        return plain_bytes, pil_img, pil_img

    # --- Build annotated image with part highlight ---
    seg_mask = obj_record.get_mask(view_id)  # (H_orig, W_orig) int16
    # Resize mask to match image (nearest to preserve IDs)
    seg_pil = Image.fromarray(seg_mask.astype(np.int16))
    seg_pil = seg_pil.resize((518, 518), Image.NEAREST)
    seg_arr = np.array(seg_pil)

    # Binary mask: True where target part is
    part_binary = np.zeros_like(seg_arr, dtype=bool)
    for pid in edit_part_ids:
        part_binary |= (seg_arr == pid)

    if not part_binary.any():
        # Part not visible in this view, return plain
        return plain_bytes, pil_img, pil_img

    # Semi-transparent green overlay on the part
    annotated = pil_img.copy()
    overlay = np.array(annotated).copy()
    overlay[part_binary] = (
        overlay[part_binary] * 0.6 + np.array([0, 200, 0]) * 0.4
    ).astype(np.uint8)

    # Red contour: dilate mask - erode mask = boundary
    from scipy import ndimage
    dilated = ndimage.binary_dilation(part_binary, iterations=3)
    eroded = ndimage.binary_erosion(part_binary, iterations=1)
    contour = dilated & ~eroded
    overlay[contour] = [255, 50, 50]

    annotated = Image.fromarray(overlay)
    buf2 = io.BytesIO()
    annotated.save(buf2, format="PNG")
    annotated_bytes = buf2.getvalue()

    return annotated_bytes, annotated, pil_img


def call_vlm_edit(client, img_bytes: bytes, edit_prompt: str,
                  after_part_desc: str, model: str,
                  has_annotation: bool = False) -> "Image.Image | None":
    """Call VLM to produce an edited image."""
    import base64
    from PIL import Image

    b64 = base64.b64encode(img_bytes).decode('utf-8')
    if has_annotation:
        text_input = (
            "The image has a green highlighted region with a red outline. "
            "This marks the ONLY part you should edit. "
            "Do NOT modify anything outside the highlighted region. "
            f"Edit that specific part with: {edit_prompt}"
        )
    else:
        text_input = f"Edit the image provided with the editing prompt: {edit_prompt}"
    if after_part_desc:
        text_input += f"\nThe edited part should look like: {after_part_desc}"
    text_input += (
        "\nIMPORTANT: Output a clean edited image WITHOUT any annotations, "
        "highlights, outlines, or overlays. The result should look natural."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_input},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }],
        )
        msg = response.choices[0].message

        # Try message.images (Gemini style)
        images = getattr(msg, 'images', None)
        if images:
            img0 = images[0]
            url = img0['image_url']['url'] if isinstance(img0, dict) else img0.image_url.url
            img_data = base64.b64decode(url.split(",", 1)[1])
            return Image.open(io.BytesIO(img_data))

        # Fallback: content list with image_url
        for part in msg.content if isinstance(msg.content, list) else []:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:image"):
                    img_data = base64.b64decode(url.split(",", 1)[1])
                    return Image.open(io.BytesIO(img_data))

        # Fallback: content is data URL string
        content = msg.content
        if isinstance(content, str) and content.startswith("data:image"):
            img_data = base64.b64decode(content.split(",", 1)[1])
            return Image.open(io.BytesIO(img_data))

        return None
    except Exception as e:
        print(f"  VLM error: {e}")
        return None


def process_one(spec: EditSpec, dataset, client, output_dir: Path,
                model: str, logger) -> dict:
    """Process a single edit spec: select view → edit → save."""
    edit_id = spec.edit_id
    result = {"edit_id": edit_id, "obj_id": spec.obj_id}

    try:
        obj = dataset.load_object(spec.shard, spec.obj_id)
        if spec.edit_type == "deletion":
            edit_part_ids = spec.remove_part_ids
        elif spec.edit_type == "global":
            edit_part_ids = []  # no specific part — whole object
        else:
            edit_part_ids = [spec.old_part_id]

        # 1. Select best view
        best_view = select_best_view(obj, edit_part_ids or
                                     [p.part_id for p in obj.parts])
        result["view_id"] = best_view

        # 2. Prepare input image (with part annotation for VLM)
        annotated_bytes, annotated_pil, plain_pil = prepare_input_image(
            obj, best_view, edit_part_ids)
        has_annotation = (annotated_pil is not plain_pil)

        # Save both: plain for phase2.5 conditioning, annotated for debug
        input_path = output_dir / f"{edit_id}_input.png"
        plain_pil.save(str(input_path))
        if has_annotation:
            debug_path = output_dir / f"{edit_id}_annotated.png"
            annotated_pil.save(str(debug_path))

        # 3. Call VLM with annotated image (part region highlighted)
        edited = call_vlm_edit(
            client, annotated_bytes, spec.edit_prompt,
            spec.after_part_desc, model,
            has_annotation=has_annotation)

        if edited is not None:
            edited = edited.resize((518, 518))
            out_path = output_dir / f"{edit_id}_edited.png"
            edited.save(str(out_path))
            result["status"] = "success"
            result["edited_image"] = str(out_path)
            result["input_image"] = str(input_path)
            logger.info(f"  {edit_id}: OK -> {out_path}")
        else:
            result["status"] = "failed"
            result["reason"] = "VLM returned no image"
            logger.warning(f"  {edit_id}: VLM returned no image")

        obj.close()
    except Exception as e:
        result["status"] = "failed"
        result["reason"] = str(e)
        logger.error(f"  {edit_id}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2D: Batch 2D image editing for modification specs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel VLM API calls")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Override image edit model name")
    parser.add_argument("--specs", type=str, default=None,
                        help="Path to edit_specs JSONL "
                             "(default: {phase1.cache_dir}/edit_specs.jsonl)")
    parser.add_argument("--edit-dir", type=str, default=None,
                        help="Output subdir name for 2D edits "
                             "(default: '2d_edits'). Use e.g. '2d_edits_action' "
                             "to avoid mixing with default-style edits")
    parser.add_argument("--type", type=str, default=None,
                        choices=["modification", "deletion", "global"],
                        help="Filter by edit type (default: modification)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag. Output goes to 2d_edits_{tag} "
                             "(overrides --edit-dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "2d_edit")
    p0 = cfg["phase0"]
    p25 = cfg.get("phase2_5", {})

    # --- VLM client ---
    from openai import OpenAI
    api_key = p0.get("vlm_api_key", "")
    if not api_key:
        # Fallback to default.yaml
        import yaml
        default_cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        if default_cfg_path.exists():
            with open(default_cfg_path) as f:
                default_cfg = yaml.safe_load(f)
            api_key = default_cfg.get("phase0", {}).get("vlm_api_key", "")
    if not api_key:
        env_var = p0.get("vlm_api_key_env", "")
        if env_var:
            import os
            api_key = os.environ.get(env_var, "")

    if not api_key:
        print("ERROR: No API key. Set phase0.vlm_api_key in config or default.yaml")
        sys.exit(1)

    client = OpenAI(
        base_url=p0.get("vlm_base_url", ""),
        api_key=api_key,
    )
    model = args.model or p25.get("image_edit_model", "gemini-2.5-flash-image")

    # --- Dataset ---
    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )

    # --- Load edit specs ---
    edit_types = [args.type] if args.type else ["modification"]
    specs_path = Path(args.specs) if args.specs else (
        Path(cfg["phase1"]["cache_dir"]) / "edit_specs.jsonl")
    mod_specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if spec.edit_type in edit_types:
                mod_specs.append(spec)

    if args.limit:
        mod_specs = mod_specs[:args.limit]

    # --- Output dir ---
    cache_dir = Path(p25.get("cache_dir", "outputs/cache/phase2_5"))
    if args.tag:
        edit_subdir = f"2d_edits_{args.tag}"
    else:
        edit_subdir = args.edit_dir or "2d_edits"
    output_dir = cache_dir / edit_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    # --- Resume ---
    done_ids: set[str] = set()
    if args.resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [s for s in mod_specs if s.edit_id not in done_ids]
    logger.info(f"Phase 2D: {len(pending)} edits to process "
                f"({len(done_ids)} already done), model={model}, "
                f"workers={args.workers}")

    if not pending:
        logger.info("All 2D edits already done")
        return

    # --- Process ---
    success, fail = 0, 0
    with open(manifest_path, "a") as fp:
        if args.workers <= 1:
            for i, spec in enumerate(pending):
                logger.info(f"[{i+1}/{len(pending)}] {spec.edit_id}: "
                            f"{spec.edit_prompt[:60]}...")
                result = process_one(spec, dataset, client, output_dir,
                                     model, logger)
                fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                fp.flush()
                if result["status"] == "success":
                    success += 1
                else:
                    fail += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
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

    logger.info(f"Phase 2D complete: {success} ok, {fail} fail -> {manifest_path}")


if __name__ == "__main__":
    main()
