#!/usr/bin/env python3
"""Standalone 2D edit utility.

This module provides 2D image editing helpers for the FLUX service.

Phase 2D: Batch 2D image editing for TRELLIS-bound edit specs.

Pre-generates edited reference images for Phase 2.5 TRELLIS,
so GPU-heavy 3D editing doesn't block on API calls.

For each TRELLIS-bound spec (modification/scale/material/global):
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

import os as _os
def _get_project_root() -> Path:
    script = Path(__file__)
    if not script.is_absolute():
        script = Path(_os.environ.get('PWD', _os.getcwd())) / script
    return script.parents[1]
_PROJECT_ROOT = _get_project_root()
sys.path.insert(0, str(_PROJECT_ROOT))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.hy3d_loader import HY3DPartDataset
from partcraft.pipeline_v3.specs import EditSpec


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

    Returns (png_bytes, pil_img). No mask annotation — the edit prompt
    provides sufficient semantic clarity for the VLM.
    """
    from PIL import Image

    view_bytes = obj_record.get_image_bytes(view_id)
    pil_img = Image.open(io.BytesIO(view_bytes)).convert("RGBA")
    pil_img = pil_img.resize((518, 518))
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    pil_img = Image.alpha_composite(bg, pil_img).convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue(), pil_img


def check_edit_server(base_url: str):
    """Check that the image edit server is reachable."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                return True
    except Exception:
        pass
    return False


def _build_edit_prompt(edit_prompt: str, after_part_desc: str,
                       old_part_label: str = "",
                       before_part_desc: str = "",
                       edit_type: str = "modification",
                       edit_params: dict | None = None) -> str:
    """Build a constrained prompt for 2D image editing.

    Adapts prompt structure to the edit type:
      - deletion: instruct removal, generate clean closure
      - scale: instruct part-only size/proportion change
      - color: pure hue swap on a single part (uses
        edit_params['target_color'] when available)
      - modification / material: edit specific part(s),
        preserve others (material falls through to the
        modification branch - already at ~62% pass)
      - global: instruct whole-object style change
    """
    et = edit_type.lower()

    if et == "deletion":
        # Deletion: remove part(s), generate clean object without them
        if old_part_label:
            target = f"the '{old_part_label}'"
        else:
            target = "the specified part"
        text = (
            f"This is a 3D rendered object on a white background. "
            f"{edit_prompt}. "
            f"Generate the same object WITHOUT {target}. "
            f"Fill in the area where {target} was with a smooth, "
            f"natural surface continuous with the surrounding geometry."
        )
        if after_part_desc:
            text += f"\nThe result should look like: {after_part_desc}"
        text += (
            "\nIMPORTANT constraints:"
            "\n- Keep the exact same camera viewpoint and angle."
            "\n- Keep the white background completely unchanged."
            "\n- Keep ALL other parts of the object exactly as they are."
            "\n- The removed area should blend naturally with surrounding surfaces."
        )
    elif et == "global":
        # Global: whole-object style/theme change
        text = (
            f"This is a 3D rendered object on a white background. "
            f"Apply the following style change to the ENTIRE object: "
            f"{edit_prompt}"
        )
        if after_part_desc:
            text += f"\nThe result should look like: {after_part_desc}"
        text += (
            "\nIMPORTANT constraints:"
            "\n- Keep the exact same camera viewpoint and angle."
            "\n- Keep the white background completely unchanged."
            "\n- Keep the overall shape, pose, and position unchanged."
            "\n- Only change the style, texture, or color as instructed."
        )
    elif et == "scale":
        # Scale: resize target part while preserving all other parts.
        if old_part_label:
            target = f"the '{old_part_label}' part"
        else:
            target = "the specified part"
        text = (
            f"This is a 3D rendered object on a white background. "
            f"Resize ONLY {target} of this object. "
            f"Editing instruction: {edit_prompt}"
        )
        if before_part_desc:
            text += f"\nThe part currently looks like: {before_part_desc}"
        if after_part_desc:
            text += f"\nAfter resizing, it should look like: {after_part_desc}"
        text += (
            "\nIMPORTANT constraints:"
            "\n- Keep the exact same camera viewpoint and angle."
            "\n- Keep the white background completely unchanged."
            "\n- Keep ALL non-target parts of the object exactly as they are."
            "\n- Keep the target part identity and style, but change only its size/proportions."
            "\n- Do NOT move or rotate the object."
        )
    elif et == "color":
        # Color: pure HUE swap on a single part. Mirrors the material-style
        # success pattern observed on shard07 (material 61.5% pass vs color
        # 13.5% pass under the old shared modification template).
        # Key changes vs the old template:
        #   * Verb changed from "Repaint" to "Recolor" - empirically "Repaint"
        #     pushes FLUX toward redrawing the part (causes geometry
        #     distortion / "melted helmet" failures in shard07).
        #   * Inline the canonical target_color from edit_params (same recipe
        #     that makes "Change to polished walnut wood" work for material).
        #   * Explicit guarantee: keep material/finish/geometry - only hue.
        if old_part_label:
            target = f"the '{old_part_label}' part"
        else:
            target = "the specified part"
        target_color = ((edit_params or {}).get("target_color") or "").strip()
        if target_color:
            text = (
                f"This is a 3D rendered object on a white background. "
                f"Recolor ONLY {target} to {target_color}. "
                f"Keep the same surface material, finish, lighting, and "
                f"geometry - only swap the hue/shade of that part."
            )
        else:
            # Fallback when edit_params lacks target_color: use the original
            # natural-language instruction but with the safer "Recolor" verb.
            text = (
                f"This is a 3D rendered object on a white background. "
                f"Recolor ONLY {target} of this object. "
                f"Editing instruction: {edit_prompt}"
            )
        if before_part_desc:
            text += f"\nThe part currently looks like: {before_part_desc}"
        if after_part_desc:
            text += f"\nAfter recoloring, it should look like: {after_part_desc}"
        text += (
            "\nIMPORTANT constraints:"
            "\n- Keep the exact same camera viewpoint and angle."
            "\n- Keep the white background completely unchanged."
            "\n- Keep ALL other parts of the object exactly as they are."
            "\n- Do NOT change the geometry, material, or finish of the "
            "target part - only its color."
            "\n- Apply the new color uniformly across the entire target part."
        )
    else:
        # Modification: edit specific part(s)
        if old_part_label:
            target = f"the '{old_part_label}' part"
        else:
            target = "the specified part"
        text = (
            f"This is a 3D rendered object on a white background. "
            f"Edit ONLY {target} of this object. "
            f"Editing instruction: {edit_prompt}"
        )
        if before_part_desc:
            text += f"\nThe part currently looks like: {before_part_desc}"
        if after_part_desc:
            text += f"\nAfter editing, it should look like: {after_part_desc}"
        text += (
            "\nIMPORTANT constraints:"
            "\n- Keep the exact same camera viewpoint and angle."
            "\n- Keep the white background completely unchanged."
            "\n- Keep ALL other parts of the object exactly as they are."
            "\n- Do NOT change the overall shape, pose, or position of the object."
            "\n- Only modify the target part as instructed."
        )
    return text


def call_local_edit(base_url: str, img_bytes: bytes, edit_prompt: str,
                    after_part_desc: str,
                    old_part_label: str = "",
                    before_part_desc: str = "",
                    edit_type: str = "modification",
                    edit_params: dict | None = None) -> "Image.Image | None":
    """Edit image via local HTTP image edit server."""
    import base64
    import urllib.request
    from PIL import Image

    text_input = _build_edit_prompt(
        edit_prompt, after_part_desc, old_part_label, before_part_desc,
        edit_type=edit_type, edit_params=edit_params)

    image_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = json.dumps({"image_b64": image_b64, "prompt": text_input}).encode()

    try:
        req = urllib.request.Request(
            f"{base_url}/edit",
            data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())

        if data.get("status") == "ok":
            img_data = base64.b64decode(data["image_b64"])
            return Image.open(io.BytesIO(img_data))
        else:
            print(f"  Edit server error: {data.get('msg', 'unknown')}")
            return None
    except Exception as e:
        print(f"  Edit server request failed: {e}")
        return None


def call_vlm_edit(client, img_bytes: bytes, edit_prompt: str,
                  after_part_desc: str, model: str,
                  old_part_label: str = "",
                  before_part_desc: str = "",
                  edit_type: str = "modification",
                  edit_params: dict | None = None,
                  **kwargs) -> "Image.Image | None":
    """Call VLM to produce an edited image via OpenAI-compatible API."""
    import base64
    from PIL import Image

    b64 = base64.b64encode(img_bytes).decode('utf-8')
    text_input = _build_edit_prompt(
        edit_prompt, after_part_desc, old_part_label, before_part_desc,
        edit_type=edit_type, edit_params=edit_params)

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
                model: str, logger, edit_server_url=None) -> dict:
    """Process a single edit spec: select view → edit → save.

    Args:
        client: OpenAI client for API backend (can be None if edit_server_url set).
        edit_server_url: Base URL of local image edit server (e.g. http://localhost:8001).
    """
    edit_id = spec.edit_id
    result = {"edit_id": edit_id, "obj_id": spec.obj_id}

    try:
        obj = dataset.load_object(spec.shard, spec.obj_id)
        if spec.edit_type == "deletion":
            edit_part_ids = spec.selected_part_ids
        elif spec.edit_type == "global":
            edit_part_ids = []  # no specific part — whole object
        elif spec.edit_type == "modification" and spec.selected_part_ids:
            # Group modification: use all parts in the group
            edit_part_ids = spec.selected_part_ids
        else:
            edit_part_ids = [spec.selected_part_ids[0]] if spec.selected_part_ids else []

        # 1. Select best view — prefer Step 1's orthogonal selection
        if hasattr(spec, 'npz_view') and spec.npz_view >= 0:
            best_view = spec.npz_view
        else:
            best_view = select_best_view(obj, edit_part_ids or
                                         [p.part_id for p in obj.parts])
        result["view_id"] = best_view

        # 2. Prepare input image (plain, no mask annotation)
        img_bytes, pil_img = prepare_input_image(
            obj, best_view, edit_part_ids)

        input_path = output_dir / f"{edit_id}_input.png"
        pil_img.save(str(input_path))

        # 3. Edit image — local server or API
        after_desc = spec.new_parts_desc or ""
        before_desc = spec.target_part_desc or ''

        # Build part label: use all remove_labels for groups,
        # fallback to old_label for single-part edits
        remove_labels = spec.part_labels
        old_label = spec.part_labels[0] if spec.part_labels else ''
        if remove_labels and len(remove_labels) > 1:
            part_label = ", ".join(remove_labels)
        elif remove_labels:
            part_label = remove_labels[0]
        else:
            part_label = old_label

        if edit_server_url is not None:
            edited = call_local_edit(
                edit_server_url, img_bytes, spec.prompt, after_desc,
                old_part_label=part_label, before_part_desc=before_desc,
                edit_type=spec.edit_type,
                edit_params=getattr(spec, "edit_params", None))
        else:
            edited = call_vlm_edit(
                client, img_bytes, spec.prompt,
                after_desc, model,
                old_part_label=part_label, before_part_desc=before_desc,
                edit_type=spec.edit_type,
                edit_params=getattr(spec, "edit_params", None))

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
        description="Phase 2D: Batch 2D image editing for TRELLIS-bound specs")
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
                        choices=["modification", "scale", "material", "global"],
                        help="Filter by edit type (default: modification)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag. Output goes to 2d_edits_{tag} "
                             "(overrides --edit-dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "2d_edit")
    p0 = (cfg.get("services") or {}).get("vlm")
    if not isinstance(p0, dict):
        raise SystemExit("[CONFIG] services.vlm is required")
    p25 = (cfg.get("services") or {}).get("image_edit") or {}
    if not isinstance(p25, dict):
        p25 = {}


    # --- Image edit backend ---
    image_edit_backend = p25.get("image_edit_backend", "api")
    client = None
    edit_server_url = None
    model = args.model or p25.get("image_edit_model", "gemini-2.5-flash-image")

    if image_edit_backend == "local_diffusers":
        edit_server_url = p25.get("image_edit_base_url", "http://localhost:8001")
        if not check_edit_server(edit_server_url):
            print(f"ERROR: Image edit server not reachable at {edit_server_url}")
            print("Start it first:  conda activate qwen_test && "
                  "python scripts/tools/image_edit_server.py --gpu 2")
            sys.exit(1)
        print(f"Image edit server OK at {edit_server_url}")
        # Use configured workers (default 1 for single-GPU sequential serving)
        cfg_workers = p25.get("image_edit_workers", 1)
        if args.workers != cfg_workers:
            logger.info(f"local_diffusers backend: workers={cfg_workers} "
                        f"(from config)")
            args.workers = cfg_workers
    else:
        from openai import OpenAI
        api_key = p0.get("vlm_api_key", "")
        if not api_key:
            import yaml
            default_cfg_path = _PROJECT_ROOT / "configs" / "default.yaml"
            if default_cfg_path.exists():
                with open(default_cfg_path) as f:
                    default_cfg = yaml.safe_load(f)
                api_key = (default_cfg.get("services") or {}).get("vlm", {}).get("vlm_api_key", "")
        if not api_key:
            env_var = p0.get("vlm_api_key_env", "")
            if env_var:
                import os
                api_key = os.environ.get(env_var, "")
        if not api_key:
            print("ERROR: No API key. Set services.vlm.vlm_api_key in config or default.yaml")
            sys.exit(1)

        image_edit_url = p25.get("image_edit_base_url") or p0.get("vlm_base_url", "")
        client = OpenAI(
            base_url=image_edit_url,
            api_key=api_key,
        )

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
    backend_label = edit_server_url or model
    logger.info(f"Phase 2D: {len(pending)} edits to process "
                f"({len(done_ids)} already done), backend={backend_label}, "
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
                            f"{spec.prompt[:60]}...")
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
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
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

    logger.info(f"Phase 2D complete: {success} ok, {fail} fail -> {manifest_path}")


if __name__ == "__main__":
    main()
