"""Phase 0: Semantic labeling + edit prompt generation.

Two modes:
  1. "dataset" mode (default): Part labels come from semantic.json (ground truth).
     VLM is called with pre-rendered 150 views to generate rich descriptions and
     edit prompts. No HY3D-Part 42-view rendering needed.
  2. "vlm" mode (legacy): Uses HY3D-Part 42-view renders + masks for VLM labeling.

Produces per-object:
  - object_desc: natural language description of the whole object
  - per-part semantic labels (e.g. "chair_leg", "car_wheel")
  - core/peripheral classification (can this part be removed?)
  - per-part editing instructions (deletion, addition, modification prompts)
"""

from __future__ import annotations

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from partcraft.io.hy3d_loader import HY3DPartDataset, ObjectRecord


def _img_bytes_to_data_url(img_bytes: bytes, fmt: str = "webp") -> str:
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:image/{fmt};base64,{b64}"


def _highlight_part_on_view(img_bytes: bytes, mask: np.ndarray, part_id: int,
                            pad: int = 30) -> bytes | None:
    """Create a crop highlighting a specific part with dimmed surroundings."""
    try:
        from PIL import Image, ImageDraw
        from scipy.ndimage import binary_dilation
    except ImportError:
        return None

    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    arr = np.array(img)

    part_mask = mask == part_id
    if part_mask.sum() < 10:
        return None

    # Dim non-part pixels
    highlight = arr.copy()
    highlight[~part_mask] = (highlight[~part_mask].astype(np.float32) * 0.3).astype(np.uint8)

    # Red border around part
    border = binary_dilation(part_mask, iterations=3) & ~part_mask
    highlight[border] = [255, 50, 50, 255]

    # Crop to part bounding box + padding
    ys, xs = np.where(part_mask)
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad, arr.shape[0])
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad, arr.shape[1])

    crop = Image.fromarray(highlight[y0:y1, x0:x1])
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def _align_labels_to_parts(part_labels: list[str], n_parts: int) -> list[str]:
    """Align semantic.json labels to HY3D part IDs.

    semantic.json often has fewer labels than HY3D parts (e.g. 10 labels for
    12 parts). We pad with "unknown_N" for the extra parts so every part_id
    gets a label entry and the VLM prompt stays consistent.
    """
    aligned = list(part_labels[:n_parts])
    for i in range(len(aligned), n_parts):
        aligned.append(f"unknown_{i}")
    return aligned


def build_vlm_request_prerender(
    obj: ObjectRecord,
    part_labels: list[str],
    prerender_dir: str,
    view_indices: list[int] | None = None,
) -> list[dict]:
    """Build VLM request using pre-rendered 150 views + ground-truth part labels.

    Uses high-quality Blender renders (512x512, 150 views) instead of
    HY3D-Part's 42 views (518x518, WebP). Part labels come from semantic.json.

    Args:
        obj: ObjectRecord (used for part count and IDs)
        part_labels: ground-truth labels from semantic.json (ordered by part_id)
        prerender_dir: path to outputs/img_Enc/{obj_id}/ with 000.png..149.png
        view_indices: which views to send to VLM (default: 4 evenly-spaced)
    """
    n_parts = len(obj.parts)
    prerender_path = Path(prerender_dir)

    if view_indices is None:
        # Pick 4 evenly-spaced views from 150
        view_indices = [0, 37, 75, 112]

    # Align labels to actual part count (pad if semantic.json has fewer)
    aligned_labels = _align_labels_to_parts(part_labels, n_parts)

    # Build part label text
    part_lines = []
    has_unknown = False
    for i, label in enumerate(aligned_labels):
        if label.startswith("unknown_"):
            part_lines.append(f'  Part {i}: (unlabeled — please identify)')
            has_unknown = True
        else:
            part_lines.append(f'  Part {i}: "{label}"')
    part_list_text = "\n".join(part_lines)

    # Collect actual available view images first
    view_images: list[tuple[int, bytes]] = []
    for vid in view_indices:
        img_path = prerender_path / f"{vid:03d}.png"
        if not img_path.exists():
            continue
        with open(img_path, "rb") as f:
            view_images.append((vid, f.read()))

    n_views = len(view_images)
    if n_views == 0:
        # No renders available — return text-only request
        return [{"type": "text", "text":
                 f"This 3D object has {n_parts} parts. No views available. "
                 "Please provide basic part descriptions based on the labels:\n"
                 f"{part_list_text}"}]

    # Extra instruction if some parts are unlabeled
    unknown_note = ""
    if has_unknown:
        unknown_note = (
            "\nNote: Some parts are unlabeled. For those, infer a reasonable "
            "semantic label from the images and provide the same fields.\n"
        )

    content = []
    content.append({"type": "text", "text": (
        f"This 3D object is shown from {n_views} viewpoints. "
        f"It has {n_parts} segmented parts:\n"
        f"{part_list_text}\n"
        f"{unknown_note}\n"
        f"For each of the {n_parts} parts (part_id 0 to {n_parts - 1}), provide:\n"
        "  1. `label`: use the given label (you may refine if clearly wrong, "
        "or assign a label for unlabeled parts)\n"
        "  2. `core`: true if removing this part would destroy the object's "
        "identity, false otherwise\n"
        "  3. `desc`: a short phrase describing appearance (shape, color, "
        "material, position)\n"
        "  4. `desc_without`: description of the WHOLE object if this part "
        "were removed. For core parts, set to empty string.\n"
        "  5. `edits`: a list of plausible editing instructions:\n"
        "     For NON-CORE parts:\n"
        '       a) One edit with type "deletion" — a natural instruction to '
        "remove this part\n"
        '       b) One edit with type "addition" — a natural instruction to '
        "add this part (assume the reader sees the object WITHOUT it)\n"
        '       c) 1-2 edits with type "modification" — change this part\'s '
        "style, material, color, or shape\n"
        "     For CORE parts:\n"
        '       a) 1-2 edits with type "modification" only\n\n'
        "Each edit object must have these fields:\n"
        '  - "type": "deletion" | "addition" | "modification"\n'
        '  - "prompt": a specific, visual, natural-language editing '
        "instruction. Mention concrete details (material, color, shape, "
        "style). Do NOT write generic instructions.\n"
        '  - "after_desc": description of the WHOLE object after the edit\n'
        '  - For modification only: also include "before_part_desc" and '
        '"after_part_desc"\n\n'
        "IMPORTANT: Output ONLY a valid JSON object, no extra text.\n"
        "The JSON must have this structure:\n"
        "{\n"
        '  "object_desc": "A detailed description of the whole object",\n'
        '  "parts": [\n'
        "    {\n"
        '      "part_id": 0,\n'
        '      "label": "backrest",\n'
        '      "core": false,\n'
        '      "desc": "curved wooden backrest with vertical slats",\n'
        '      "desc_without": "A dining chair without a backrest, showing '
        'only the seat and four legs",\n'
        '      "edits": [\n'
        '        {"type": "deletion", "prompt": "Remove the curved wooden '
        'backrest with vertical slats from the dining chair", '
        '"after_desc": "A dining chair without a backrest"},\n'
        '        {"type": "addition", "prompt": "Add a curved wooden '
        'backrest with vertical slats to the back of the dining chair", '
        '"after_desc": "A dining chair with a curved wooden backrest"},\n'
        '        {"type": "modification", "prompt": "Replace the curved '
        "wooden backrest with a modern mesh fabric backrest in dark gray"
        '", "after_desc": "A dining chair with a mesh fabric backrest", '
        '"before_part_desc": "curved wooden backrest with vertical slats'
        '", "after_part_desc": "modern dark gray mesh fabric backrest"}\n'
        "      ]\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}"
    )})

    # Add pre-rendered views
    for vid, img_bytes in view_images:
        content.append({"type": "image_url",
                        "image_url": {"url": _img_bytes_to_data_url(img_bytes, "png")}})

    return content


def build_vlm_request(obj: ObjectRecord, view_ids: list[int]) -> list[dict]:
    """Build the content parts for a single VLM call that labels all parts
    AND generates editing instructions (deletion, addition, modification)."""
    n_parts = len(obj.parts)

    content = []

    # 1) Instruction + full object views
    content.append({"type": "text", "text": (
        f"This 3D object is shown from {len(view_ids)} viewpoints. "
        f"It has {n_parts} segmented parts (highlighted individually below). "
        "For each part, provide:\n"
        "  1. `label`: a concise semantic name (e.g. 'chair_leg', 'car_hood', 'lamp_shade')\n"
        "  2. `core`: true if removing this part would destroy the object's identity, false otherwise\n"
        "  3. `desc`: a short phrase describing appearance (shape, color, material, position)\n"
        "  4. `desc_without`: description of the WHOLE object if this part were removed. "
        "For core parts, set to empty string.\n"
        "  5. `edits`: a list of plausible editing instructions for this part. "
        "For NON-CORE parts, generate:\n"
        "     a) One edit with `type: \"deletion\"` — a natural instruction to remove this part\n"
        "     b) One edit with `type: \"addition\"` — a natural instruction to add this part "
        "(assume the reader sees the object WITHOUT this part)\n"
        "     c) 1-2 edits with `type: \"modification\"` — change this part's style, material, "
        "color, or shape to something different but plausible\n"
        "  For CORE parts, generate:\n"
        "     a) 1-2 edits with `type: \"modification\"` only (core parts cannot be removed)\n\n"
        "Each edit must have:\n"
        "  - `type`: \"deletion\" | \"addition\" | \"modification\"\n"
        "  - `prompt`: a specific, visual, natural-language instruction. Include details about "
        "material, color, shape, or style. Do NOT write generic instructions like 'remove the part'.\n"
        "  - `after_desc`: description of the WHOLE object after the edit is applied\n"
        "  - For modification: also include `before_part_desc` (current appearance) and "
        "`after_part_desc` (new appearance of just the changed part)\n\n"
        "Output ONLY valid JSON:\n"
        '{"object_desc": "...", "parts": [{"part_id": 0, "label": "...", "core": true, '
        '"desc": "...", "desc_without": "", '
        '"edits": [{"type": "modification", "prompt": "Replace the curved wooden backrest '
        'with a modern mesh fabric backrest", "after_desc": "A dining chair with mesh backrest...", '
        '"before_part_desc": "curved wooden backrest with slats", '
        '"after_part_desc": "sleek mesh fabric backrest"}]}, ...]}'
    )})

    for vid in view_ids:
        img_bytes = obj.get_image_bytes(vid)
        content.append({"type": "image_url",
                        "image_url": {"url": _img_bytes_to_data_url(img_bytes)}})

    # 2) Per-part highlight crops
    for part in obj.parts:
        best_view = obj.get_best_view_for_part(part.part_id)
        img_bytes = obj.get_image_bytes(best_view)
        mask = obj.get_mask(best_view)
        crop = _highlight_part_on_view(img_bytes, mask, part.part_id)
        if crop is None:
            content.append({"type": "text", "text": f"Part {part.part_id}: (not visible in any view)"})
            continue
        content.append({"type": "text", "text": f"Part {part.part_id}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": _img_bytes_to_data_url(crop, "png")}})

    return content


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from VLM output.

    Handles: raw JSON, markdown fences (```json ... ```), surrounding prose,
    and common VLM quirks (trailing commas, comments).
    """
    text = text.strip()

    # 1. Try raw parse first (fastest path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences: ```json ... ``` or ``` ... ```
    import re
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find the outermost { ... } (handles surrounding prose)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break

    raise json.JSONDecodeError(
        f"Could not extract valid JSON from VLM response (first 200 chars): "
        f"{text[:200]}", text, 0)


def call_vlm(client, content: list[dict], model: str,
             max_retries: int = 2, timeout: float = 120) -> dict:
    """Send content to VLM and parse JSON response with retries."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                timeout=timeout,
            )
            text = response.choices[0].message.content
            return _extract_json(text)
        except json.JSONDecodeError as e:
            last_err = e
            if attempt < max_retries:
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                import time
                time.sleep(2 ** attempt)
                continue
            raise


def _validate_vlm_result(result: dict, obj: ObjectRecord) -> dict:
    """Validate and fix VLM response structure.

    Ensures:
      - 'object_desc' exists (string)
      - 'parts' is a list with valid entries
      - Each part has: part_id (int), label (str), core (bool)
      - Part IDs match actual object parts
    """
    if "object_desc" not in result or not isinstance(result["object_desc"], str):
        result["object_desc"] = ""

    valid_pids = {p.part_id for p in obj.parts}

    if "parts" not in result or not isinstance(result["parts"], list):
        # VLM returned no parts — generate fallback entries
        result["parts"] = [
            {"part_id": p.part_id, "label": f"part_{p.part_id}",
             "core": False, "desc": "", "desc_without": "", "edits": []}
            for p in obj.parts
        ]
        return result

    # Validate each part entry
    cleaned_parts = []
    seen_pids = set()
    for p in result["parts"]:
        if not isinstance(p, dict):
            continue
        pid = p.get("part_id")
        if pid is None or not isinstance(pid, int):
            continue
        if pid not in valid_pids:
            continue
        if pid in seen_pids:
            continue
        seen_pids.add(pid)

        # Ensure required fields
        p.setdefault("label", f"part_{pid}")
        p.setdefault("core", False)
        p.setdefault("desc", "")
        p.setdefault("desc_without", "")
        if not isinstance(p.get("edits"), list):
            p["edits"] = []

        # Validate each edit entry
        valid_edits = []
        for e in p["edits"]:
            if not isinstance(e, dict):
                continue
            if e.get("type") not in ("deletion", "addition", "modification"):
                continue
            e.setdefault("prompt", "")
            e.setdefault("after_desc", "")
            valid_edits.append(e)
        p["edits"] = valid_edits

        cleaned_parts.append(p)

    # Add fallback entries for missing parts
    for part in obj.parts:
        if part.part_id not in seen_pids:
            cleaned_parts.append({
                "part_id": part.part_id, "label": f"part_{part.part_id}",
                "core": False, "desc": "", "desc_without": "", "edits": [],
            })

    result["parts"] = cleaned_parts
    return result


def label_single_object(
    client, obj: ObjectRecord, cfg: dict,
    part_labels: list[str] | None = None,
    prerender_dir: str | None = None,
) -> dict:
    """Label all parts of a single object via VLM. Returns semantic dict.

    If part_labels and prerender_dir are provided, uses pre-rendered 150 views
    + ground-truth labels (dataset mode). Otherwise falls back to HY3D 42 views.
    """
    model = cfg["phase0"]["vlm_model"]

    if part_labels is not None and prerender_dir is not None:
        # Dataset mode: use pre-rendered views + known labels
        content = build_vlm_request_prerender(
            obj, part_labels, prerender_dir)
    else:
        # Legacy mode: use HY3D-Part 42 views
        view_ids = cfg["phase0"]["views_for_labeling"]
        content = build_vlm_request(obj, view_ids)

    result = call_vlm(client, content, model)

    # Validate and fix VLM response structure
    result = _validate_vlm_result(result, obj)

    # Attach obj_id for traceability
    result["obj_id"] = obj.obj_id
    result["shard"] = obj.shard
    result["num_parts"] = len(obj.parts)

    return result


def _create_vlm_client(cfg: dict):
    """Create VLM client based on backend config.

    Supports:
      - "api": remote API (Gemini, OpenAI-compatible)
      - "local": local vLLM/SGLang serving Qwen2.5-VL on GPU
    """
    from openai import OpenAI
    p0 = cfg["phase0"]
    backend = p0.get("vlm_backend", "api")

    if backend == "local":
        base_url = p0.get("local_base_url", "http://localhost:8000/v1")
        print(f"Phase 0: using local VLM at {base_url}")
        return OpenAI(base_url=base_url, api_key="dummy"), p0.get("local_model_path", "default")
    else:
        api_key = p0.get("vlm_api_key", "")
        if not api_key:
            env_var = p0.get("vlm_api_key_env", "")
            if env_var:
                api_key = os.environ.get(env_var, "")
        return OpenAI(base_url=p0["vlm_base_url"], api_key=api_key), p0["vlm_model"]


def _load_semantic_json(cfg: dict) -> dict[str, list[str]]:
    """Load ground-truth part labels from semantic.json.

    Returns: {obj_id: [label_0, label_1, ...]}
    """
    data_dir = Path(cfg["data"].get("data_dir", "data/partobjaverse_tiny"))
    if not data_dir.exists():
        # Fallback: derive from image_npz_dir
        data_dir = Path(cfg["data"]["image_npz_dir"]).parent
    sem_path = data_dir / "source" / "semantic.json"
    if not sem_path.exists():
        return {}

    with open(sem_path) as f:
        sem = json.load(f)

    # Flatten: semantic.json is {category: {obj_id: [labels]}}
    result = {}
    for category, objects in sem.items():
        if isinstance(objects, dict):
            for obj_id, labels in objects.items():
                result[obj_id] = labels
    return result


def _get_prerender_dir(cfg: dict, obj_id: str) -> str | None:
    """Get pre-rendered views directory for an object (if exists)."""
    project_root = Path(__file__).parents[2]
    render_dir = str(project_root / "data" / "img_Enc" / obj_id)
    # Check that renders actually exist
    if os.path.isdir(render_dir) and os.path.exists(os.path.join(render_dir, "000.png")):
        return render_dir
    return None


def run_phase0(cfg: dict, dataset: HY3DPartDataset | None = None,
               limit: int | None = None, force: bool = False):
    """Run Phase 0: semantic labeling for all objects.

    Uses pre-rendered 150 views + semantic.json labels when available (dataset mode).
    Falls back to HY3D-Part 42 views otherwise (legacy mode).

    Results are saved as JSONL to cache_dir/semantic_labels.jsonl.

    Args:
        force: if True, delete existing output and re-run everything.
    """
    cache_dir = Path(cfg["phase0"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "semantic_labels.jsonl"

    if force and output_path.exists():
        # Back up old results, then start fresh
        backup = output_path.with_suffix(".jsonl.bak")
        output_path.rename(backup)
        print(f"Phase 0: --force: backed up old results to {backup}")

    # Track already-processed objects
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["obj_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    if dataset is None:
        dataset = HY3DPartDataset(
            cfg["data"]["image_npz_dir"],
            cfg["data"]["mesh_npz_dir"],
            cfg["data"]["shards"],
        )

    # Load ground-truth part labels
    semantic_map = _load_semantic_json(cfg)
    use_prerender = bool(semantic_map)
    if use_prerender:
        print(f"Phase 0: dataset mode — {len(semantic_map)} objects with "
              "ground-truth labels, using pre-rendered 150 views")
    else:
        print("Phase 0: legacy mode — using HY3D-Part 42 views for VLM labeling")

    client, model_name = _create_vlm_client(cfg)
    cfg = {**cfg, "phase0": {**cfg["phase0"], "vlm_model": model_name}}

    objects = []
    for obj in dataset:
        if obj.obj_id in done_ids:
            continue
        objects.append(obj)
        if limit and len(objects) >= limit:
            break

    if not objects:
        print(f"Phase 0: all objects already labeled ({len(done_ids)} done)")
        return output_path

    print(f"Phase 0: labeling {len(objects)} objects ({len(done_ids)} already done)")

    max_workers = cfg["phase0"].get("max_workers", 4)

    with open(output_path, "a") as out_fp:
        def _process(obj: ObjectRecord) -> tuple[str, dict | None, str]:
            try:
                part_labels = semantic_map.get(obj.obj_id) if use_prerender else None
                prerender_dir = _get_prerender_dir(cfg, obj.obj_id) if part_labels else None
                result = label_single_object(
                    client, obj, cfg,
                    part_labels=part_labels,
                    prerender_dir=prerender_dir,
                )
                return obj.obj_id, result, ""
            except Exception as e:
                return obj.obj_id, None, str(e)
            finally:
                obj.close()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, obj): obj.obj_id for obj in objects}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Phase 0")
            success, fail = 0, 0
            for future in pbar:
                obj_id, result, err = future.result()
                if result is not None:
                    out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                else:
                    fail += 1
                pbar.set_postfix(ok=success, fail=fail)

    print(f"Phase 0 complete: {success} labeled, {fail} failed → {output_path}")
    return output_path
