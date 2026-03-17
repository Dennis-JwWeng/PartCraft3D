"""Semantic enrichment via VLM with 1 overview image + text labels.

Takes PartObjaverse-Tiny's structured annotations (category + part labels),
renders ONE quick overview thumbnail per object, and sends both to a VLM
for grounded description and modification prompt generation.

Key trade-off vs Phase 0:
  - Phase 0: 42 views + per-part masks → expensive, slow
  - This:    1 overview image + text labels → cheap, grounded, fast
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

CORE_KEYWORDS = frozenset({
    "body", "torso", "base", "frame", "main", "head",
    "wall", "floor", "chassis", "hull", "trunk",
})

BLENDER_PATH = "/tmp/blender-3.0.1-linux-x64/blender"
BLENDER_SCRIPT = str(Path(__file__).resolve().parents[2] / "scripts" / "blender_render.py")


def _is_core_part(label: str) -> bool:
    tokens = set(label.lower().replace("-", " ").replace("_", " ").split())
    return bool(tokens & CORE_KEYWORDS)


# ---------------------------------------------------------------------------
# Quick thumbnail rendering
# ---------------------------------------------------------------------------

def load_thumbnail_from_npz(npz_path: str, view_id: int = 0) -> bytes | None:
    """Load a pre-rendered view from HY3D-Part image NPZ.

    Returns PNG bytes, or None on failure.
    """
    try:
        import numpy as np
        from PIL import Image as _Image

        npz = np.load(npz_path, allow_pickle=True)
        key = f"{view_id:03d}.webp"
        if key not in npz:
            # Try first available image
            img_keys = [k for k in npz.files if k.endswith(('.webp', '.png'))]
            if not img_keys:
                return None
            key = img_keys[0]

        img = _Image.open(io.BytesIO(npz[key].tobytes())).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to load thumbnail from NPZ: {e}")
        return None


def render_thumbnail(glb_path: str, resolution: int = 512) -> bytes | None:
    """Render 1 overview image from a GLB using Blender (fast, ~3s).

    Returns PNG bytes, or None on failure.
    """
    if not os.path.exists(BLENDER_PATH):
        return _render_thumbnail_pyrender(glb_path, resolution)

    with tempfile.TemporaryDirectory(prefix="thumb_") as tmpdir:
        views = json.dumps([{
            "yaw": 0.5, "pitch": 0.3, "radius": 2.5,
            "fov": 1.047,
        }])
        cmd = [
            BLENDER_PATH, "-b", "-P", BLENDER_SCRIPT, "--",
            "--object", glb_path,
            "--output_folder", tmpdir,
            "--views", views,
            "--resolution", str(resolution),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            png_path = os.path.join(tmpdir, "000.png")
            if os.path.exists(png_path):
                with open(png_path, "rb") as f:
                    return f.read()
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Blender thumbnail failed: {e}")

    return _render_thumbnail_pyrender(glb_path, resolution)


def _render_thumbnail_pyrender(glb_path: str, resolution: int = 512) -> bytes | None:
    """Fallback: render with pyrender (faster but lower quality)."""
    try:
        import trimesh
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        import numpy as np
        from PIL import Image

        scene_tm = trimesh.load(glb_path)
        scene = pyrender.Scene(bg_color=[255, 255, 255, 0],
                               ambient_light=[0.6, 0.6, 0.6])

        if isinstance(scene_tm, trimesh.Scene):
            for name, geom in scene_tm.geometry.items():
                if hasattr(geom, 'faces'):
                    mesh = pyrender.Mesh.from_trimesh(geom)
                    scene.add(mesh)
        else:
            scene.add(pyrender.Mesh.from_trimesh(scene_tm))

        camera = pyrender.PerspectiveCamera(yfov=1.047)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 2.5  # z distance
        cam_pose[1, 3] = 0.3  # slight elevation
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=cam_pose)

        renderer = pyrender.OffscreenRenderer(resolution, resolution)
        color, _ = renderer.render(scene)
        renderer.delete()

        img = Image.fromarray(color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"pyrender thumbnail failed: {e}")
        return None


def _png_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# LLM/VLM prompt construction and calling
# ---------------------------------------------------------------------------

def _build_prompt(category: str, labels: list[str],
                  style: str = "default") -> str:
    """Build text portion of the VLM prompt.

    Args:
        style: "default" for original detailed prompts,
               "action" for 3DEditVerse-style short action-oriented prompts.
    """
    parts_str = ", ".join(f"part_{i}={label}" for i, label in enumerate(labels))

    if style == "action":
        return _build_prompt_action(category, parts_str)
    return _build_prompt_default(category, parts_str)


def _build_prompt_default(category: str, parts_str: str) -> str:
    """Original detailed prompt style."""
    return f"""Look at this 3D object image. Given its category and part list, generate JSON with a description and one style modification per part.

Category: {category}
Parts: {parts_str}

Return JSON only, no fences:
{{"object_desc":"vivid 1-sentence description based on what you see","parts":[{{"part_id":0,"label":"body","is_core":true,"desc":"what this part actually looks like","desc_without":"","modifications":[{{"prompt":"Replace X with Y","after_desc":"object after edit","before_part_desc":"current look","after_part_desc":"new look"}}]}}]}}

Rules:
- Describe what you SEE in the image, not what you imagine
- is_core=true for structural parts, desc_without="" if core
- 1 modification per part, specific about material/style/color changes
- Keep all values SHORT (under 20 words each)"""


def _build_prompt_action(category: str, parts_str: str) -> str:
    """3DEditVerse-style: short, action-oriented prompts strictly within part scope.

    Each non-core part gets 3 modifications:
      1. Style change (material/color/texture)
      2. Another style change (different from #1)
      3. Type/shape swap (replace with a logically compatible alternative)
    Core parts get 2 style modifications only (no shape swap).
    """
    return f"""Look at this 3D object image. Given its category and part list, generate JSON with a description and modifications per part.

Category: {category}
Parts: {parts_str}

Return JSON only, no fences:
{{"object_desc":"1-sentence description","parts":[{{"part_id":0,"label":"blade","is_core":false,"desc":"what this part looks like","desc_without":"object without this part","modifications":[{{"prompt":"Change the blade material to gold","mod_type":"style","after_desc":"object after edit","before_part_desc":"current look","after_part_desc":"new look"}},{{"prompt":"Make the blade rusty","mod_type":"style","after_desc":"object after edit","before_part_desc":"current look","after_part_desc":"new look"}},{{"prompt":"Replace the blade with an axe head","mod_type":"swap","after_desc":"object after edit","before_part_desc":"a steel blade","after_part_desc":"a heavy axe head"}}]}}]}}

CRITICAL RULES:
- Each prompt MUST include the part label so the edit target is unambiguous
  BAD: "Make it rusty"  (what is "it"?)
  GOOD: "Make the blade rusty"
  BAD: "Change material to gold"  (which part?)
  GOOD: "Change the blade material to gold"
- Each prompt must be UNDER 10 WORDS
- Use action verbs: "Change the X to", "Make the X", "Replace the X with"
- ONLY describe the change to THIS SPECIFIC PART, never mention the whole object
  BAD: "Change the sword's blade to gold"  (mentions "sword")
  GOOD: "Change the blade material to gold"

TWO TYPES OF MODIFICATION:
1. mod_type="style": material, color, or texture change. Part keeps its shape.
   Examples: "Make the blade wooden", "Change the hat color to red"
2. mod_type="swap": replace with a DIFFERENT but logically compatible item.
   The replacement must fit in the SAME spatial region and make sense on this object.
   Examples:
     - blade → "Replace the blade with an axe head"
     - hat → "Replace the hat with a top hat"
     - wheel → "Replace the wheels with tank treads"
     - tail → "Replace the tail with a fish tail"
   BAD swaps: "Replace with a car" (too large), "Replace with nothing" (that's deletion)

PER-PART RULES:
- Non-core parts: 2 style + 1 swap = 3 modifications
- Core/structural parts (is_core=true): 2 style modifications only, NO swap
  Core parts cannot change shape — only material/color/texture
- desc_without="" if is_core=true
- Keep ALL string values VERY SHORT (under 10 words each)
- The edit must stay WITHIN the part's own spatial boundary
- Output compact JSON, no extra whitespace"""


def _build_messages(category: str, labels: list[str],
                    thumbnail_bytes: bytes | None,
                    style: str = "default") -> list[dict]:
    """Build VLM messages with optional image."""
    text = _build_prompt(category, labels, style=style)

    if thumbnail_bytes:
        return [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": _png_to_data_url(thumbnail_bytes)}},
                {"type": "text", "text": text},
            ]
        }]
    else:
        # Fallback: text-only
        return [{"role": "user", "content": text}]


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response with fallbacks."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _call_vlm(client, model: str, category: str, labels: list[str],
              thumbnail_bytes: bytes | None,
              max_retries: int = 2,
              style: str = "default") -> dict | None:
    """Call VLM with image + text, with retries."""
    messages = _build_messages(category, labels, thumbnail_bytes, style=style)

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=16384,
                timeout=120,
            )
            msg = resp.choices[0].message
            text = msg.content
            # Debug: log response structure if content is empty/None
            if not text:
                # Try extracting from thinking response structure
                raw = getattr(msg, 'model_extra', {}) or {}
                print(f"  [DEBUG] Empty content. Keys: {list(raw.keys())}"
                      f" | role={msg.role}"
                      f" | type(content)={type(msg.content)}", flush=True)
                # Some proxies put thinking output + real output together
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'text') and part.text:
                            text = part.text
                            break
            if not text:
                logger.warning(f"VLM returned empty content (attempt {attempt+1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
            result = _extract_json(text)
            if result and "parts" in result:
                return result
            logger.warning(f"VLM returned invalid JSON (attempt {attempt+1}): "
                              f"{text[:300] if text else '(empty)'}")
        except Exception as e:
            logger.warning(f"VLM call failed (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_enrichment(category: str, labels: list[str]) -> dict:
    """Rule-based fallback if VLM fails."""
    cat_clean = category.lower().replace("&&", " and ").replace("-", " ")
    parts = []
    for i, label in enumerate(labels):
        is_core = _is_core_part(label)
        lbl = label.lower().replace("_", " ")
        parts.append({
            "part_id": i,
            "label": label.lower().replace(" ", "_"),
            "is_core": is_core,
            "desc": lbl,
            "desc_without": "" if is_core else f"The {cat_clean} without its {lbl}",
            "modifications": [{
                "prompt": f"Change the style of the {lbl}",
                "after_desc": f"The {cat_clean} with a restyled {lbl}",
                "before_part_desc": lbl,
                "after_part_desc": f"restyled {lbl}",
            }],
        })
    return {
        "object_desc": f"A 3D {cat_clean} object",
        "parts": parts,
    }


# ---------------------------------------------------------------------------
# Per-object enrichment
# ---------------------------------------------------------------------------

def _result_to_phase0_record(result: dict, uid: str, category: str,
                              shard: str = "00",
                              prompt_style: str = "default") -> dict:
    """Convert VLM result dict to Phase 0 output format.

    Args:
        prompt_style: "action" generates short del/add prompts from label,
                      "default" uses full desc (legacy).
    """
    phase0_parts = []
    for p in result.get("parts", []):
        pid = p.get("part_id", 0)
        label = p.get("label", f"part_{pid}").lower().replace(" ", "_")
        is_core = p.get("is_core", _is_core_part(label))

        edits = []
        if not is_core:
            desc = p.get("desc", label)
            # Action style: short label-based prompts
            # Default style: verbose desc-based prompts
            label_clean = label.replace("_", " ")
            if prompt_style == "action":
                del_prompt = f"Remove the {label_clean}"
                add_prompt = f"Add a {label_clean}"
            else:
                del_prompt = f"Remove the {desc} from the object"
                add_prompt = f"Add a {desc} to the object"

            edits.append({
                "type": "deletion",
                "prompt": del_prompt,
                "after_desc": p.get("desc_without", ""),
                "before_part_desc": desc,
                "after_part_desc": "",
            })
            edits.append({
                "type": "addition",
                "prompt": add_prompt,
                "after_desc": result.get("object_desc", ""),
                "before_part_desc": "",
                "after_part_desc": desc,
            })

        for mod in p.get("modifications", []):
            edits.append({
                "type": "modification",
                "mod_type": mod.get("mod_type", "style"),  # "style" or "swap"
                "prompt": mod.get("prompt", ""),
                "after_desc": mod.get("after_desc", ""),
                "before_part_desc": mod.get("before_part_desc", p.get("desc", "")),
                "after_part_desc": mod.get("after_part_desc", ""),
            })

        phase0_parts.append({
            "part_id": pid,
            "label": label,
            "core": is_core,
            "desc": p.get("desc", label),
            "desc_without": p.get("desc_without", "") if not is_core else "",
            "edits": edits,
        })

    return {
        "obj_id": uid,
        "shard": shard,
        "num_parts": len(phase0_parts),
        "object_desc": result.get("object_desc", f"A 3D {category} object"),
        "parts": phase0_parts,
    }


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def _resolve_glb_path(uid: str, glb_sources: list[Path]) -> str | None:
    """Find the GLB file for a UID across possible source locations."""
    for src in glb_sources:
        # Direct file
        p = src / f"{uid}.glb"
        if p.exists():
            return str(p)
        # Inside extracted zip directory
        p = src / uid / f"{uid}.glb"
        if p.exists():
            return str(p)
    return None


def enrich_semantic_labels(
    cfg: dict,
    semantic_json_path: str | Path,
    output_path: str | Path,
    image_npz_dir: str | Path | None = None,
    shard: str = "00",
    glb_dir: str | Path | None = None,
    limit: int = 0,
    max_workers: int = 4,
    prompt_style: str = "default",
) -> Path:
    """Enrich all objects via VLM with 1 overview image + text labels.

    Image priority: pre-rendered NPZ > GLB re-render > text-only fallback.

    Args:
        cfg: pipeline config (uses phase0.vlm_model, vlm_base_url, vlm_api_key)
        semantic_json_path: path to PartObjaverse-Tiny_semantic.json
        output_path: where to write semantic_labels.jsonl
        image_npz_dir: directory with pre-rendered {shard}/{uid}.npz (preferred)
        shard: data shard for NPZ lookup
        glb_dir: fallback GLB directory for on-the-fly rendering
        limit: max objects (0=all)
        max_workers: concurrent VLM calls
        prompt_style: "default" for detailed prompts,
                      "action" for 3DEditVerse-style short action-oriented prompts

    Returns:
        Path to the output JSONL file.
    """
    from openai import OpenAI

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup VLM client
    p0 = cfg.get("phase0", {})
    model = p0.get("vlm_model", "gemini-2.5-flash")
    base_url = p0.get("vlm_base_url", "")
    api_key = p0.get("vlm_api_key", "")
    if not api_key:
        env_var = p0.get("vlm_api_key_env", "")
        if env_var:
            api_key = os.environ.get(env_var, "")

    if not api_key:
        # Try loading from default.yaml as fallback
        try:
            import yaml
            default_cfg_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
            if default_cfg_path.exists():
                with open(default_cfg_path) as _f:
                    _dcfg = yaml.safe_load(_f)
                api_key = _dcfg.get("phase0", {}).get("vlm_api_key", "")
        except Exception:
            pass

    if not api_key:
        raise ValueError("No API key. Set phase0.vlm_api_key in config or default.yaml")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Load semantic annotations
    with open(semantic_json_path) as f:
        semantic_json = json.load(f)

    uid_info: dict[str, tuple[str, list[str]]] = {}
    for category, objects in semantic_json.items():
        for uid, labels in objects.items():
            uid_info[uid] = (category, labels)

    all_uids = sorted(uid_info.keys())
    if limit > 0:
        all_uids = all_uids[:limit]

    # Resolve image sources
    npz_dir = Path(image_npz_dir) if image_npz_dir else None
    glb_sources = [Path(glb_dir)] if glb_dir else []

    # Objaverse GLB mapping as fallback
    mapping_path = Path(semantic_json_path).parent / "objaverse_mapping.json"
    glb_mapping: dict[str, str] = {}
    if mapping_path.exists():
        try:
            mapping_data = json.load(open(mapping_path))
            for uid, info in mapping_data.get("objects", {}).items():
                glb = info.get("objaverse_glb")
                if glb and os.path.exists(glb):
                    glb_mapping[uid] = glb
        except Exception:
            pass

    # Resume
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["obj_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            logger.info(f"Resuming: {len(done_ids)} already done")

    pending = [uid for uid in all_uids if uid not in done_ids]
    if not pending:
        logger.info("All objects already enriched")
        return output_path

    # Prepare tasks: resolve thumbnail source for each object
    def _get_thumbnail(uid: str) -> bytes | None:
        # Priority 1: pre-rendered NPZ
        if npz_dir:
            npz_path = npz_dir / shard / f"{uid}.npz"
            if npz_path.exists():
                thumb = load_thumbnail_from_npz(str(npz_path), view_id=0)
                if thumb:
                    return thumb
        # Priority 2: GLB re-render
        glb = glb_mapping.get(uid) or _resolve_glb_path(uid, glb_sources)
        if glb:
            return render_thumbnail(glb, resolution=512)
        return None

    n_with_npz = sum(1 for uid in pending
                     if npz_dir and (npz_dir / shard / f"{uid}.npz").exists())
    logger.info(f"Enriching {len(pending)} objects via VLM ({model}), "
                f"{n_with_npz} with pre-rendered images, "
                f"prompt_style={prompt_style}")

    from tqdm import tqdm

    success, fail = 0, 0
    with open(output_path, "a") as out_fp:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for uid in pending:
                category, labels = uid_info[uid]
                thumbnail = _get_thumbnail(uid)
                fut = pool.submit(
                    _call_vlm, client, model, category, labels, thumbnail,
                    style=prompt_style)
                futures[fut] = (uid, category, labels, thumbnail is not None)

            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="Enriching", unit="obj")
            for fut in pbar:
                uid, category, labels, had_image = futures[fut]
                try:
                    result = fut.result()
                    if result is None:
                        result = _fallback_enrichment(category, labels)
                        fail += 1
                    else:
                        success += 1

                    record = _result_to_phase0_record(
                        result, uid, category, shard,
                        prompt_style=prompt_style)
                    out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fp.flush()

                    n_mods = sum(
                        len([e for e in p["edits"] if e["type"] == "modification"])
                        for p in record["parts"]
                    )
                    n_swap = sum(
                        1 for p in record["parts"] for e in p["edits"]
                        if e.get("mod_type") == "swap"
                    )
                    img_tag = "img" if had_image else "txt"
                    pbar.set_postfix(
                        ok=success, fail=fail,
                        last=f"{uid[:8]}[{img_tag}] {n_mods}m/{n_swap}s")
                except Exception as e:
                    fail += 1
                    pbar.set_postfix(ok=success, fail=fail,
                                     err=str(e)[:40])
            pbar.close()

    logger.info(f"Enrichment complete: {success} succeeded, {fail} failed -> {output_path}")
    return output_path
