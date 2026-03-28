"""Semantic enrichment via VLM — orthogonal 4-view + text labels.

Primary mode (visual, when dataset available):
  4 orthogonal plain views + text labels → VLM groups parts + generates edits (1 call)

Fallback mode (single-call, no dataset):
  1 overview thumbnail + text labels → VLM generates per-part edits (1 call)
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

BLENDER_PATH = "/home/artgen/software/blender-3.3.1-linux-x64/blender"
BLENDER_SCRIPT = str(Path(__file__).parents[2] / "scripts" / "blender_render.py")


def _is_core_part(label: str) -> bool:
    tokens = set(label.lower().replace("-", " ").replace("_", " ").split())
    return bool(tokens & CORE_KEYWORDS)


# ---------------------------------------------------------------------------
# Orthogonal 4-view helpers
# ---------------------------------------------------------------------------

def _select_orthogonal_views(obj, target_elev: float = 25.0) -> list[int]:
    """Select 4 orthogonal views at ~target_elev° elevation.

    Returns view indices for [front(0°), right(90°), back(180°), left(-90°)].
    Only considers views that actually have images in the NPZ
    (obj.view_indices). Falls back to all frames if view_indices is empty.
    """
    import math
    import numpy as np

    transforms = obj.get_transforms()
    frames = transforms["frames"]
    # Only search among views that have actual images
    available = set(obj.view_indices) if obj.view_indices else set(range(len(frames)))
    target_azims = [0, 90, 180, -90]
    selected = []

    for t_azim in target_azims:
        best_idx, best_dist = min(available), float('inf')
        for i in available:
            if i >= len(frames):
                continue
            frame = frames[i]
            c2w = np.array(frame["transform_matrix"])
            pos = c2w[:3, 3]
            r = np.linalg.norm(pos)
            if r < 1e-6:
                continue
            elev = math.degrees(math.asin(pos[2] / r))
            azim = math.degrees(math.atan2(pos[1], pos[0]))
            da = abs(azim - t_azim)
            if da > 180:
                da = 360 - da
            de = abs(elev - target_elev)
            dist = math.sqrt(da ** 2 + de ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        selected.append(best_idx)

    return selected


def _render_plain_views(obj, view_ids: list[int]) -> list[bytes]:
    """Render plain views composited on white background. No mask overlay."""
    from PIL import Image

    results = []
    for vid in view_ids:
        img_bytes = obj.get_image_bytes(vid)
        img_rgba = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        img_rgb = Image.alpha_composite(bg, img_rgba).convert("RGB")
        buf = io.BytesIO()
        img_rgb.save(buf, format="PNG")
        results.append(buf.getvalue())
    return results


# ---------------------------------------------------------------------------
# Orthogonal 4-view VLM prompt (group-level edits)
# ---------------------------------------------------------------------------

def _build_orthogonal_prompt(category: str, labels: list[str]) -> str:
    """Build prompt for 4-view orthogonal enrichment with group-level edits."""
    parts_str = ", ".join(f"p{i}={lbl}" for i, lbl in enumerate(labels))
    n = len(labels) - 1
    return f"""You see 4 views of a 3D {category} from slightly above:
- View 0: front
- View 1: right side
- View 2: back
- View 3: left side

Parts: {parts_str}

Group semantically related parts and generate group-level editing instructions.

Return JSON only, no fences:
{{"object_desc":"A warrior figure with a monkey on its back","part_groups":[
  {{"group_name":"warrior body","part_ids":[0,2,4,5],"is_core":true,"desc":"warrior torso, head, arms and legs"}},
  {{"group_name":"monkey","part_ids":[6,7,8],"is_core":false,
    "desc":"small monkey riding on back","desc_without":"warrior without monkey",
    "best_view_idx":2,
    "deletion":{{"prompt":"Remove the monkey from the warrior's back","after_desc":"warrior standing alone"}},
    "swaps":[{{"prompt":"Replace the monkey with a parrot","before_desc":"small gray monkey","after_desc":"colorful parrot"}}]}},
  {{"group_name":"weapon","part_ids":[3],"is_core":false,
    "desc":"curved metallic knife","desc_without":"warrior with empty hand",
    "best_view_idx":0,
    "deletion":{{"prompt":"Remove the knife from the hand","after_desc":"warrior with empty hand"}},
    "swaps":[{{"prompt":"Replace the knife with a battle axe","before_desc":"curved knife","after_desc":"heavy battle axe"}}],
    "materials":[{{"prompt":"Change the knife to rusty iron","after_desc":"rusty iron knife"}}]}},
  {{"group_name":"accessories","part_ids":[1,9,10,11],"is_core":false,"desc":"clothing and straps"}}
],
"global_edits":[{{"prompt":"Make the entire object wooden","after_desc":"wooden carved version","best_view_idx":0}},{{"prompt":"Transform into sci-fi style","after_desc":"metallic sci-fi version","best_view_idx":1}}]}}

RULES:
- part_groups: Group parts by semantic relatedness (e.g. monkey_body + monkey_head + monkey_tail → "monkey"). A single part can be its own group.
  - Every part_id (0 to {n}) must appear in exactly one group.
  - is_core=true for structural / base parts (body, torso, base, frame, chassis, hull) — NO edits.
  - Groups without clear or distinctive semantics (generic straps, unnamed bits): omit deletion and swaps fields.
  - Non-core groups with clear semantics: provide deletion + 1-2 shape swaps.
- best_view_idx: 0=front, 1=right, 2=back, 3=left. Which view shows this group best for editing.
- desc: what these parts look like together (under 10 words).
- desc_without: what the WHOLE OBJECT looks like without this group (under 15 words).
- deletion.prompt: "Remove the X from the Y" (under 15 words, action verb).
- deletion.after_desc: what the object looks like after removal (under 15 words).
- swaps: 1-2 SHAPE replacements (NOT color/material changes).
  prompt: "Replace the X with a Y" (under 15 words).
  before_desc / after_desc: part appearance before/after (under 8 words each).
  BAD: "Make the blade golden"
  GOOD: "Replace the blade with an axe head"
- materials: 0-1 MATERIAL/TEXTURE changes for this group (shape stays the same).
  prompt: "Change the X to Y material" (under 15 words).
  after_desc: what the part looks like after material change (under 8 words).
  GOOD: "Change the knife to rusty iron", "Make the legs wooden"
- global_edits: 2-3 whole-object style/material/theme changes (no add/remove parts). Include best_view_idx (0-3) for the view that best represents the object for this edit.
- object_desc: 1 sentence, under 15 words."""


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------

def _vlm_call_with_images(client, model: str, prompt: str,
                           image_bytes_list: list[bytes],
                           max_retries: int = 2) -> dict | None:
    """Call VLM with one or more images + text prompt, return parsed JSON."""
    content = []
    for img_bytes in image_bytes_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": _png_to_data_url(img_bytes)},
        })
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
                timeout=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = resp.choices[0].message.content
            if not text:
                logger.warning(f"VLM returned empty (attempt {attempt+1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
            result = _extract_json(text)
            if result:
                return result
            logger.warning(f"VLM invalid JSON (attempt {attempt+1}): "
                           f"{text[:200]}")
        except Exception as e:
            logger.warning(f"VLM call failed (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


def _enrich_one_object_visual(client, model: str, obj, category: str,
                               labels: list[str],
                               debug_dir: Path | None = None) -> dict | None:
    """Single-call enrichment with 4 orthogonal views.

    Sends 4 plain views (front/right/back/left at ~25° elevation) plus
    part text labels.  VLM returns part groups with group-level edits
    and best_view_idx — no mask rendering needed.
    """
    ortho_views = _select_orthogonal_views(obj)
    view_images = _render_plain_views(obj, ortho_views)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        view_labels = ["front", "right", "back", "left"]
        for i, img_bytes in enumerate(view_images):
            (debug_dir / f"ortho_{view_labels[i]}_v{ortho_views[i]}.png"
             ).write_bytes(img_bytes)

    prompt = _build_orthogonal_prompt(category, labels)
    result = _vlm_call_with_images(client, model, prompt, view_images)

    if not result or "part_groups" not in result:
        logger.warning(f"Orthogonal enrichment failed for {obj.obj_id}")
        return None

    # Attach metadata for downstream record builder
    result["orthogonal_views"] = ortho_views
    result["_labels"] = labels
    return result


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

def _build_prompt(category: str, labels: list[str]) -> str:
    """Build text portion of the VLM prompt (action style: swap + global)."""
    parts_str = ", ".join(f"part_{i}={label}" for i, label in enumerate(labels))
    return _build_prompt_action(category, parts_str)


def _build_prompt_action(category: str, parts_str: str) -> str:
    """Action-oriented prompts: per-part shape swaps + whole-object global edits.

    Each non-core part gets 1-2 shape swap modifications.
    Core parts get 0 modifications (structural, cannot swap).
    The whole object gets 2-3 global style/theme edits.
    """
    return f"""Look at this 3D object image. Given its category and part list, generate JSON with descriptions, per-part edits (deletion, addition, swap), and global style edits.

Category: {category}
Parts: {parts_str}

Return JSON only, no fences:
{{"object_desc":"1-sentence description","parts":[{{"part_id":0,"label":"blade","is_core":false,"desc":"a long sharp steel blade","desc_without":"a sword hilt without a blade, showing only the handle and crossguard","deletion":{{"prompt":"Remove the steel blade from the sword","after_desc":"a sword hilt without a blade"}},"addition":{{"prompt":"Attach a long steel blade to the hilt","after_desc":"a complete sword with a sharp steel blade"}},"swaps":[{{"prompt":"Replace the blade with an axe head","after_desc":"a weapon with an axe head on the hilt","before_part_desc":"a long sharp steel blade","after_part_desc":"a heavy axe head"}},{{"prompt":"Replace the blade with a wooden club","after_desc":"a weapon with a thick wooden club","before_part_desc":"a long sharp steel blade","after_part_desc":"a thick wooden club"}}]}},{{"part_id":1,"label":"hilt","is_core":true,"desc":"leather-wrapped sword hilt with crossguard","desc_without":"","deletion":null,"addition":null,"swaps":[]}}],"global_edits":[{{"prompt":"Make the entire object wooden","after_desc":"a wooden version with natural wood grain"}},{{"prompt":"Transform into a futuristic sci-fi style","after_desc":"a sleek metallic sci-fi version with glowing accents"}},{{"prompt":"Make the object look ancient and weathered","after_desc":"an aged version with rust, cracks and patina"}}]}}

CRITICAL RULES FOR ALL PROMPTS:
- Each prompt MUST be specific and visual — describe WHAT is being done
  BAD: "Remove the part"  (too generic)
  GOOD: "Remove the steel blade from the sword"
  BAD: "Add a blade"  (no context)
  GOOD: "Attach a long steel blade to the hilt"
- Each prompt must be UNDER 15 WORDS
- Use action verbs: "Remove", "Detach", "Add", "Attach", "Replace X with"

PER-PART FIELDS:
For NON-CORE parts (is_core=false):
  - "deletion": {{"prompt": "...", "after_desc": "..."}}
    Describe removing this part. after_desc = what object looks like WITHOUT this part.
    The after_desc MUST describe the remaining structure (not just "object without X").
  - "addition": {{"prompt": "...", "after_desc": "..."}}
    Describe adding this part. Assume the reader sees the object WITHOUT this part.
    after_desc = what object looks like WITH this part added back (≈ the original).
  - "swaps": 1-2 shape replacements (see below)
For CORE parts (is_core=true):
  - "deletion": null, "addition": null, "swaps": []
  - desc_without = ""

SHAPE SWAPS (swaps array):
- Replace part with a DIFFERENT but logically compatible shape/type
- The replacement must fit in the SAME spatial region
- Examples:
    blade → "Replace the blade with an axe head"
    wheel → "Replace the wheels with tank treads"
    leg → "Replace the legs with curved cabriole legs"
  BAD: "Make the blade golden" (color change, not shape swap)
  BAD: "Replace with nothing" (that's deletion)

GLOBAL EDITS (whole-object style/theme changes):
- 2-3 diverse edits that change the ENTIRE object's style, material theme, or era
- Do NOT add or remove parts — only change the overall look
- Examples: "Make the entire object wooden", "Transform into steampunk style"
- Each needs: prompt (under 12 words), after_desc (1 sentence)

OUTPUT RULES:
- Keep ALL string values VERY SHORT (under 10 words each)
- desc_without MUST be non-empty for non-core parts (describe the remaining object)
- Output compact JSON, no extra whitespace"""


def _build_messages(category: str, labels: list[str],
                    thumbnail_bytes: bytes | None) -> list[dict]:
    """Build VLM messages with optional image."""
    text = _build_prompt(category, labels)

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
              max_retries: int = 2) -> dict | None:
    """Call VLM with image + text, with retries."""
    messages = _build_messages(category, labels, thumbnail_bytes)

    for attempt in range(max_retries + 1):
        try:
            # chat_template_kwargs disables Qwen3.5 thinking mode
            # when served via SGLang — ignored by other backends.
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
                timeout=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = resp.choices[0].message
            text = msg.content
            # Debug: log response structure if content is empty/None
            if not text:
                # Try extracting from thinking response structure
                raw = getattr(msg, 'model_extra', {}) or {}
                logger.debug(f"Empty content. Keys: {list(raw.keys())}"
                             f" | role={msg.role}"
                             f" | type(content)={type(msg.content)}")
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
        part = {
            "part_id": i,
            "label": label.lower().replace(" ", "_"),
            "is_core": is_core,
            "desc": lbl,
            "desc_without": "" if is_core else f"The {cat_clean} without its {lbl}",
        }
        if not is_core:
            part["deletion"] = {
                "prompt": f"Remove the {lbl} from the {cat_clean}",
                "after_desc": f"The {cat_clean} without its {lbl}",
            }
            part["addition"] = {
                "prompt": f"Add a {lbl} to the {cat_clean}",
                "after_desc": f"A {cat_clean} with a {lbl}",
            }
            part["swaps"] = [{
                "prompt": f"Replace the {lbl} with a different {lbl}",
                "after_desc": f"The {cat_clean} with a different {lbl}",
                "before_part_desc": lbl,
                "after_part_desc": f"a different style {lbl}",
            }]
        else:
            part["deletion"] = None
            part["addition"] = None
            part["swaps"] = []
        parts.append(part)
    return {
        "object_desc": f"A 3D {cat_clean} object",
        "parts": parts,
        "global_edits": [
            {"prompt": f"Make the {cat_clean} wooden",
             "after_desc": f"A wooden {cat_clean} with natural wood grain"},
            {"prompt": f"Make the {cat_clean} look metallic",
             "after_desc": f"A shiny metallic {cat_clean}"},
        ],
    }


# ---------------------------------------------------------------------------
# Per-object enrichment
# ---------------------------------------------------------------------------

def _result_groups_to_record(result: dict, uid: str, category: str,
                              shard: str = "00",
                              actual_part_ids: list[int] | None = None,
                              ) -> dict:
    """Convert group-based VLM result (from orthogonal enrichment) to record.

    Output format has per-part metadata in ``parts`` and editing instructions
    in ``group_edits`` (each group is an edit unit).

    Args:
        actual_part_ids: Real NPZ part_ids in the same order as the labels
            list.  VLM ``part_ids`` are label-list indices (0..N-1); this
            mapping converts them back to NPZ part_ids so downstream mask
            building uses the correct IDs.
    """
    obj_desc = result.get("object_desc", f"A 3D {category} object")
    labels = result.get("_labels", [])

    # Build label-index → actual NPZ part_id mapping
    # If actual_part_ids not provided, assume identity (0, 1, 2, ...)
    if actual_part_ids is not None and len(actual_part_ids) == len(labels):
        idx_to_pid = actual_part_ids
    else:
        idx_to_pid = list(range(len(labels)))

    def _map_pids(vlm_indices: list[int]) -> list[int]:
        """Map VLM label indices to actual NPZ part_ids."""
        return [idx_to_pid[i] for i in vlm_indices if i < len(idx_to_pid)]

    # Map label-index → group info (using label indices for lookup)
    part_group_map: dict[int, str] = {}   # keyed by actual part_id
    core_pids: set[int] = set()           # actual part_ids
    part_descs: dict[int, str] = {}       # keyed by actual part_id

    for group in result.get("part_groups", []):
        gname = group.get("group_name", "")
        is_core = group.get("is_core", False)
        gdesc = group.get("desc", "")
        for label_idx in group.get("part_ids", []):
            real_pid = idx_to_pid[label_idx] if label_idx < len(idx_to_pid) else label_idx
            part_group_map[real_pid] = gname
            part_descs[real_pid] = gdesc
            if is_core:
                core_pids.add(real_pid)

    # Build parts list using actual NPZ part_ids
    phase0_parts = []
    for label_idx, lbl in enumerate(labels):
        real_pid = idx_to_pid[label_idx] if label_idx < len(idx_to_pid) else label_idx
        is_core = real_pid in core_pids or _is_core_part(lbl)
        phase0_parts.append({
            "part_id": real_pid,
            "label": lbl.lower().replace(" ", "_"),
            "core": is_core,
            "desc": part_descs.get(real_pid, lbl.replace("_", " ")),
            "group": part_group_map.get(real_pid, ""),
        })

    # Resolve orthogonal view indices (VLM returns 0-3 → map to NPZ view ids)
    ortho_views = result.get("orthogonal_views", [])

    # Build group edits (map VLM label indices → actual NPZ part_ids)
    group_edits_out: list[dict] = []
    for group in result.get("part_groups", []):
        if group.get("is_core", False):
            continue

        gname = group.get("group_name", "")
        vlm_part_ids = group.get("part_ids", [])
        real_part_ids = _map_pids(vlm_part_ids)
        gdesc = group.get("desc", "")
        desc_without = group.get("desc_without", "")
        best_view_idx = group.get("best_view_idx", 0)
        # Map 0-3 (front/right/back/left) → actual NPZ view index
        best_view = (ortho_views[best_view_idx]
                     if best_view_idx < len(ortho_views) else 0)

        deletion = group.get("deletion")
        swaps = group.get("swaps", [])
        materials = group.get("materials", [])

        if not deletion and not swaps and not materials:
            continue  # No edits for groups without clear semantics

        edits: list[dict] = []
        if deletion:
            edits.append({
                "type": "deletion",
                "prompt": deletion.get("prompt", f"Remove the {gdesc}"),
                "after_desc": deletion.get("after_desc", desc_without),
                "before_part_desc": gdesc,
                "after_part_desc": "",
            })
            # Auto-generate addition (reverse of deletion)
            edits.append({
                "type": "addition",
                "prompt": f"Add {gdesc} to the {category}",
                "after_desc": obj_desc,
                "before_part_desc": "",
                "after_part_desc": gdesc,
            })

        for swap in swaps:
            edits.append({
                "type": "modification",
                "mod_type": "swap",
                "prompt": swap.get("prompt", ""),
                "after_desc": swap.get("after_desc", ""),
                "before_part_desc": swap.get("before_desc", gdesc),
                "after_part_desc": swap.get("after_desc", ""),
            })

        for mat in materials:
            edits.append({
                "type": "material",
                "mod_type": "material",
                "prompt": mat.get("prompt", ""),
                "after_desc": mat.get("after_desc", ""),
                "before_part_desc": gdesc,
                "after_part_desc": mat.get("after_desc", ""),
            })

        group_edits_out.append({
            "group_name": gname,
            "part_ids": real_part_ids,
            "desc": gdesc,
            "desc_without": desc_without,
            "best_view_idx": best_view_idx,
            "best_view": best_view,
            "edits": edits,
        })

    # Global edits
    global_edits: list[dict] = []
    for ge in result.get("global_edits", []):
        prompt = ge.get("prompt", "")
        if prompt:
            global_edits.append({
                "type": "global",
                "prompt": prompt,
                "after_desc": ge.get("after_desc", ""),
            })

    record: dict = {
        "obj_id": uid,
        "shard": shard,
        "num_parts": len(phase0_parts),
        "object_desc": obj_desc,
        "orthogonal_views": result.get("orthogonal_views", []),
        "parts": phase0_parts,
        "group_edits": group_edits_out,
    }
    if global_edits:
        record["global_edits"] = global_edits
    return record


def _result_to_phase0_record(result: dict, uid: str, category: str,
                              shard: str = "00",
                              actual_part_ids: list[int] | None = None,
                              ) -> dict:
    """Convert VLM result dict to output record.

    Dispatches to group-based format if ``part_groups`` present,
    otherwise uses legacy per-part format.
    Only keeps swap modifications (no style/color changes).
    Passes through global edits for whole-object style changes.
    """
    # --- Dispatch: group-based format (orthogonal enrichment) ---
    if "part_groups" in result:
        return _result_groups_to_record(result, uid, category, shard,
                                        actual_part_ids=actual_part_ids)

    # --- Legacy per-part format ---
    obj_desc = result.get("object_desc", f"A 3D {category} object")

    phase0_parts = []
    for p in result.get("parts", []):
        pid = p.get("part_id", 0)
        label = p.get("label", f"part_{pid}").lower().replace(" ", "_")
        is_core = p.get("is_core", _is_core_part(label))
        label_clean = label.replace("_", " ")
        desc = p.get("desc", label)

        edits = []
        if not is_core:
            # --- Deletion: use VLM-generated prompt, fallback to template ---
            del_data = p.get("deletion") or {}
            del_prompt = del_data.get("prompt") or f"Remove the {label_clean}"
            del_after = del_data.get("after_desc") or p.get("desc_without", "")
            edits.append({
                "type": "deletion",
                "prompt": del_prompt,
                "after_desc": del_after,
                "before_part_desc": desc,
                "after_part_desc": "",
            })

            # --- Addition: use VLM-generated prompt, fallback to template ---
            add_data = p.get("addition") or {}
            add_prompt = add_data.get("prompt") or f"Add a {label_clean}"
            add_after = add_data.get("after_desc") or obj_desc
            edits.append({
                "type": "addition",
                "prompt": add_prompt,
                "after_desc": add_after,
                "before_part_desc": "",
                "after_part_desc": desc,
            })

        # --- Swap modifications (shape replacement only) ---
        for swap in p.get("swaps", []):
            prompt = swap.get("prompt", "")
            if not prompt:
                continue
            edits.append({
                "type": "modification",
                "mod_type": "swap",
                "prompt": prompt,
                "after_desc": swap.get("after_desc", ""),
                "before_part_desc": swap.get("before_part_desc", desc),
                "after_part_desc": swap.get("after_part_desc", ""),
            })

        # Backward compat: also check old "modifications" key
        for mod in p.get("modifications", []):
            if mod.get("mod_type") != "swap":
                continue
            prompt = mod.get("prompt", "")
            if not prompt:
                continue
            edits.append({
                "type": "modification",
                "mod_type": "swap",
                "prompt": prompt,
                "after_desc": mod.get("after_desc", ""),
                "before_part_desc": mod.get("before_part_desc", desc),
                "after_part_desc": mod.get("after_part_desc", ""),
            })

        phase0_parts.append({
            "part_id": pid,
            "label": label,
            "core": is_core,
            "desc": desc,
            "desc_without": p.get("desc_without", "") if not is_core else "",
            "edits": edits,
        })

    # Global edits (whole-object style/theme changes)
    global_edits = []
    for ge in result.get("global_edits", []):
        prompt = ge.get("prompt", "")
        if not prompt:
            continue
        global_edits.append({
            "type": "global",
            "prompt": prompt,
            "after_desc": ge.get("after_desc", ""),
        })

    record = {
        "obj_id": uid,
        "shard": shard,
        "num_parts": len(phase0_parts),
        "object_desc": obj_desc,
        "parts": phase0_parts,
    }
    if global_edits:
        record["global_edits"] = global_edits
    return record


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
    visual_grounding: bool = True,
    dataset=None,
    debug: bool = False,
    object_ids: list[str] | None = None,
) -> Path:
    """Enrich all objects via VLM.

    Two modes:
      - visual_grounding=True + dataset provided: two-phase VLM with
        color-tinted overview + per-group highlights (better quality)
      - Otherwise: single-call with 1 thumbnail + text labels (faster)

    Args:
        cfg: pipeline config (uses phase0.vlm_model, vlm_base_url, vlm_api_key)
        semantic_json_path: path to PartObjaverse-Tiny_semantic.json
        output_path: where to write semantic_labels.jsonl
        image_npz_dir: directory with pre-rendered {shard}/{uid}.npz (preferred)
        shard: data shard for NPZ lookup
        glb_dir: fallback GLB directory for on-the-fly rendering
        limit: max objects (0=all)
        max_workers: concurrent VLM calls
        visual_grounding: use two-phase visual grounding when dataset available
        dataset: HY3DPartDataset instance (required for visual grounding)

    Returns:
        Path to the output JSONL file.
    """
    from openai import OpenAI

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup VLM client
    p0 = cfg.get("phase0", {})
    model = p0.get("vlm_model", "gemini-3.1-flash-lite-preview")
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
            default_cfg_path = Path(__file__).parents[2] / "configs" / "default.yaml"
            if default_cfg_path.exists():
                with open(default_cfg_path) as _f:
                    _dcfg = yaml.safe_load(_f)
                api_key = _dcfg.get("phase0", {}).get("vlm_api_key", "")
        except Exception:
            pass

    if not api_key:
        raise ValueError("No API key. Set phase0.vlm_api_key in config or default.yaml")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Load object list: from semantic.json or from dataset
    # uid_info maps uid → (category, labels, actual_part_ids)
    # actual_part_ids: real NPZ part_ids in same order as labels
    uid_info: dict[str, tuple[str, list[str], list[int]]] = {}
    if semantic_json_path and Path(semantic_json_path).exists():
        with open(semantic_json_path) as f:
            semantic_json = json.load(f)
        for category, objects in semantic_json.items():
            for uid, labels in objects.items():
                # Legacy: no actual_part_ids, assume identity
                uid_info[uid] = (category, labels, list(range(len(labels))))
    elif dataset is not None:
        # Derive UIDs and labels from dataset NPZ files
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
            uid_info[obj_id] = ("object", labels, actual_pids)
            obj_rec.close()
    else:
        raise ValueError("Need either semantic_json_path or dataset")

    all_uids = sorted(uid_info.keys())
    if object_ids:
        allow = set(object_ids)
        all_uids = [u for u in all_uids if u in allow]
    if limit > 0:
        all_uids = all_uids[:limit]

    # Resolve image sources
    npz_dir = Path(image_npz_dir) if image_npz_dir else None
    glb_sources = [Path(glb_dir)] if glb_dir else []
    glb_mapping: dict[str, str] = {}

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

    # Determine if visual grounding is usable
    use_visual = visual_grounding and dataset is not None
    if visual_grounding and dataset is None:
        logger.info("visual_grounding=True but no dataset provided, "
                     "falling back to single-call mode")

    # Debug output directory for visual grounding (only when --debug)
    cache_dir = output_path.parent
    debug_base = (cache_dir / "debug_enricher") if (debug and use_visual) else None

    n_with_npz = sum(1 for uid in pending
                     if npz_dir and (npz_dir / shard / f"{uid}.npz").exists())
    mode_str = "orthogonal 4-view (group edits)" if use_visual else "single-call"
    logger.info(f"Enriching {len(pending)} objects via VLM ({model}), "
                f"mode={mode_str}, "
                f"{n_with_npz} with pre-rendered images")

    from tqdm import tqdm

    success, fail = 0, 0

    if use_visual:
        # --- Visual-grounded mode: sequential (each object needs multiple VLM calls) ---
        pbar = tqdm(pending, desc="Enriching (visual)", unit="obj")
        with open(output_path, "a") as out_fp:
            for uid in pbar:
                category, labels, actual_pids = uid_info[uid]
                try:
                    obj = dataset.load_object(shard, uid)
                except Exception as e:
                    logger.warning(f"Cannot load {uid} from dataset: {e}, "
                                   f"falling back to single-call")
                    obj = None

                result = None
                mode_tag = "vis"

                if obj is not None:
                    debug_dir = debug_base / uid if debug_base else None
                    try:
                        result = _enrich_one_object_visual(
                            client, model, obj, category, labels,
                            debug_dir=debug_dir)
                    except Exception as e:
                        logger.warning(f"Visual enrichment failed for {uid}: {e}")
                    finally:
                        obj.close()

                if result is None:
                    # Fallback to single-call legacy mode
                    mode_tag = "leg"
                    thumbnail = _get_thumbnail(uid)
                    result = _call_vlm(client, model, category, labels, thumbnail)

                if result is None:
                    result = _fallback_enrichment(category, labels)
                    fail += 1
                    mode_tag = "fb"
                else:
                    success += 1

                record = _result_to_phase0_record(
                    result, uid, category, shard,
                    actual_part_ids=actual_pids)
                out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fp.flush()

                # Count edits (handle both group and per-part formats)
                if "group_edits" in record:
                    n_grp = len(record["group_edits"])
                    n_edits = sum(len(g["edits"])
                                  for g in record["group_edits"])
                else:
                    n_grp = 0
                    n_edits = sum(len(p.get("edits", []))
                                  for p in record["parts"])
                n_glb = len(record.get("global_edits", []))
                pbar.set_postfix(
                    ok=success, fail=fail,
                    last=f"{uid[:8]}[{mode_tag}] "
                         f"{n_grp}grp/{n_edits}ed/{n_glb}g")
        pbar.close()
    else:
        # --- Legacy single-call mode: concurrent ---
        with open(output_path, "a") as out_fp:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for uid in pending:
                    category, labels, actual_pids = uid_info[uid]
                    thumbnail = _get_thumbnail(uid)
                    fut = pool.submit(
                        _call_vlm, client, model, category, labels, thumbnail)
                    futures[fut] = (uid, category, labels, actual_pids,
                                    thumbnail is not None)

                pbar = tqdm(as_completed(futures), total=len(futures),
                            desc="Enriching", unit="obj")
                for fut in pbar:
                    uid, category, labels, actual_pids, had_image = futures[fut]
                    try:
                        result = fut.result()
                        if result is None:
                            result = _fallback_enrichment(category, labels)
                            fail += 1
                        else:
                            success += 1

                        record = _result_to_phase0_record(
                            result, uid, category, shard,
                            actual_part_ids=actual_pids)
                        out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_fp.flush()

                        n_edits = sum(len(p.get("edits", []))
                                      for p in record["parts"])
                        n_glb = len(record.get("global_edits", []))
                        img_tag = "img" if had_image else "txt"
                        pbar.set_postfix(
                            ok=success, fail=fail,
                            last=f"{uid[:8]}[{img_tag}] {n_edits}ed/{n_glb}g")
                    except Exception as e:
                        fail += 1
                        pbar.set_postfix(ok=success, fail=fail,
                                         err=str(e)[:40])
                pbar.close()

    logger.info(f"Enrichment complete: {success} succeeded, {fail} failed -> {output_path}")
    return output_path
