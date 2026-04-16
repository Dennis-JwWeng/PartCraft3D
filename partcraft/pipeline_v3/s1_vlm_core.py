"""VLM helpers for pipeline_v3 — Mode E (Mode B text generation + alignment gate).

Active code paths
-----------------
Mode B (text_semantic)  — text-only edit generation from part captions
  SYSTEM_PROMPT_B, USER_PROMPT_TEXT_SEMANTIC, build_semantic_list,
  call_vlm_text_async

Mode E alignment gate   — per-edit VLM image+text judge (gate_text_align step)
  SYSTEM_PROMPT_ALIGN_GATE, build_align_gate_user_prompt, parse_align_gate_output,
  call_vlm_async

Shared utilities
  VIEW_INDICES, extract_json_object, quota_for, validate_simple,
  render_overview_png, _sample_global_note, _GLOBAL_STYLE_POOL

Commented-out sections (preserved for reference)
  Legacy v2-era SYSTEM_PROMPT / USER_PROMPT_TEMPLATE  (live in pipeline_v2.s1_vlm_core)
  Mode A "image_semantic" — build_image_semantic_menu, SYSTEM_PROMPT_A
  Mode C "image_only"    — build_image_only_menu, SYSTEM_PROMPT_C
  Mode D "two-stage"     — SYSTEM_PROMPT_S1/S2, build_s1/s2_user_prompt, parse_s1_output
  validate()             — legacy validator (imported from pipeline_v2.s1_vlm_core)
"""
from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from partcraft.render.overview import (
    VIEW_INDICES, _PALETTE,
    extract_parts, load_views_from_npz, run_blender, stitch_two_rows,
)

# SYSTEM_PROMPT = """You are a 3D Spatial Reasoning Engine generating a JSON dataset for 3D part editing. \
# You are given a 5x2 grid: TOP row = 5 RGB photos of one 3D object from 5 cameras; \
# BOTTOM row = the same 5 cameras re-rendered with each editable part in a fixed \
# palette color.
#
# CRITICAL: maintain 3D OBJECT-SPACE consistency. An object's anatomical "left" is \
# a fixed physical part — it does NOT change just because the camera moved it to \
# the image-right. Whenever you write a directional word, you MUST first reason \
# about which camera you are looking through and apply the mirror rule explicitly.
#
# Output ONE valid JSON object — no prose, no markdown. Begin with '{' and end with '}'."""
#
#
# USER_PROMPT_TEMPLATE = """[Image: 5×2 grid. TOP row = 5 RGB photos. BOTTOM row =
# same 5 cameras re-rendered with each editable part in a fixed palette color
# (same column = same camera). The palette colors are INTERNAL labels for you
# only — they are NOT properties of the real object.]
#
# # CAMERA GEOMETRY (fixed for every object — memorize this)
#
# Views 0,1,2,3 form a horizontal ring around the object at 90° yaw increments,
# all looking slightly downward (elev ≈ +30°):
#   • view (k+2) mod 4  is the 180° OPPOSITE of view k  (back-to-back cameras)
#   • view (k+1) mod 4  is the 90° rotation of view k   (perpendicular / profile)
# View 4 is a low camera looking UP from below the object.
#
# This geometry is intrinsic to the cameras, NOT to the object. The object itself
# has an arbitrary world orientation — you must decide which camera happens to
# face the object's front by LOOKING at the photos.
#
# Parts (id, palette color in bottom row, cluster_size):
# {part_menu}
#
# # CORE LOGIC: THE 3D SPATIAL RULEBOOK
#
# P1. ID-FIRST TRUTH. selected_part_ids are the absolute ground truth on which
#     part is being edited. Your text MUST visually match the highlighted parts.
#     If unsure what a part is, describe its shape/structure, do not guess a name.
#
# P2. ANTI-VIEWPORT RULE. Bare directional words ("left", "right", "front",
#     "back", "leftmost", "rightmost") are FORBIDDEN as standalone descriptors.
#     Use either:
#       (a) view-invariant cues — shape, size, function, or structural relation
#           to another part ("the ear above the raised paw", "the wheel under
#           the driver seat"), OR
#       (b) the object-anatomical form defined in P3+P4 below.
#
# P3. CANONICAL FRONT (mandatory field).
#     object.canonical_front: ONE structural sentence describing what visually
#         marks the object's intrinsic forward direction (e.g. "the side with
#         the snout and eyes", "the side with the headlights and windshield",
#         "the side where you sit on the chair"), OR null if the object has no
#         unambiguous orientation (sphere, vase, symmetric drum, etc.).
#     object.frontal_view_index: int in [0..4], the view_index whose camera
#         most directly faces canonical_front (the camera you would describe
#         as "looking the object in the face"). Set to null iff
#         canonical_front is null.
#
#     When canonical_front is null, NO directional words at all (P4 disabled).
#
# P4. MIRROR RULE (only valid when canonical_front and frontal_view_index are set).
#     Let F = frontal_view_index. Then because the camera and the object are
#     facing each other:
#       • In view F   : image-LEFT half = object's anatomical RIGHT side,
#                        image-RIGHT half = object's anatomical LEFT side. (MIRROR)
#       • In view (F+2) mod 4 (back view): image-side = object-side. (NO MIRROR)
#       • In view (F±1) mod 4 (profile views): the object is sideways — you
#         CANNOT read anatomical left/right from these views. Do NOT pick a
#         profile view as view_index for any edit that uses left/right.
#       • View 4 (bottom-up): also unreliable for left/right; use other cues.
#
#     So any anatomical-left/right edit MUST have view_index ∈ {{F, (F+2) mod 4}}.
#
#     EVERY use of an anatomical "left" / "right" in prompt or target_part_desc
#     MUST be tagged with "(object's anatomical left/right)". The rationale
#     field MUST cite the mirror reasoning explicitly, e.g.:
#        "frontal_view_index=1; target visible in view 1 on the image-RIGHT
#         half → mirror → object's anatomical LEFT ear."
#
# P5. LEVEL HIERARCHY. The part menu shows ``level=N`` for each part (lower N
#     = closer to the object root). Parts at the same level are siblings in the
#     part hierarchy. Use level to reason about part relationships — e.g. parts
#     at the same level may be symmetric instances (wheels, legs, wings).
#
#
# P6. NO PALETTE COLORS. The palette names (red, orange, yellow, lime, green,
#     teal, cyan, blue, navy, purple, magenta, pink, brown, tan, black, gray)
#     are INTERNAL labels and MUST NOT appear in any output text field. To
#     describe real color, use the appearance from the TOP row photos
#     ("the dark wooden seat", "the chrome pipe").
#
# # OUTPUT — one JSON object
#
# object:
#   full_desc            full English description of the object
#   full_desc_stage1     geometry-only version (no colors/materials/finish words)
#   full_desc_stage2     texture-only version (no shape/count/layout words)
#   canonical_front      ONE structural sentence OR null  (see P3)
#   frontal_view_index   int in [0..4] OR null            (see P3)
#   parts                [{{part_id, color, name}}, ...] for every menu entry;
#                        name = your short semantic label, "(artifact)" or
#                        "(invisible)" for noise / unseen parts.
#
# edits: EXACTLY {n_total} entries with these per-type counts
#   - {n_deletion} deletion
#   - {n_modification} modification
#   - {n_scale} scale
#   - {n_material} material
#   - {n_color} color
#   - {n_global} global       (selected_part_ids = [])
#
# Each edit MUST list these fields IN THIS ORDER (the rationale comes FIRST so
# you reason before you write the prompt):
#   rationale           ONE sentence.
#                       • If the edit uses anatomical left/right: MUST include
#                         the literal value "frontal_view_index=N" and the
#                         mirror calculation, e.g.:
#                         "frontal_view_index=1; in view 1 the target ear is on
#                          the image-RIGHT half → mirror → object's anatomical
#                          LEFT ear (part_id 5)."
#                       • For global edits: MUST name the source style category
#                         (Rendering / Historical / Genre), e.g.:
#                         "Choosing 'ukiyo-e woodblock print' from the
#                          Historical category."
#                       • For all other edits: a single short reason.
#   edit_type           one of: deletion | modification | scale | material | color | global
#                       DECISION: deletion=remove part · modification=shape/identity change ·
#                       scale=size only · material=substance only · color=hue only · global=art style
#   selected_part_ids   list of int part_ids; empty ONLY for global
#   prompt              imperative starting with Remove/Delete/Add/Change/
#                       Replace/Make/Scale/Resize. NO part_id numbers,
#                       NO palette color names. Obeys P2/P4/P6. Any anatomical
#                       left/right MUST be tagged "(object's anatomical L/R)".
#   target_part_desc    short visual description of the target part(s) — same
#                       forbidden-word rules as prompt.
#   view_index          int in [0..4]: the view where the target is most
#                       visible. If the edit uses anatomical left/right this
#                       MUST equal frontal_view_index OR (frontal_view_index+2)
#                       mod 4. (For global, pick the best overall view.)
#
# # MODIFICATION EDITS — SHAPE MORPH OR FUNCTIONAL REPLACEMENT
#   A modification either (a) changes a part's geometry while keeping its identity, OR
#   (b) substitutes it with a completely different but logically equivalent object in
#   the same structural slot. Both are valid and encouraged.
#
#   (a) Shape morph — same functional identity, different geometry:
#       • straight sword blade    →  curved saber blade
#       • cylindrical barrel      →  hexagonal prism barrel
#       • spherical head          →  cubic head
#       • upright rabbit ears     →  floppy drooping ears
#       • rectangular door panel  →  arched gothic door panel
#
#   (b) Functional replacement — completely different object, same structural role:
#       • sword blade             →  axe head
#       • circular wheel          →  triangular wheel
#       • vertical antenna        →  parabolic satellite dish
#       • cylindrical chair leg   →  hairpin metal rod leg
#       • vertical stabilizer     →  swept-back winglet
#       • rectangular table top   →  circular table top
#
#   new_part_desc MUST name the new object AND describe its key geometry, e.g.:
#     "a broad wedge-shaped axe head" · "a flat triangular wheel" · "a parabolic dish"
#     "hairpin-bent thin metal rod" · "a swept-back delta-shaped winglet"
#
#   TYPE BOUNDARY — pick modification ONLY if the geometry or identity changes:
#     • Changing ONLY colour?              → use "color"      (not modification)
#     • Changing ONLY surface material?   → use "material"   (not modification)
#     • Removing the part entirely?       → use "deletion"   (not modification)
#     • Resizing without shape change?    → use "scale"      (not modification)
#
#   STRICTLY FORBIDDEN in modification: changing only color, surface finish, or
#   material. The new_part_desc MUST describe a geometry or identity change.
#
#   edit_params         deletion: {{}}
#                       modification: {{"new_part_desc": "..."}}
#                       scale:        {{"factor": float in [0.3, 0.85]}}
#                                     Shrink only. Prefer large/dominant parts (main body, primary limbs).
#                                     Do NOT enlarge small decorative parts.
#                       material:     {{"target_material": "..."}}
#                                     Target must be a specific surface substance or finish, e.g.:
#                                     "polished walnut wood", "brushed stainless steel",
#                                     "frosted borosilicate glass", "hand-stitched leather",
#                                     "poured concrete", "translucent amber resin".
#                                     FORBIDDEN in target_material: style/aesthetic words
#                                     (cartoon, vintage, futuristic, minimalist, steampunk,
#                                     cyberpunk) — those belong in "global" edits.
#                       color:        {{"target_color": "..."}}
#                                     Target must be a specific, descriptive colour phrase, e.g.:
#                                     "deep crimson red", "matte charcoal black", "cobalt blue",
#                                     "ivory cream white", "forest green", "warm amber orange".
#                                     FORBIDDEN: bare internal palette names (red, orange, lime, …)
#                                     — always qualify: "vivid lime green", not "lime".
#                                     Do NOT change the surface material or finish; use "material"
#                                     for that.
#                       global:       {{"target_style": "..."}}
#   after_desc_full / after_desc_stage1 / after_desc_stage2
#                       object after the edit. For deletion: ALL three null.
#                       For others: all three filled, stage1 has no
#                       colors/materials, stage2 has no shape changes.
#   new_parts_desc / new_parts_desc_stage1 / new_parts_desc_stage2
#                       modification only: describe the new replacement parts.
#                       null for non-modification edits.
#   confidence          "high" | "medium" | "low"
#
# # COLOR EDITS — HUE AND SHADE CHANGES ONLY
#   A color edit repaints one or more parts with a new hue or shade while keeping the
#   surface material and geometry unchanged.
#   Think: what color contrast or accent would improve the object?
#   Examples:
#     • beige seat → deep burgundy red seat
#     • silver handle → matte charcoal black handle
#     • white lamp shade → warm amber orange shade
#   STRICTLY FORBIDDEN in color: changing surface material or finish (use "material"),
#   changing geometry (use "modification"), or changing the whole object (use "global").
#   Use descriptive colour phrases — never bare internal palette names.
#   The new_part_desc is NOT required for color edits (there is no shape change).
#
# # GLOBAL STYLE EDITS — ARTISTIC / RENDERING AESTHETIC ONLY
#
#   A global edit transforms the ENTIRE object's artistic or rendering aesthetic.
#   It must change how the object looks as a *visual artwork* — NOT what material
#   it is made of.
#
#   STRICTLY FORBIDDEN in global target_style:
#     • Surface-material words: gold, silver, metal, wood, stone, clay, glass,
#       ceramic, rubber, plastic, ice, crystal, fabric, concrete, leather.
#       → Those belong in "material" edits.
#     • Generic quality descriptors: "realistic", "detailed", "high quality".
#     • Near-duplicate styles: "cartoon", "cartoonish", "toon" count as ONE choice.
#
#   VALID target_style — for this object you MUST use ONLY the per-object style roster
#   injected below (it is randomised per object to enforce diversity).
#
# {global_roster}
#
#   DIVERSITY RULE: For this object's {{n_global}} global edits, each target_style
#   MUST come from a DIFFERENT category row of the roster above.
#   The rationale for every global edit MUST name the source category, e.g.:
#   "Choosing 'watercolour wash' from the Rendering category."
#
# # HARD RULES (violations drop that edit)
#
# R1. selected_part_ids ⊆ part menu ids; never target parts with cluster_size<30
#     UNLESS the part appears to be the primary body of the object — infer
#     valid semantic parts regardless of cluster_size);
#     never target parts you cannot see in the bottom row.
# R2. Each edit is distinct: no two with same edit_type AND same
#     selected_part_ids.
# R3. Never delete or extreme-scale a part that forms the structural body —
#     the object should remain recognizable.
# R4. prompt and target_part_desc must obey P2, P4, P5, P6.
# R5. Non-deletion edits fill all three after_desc_*. Deletion edits set
#     all three to null.
# R6. view_index ∈ [0,4] and the target must be clearly visible in that view.
# R7. If canonical_front is null, NO directional words anywhere; use group
#     edits or structural anchors only.
# R8. If an edit uses anatomical left/right, view_index ∈ {{F, (F+2) mod 4}}
#     where F = frontal_view_index, and the rationale must cite the mirror
#     reasoning explicitly.
#
# # OUTPUT FORMAT
#
# ONE JSON object. Begin with '{{', end with '}}'. No prose, no markdown."""
#
#

# def build_image_semantic_menu(
#     mesh_npz: Path,
#     img_npz: Path,
#     anno_obj_dir: "Path | None" = None,
# ) -> tuple[list[int], str]:
#     """Semantic part menu — for image_semantic mode (Mode A).
#
#     Columns: part_id | palette-colour | description
#     The palette colour matches the colour-coded BOTTOM-row image overlay.
#
#     Format per line:
#         part_{id:<3d}   {colour}   "{description}"
#     """
#     z = np.load(img_npz, allow_pickle=True)
#     sm = json.loads(bytes(z["split_mesh.json"]).decode())
#     clusters = sm.get("valid_clusters", {})
#     z2 = np.load(mesh_npz, allow_pickle=True)
#     import re as _re
#
#     def _parse_pid(k: str) -> int | None:
#         m = _re.search(r"\d+", k)
#         return int(m.group()) if m else None
#
#     pids = sorted(
#         pid
#         for k in z2.files
#         if k.startswith("part_") and (k.endswith(".glb") or k.endswith(".ply"))
#         if (pid := _parse_pid(k)) is not None
#     )
#
#     # Load per-part captions from embedded part_captions.json
#     part_captions: dict[int, list[str]] = {}
#     if "part_captions.json" in z2.files:
#         try:
#             raw_caps: dict[str, list] = json.loads(bytes(z2["part_captions.json"]).decode())
#             part_captions = {int(k): v for k, v in raw_caps.items()}
#         except Exception:
#             pass
#
#     import re as _re2
#     _ADET = _re2.compile(r'^(?:a |an |the )+', _re2.I)
#
#     def _caption(pid: int) -> str:
#         caps = part_captions.get(pid, [])
#         if caps and isinstance(caps[0], str) and caps[0].strip():
#             s = caps[0].strip().rstrip(".")
#             return _ADET.sub("", s).strip() or f"part_{pid}"
#         return f"part_{pid}"
#
#     lines = []
#     for pid in pids:
#         color = _PALETTE_NAMES[pid % len(_PALETTE_NAMES)]
#         desc = _caption(pid)
#         base = f'  part_{pid:<3d}   {color:<8s}  "{desc}"'
#         lines.append(base)
#     return pids, "\n".join(lines)
#
#
# ────────────────────────────── overview render ─────────────────────────────

def render_overview_png(mesh_npz: Path, img_npz: Path, blender: str) -> bytes:
    """Render the 4×2 overview and return PNG bytes."""
    top_imgs, frames = load_views_from_npz(img_npz, VIEW_INDICES)
    H = top_imgs[0].shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        part_ids = extract_parts(mesh_npz, tmp)
        max_pid = max(part_ids) + 1
        pid_palette = [[200, 200, 200]] * max_pid
        for pid in part_ids:
            pid_palette[pid] = _PALETTE[pid % len(_PALETTE)]
        bot_imgs = run_blender(tmp, blender, H, pid_palette, frames)
    final = stitch_two_rows(top_imgs, bot_imgs)
    ok, buf = cv2.imencode(".png", final)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


# ─────────────────────────────── VLM call ───────────────────────────────────

# def call_vlm(image_png: bytes, system: str, user: str,
#              url: str, model: str, max_tokens: int = 4096) -> str:
#     """Synchronous single-call (kept for non-async path)."""
#     from openai import OpenAI
#     client = OpenAI(base_url=url, api_key="EMPTY")
#     return _do_call_sync(client, image_png, system, user, model, max_tokens)
#
#
# def _do_call_sync(client, image_png, system, user, model, max_tokens):
#     b64 = base64.b64encode(image_png).decode()
#     resp = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url",
#                      "image_url": {"url": f"data:image/png;base64,{b64}"}},
#                     {"type": "text", "text": user},
#                 ],
#             },
#         ],
#         temperature=0.3,
#         max_tokens=max_tokens,
#         timeout=300,
#         extra_body={"chat_template_kwargs": {"enable_thinking": False}},
#     )
#     return resp.choices[0].message.content or ""
#

async def call_vlm_async(client, image_png, system, user, model, max_tokens=4096):
    b64 = base64.b64encode(image_png).decode()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": user},
                ],
            },
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        timeout=600,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


# ─────────────────────────────── parsing ────────────────────────────────────

def extract_json_object(text: str) -> dict | None:
    """Find the outermost balanced { ... } and parse it."""
    text = text.strip()
    # strip fences
    if text.startswith("```"):
        end = text.find("```", 3)
        if end > 0:
            inner = text[3:end].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            text = inner
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if in_str:
            if c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


REQUIRED_AFTER_FIELDS = ("after_desc_full", "after_desc_stage1", "after_desc_stage2")
EDIT_TYPES = ("deletion", "modification", "scale", "material", "color", "global")
N_VIEWS = 5  # must match len(VIEW_INDICES) in render_part_overview
MAX_PARTS = 16  # objects with more parts are skipped


def quota_for(n_parts: int) -> dict:
    """Per-edit-type quotas based on number of (valid) parts.

    Deletion and modification use semantic-group ceilings: parts sharing the
    same functional role (all legs, both wheels) count as ONE group → one edit.
    The formulas approximate "number of distinct semantic scenarios":
      del = ceil(n_parts / 2.5), cap 8   (main body excluded → fewer than mod)
      mod = ceil(n_parts / 2),   cap 8   (body can be modified → slightly more)

    Scale / material / color are capped at 2, stepping from 1 (< 8 parts) to
    2 (≥ 8 parts).  Global is always 2.

    Example totals (del + mod + scale + mat + clr + global):
      n=4  → 2+2+1+1+1+2 = 9
      n=8  → 4+4+2+2+2+2 = 16
      n=12 → 5+6+2+2+2+2 = 19
      n=16 → 7+8+2+2+2+2 = 23
    """
    import math as _math
    attr  = 1 if n_parts < 8 else 2
    del_q = max(1, min(8, _math.ceil(n_parts / 2.5)))
    mod_q = max(1, min(8, _math.ceil(n_parts / 2)))
    return {
        "deletion":     del_q,
        "modification": mod_q,
        "scale":        attr,
        "material":     attr,
        "color":        attr,
        "global":       2,
    }


# def validate(parsed: dict, valid_pids: set[int], quota: dict | None = None) -> dict:
#     """Lightweight check: returns {ok, errors, warnings, n_kept_edits}."""
#     out = {"ok": False, "errors": [], "warnings": [], "n_kept_edits": 0}
#     if not isinstance(parsed, dict):
#         out["errors"].append("not a dict")
#         return out
#     if "object" not in parsed or "edits" not in parsed:
#         out["errors"].append("missing object/edits keys")
#         return out
#     obj = parsed["object"]
#     for k in ("full_desc", "full_desc_stage1", "full_desc_stage2", "parts"):
#         if k not in obj:
#             out["errors"].append(f"object missing {k}")
#     edits = parsed["edits"]
#     if not isinstance(edits, list):
#         out["errors"].append("edits is not a list")
#         return out
#     type_count: dict[str, int] = {}
#     kept = 0
#     _invalid_indices: set[int] = set()
#     for i, e in enumerate(edits):
#         problems = []
#         et = e.get("edit_type")
#         if et not in EDIT_TYPES:
#             problems.append(f"bad edit_type={et}")
#         pids = e.get("selected_part_ids", [])
#         if not isinstance(pids, list) or any(p not in valid_pids for p in pids):
#             problems.append(f"invalid selected_part_ids={pids}")
#         if et == "global" and pids:
#             problems.append("global edit has selected_part_ids")
#         if et != "global" and not pids:
#             problems.append("non-global edit has empty selected_part_ids")
#         if et != "deletion":
#             for k in REQUIRED_AFTER_FIELDS:
#                 if not e.get(k):
#                     problems.append(f"missing {k}")
#         if any(f"part_{p}" in prompt for p in valid_pids):
#             problems.append("prompt mentions part_id")
#         vi = e.get("view_index")
#         if not isinstance(vi, int) or vi < 0 or vi >= N_VIEWS:
#             problems.append(f"invalid view_index={vi}")
#         if problems:
#             out["warnings"].append({"edit_index": i, "problems": problems})
#             _invalid_indices.add(i)
#         else:
#             kept += 1
#             type_count[et] = type_count.get(et, 0) + 1
#     out["n_kept_edits"] = kept
#     # R2 cross-edit check: no two valid edits with same (edit_type, selected_part_ids).
#     # Global edits are exempt: they always have selected_part_ids=[] and differ only by
#     # target_style; multiple globals are allowed by quota for large objects.
#     seen_signatures: set[tuple] = set()
#     for i, e in enumerate(edits):
#         et = e.get("edit_type")
#         if et == "global":
#             continue
#         if i in _invalid_indices:
#             continue
#         pids = tuple(sorted(e.get("selected_part_ids", [])))
#         sig = (et, pids)
#         if sig in seen_signatures:
#             out["warnings"].append({
#                 "edit_index": i,
#                 "problems": [f"R2 violation: duplicate (edit_type={et}, selected_part_ids={list(pids)})"],
#             })
#         else:
#             seen_signatures.add(sig)
#     out["type_counts"] = type_count
#     out["expected_dist"] = quota or {}
#     target = sum((quota or {}).values()) if quota else len(edits)
#     # Allow 70% recovery as success threshold
#     out["ok"] = kept >= max(1, int(target * 0.7)) and not out["errors"]
#     return out
#

# ─────────────────────────────── main ───────────────────────────────────────


__all__ = [
    # ── shared utilities ──────────────────────────────────────────────
    "VIEW_INDICES", "MAX_PARTS", "EDIT_TYPES", "N_VIEWS",
    "render_overview_png",
    "call_vlm_async",
    "extract_json_object", "quota_for",
    # ── Mode B: text_semantic (edit generation, no image) ─────────────
    "SYSTEM_PROMPT_B", "USER_PROMPT_TEXT_SEMANTIC",
    "build_semantic_list",
    "call_vlm_text_async",
    "validate_simple",
    "_GLOBAL_STYLE_POOL", "_sample_global_note",
    # ── Mode E: alignment gate ────────────────────────────────────────
    "SYSTEM_PROMPT_ALIGN_GATE",
    "build_align_gate_user_prompt",
    "parse_align_gate_output",
    "run_gate_quality",
    "build_gate_image",
    "run_gate_text_align",
    # ── Quality judge prompts ─────────────────────────────────────────
    "JUDGE_SYSTEM_PROMPTS",
    "JUDGE_SYSTEM_PROMPT_DELETION",
    "JUDGE_SYSTEM_PROMPT_MODIFICATION",
    "JUDGE_SYSTEM_PROMPT_SCALE",
    "JUDGE_SYSTEM_PROMPT_MATERIAL",
    "JUDGE_SYSTEM_PROMPT_COLOR",
    "JUDGE_SYSTEM_PROMPT_GLOBAL",
    "JUDGE_SYSTEM_PROMPT_ADDITION",
    "JUDGE_SYSTEM_PROMPT_GENERIC",
    "build_judge_user_prompt",
    "parse_judge_output",
    "extract_judge_json",
]


# # ═══════════════════════════════════════════════════════════════════════
# #  SIMPLIFIED PROMPT SYSTEM  (pipeline_v3)
# #
# #  Three prompt modes — each has its own static system prompt (KV-cacheable)
# #  and a thin per-object user prompt (part menu + quota line + global roster).
# #
# #  Mode A  "image_semantic"  — 5×2 grid image + semantic menu
# #  Mode B  "text_semantic"   — semantic part list only (no image)
# #  Mode C  "image_only"      — 5×2 grid image + colour-only menu
# #
# #  Architecture:
# #    _SYSTEM_CORE          shared invariant rules + schema (never has {} placeholders)
# #    _PREAMBLE_{A,B,C}     mode-specific input declaration (2–8 lines)
# #    SYSTEM_PROMPT_{A,B,C} = preamble + core  → KV-cached per mode
# #    USER_PROMPT_*         only per-object variables: {part_menu}, {n_*}, {global_note}
# # ═══════════════════════════════════════════════════════════════════════
#
# # ── Shared core: rules, edit-type guidance, output schema ─────────────
# # No {placeholders} here — purely static, eligible for KV-cache on all modes.
# _SYSTEM_CORE = """
# # EDIT TYPES
#   deletion      — remove a complete part or semantic group (object stays recognisable)
#   modification  — replace a part or group with a new shape OR a functionally different
#                   object in the same structural role (geometry AND/OR identity change)
#   scale         — resize a dominant part (shrink only: factor 0.3–0.85)
#   material      — change a part's surface substance or finish (shape preserved)
#   color         — repaint a part with a new hue or shade (shape + material preserved)
#   global        — transform the whole object's artistic/rendering style
#
# # TYPE DECISION GUIDE — pick the FIRST rule that matches
#   1. Removing the part entirely?                     → deletion
#   2. Changing ONLY the size (shrink)?                → scale
#   3. Changing ONLY the surface material/texture?     → material
#   4. Changing ONLY the hue/colour?                   → color
#   5. Changing the whole object's artistic aesthetic? → global
#   6. Replacing the part's shape OR swapping it
#      for a logically equivalent but different object → modification
#
#   Key distinctions:
#   • modification changes WHAT the part IS or HOW it is shaped
#   • material changes what it is MADE OF (substance) — geometry stays identical
#   • color changes its HUE — geometry and material both stay identical
#   Never mix: "a red wooden leg" = TWO edits (color + material), not one.
#
# # RULES
# R1. selected_part_ids must only contain IDs from the part menu.
# R2. No two edits may share the same edit_type AND selected_part_ids.
# R3. global edits use selected_part_ids = [].
# R4. view_index ∈ 0–4: the view where the target is most clearly visible (0 for global).
#
# # SEMANTIC GROUPING — applies to BOTH deletion and modification
#   Parts sharing the same functional role (all four chair legs, both wheels, all fin
#   panels, all support struts) form ONE semantic group.  Identify groups by shared
#   function: parts with similar names or descriptions are typically symmetric
#   instances of the same group.
#
#   For DELETION:
#   • Treat the entire group as ONE edit — list ALL member part_ids in
#     selected_part_ids.  NEVER generate separate deletions per group member.
#   • Never target the primary structural body for deletion — the part whose
#     removal makes the object unrecognisable (chair seat, car chassis, sword hilt,
#     lamp base).  Ask: "would the object still be identifiable without this part?"
#     If no → skip it.
#   • Each deletion should be a plausible user action: "remove all four legs",
#     "remove both armrests", "remove the decorative trim ring".
#
#   For MODIFICATION:
#   • When the same shape/identity change applies uniformly to a group (all legs →
#     hairpin rods, both wheels → triangular), group all IDs into ONE edit.
#   • Unlike deletion, the primary structural body CAN be a modification target.
#   • Individual unique parts (seat, backrest, steering wheel) with no semantic
#     siblings are naturally single-ID modification edits.
#
#   RESULT: edit count reflects semantic groups, not raw part count — each edit
#   is meaningful and non-redundant.
#
# # DELETION EDITS
#   A deletion removes one part or semantic group; the remaining object is still
#   recognisable as the same object category.
#   edit_params must be {}. after_desc must be null.
#
# # MODIFICATION — two valid sub-types (both use edit_params.new_part_desc):
#   (a) Shape morph — same functional identity, different geometry:
#         cylindrical barrel → hexagonal prism barrel
#         straight sword blade → curved saber blade
#         spherical head → cubic head   ·   upright ears → floppy drooping ears
#   (b) Functional replacement — completely different object in the same structural slot:
#         sword blade → axe head   ·   circular wheel → triangular wheel
#         round antenna → satellite dish   ·   vertical stabilizer → swept-back tail fin
#   new_part_desc must name the new object/shape AND its key geometry, e.g.:
#     "a broad wedge-shaped axe head" · "a triangular flat wheel" · "a parabolic dish"
#   FORBIDDEN in modification: changing ONLY colour (→ color) or ONLY material (→ material).
#
# # MATERIAL — target_material: specific surface substance, e.g.
#   "polished walnut wood" · "brushed stainless steel" · "frosted borosilicate glass"
#   · "hand-stitched leather" · "poured concrete" · "translucent amber resin"
#   Forbidden: style/aesthetic words (cartoon, vintage, futuristic …)
#
# # COLOR — target_color: descriptive colour phrase (hue/shade only), e.g.
#   "deep crimson red" · "matte charcoal black" · "cobalt blue" · "ivory cream white"
#   · "forest green" · "warm amber orange" · "pale lavender" · "glossy navy blue"
#   Forbidden: bare palette names (red, orange, lime, …) — always qualify them.
#   Do NOT change material/finish; use "material" for that.
#
# # GLOBAL — target_style: a specific artistic or rendering aesthetic.
#   The allowed styles for THIS object are listed under "GLOBAL STYLE ROSTER" in the
#   user message — choose target_style values ONLY from that per-object list.
#   Each global edit must use a style from a DIFFERENT category row of the roster.
#   Forbidden: surface-material words (gold, wood, metal) — use "material" for those.
#
# # OUTPUT — one JSON object
# {
#   "object": {
#     "full_desc": "complete English description",
#     "parts": [{"part_id": <int>, "name": "<semantic label>"}, ...]
#   },
#   "edits": [
#     {
#       "edit_type":         "deletion | modification | scale | material | color | global",
#       "selected_part_ids": [<int>, ...],
#       "prompt":            "<imperative verb phrase>",
#       "target_part_desc":  "<visual description of target part(s)>",
#       "view_index":        <int 0-4>,
#       "edit_params": {
#         // deletion:     {}
#         // modification: {"new_part_desc": "<geometry + identity description>"}
#         // scale:        {"factor": <0.3-0.85>}
#         // material:     {"target_material": "<surface substance>"}
#         // color:        {"target_color": "<descriptive colour phrase>"}
#         // global:       {"target_style": "<artistic style>"}
#       },
#       "after_desc": "<object after edit; null for deletion>"
#     }
#   ]
# }"""
#
# # ── Mode-specific preambles (static, no placeholders) ─────────────────
# # Actual part-menu column formats (no "level" column anywhere):
# #   Mode A  part_id | palette-colour | description
# #   Mode B  part_id | description
# #   Mode C  part_id | palette-colour
# # All three preambles share the same block order for easy comparison:
# #   identity → INPUT (with column format) → PALETTE RULE* → VIEW RULE → PART ID RULE*
# #   (* = absent in Mode B / only in Mode C respectively)
#
# _PREAMBLE_A = """You are a 3D-object edit-set generator. Output ONE valid JSON object — no prose, no markdown.
#
# INPUT (Mode A — image + semantic menu):
#   You will receive:
#     • A 5×2 grid image: TOP row = 5 RGB photos (view 0–4), BOTTOM row = same 5
#       cameras re-rendered with parts colour-coded by palette ID (column = camera)
#     • A semantic part menu — columns: part_id | palette-colour | description
#
# PALETTE RULE: Palette colour names (red, orange, yellow, lime, green, teal, cyan,
#   blue, navy, purple, magenta, pink, brown, tan, black, gray) are INTERNAL labels
#   matching the BOTTOM-row highlights — do NOT use them in any output text field.
#   Describe real colour/appearance using the TOP-row RGB photos.
#
# VIEW RULE: Use the images to select view_index (0–4) where each target part is
#   most clearly visible.
# """
#
# _PREAMBLE_B = """You are a 3D-object edit-set generator. Output ONE valid JSON object — no prose, no markdown.
#
# INPUT (Mode B — text-only semantic list, no image):
#   You will receive:
#     • A semantic part list — columns: part_id | description
#     • NO image is provided.
#
# VIEW RULE: Without an image, view_index is a structural best-estimate.
#   Use 0 for parts typically facing front, 4 for parts on the bottom.
#   Parts with similar names or descriptions are likely symmetric instances
#   (legs, wheels, fins) — group them into one edit accordingly.
# """
#
# _PREAMBLE_C = """You are a 3D-object edit-set generator. Output ONE valid JSON object — no prose, no markdown.
#
# INPUT (Mode C — image + colour-only menu):
#   You will receive:
#     • A 5×2 grid image: TOP row = 5 RGB photos (view 0–4), BOTTOM row = same 5
#       cameras re-rendered with parts colour-coded by palette ID (column = camera)
#     • A colour-only part menu — columns: part_id | palette-colour
#       (no text descriptions — identify parts by matching palette colour in the
#        BOTTOM row to the corresponding region in the TOP-row photos)
#
# PALETTE RULE: Palette colour names (red, orange, yellow, lime, green, teal, cyan,
#   blue, navy, purple, magenta, pink, brown, tan, black, gray) are INTERNAL labels
#   matching the BOTTOM-row highlights — do NOT use them in any output text field.
#   Describe real colour/appearance using the TOP-row RGB photos.
#
# VIEW RULE: Use the images to select view_index (0–4) where each target part is
#   most clearly visible.
#
# PART ID RULE: To identify what part_N is, locate its palette colour in the BOTTOM
#   row and examine the matching region in the TOP-row RGB photos.
# """
#
# # ── Assembled system prompts (preamble + core) — one per mode ─────────
# # These are purely static strings: no {placeholders}. They are the same
# # for every object within a mode, making them eligible for KV-cache reuse.
# SYSTEM_PROMPT_A = _PREAMBLE_A + _SYSTEM_CORE   # image + semantic menu
# # SYSTEM_PROMPT_B — fully standalone, no image input
# # Input: caption-based semantic part list (part_id | description)
# # No image → no view_index in output schema.
SYSTEM_PROMPT_B = """You are a 3D-object edit-set generator. Output ONE valid JSON object — no prose, no markdown fences.

INPUT (caption list — no image):
  You will receive a semantic part list with columns: part_id | description.
  Reason about parts purely from their text descriptions.

# EDIT TYPES
  deletion      — remove a complete part or semantic group (object stays recognisable)
  modification  — replace a part or group with a new shape OR a functionally different
                  object in the same structural role (geometry AND/OR identity change)
  scale         — resize a dominant part (shrink only: factor 0.3–0.85)
  material      — change a part's surface substance or finish (shape preserved)
  color         — repaint a part with a new hue or shade (shape + material preserved)
  global        — transform the whole object's artistic/rendering style

# TYPE DECISION GUIDE — pick the FIRST rule that matches
  1. Removing the part entirely?                     → deletion
  2. Changing ONLY the size (shrink)?                → scale
  3. Changing ONLY the surface material/texture?     → material
  4. Changing ONLY the hue/colour?                   → color
  5. Changing the whole object's artistic aesthetic? → global
  6. Replacing the part's shape OR swapping it
     for a logically equivalent but different object → modification

  Key distinctions:
  • modification changes WHAT the part IS or HOW it is shaped
  • material changes what it is MADE OF (substance) — geometry stays identical
  • color changes its HUE — geometry and material both stay identical
  Never mix: "a red wooden leg" = TWO edits (color + material), not one.

# RULES
  R1. selected_part_ids must only contain IDs from the part list.
  R2. No two edits may share the same edit_type AND selected_part_ids.
  R3. global edits use selected_part_ids = [].

# SEMANTIC GROUPING — applies to BOTH deletion and modification
  Two kinds of parts should be grouped into a single edit:

  (a) Repeated instances — parts sharing the same functional role that appear
      multiple times on the object:
        all four chair legs · both wheels · all fin panels · all support struts
      Clue: similar names or descriptions that differ only by position or index.

  (b) Semantic constituents — distinct parts that together form one coherent
      semantic unit, such that acting on only a subset would leave an incomplete
      or incoherent result:
        a character head → ears + eyes + nose + mouth → ONE deletion
        a human hand → fingers + palm → ONE edit
        a face panel → all facial feature parts → ONE edit
      Clue: the parts collectively define a single recognisable sub-object;
      removing or changing just one of them would produce a broken result.

  For DELETION:
  • Treat the entire group as ONE edit — list ALL member part_ids in selected_part_ids.
  • Never target the primary structural body for deletion.
  • Each deletion should be a plausible user action.

  For MODIFICATION:
  • When the same change applies uniformly to a group, list all IDs in one edit.
  • Unlike deletion, the primary structural body CAN be a modification target.

# DELETION EDITS
  edit_params must be {}. after_desc must be null.

# MODIFICATION — edit_params.new_part_desc: name the new object/shape AND key geometry.
  (a) Shape morph — same role, different geometry
  (b) Functional replacement — different object in the same structural slot
  FORBIDDEN: changing ONLY colour (→ color) or ONLY material (→ material).

# MATERIAL — target_material: specific surface substance, e.g.
  "polished walnut wood" · "brushed stainless steel" · "frosted borosilicate glass"

# COLOR — target_color: descriptive colour phrase (hue/shade only), e.g.
  "deep crimson red" · "matte charcoal black" · "cobalt blue"
  Forbidden: bare palette names — always qualify them.

# GLOBAL — target_style: choose ONLY from the "GLOBAL STYLE ROSTER" in the user message.
  Each global edit must use a style from a DIFFERENT category row of the roster.

LANGUAGE: All output fields must be in English only. Do not use any Chinese or other non-English characters.

# OUTPUT — one JSON object (no view_index — there is no image)
{
  "object": {
    "full_desc": "<complete English description of the object>",
    "parts": [{"part_id": <int>, "name": "<semantic label>"}, ...]
  },
  "edits": [
    {
      "edit_type":         "deletion | modification | scale | material | color | global",
      "selected_part_ids": [<int>, ...],
      "prompt":            "<imperative verb phrase>",
      "target_part_desc":  "<description of target part(s) from the caption>",
      "edit_params": {
        // deletion:     {}
        // modification: {"new_part_desc": "<geometry + identity description>"}
        // scale:        {"factor": <0.3-0.85>}
        // material:     {"target_material": "<surface substance>"}
        // color:        {"target_color": "<descriptive colour phrase>"}
        // global:       {"target_style": "<artistic style>"}
      },
      "after_desc": "<object after edit; null for deletion>"
    }
  ]
}"""

# SYSTEM_PROMPT_C = _PREAMBLE_C + _SYSTEM_CORE   # image + colour-only menu
#
# ── User prompts: only the per-object variable parts ──────────────────
_QUOTA_LINE = "Generate EXACTLY {n_total} edits — {n_deletion} deletion · {n_modification} modification · {n_scale} scale · {n_material} material · {n_color} color · {n_global} global{global_note}"

# # Mode A: image + semantic menu (palette colour + description)
# USER_PROMPT_IMAGE_SEMANTIC = """[Image: 5 RGB photos (top row) + same 5 views re-rendered with parts colour-coded by ID (bottom row). Palette colours are INTERNAL labels — do NOT use them in output text.]
#
# # PART MENU  (id · palette-colour · description)
# {part_menu}
#
# """ + _QUOTA_LINE
#
# Mode B: text-only semantic list, no image
USER_PROMPT_TEXT_SEMANTIC = """# PART LIST  (id · description)
{part_menu}

""" + _QUOTA_LINE

# # Mode C: image + colour-only menu (no descriptions — VLM reasons from image)
# USER_PROMPT_IMAGE_ONLY = """[Image: 5 RGB photos (top row) + same 5 views re-rendered with parts colour-coded by ID (bottom row). Palette colours are INTERNAL labels — do NOT use them in output text.]
#
# # PART MENU  (id · palette-colour)
# {part_menu}
#
# """ + _QUOTA_LINE
#
# PROMPT_MODES = ("image_semantic", "text_semantic", "image_only")
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  Menu builders for the simplified modes
# # ═══════════════════════════════════════════════════════════════════════
#
# def build_image_only_menu(
#     mesh_npz: Path,
#     img_npz: Path,
# ) -> tuple[list[int], str]:
#     """Colour-only part menu — for image_only mode (Mode C).
#
#     No semantic descriptions or level — the VLM must reason purely from
#     the colour-coded image overlay.
#
#     Format per line:
#         part_{id:<3d}   {colour}
#     """
#     z2 = __import__('numpy').load(mesh_npz, allow_pickle=True)
#     z = __import__('numpy').load(img_npz, allow_pickle=True)
#     sm = json.loads(bytes(z["split_mesh.json"]).decode())
#     clusters = sm.get("valid_clusters", {})
#     import re as _re
#
#     def _parse_pid(k: str) -> int | None:
#         m = _re.search(r"\d+", k)
#         return int(m.group()) if m else None
#
#     pids = sorted(
#         pid
#         for k in z2.files
#         if k.startswith("part_") and (k.endswith(".glb") or k.endswith(".ply"))
#         if (pid := _parse_pid(k)) is not None
#     )
#
#     lines = []
#     for pid in pids:
#         color = _PALETTE_NAMES[pid % len(_PALETTE_NAMES)]
#         base = f"  part_{pid:<3d}   {color}"
#         lines.append(base)
#     return pids, "\n".join(lines)
#
#
def build_semantic_list(
    mesh_npz: Path,
    img_npz: Path,
    anno_obj_dir: "Path | None" = None,
) -> tuple[list[int], str]:
    """Text-only semantic part list — for text_semantic mode (Mode B).

    Columns: part_id | description
    Uses PartVerse text_captions when available; falls back to "part_N".
    No palette colours.

    Format per line:
        part_{id:<3d}  "{name}"
    """
    import re as _re
    import numpy as _np

    z = _np.load(img_npz, allow_pickle=True)
    sm = json.loads(bytes(z["split_mesh.json"]).decode())
    clusters = sm.get("valid_clusters", {})
    pid_to_name_raw = sm.get("part_id_to_name", [])

    z2 = _np.load(mesh_npz, allow_pickle=True)

    def _parse_pid(k: str) -> int | None:
        m = _re.search(r"\d+", k)
        return int(m.group()) if m else None

    pids = sorted(
        pid
        for k in z2.files
        if k.startswith("part_") and (k.endswith(".glb") or k.endswith(".ply"))
        if (pid := _parse_pid(k)) is not None
    )

    # Load per-part captions: prefer embedded part_captions.json (from PartVerse
    # text_captions.json repacked by repack_mesh_add_anno.py), fall back to the
    # image-captioning sentences in split_mesh.json part_id_to_name.
    part_captions: dict[int, list[str]] = {}
    if "part_captions.json" in z2.files:
        try:
            raw_caps: dict[str, list] = json.loads(bytes(z2["part_captions.json"]).decode())
            part_captions = {int(k): v for k, v in raw_caps.items()}
        except Exception:
            pass

    import re as _re2
    _STRIP_TAIL = _re2.compile(r'\._\d+\s*$')
    _OF_THE     = _re2.compile(r'(?:of the|of a|of an)\s+(.+)$', _re2.I)
    _ADET       = _re2.compile(r'^(?:a |an |the )+', _re2.I)

    def _name(pid: int) -> str:
        # Priority 1: PartVerse text_captions short description
        if pid in part_captions:
            captions = part_captions[pid]
            if captions and isinstance(captions[0], str) and captions[0].strip():
                s = captions[0].strip().rstrip(".")
                s = _ADET.sub("", s).strip()
                if s:
                    return s
        # Priority 2: split_mesh.json image caption (noisy, apply heuristics)
        if isinstance(pid_to_name_raw, list) and pid < len(pid_to_name_raw):
            raw = pid_to_name_raw[pid]
            if isinstance(raw, str) and raw.strip():
                s = _STRIP_TAIL.sub("", raw).strip().rstrip(".")
                m = _OF_THE.search(s)
                if m:
                    s = m.group(1).strip().rstrip(".")
                s = _ADET.sub("", s).strip()
                if s:
                    return s
        return f"part_{pid}"

    lines = []
    for pid in pids:
        name = _name(pid)
        base = f'  part_{pid:<3d}  "{name}"'
        lines.append(base)
    return pids, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Text-only VLM call (Modes 1 & 3 — no image attached)
# ═══════════════════════════════════════════════════════════════════════

async def call_vlm_text_async(
    client,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 4096,
) -> str:
    """Async VLM call with no image — for text_menu and text_semantic modes."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        timeout=300,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════════
#  Global-style randomisation — prevents VLM always picking the same few
# ═══════════════════════════════════════════════════════════════════════

# Full pool by category.  NEVER use bare items from this list in prompts;
# always go through _sample_global_note() so each object sees a shuffled
# subset — this breaks the position-bias that makes Qwen pick cel-shading
# and LEGO every time.
_GLOBAL_STYLE_POOL: dict[str, list[str]] = {
    "Rendering": [
        "cel-shading",
        "flat-shading with bold outlines",
        "wireframe-outline with coloured faces",
        "watercolour wash",
        "oil-painting impasto",
        "impressionist brushstroke",
        "pointillist dots",
        "charcoal sketch",
        "ink-wash sumi-e",
        "stained-glass mosaic",
        "neon-glow bloom",
        "risograph screen-print",
        "low-poly faceted geometry",
    ],
    "Historical": [
        "Art Nouveau organic flowing lines",
        "Art Deco bold geometric",
        "ukiyo-e woodblock print",
        "Bauhaus functional",
        "brutalist concrete aesthetic",
        "baroque gilded ornament",
        "gothic cathedral tracery",
        "ancient terracotta figurine",
        "Ming dynasty blue-and-white porcelain",
        "Byzantine gold-leaf mosaic",
        "medieval illuminated manuscript",
        "Edo-period lacquerware",
    ],
    "Genre": [
        "cyberpunk neon-and-chrome",
        "steampunk brass-and-gears",
        "solarpunk organic-tech",
        "vaporwave pastel grid",
        "retro-1980s pixel-art",
        "lo-fi cassette-tape grain",
        "biomechanical flesh-and-machine",
        "origami paper-fold geometry",
        "LEGO brick construction",
        "Islamic geometric mosaic tile",
        "psychedelic tie-dye swirl",
        "dieselpunk rust-and-rivets",
        "cottagecore hand-embroidered",
    ],
}

_POOL_CATS = list(_GLOBAL_STYLE_POOL.keys())  # ["Rendering", "Historical", "Genre"]


def _sample_global_note(variety_seed: int, n_global: int) -> str:
    """Return a per-object mandatory style roster injected into the user prompt.

    Shows 2-3 randomly-ordered choices per category (seeded by variety_seed).
    The VLM is told it MUST pick from these specific options, eliminating
    the position-bias that causes it to always choose the same popular styles.

    n_global ≤ 3: one category slot each, show 2 choices per category.
    n_global = 4: one category has 2 slots, show 3 choices for it.
    """
    import random as _rng
    rng = _rng.Random(variety_seed)

    # Shuffle the pool independently per category
    shuffled: dict[str, list[str]] = {}
    for cat, styles in _GLOBAL_STYLE_POOL.items():
        s = list(styles)
        rng.shuffle(s)
        shuffled[cat] = s

    # Decide how many slots per category (n_global slots across 3 cats)
    slots = {cat: 1 for cat in _POOL_CATS}
    if n_global > len(_POOL_CATS):
        extra_cat = rng.choice(_POOL_CATS)
        slots[extra_cat] = 2

    # Build the note: show (slots[cat] + 1) choices per category so VLM
    # has some flexibility while still seeing a shuffled non-default list
    lines = [
        "\n# GLOBAL STYLE ROSTER (mandatory for this object — choose ONLY from these):"
    ]
    for cat in _POOL_CATS:
        n_show = slots[cat] + 1          # 2 or 3 options shown
        choices = shuffled[cat][:n_show]
        lines.append(f"  {cat}: " + "  ·  ".join(choices))
    if n_global > 1:
        lines.append(
            f"  Use a DIFFERENT category row for each of the {n_global} global edits."
        )
    return "\n".join(lines)


# def build_prompt_for_mode(
#     mode: str,
#     pids: list[int],
#     part_menu: str,
#     quota: dict,
#     *,
#     variety_seed: int | None = None,
# ) -> tuple[str, str]:
#     """Return (system, user) prompt strings for the given mode.
#
#     Args:
#         mode:           one of PROMPT_MODES:
#                           "image_semantic" — image + semantic menu (Mode A)
#                           "text_semantic"  — semantic menu only, no image (Mode B)
#                           "image_only"     — image + colour-only menu (Mode C)
#         pids:           list of valid part IDs
#         part_menu:      pre-built menu string from the matching builder
#         quota:          output of quota_for()
#         variety_seed:   integer seed for per-object style randomisation.
#                         Pass hash(obj_id) or any per-object integer. When
#                         None, falls back to a seed derived from pids (less
#                         diverse). Always provide this for production runs.
#     """
#     if mode not in PROMPT_MODES:
#         raise ValueError(f"Unknown prompt mode: {mode!r}. Choose from {PROMPT_MODES}")
#
#     n_global = quota.get("global", 0)
#     seed = variety_seed if variety_seed is not None else hash(tuple(sorted(pids)))
#     global_note = _sample_global_note(seed, n_global) if n_global > 0 else ""
#
#     n_total = sum(quota.values())
#     fmt = dict(
#         part_menu=part_menu,
#         n_total=n_total,
#         n_deletion=quota.get("deletion", 0),
#         n_modification=quota.get("modification", 0),
#         n_scale=quota.get("scale", 0),
#         n_material=quota.get("material", 0),
#         n_color=quota.get("color", 0),
#         n_global=n_global,
#         global_note=global_note,
#     )
#
#     if mode == "image_semantic":
#         user = USER_PROMPT_IMAGE_SEMANTIC.format(**fmt)
#     elif mode == "text_semantic":
#         user = USER_PROMPT_TEXT_SEMANTIC.format(**fmt)
#     else:  # image_only
#         user = USER_PROMPT_IMAGE_ONLY.format(**fmt)
#
#     sys_map = {
#         "image_semantic": SYSTEM_PROMPT_A,
#         "text_semantic":  SYSTEM_PROMPT_B,
#         "image_only":     SYSTEM_PROMPT_C,
#     }
#     return sys_map[mode], user
#
#
# ═══════════════════════════════════════════════════════════════════════
#  validate_simple — lighter validator for the simplified output schema
# ═══════════════════════════════════════════════════════════════════════

def validate_simple(parsed: dict, valid_pids: set[int], quota: dict | None = None) -> dict:
    """Validator for the simplified schema (no stage1/2 fields required).

    Required per edit: edit_type, selected_part_ids, prompt, view_index.
    after_desc is required for non-deletion edits (null/missing → warning, not error).
    edit_params presence is checked per type.
    """
    out: dict = {"ok": False, "errors": [], "warnings": [], "n_kept_edits": 0}
    if not isinstance(parsed, dict):
        out["errors"].append("not a dict")
        return out
    if "edits" not in parsed:
        out["errors"].append("missing 'edits' key")
        return out

    edits = parsed["edits"]
    if not isinstance(edits, list):
        out["errors"].append("edits is not a list")
        return out

    type_count: dict[str, int] = {}
    kept = 0
    _invalid: set[int] = set()

    for i, e in enumerate(edits):
        problems: list[str] = []

        et = e.get("edit_type")
        if et not in EDIT_TYPES:
            problems.append(f"bad edit_type={et!r}")

        pids = e.get("selected_part_ids", [])
        if not isinstance(pids, list) or any(p not in valid_pids for p in pids):
            problems.append(f"invalid selected_part_ids={pids}")
        if et == "global" and pids:
            problems.append("global edit must have selected_part_ids=[]")
        if et != "global" and not pids:
            problems.append("non-global edit has empty selected_part_ids")


        # view_index is optional for text-only modes (Mode B) — skip if absent
        vi = e.get("view_index")
        if vi is not None and (not isinstance(vi, int) or vi < 0 or vi >= N_VIEWS):
            problems.append(f"view_index must be 0-{N_VIEWS - 1}, got {vi!r}")

        # after_desc required for non-deletion (warning, not hard error)
        if et != "deletion" and not e.get("after_desc"):
            problems.append("after_desc missing for non-deletion edit")

        # edit_params type checks
        ep = e.get("edit_params") or {}
        if et == "modification" and not ep.get("new_part_desc"):
            problems.append("modification missing edit_params.new_part_desc")
        if et == "scale":
            f = ep.get("factor")
            if not isinstance(f, (int, float)) or not (0.3 <= f <= 0.85):
                problems.append(f"scale factor must be 0.3-0.85, got {f!r}")
        if et == "material" and not ep.get("target_material"):
            problems.append("material missing edit_params.target_material")
        if et == "color" and not ep.get("target_color"):
            problems.append("color missing edit_params.target_color")
        if et == "global" and not ep.get("target_style"):
            problems.append("global missing edit_params.target_style")

        if problems:
            out["warnings"].append({"edit_index": i, "problems": problems})
            _invalid.add(i)
        else:
            kept += 1
            type_count[et] = type_count.get(et, 0) + 1

    # R4 duplicate check
    seen: set[tuple] = set()
    for i, e in enumerate(edits):
        et = e.get("edit_type")
        if et == "global" or i in _invalid:
            continue
        sig = (et, tuple(sorted(e.get("selected_part_ids", []))))
        if sig in seen:
            out["warnings"].append({
                "edit_index": i,
                "problems": [f"R4 duplicate (edit_type={et}, selected_part_ids={list(sig[1])})"],
            })
        else:
            seen.add(sig)

    out["n_kept_edits"] = kept
    out["type_counts"] = type_count
    out["expected_dist"] = quota or {}
    target = sum((quota or {}).values()) if quota else len(edits)
    out["ok"] = kept >= max(1, int(target * 0.7)) and not out["errors"]
    return out


# # ═══════════════════════════════════════════════════════════════════════
# #  TWO-STAGE PIPELINE  (Mode D — image→semantics then text→edits)
# #
# #  Stage 1  call_vlm_async(image + colour menu)  → s1_parts JSON
# #  Stage 2  call_vlm_text_async(s1_parts text)   → edit JSON   (no image)
# #
# #  Key invariant: the image is ONLY seen in Stage 1. Stage 2 is fully
# #  text-driven so it cannot be misled by PartVerse caption noise.
# # ═══════════════════════════════════════════════════════════════════════
#
# # ── Stage 1: visual part-semantic reconstruction ─────────────────────
#
# SYSTEM_PROMPT_S1 = """\
# You are a 3D-part semantic labeller. Given a colour-coded overview image of a 3D object, assign a precise functional name to every visible part.
#
# INPUT:
#   • A 5×2 grid image:
#       TOP row    = 5 RGB photos (views 0–4, same camera each column)
#       BOTTOM row = same 5 views re-rendered with parts colour-coded by palette ID
#   • A colour-only part menu: part_id | palette-colour
#
# TASK:
#   For each entry in the menu, locate its palette colour in the BOTTOM row,
#   examine the matching region in the TOP-row RGB photos, and decide what
#   structural role that region plays in the overall object.
#
# OUTPUT: ONE valid JSON object — no prose, no markdown fences.
# {
#   "object_desc": "<one sentence describing the whole object>",
#   "parts": [
#     {
#       "part_id": <int>,
#       "name": "<concise functional label, 1-4 words, e.g. 'front left wheel', 'gun barrel', 'left arm'>",
#       "view_index": <0-4, the column where this part is most clearly visible>,
#       "appearance": "<brief visual description from the TOP-row RGB photos, 5-12 words>"
#     }
#   ]
# }
#
# RULES:
#   R1. name is a SHORT FUNCTIONAL LABEL — not a visual description.
#       Good: "rear left leg"  "trigger guard"  "left winglet"
#       Bad:  "cylindrical green component"  "blue faceted object"
#   R2. Symmetric/repeated instances must be individually named with a
#       positional qualifier: "front left wheel" not just "wheel".
#   R3. For each part, scan ALL 5 bottom-row columns for its palette colour.
#       If you CANNOT clearly locate the colour in ANY of the 5 views:
#         • Set "name" to null and "view_index" to -1.
#         • Do NOT guess or invent a name from context or position.
#       Occluded / invisible parts will be excluded from editing — accuracy
#       matters more than completeness.
#   R4. Do NOT use palette colour names (red, orange, yellow, lime …) in any
#       output field — use the real appearance from the TOP row.
#   R5. Output parts in ascending part_id order.
# """
#
# # ── Stage 2: text-only edit generation ───────────────────────────────
# # Reuses _PREAMBLE_B (text-only preamble) + _SYSTEM_CORE.
# # SYSTEM_PROMPT_S2 is identical to SYSTEM_PROMPT_B in behaviour.
# # We give it a distinct name to signal intent.
# SYSTEM_PROMPT_S2 = _PREAMBLE_B + _SYSTEM_CORE  # same as SYSTEM_PROMPT_B
#
#
# # ── User-prompt builders ──────────────────────────────────────────────
#
# def build_s1_user_prompt(pids: list[int]) -> str:
#     """Colour-only menu for Stage 1 (identical format to build_image_only_menu
#     but takes pre-computed pids directly, no NPZ access needed)."""
#     lines = [f"  part_{pid:<3d}   {_PALETTE_NAMES[pid % len(_PALETTE_NAMES)]}"
#              for pid in sorted(pids)]
#     return "Colour-only part menu:\n" + "\n".join(lines)
#
#
# def build_s2_user_prompt(
#     s1_parts: list[dict],
#     object_desc: str,
#     quota: dict,
#     *,
#     variety_seed: int | None = None,
# ) -> str:
#     """Format Stage 1 visible-part output into a Stage 2 (text-only) user prompt.
#
#     Accepts the contents of d["parts_visible"] — null-name parts are already
#     filtered out by parse_s1_output and must NOT be passed here.
#
#     Each part line:  part_{id}   "{name}"   [view: N]  — appearance
#     """
#     pids = [p["part_id"] for p in s1_parts]
#     n_global = quota.get("global", 0)
#     seed = variety_seed if variety_seed is not None else hash(tuple(sorted(pids)))
#     global_note = _sample_global_note(seed, n_global) if n_global > 0 else ""
#
#     part_lines = []
#     for p in sorted(s1_parts, key=lambda x: x["part_id"]):
#         pid  = p["part_id"]
#         name = p.get("name", f"part_{pid}")
#         vi   = p.get("view_index", 0)
#         app  = p.get("appearance", "")
#         vi_str = str(vi) if vi >= 0 else "hidden"
#         line = f'  part_{pid:<3d}   "{name}"   [view: {vi_str}]'
#         if app:
#             line += f"  — {app}"
#         part_lines.append(line)
#
#     parts_block = "\n".join(part_lines)
#     n_total = sum(quota.values())
#
#     return (
#         f"Object: {object_desc}\n\n"
#         f"Parts:\n{parts_block}\n\n"
#         f"Generate at most {n_total} edits total "
#         f"({quota.get('deletion',0)} deletion, "
#         f"{quota.get('modification',0)} modification, "
#         f"{quota.get('scale',0)} scale, "
#         f"{quota.get('material',0)} material, "
#         f"{quota.get('color',0)} color, "
#         f"{quota.get('global',0)} global)."
#         + (f"\n\nGlobal style roster:\n{global_note}" if global_note else "")
#     )
#
#
# def parse_s1_output(raw: str) -> dict | None:
#     """Extract and lightly validate the Stage 1 JSON.
#
#     Returns the parsed dict on success, None on failure.
#     Adds two keys to the dict:
#       "parts_visible"  — parts whose name is non-null (will be sent to Stage 2)
#       "parts_hidden"   — parts whose name is null (invisible, excluded from Stage 2)
#     """
#     d = extract_json_object(raw)   # already returns dict | None
#     if not isinstance(d, dict):
#         return None
#     parts = d.get("parts")
#     if not isinstance(parts, list) or not parts:
#         return None
#     if not all("part_id" in p for p in parts):
#         return None
#     visible = [p for p in parts if p.get("name") not in (None, "null", "")]
#     hidden  = [p for p in parts if p.get("name") in (None, "null", "")]
#     d["parts_visible"] = visible
#     d["parts_hidden"]  = hidden
#     # Require at least one visible part
#     if not visible:
#         return None
#     return d
#
#
# ═══════════════════════════════════════════════════════════════════════
#  Alignment Gate (Mode E) — image+text call, judges edit↔part alignment
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_ALIGN_GATE = """You are a 3D-edit alignment judge.

INPUT (two possible layouts):
  [Non-global edits] 5×2 grid image:
      TOP row    = 5 views of the 3D object
      BOTTOM row = 5 highlight renders — selected (target) parts are RED,
                   all other parts are GREY, background is WHITE.
      Column layout:
        col 0          = the view with the HIGHEST target-part visibility
                         (computed from pixel counts)
        cols 1–4       = fixed views 0–3 of the standard overview
                         (same order as the original 5-view overview, bottom view excluded)
  [Global edits] 5×1 strip image:
      A single row of 5 RGB photos (no bottom row).
      The whole object is the edit target.
  • Edit instruction and edit type in text.

TASK (non-global):
  1. Locate the red region(s) in the bottom row — note their exact shape, size, and position within each column.
  2. For each column, look directly ABOVE the red region in the top-row RGB: what physical object/component occupies that exact spatial area?
     Do NOT guess from the scene context — trace the boundary carefully.
  3. Based on steps 1–2, identify what the target part actually IS (its shape, material, label).
  4. Compare the identified part against the edit instruction and target_part_desc.
     Judge alignment: does the instruction make semantic sense for that exact part?
  5. Choose which column (0–4) shows the target part most clearly for downstream editing.

TASK (global):
  1. You only see RGB photos — the whole object is the edit target.
  2. Judge alignment based on whether the edit type/instruction applies to the whole object.
  3. Choose which column (0–4) shows the overall object most clearly for editing.

LANGUAGE: All output fields must be written in English only. Do not use any Chinese or other non-English characters anywhere in the JSON.

OUTPUT: ONE valid JSON object — no prose, no markdown fences.
{
  "aligned":   <true|false>,
  "reason":    "<1-2 sentences>",
  "best_view": <0-4, column index in THIS image where target is clearest>
}

RULES:
  R1. aligned=true  iff ALL of the following hold:
        • The red-highlighted parts visually match the instruction’s stated target
          (correct parts, not too many and not too few).
        • The edit type is appropriate for those parts.
        • The instruction is specific and unambiguous.
  R2. aligned=false in ANY of these cases:
        • Wrong selection — highlighted region does not correspond to the named
          target in the instruction.
        • Over-selection — far more parts are highlighted than the instruction
          refers to.
        • Under-selection — the instruction targets multiple distinct parts but
          only a subset is highlighted.
        • Fully occluded — red is absent in ALL 5 columns.
        • Unclear prompt — the instruction is so vague or self-contradictory that
          a downstream editor cannot act on it.
  R3. best_view = the column where the target is most visible in the top-row RGB
      and best suited for downstream image editing.
  R4. For global edits, always aligned=true unless the instruction is incoherent.
      Choose best_view from the RGB photos based on overall object visibility.
"""

def build_align_gate_user_prompt(
    edit_type: str,
    prompt: str,
    selected_part_ids: list[int],
) -> str:
    """User prompt for the alignment gate VLM call (text portion only)."""
    parts_str = ", ".join(f"part_{p}" for p in sorted(selected_part_ids)) or "(none)"
    return (
        f"Edit type: {edit_type}\n"
        f'Instruction: "{prompt}"\n'
        f"Selected parts: {parts_str}"
    )


def parse_align_gate_output(raw: str) -> dict | None:
    """Parse alignment gate VLM response.

    Returns dict with at least {"aligned": bool, "reason": str, "best_view": int}
    or None on failure.
    """
    d = extract_json_object(raw)
    if not isinstance(d, dict):
        return None
    if not isinstance(d.get("aligned"), bool):
        return None
    bv = d.get("best_view")
    if not (type(bv) is int):
        d["best_view"] = 0   # safe default; also coerces bool true/false from model
    return d


# ============================================================================
# Final quality gate (gate_quality / gate_e) — visual VLM judge
# ============================================================================
# Moved from sq3_qc_e.py.  Evaluates each edit by comparing before/after
# views in a collage image using the judge VLM.
#
# Public entry point: run_gate_quality(ctxs, *, vlm_urls, vlm_model, cfg, ...)
# ============================================================================

import asyncio as _asyncio
import base64 as _base64
import logging
# cv2 imported lazily inside _qe_* functions to avoid hard dep at module load

# Pipeline submodule imports for gate_quality are done lazily inside
# each function to avoid the specs.py -> s1_vlm_core.py circular import.

_QE_DEFS = {
    # edit_type → default thresholds (overridden by qc.thresholds_by_type in config)
    "deletion":     {"min_visual_quality": 3, "require_preserve_other": True},
    "modification": {"min_visual_quality": 3, "require_preserve_other": True},
    "scale":        {"min_visual_quality": 3, "require_preserve_other": True},
    "material":     {"min_visual_quality": 3, "require_preserve_other": True},
    # color: geometry unchanged by design; preserve_other checks shape stability
    "color":        {"min_visual_quality": 3, "require_preserve_other": True},
    # global: structure must remain recognisable
    "global":       {"min_visual_quality": 3, "require_preserve_other": True},
    "addition":     {"min_visual_quality": 3, "require_preserve_other": True},
}

_LOG_QE = logging.getLogger("pipeline_v3.gate_quality")


def _qe_passes(judge_json: dict, edit_type: str, thresholds: dict) -> bool:
    """Return True if the judge result meets the QC thresholds for *edit_type*."""
    t = {**_QE_DEFS.get(edit_type, {}), **(thresholds.get(edit_type) or {})}
    if not judge_json.get("edit_executed", False):
        return False
    try:
        vq = int(judge_json.get("visual_quality", 0))
    except (TypeError, ValueError):
        vq = 0
    if vq < t.get("min_visual_quality", 3):
        return False
    if not judge_json.get("correct_region", False):
        return False
    if t.get("require_preserve_other") and not judge_json.get("preserve_other", False):
        return False
    return True


def _qe_load_before_imgs(ctx) -> "list | None":
    """Load 5 before-state BGR images from the images NPZ at VIEW_INDICES."""
    from .specs import VIEW_INDICES  # lazy — avoid circular import at module level
    if ctx.image_npz is None or not ctx.image_npz.is_file():
        return None
    try:
        from partcraft.render.overview import load_views_from_npz
        imgs, _ = load_views_from_npz(ctx.image_npz, VIEW_INDICES)
        return imgs
    except Exception:
        return None


def _qe_load_after_previews(edit_dir: "Path") -> "list | None":
    """Load preview_0.png … preview_4.png from *edit_dir*. Returns None on any missing file."""
    import numpy as np
    import cv2 as _cv2
    imgs = []
    for i in range(5):
        p = edit_dir / f"preview_{i}.png"
        if not p.is_file():
            return None
        img = _cv2.imread(str(p))
        if img is None:
            return None
        imgs.append(img)
    return imgs


def _qe_make_collage(before_imgs: list, after_imgs: list) -> "bytes | None":
    """Build a 2-row × 5-col PNG collage (top = before, bottom = after)."""
    import cv2 as _cv2
    import numpy as np
    h = 256

    def _r(x):
        s = h / x.shape[0]
        return _cv2.resize(x, (int(x.shape[1] * s), h))

    try:
        row_b = np.hstack([_r(img) for img in before_imgs])
        row_a = np.hstack([_r(img) for img in after_imgs])
        w = max(row_b.shape[1], row_a.shape[1])
        if row_b.shape[1] < w:
            row_b = np.pad(row_b, ((0, 0), (0, w - row_b.shape[1]), (0, 0)))
        if row_a.shape[1] < w:
            row_a = np.pad(row_a, ((0, 0), (0, w - row_a.shape[1]), (0, 0)))
        ok, buf = _cv2.imencode(".png", np.vstack([row_b, row_a]))  # noqa (cv2 imported above)
        return buf.tobytes() if ok else None
    except Exception:
        return None


def _qe_iter_add_edits(ctx):
    """Yield (edit_id, meta_dict) for addition edits from edits_3d/*/meta.json."""
    if not ctx.edits_3d_dir.is_dir():
        return
    for add_dir in sorted(ctx.edits_3d_dir.iterdir()):
        if not add_dir.is_dir():
            continue
        meta_path = add_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("edit_type") == "addition":
            yield add_dir.name, meta



# ============================================================================
# ============================================================================
# Quality judge prompts  (single source of truth for all VLM judge prompts)
#
# Pattern mirrors the alignment gate:
#   JUDGE_SYSTEM_PROMPTS[edit_type]  — large, directly readable system prompt
#   build_judge_user_prompt(...)     — tiny function, fills per-edit variables
#   parse_judge_output(raw)          — extracts the JSON response
# ============================================================================

# Schema appended to every system prompt so the model always sees output rules.
_JUDGE_OUTPUT_SCHEMA = """
OUTPUT: ONE valid JSON object only — no prose, no markdown fences.
First character must be "{" and last must be "}".

{
  "edit_executed":    <true|false>,
  "correct_region":   <true|false>,
  "preserve_other":   <true|false>,
  "visual_quality":   <1-5>,
  "artifact_free":    <true|false>,
  "reason":           "<one sentence explaining your verdict>",
  "prompt_quality":   <1-5>,
  "improved_prompt":  "<imperative rewrite of the original prompt, always fill>",
  "improved_after_desc": "<concise description of the AFTER object, always fill>"
}

Fields:
  edit_executed    — Did the intended edit visibly happen?
  correct_region   — Was the change applied to the correct region / part?
  preserve_other   — Are all other parts / the overall structure intact?
  visual_quality   — 1=terrible  2=poor  3=acceptable  4=good  5=excellent
  artifact_free    — No floating blobs, broken geometry, seam artefacts?
  reason           — One sentence justifying your overall verdict.
  prompt_quality   — How precisely does the ORIGINAL prompt describe what actually
                     happened? 1=wrong/missing  2=vague  3=roughly correct
                                4=accurate  5=precise
  improved_prompt  — Rewrite the prompt to exactly match what you observe.
                     Imperative ("Remove the ...", "Change ... to ...").
  improved_after_desc — Concise description of the complete AFTER object.
"""

# ── Per-type system prompts ───────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT_DELETION = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

EDIT TYPE: deletion
A deletion edit removes one or more named parts from the object.
The AFTER model should have those parts entirely gone, with the remaining
geometry intact and the removal site clean.

WHAT TO JUDGE:
  edit_executed   — Is the target part GONE from the AFTER model?
                    true if the part is absent; false if it is still present.
  correct_region  — Was the CORRECT part removed (matching the target label
                    and instruction), not some other part?
  preserve_other  — Are ALL remaining parts intact? No floating fragments,
                    no gaps, no broken surfaces left at the removal site?
  visual_quality  — Overall quality of the AFTER model.
                    Penalise holes, seam artefacts, distorted remaining parts.
  artifact_free   — No floating blobs, stray geometry, or jagged seams
                    at the removal site?
""" + _JUDGE_OUTPUT_SCHEMA

JUDGE_SYSTEM_PROMPT_MODIFICATION = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

EDIT TYPE: modification
A modification edit changes the SHAPE, SILHOUETTE, or FUNCTIONAL ROLE of a
specific part — NOT its colour or surface material.
The AFTER model should show a visibly different form for that part while all
other parts remain geometrically unchanged.

WHAT TO JUDGE:
  edit_executed   — Is the target part's SHAPE or FORM visibly different in AFTER?
                    Return false if only colour/texture changed, or if no change
                    is visible at all.
  correct_region  — Was the shape change applied to the correct part (as named
                    in the instruction), not to some neighbouring part?
  preserve_other  — Are all OTHER parts unchanged in shape and position?
                    Minor shading differences due to lighting are acceptable;
                    geometry must be intact.
  visual_quality  — Quality of the modified AFTER model.
                    Penalise implausible geometry, broken meshes, or shape
                    changes that contradict the instruction.
  artifact_free   — No floating blobs, intersecting geometry, or broken normals?
""" + _JUDGE_OUTPUT_SCHEMA

JUDGE_SYSTEM_PROMPT_SCALE = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

EDIT TYPE: scale
A scale edit SHRINKS a specific part while keeping all other geometry unchanged.
The AFTER model should show the target part noticeably smaller, still attached
and positioned correctly, with no floating connectors.

WHAT TO JUDGE:
  edit_executed   — Is the target part visibly SMALLER in AFTER than in BEFORE?
                    Return false if the size is unchanged or the part grew.
  correct_region  — Was the CORRECT part scaled, not a neighbouring part?
  preserve_other  — Are all OTHER parts at their original size and position?
  visual_quality  — Quality of the scaled AFTER model.
                    Penalise disproportionate scaling, floating connectors,
                    or visible seams where the part was detached.
  artifact_free   — No floating or disconnected geometry after scaling?
""" + _JUDGE_OUTPUT_SCHEMA

JUDGE_SYSTEM_PROMPT_MATERIAL = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

EDIT TYPE: material
A material edit changes the SURFACE MATERIAL or FINISH of a specific part
(e.g. wood → steel, plastic → glass) while leaving the geometry entirely
unchanged. Only texture/shading should differ between BEFORE and AFTER.

WHAT TO JUDGE:
  edit_executed   — Is the surface material / texture of the target part
                    visibly different in AFTER?
  correct_region  — Did the material change happen on the CORRECT part only,
                    with no spillover onto adjacent parts?
  preserve_other  — Is the geometry of ALL parts intact — no shape changes,
                    no new holes, no parts removed?
  visual_quality  — Quality of the material change.
                    Does it look like the expected material? Is the
                    shading / texture plausible and consistent?
  artifact_free   — No UV seams, tiling artefacts, or broken surfaces?
""" + _JUDGE_OUTPUT_SCHEMA

JUDGE_SYSTEM_PROMPT_COLOR = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

EDIT TYPE: color
A colour edit changes the HUE or SHADE of a specific part while leaving its
shape and material type unchanged. Only the colour should differ between BEFORE
and AFTER.

WHAT TO JUDGE:
  edit_executed   — Is the target part's colour visibly different in AFTER —
                    a clear hue or tonal shift from BEFORE?
                    Return false if the part looks the same colour.
  correct_region  — Did the colour change apply to the CORRECT part only?
                    Return false if the wrong part changed colour, or if colour
                    bled noticeably onto neighbouring parts.
  preserve_other  — Is the GEOMETRY of ALL parts intact — no shape changes,
                    no holes, no parts added or removed?
                    Minor lighting/shading differences from the colour change
                    are acceptable.
  visual_quality  — Quality of the colour change.
                    Does the new colour look natural on the part? Is it
                    consistent across all visible faces in all 5 views?
  artifact_free   — No colour bleeding, hard colour edges, or rendering seams
                    at the boundary of the recoloured region?
""" + _JUDGE_OUTPUT_SCHEMA

JUDGE_SYSTEM_PROMPT_GLOBAL = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

EDIT TYPE: global
A global edit changes the ENTIRE OBJECT's artistic or rendering aesthetic
(e.g. cel-shading, Art Deco, weathered metal, low-poly) — NOT a material or
colour change to individual parts. The overall form and part structure should
remain recognisable while the visual style changes uniformly.

WHAT TO JUDGE:
  edit_executed   — Is the overall rendering / art style of the object visibly
                    changed in AFTER? Return false if the object looks identical
                    to BEFORE.
  correct_region  — Is the style applied CONSISTENTLY across the ENTIRE object,
                    with no partial areas remaining in the original style?
  preserve_other  — Is the object's underlying GEOMETRY and STRUCTURE still
                    recognisable — same number of parts, overall form preserved?
                    Return false if geometry was destroyed or heavily distorted.
  visual_quality  — How well does the style transfer look?
                    Penalise inconsistent style application, artefacts, or a
                    style that does not match the target description.
  artifact_free   — No floating geometry, broken normals, or severe rendering
                    artefacts introduced by the style transfer?
""" + _JUDGE_OUTPUT_SCHEMA

JUDGE_SYSTEM_PROMPT_ADDITION = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

EDIT TYPE: addition
An addition edit ADDS a new physical element to the object. The AFTER model
should contain a new part that was not present in BEFORE, placed in a
reasonable position and blending naturally with the original object.

WHAT TO JUDGE:
  edit_executed   — Is there a NEW element in AFTER that was NOT present in
                    BEFORE?
  correct_region  — Is the new element placed in a REASONABLE position —
                    attached or adjacent to the correct existing part, not
                    floating arbitrarily in space?
  preserve_other  — Are all ORIGINAL parts still present and intact?
  visual_quality  — Quality of the addition.
                    Does it blend naturally with the original object's style
                    and scale? Does it match the description in the instruction?
  artifact_free   — No interpenetrating geometry, floating blobs, or broken
                    surfaces at the attachment point?
""" + _JUDGE_OUTPUT_SCHEMA

# Fallback for unknown / future edit types
JUDGE_SYSTEM_PROMPT_GENERIC = """You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
  TOP row    — BEFORE the edit (original object, 5 views)
  BOTTOM row — AFTER the edit  (edited object,   5 views)

Judge whether the described edit was executed correctly and cleanly.

WHAT TO JUDGE:
  edit_executed   — Did the described edit visibly happen?
  correct_region  — Was the change applied to the correct region / part?
  preserve_other  — Are all other parts preserved and intact?
  visual_quality  — Overall quality of the AFTER model (1–5).
  artifact_free   — No floating blobs or broken surfaces?
""" + _JUDGE_OUTPUT_SCHEMA

# Lookup table: edit_type (lowercase) → system prompt string
JUDGE_SYSTEM_PROMPTS: dict = {
    "deletion":     JUDGE_SYSTEM_PROMPT_DELETION,
    "modification": JUDGE_SYSTEM_PROMPT_MODIFICATION,
    "scale":        JUDGE_SYSTEM_PROMPT_SCALE,
    "material":     JUDGE_SYSTEM_PROMPT_MATERIAL,
    "color":        JUDGE_SYSTEM_PROMPT_COLOR,
    "global":       JUDGE_SYSTEM_PROMPT_GLOBAL,
    "addition":     JUDGE_SYSTEM_PROMPT_ADDITION,
}


def build_judge_user_prompt(
    edit_type: str,
    edit_prompt: str,
    object_desc: str,
    part_label: str,
    target_part_desc: str = "",
    edit_params: "dict | None" = None,
) -> str:
    """Build the per-edit USER message for the quality judge VLM call.

    The system message is looked up from ``JUDGE_SYSTEM_PROMPTS[edit_type]``
    (or the generic fallback).  This function only assembles the small
    per-edit context: object description, instruction, target info.

    Args:
        edit_type:        PartCraft canonical type string.
        edit_prompt:      Original phase-1 generation prompt.
        object_desc:      Full object description from parsed.json.
        part_label:       Human-readable label(s) of the target part(s).
        target_part_desc: Visual description of the part before editing.
        edit_params:      Extra fields (target_color, target_material,
                          target_style, new_part_desc, factor …).
    """
    ep = edit_params or {}
    et = edit_type.lower()
    lines = [
        f"Object: {object_desc}",
        f'Edit instruction: "{edit_prompt}"',
        f"Target part: {part_label}",
    ]
    if target_part_desc:
        lines.append(f'Target part description (before): "{target_part_desc}"')
    # Type-specific extra context
    if et == "modification" and ep.get("new_part_desc"):
        lines.append(f'Expected shape after: "{ep["new_part_desc"]}"')
    elif et == "scale" and ep.get("factor"):
        lines.append(f'Scale factor (shrink): {ep["factor"]}')
    elif et == "material" and ep.get("target_material"):
        lines.append(f'Expected material/finish: "{ep["target_material"]}"')
    elif et == "color" and ep.get("target_color"):
        lines.append(f'Expected colour: "{ep["target_color"]}"')
    elif et == "global" and ep.get("target_style"):
        lines.append(f'Expected style: "{ep["target_style"]}"')
    return "\n".join(lines)


def parse_judge_output(raw: str) -> "dict | None":
    """Extract the quality judge JSON from a VLM response string.

    Returns a dict with at least the required fields, or None on failure.
    """
    required = {"edit_executed", "visual_quality", "reason"}
    result = extract_judge_json(raw)
    if result is None:
        return None
    if not required.issubset(result.keys()):
        return None
    return result


# ---------------------------------------------------------------------------
# JSON extractor (self-contained — no vlm_filter dependency)
# ---------------------------------------------------------------------------

def _judge_balanced_brace(s: str, start: int) -> "str | None":
    """Return the balanced JSON-object substring starting at s[start], or None."""
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth, in_string, escape = 0, False, False
    for j in range(start, len(s)):
        c = s[j]
        if escape:
            escape = False; continue
        if in_string:
            if c == "\\": escape = True
            elif c == '"': in_string = False
        else:
            if c == '"': in_string = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:j + 1]
    return None


def extract_judge_json(content: str) -> "dict | None":
    """Extract the first valid JSON object from a VLM response string."""
    import json as _json
    required = {"edit_executed", "visual_quality", "reason"}
    candidates = []
    for i, c in enumerate(content):
        if c == "{":
            sub = _judge_balanced_brace(content, i)
            if sub:
                candidates.append(sub)
    for blob in candidates:
        try:
            obj = _json.loads(blob)
            if required.issubset(obj.keys()):
                return obj
        except Exception:
            pass
    # last-resort: grab last {...} in content
    last = content.rfind("{")
    if last >= 0:
        sub = _judge_balanced_brace(content, last)
        if sub:
            try:
                obj = _json.loads(sub)
                if required.issubset(obj.keys()):
                    return obj
            except Exception:
                pass
    return None


async def _qe_call_vlm_judge(
    client,
    model: str,
    img_bytes: bytes,
    edit_prompt: str,
    edit_type: str,
    object_desc: str,
    part_label: str,
    target_part_desc: str = "",
    edit_params: "dict | None" = None,
    max_retries: int = 4,
    max_tokens: int = 1024,
) -> "dict | None":
    """Async VLM judge call with retry.

    Uses ``JUDGE_SYSTEM_PROMPTS[edit_type]`` as the system message and
    ``build_judge_user_prompt(...)`` to build the per-edit user message,
    mirroring the alignment-gate pattern.
    """
    b64 = _base64.b64encode(img_bytes).decode("utf-8")
    system_prompt = JUDGE_SYSTEM_PROMPTS.get(
        edit_type.lower(), JUDGE_SYSTEM_PROMPT_GENERIC
    )
    user_text = build_judge_user_prompt(
        edit_type, edit_prompt, object_desc, part_label,
        target_part_desc=target_part_desc,
        edit_params=edit_params or {},
    )
    strict_suffix = (
        "\n\nIf you already wrote analysis above, IGNORE it for the parser: "
        "output ONE new line that is ONLY the JSON object, starting with { and ending with }."
    )

    for attempt in range(max_retries + 1):
        text = user_text + (strict_suffix if attempt > 0 else "")
        sys_msg = system_prompt
        if attempt > 0:
            sys_msg += "\n\nYour reply must be a single JSON object; no chain-of-thought."
        try:
            create_kw: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": text},
                    ]},
                ],
                temperature=0.1, max_tokens=max_tokens, timeout=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            try:
                resp = await client.chat.completions.create(**create_kw)
            except TypeError:
                create_kw.pop("extra_body", None)
                resp = await client.chat.completions.create(**create_kw)

            content = resp.choices[0].message.content
            if not content:
                if attempt < max_retries:
                    await _asyncio.sleep(2 * (attempt + 1))
                continue
            result = extract_judge_json(content)
            if result is not None:
                return result
        except Exception as e:
            _LOG_QE.warning("[gate_quality] VLM error (attempt %d/%d): %s",
                            attempt + 1, max_retries + 1, e)
        if attempt < max_retries:
            await _asyncio.sleep(2 * (attempt + 1))
    return None


async def _qe_judge_one(
    client,
    vlm_model: str,
    edit_id: str,
    edit_type: str,
    prompt: str,
    obj_desc: str,
    part_label: str,
    before_imgs: list,
    edit_dir,
    thresholds: dict,
    ctx,
    target_part_desc: str = "",
    edit_params: "dict | None" = None,
    swap_collage: bool = False,
) -> "tuple[bool, int, int]":
    """Judge one edit.  Returns ``(ok, n_pass_delta, n_fail_delta)``.

    *swap_collage=True* is used for addition edits: the preview stores the
    before-addition state (object minus part) while *before_imgs* from the
    images NPZ represents the after-addition target (original complete object).
    """
    from .qc_io import update_edit_gate as _update_edit_gate
    from .edit_status_io import update_edit_stage as _update_edit_stage
    after_imgs = _qe_load_after_previews(edit_dir)
    if after_imgs is None:
        _update_edit_gate(ctx, edit_id, edit_type, "E",
                          vlm_result={"pass": False, "score": 0.0,
                                      "reason": "missing_previews"})
        _update_edit_stage(ctx, edit_id, edit_type, "gate_e", status="fail")
        return False, 0, 1

    coll = _qe_make_collage(after_imgs, before_imgs) if swap_collage else            _qe_make_collage(before_imgs, after_imgs)
    if coll is None:
        _update_edit_gate(ctx, edit_id, edit_type, "E",
                          vlm_result={"pass": False, "score": 0.0,
                                      "reason": "collage_failed"})
        _update_edit_stage(ctx, edit_id, edit_type, "gate_e", status="fail")
        return False, 0, 1

    j = await _qe_call_vlm_judge(
        client, vlm_model, coll,
        edit_prompt=prompt, edit_type=edit_type,
        object_desc=obj_desc, part_label=part_label,
        target_part_desc=target_part_desc, edit_params=edit_params,
    )
    if j is None:
        _update_edit_gate(ctx, edit_id, edit_type, "E",
                          vlm_result={"pass": False, "score": 0.0,
                                      "reason": "vlm_no_response"})
        _update_edit_stage(ctx, edit_id, edit_type, "gate_e", status="fail")
        return False, 0, 1

    ok = _qe_passes(j, edit_type, thresholds)
    _update_edit_gate(ctx, edit_id, edit_type, "E",
                      vlm_result={"pass": ok,
                                  "score": round(j.get("visual_quality", 0) / 5.0, 2),
                                  "reason": j.get("reason", "")})
    _update_edit_stage(ctx, edit_id, edit_type, "gate_e",
                       status="pass" if ok else "fail")
    return ok, (1 if ok else 0), (0 if ok else 1)


async def _qe_process_one(
    ctx,
    vlm_url: str,
    vlm_model: str,
    thresholds: dict,
    force: bool,
    log: "logging.Logger",
) -> dict:
    """Judge all edits for one object."""
    from .specs import iter_all_specs as _iter_all_specs
    from .status import update_step as _update_step, STATUS_OK as _STATUS_OK, step_done as _step_done
    from .qc_io import is_edit_qc_failed as _is_edit_qc_failed
    if not force and _step_done(ctx, "sq3_qc_E"):
        return {"obj_id": ctx.obj_id, "skipped": True}

    n_pass = n_fail = n_skip = 0
    before_imgs = _qe_load_before_imgs(ctx)
    if before_imgs is None:
        log.warning("[gate_quality] %s: cannot load before images from image_npz",
                    ctx.obj_id)
        _update_step(ctx, "sq3_qc_E", status=_STATUS_OK,
                     n_pass=0, n_fail=0, n_skip=0,
                     reason="missing_image_npz")
        return {"obj_id": ctx.obj_id, "n_pass": 0, "n_fail": 0}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")

    for spec in _iter_all_specs(ctx):
        if not force and _is_edit_qc_failed(ctx, spec.edit_id):
            n_skip += 1
            continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        _, dp, df = await _qe_judge_one(
            client, vlm_model,
            spec.edit_id, spec.edit_type,
            spec.prompt, spec.object_desc,
            ", ".join(spec.part_labels),
            before_imgs, edit_dir, thresholds, ctx,
            target_part_desc=spec.target_part_desc,
            edit_params=spec.edit_params,
        )
        n_pass += dp; n_fail += df

    for add_id, meta in _qe_iter_add_edits(ctx):
        if not force and _is_edit_qc_failed(ctx, add_id):
            n_skip += 1
            continue
        _, dp, df = await _qe_judge_one(
            client, vlm_model,
            add_id, "addition",
            meta.get("prompt", ""),
            meta.get("object_desc", ""),
            ", ".join(meta.get("part_labels", [])),
            before_imgs, ctx.edit_3d_dir(add_id), thresholds, ctx,
            target_part_desc=meta.get("target_part_desc", ""),
            edit_params=meta.get("edit_params", {}),
            swap_collage=True,
        )
        n_pass += dp; n_fail += df

    _update_step(ctx, "sq3_qc_E", status=_STATUS_OK,
                 n_pass=n_pass, n_fail=n_fail, n_skip=n_skip)
    log.info("[gate_quality] %s done: pass=%d fail=%d skip=%d",
             ctx.obj_id, n_pass, n_fail, n_skip)
    return {"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail}


async def run_gate_quality(
    ctxs,
    *,
    vlm_urls: "list[str]",
    vlm_model: str,
    cfg: dict,
    force: bool = False,
    concurrency: int = 8,
    logger: "logging.Logger | None" = None,
) -> "list[dict]":
    """Async entry point for the final quality gate (gate_quality / gate_e).

    Distributes objects across *vlm_urls* round-robin.  For each object:

    1. Loads 5 before-state views from *ctx.image_npz* at ``VIEW_INDICES``.
    2. Loads 5 after-state ``preview_{0..4}.png`` from each edit's 3D output dir.
    3. Builds a 2-row × 5-col collage and calls the VLM judge.
    4. Records the result in ``edit_status.json`` (``gate_e`` field).

    Args:
        vlm_urls:    One URL per VLM server (round-robin across objects).
        concurrency: Max simultaneous objects being judged.
        cfg:         Pipeline config dict (for ``qc.thresholds_by_type``).
    """
    if not vlm_urls:
        raise ValueError("vlm_urls must not be empty")
    log = logger or _LOG_QE
    thresholds = (cfg.get("qc") or {}).get("thresholds_by_type") or _QE_DEFS
    ctxs = list(ctxs)
    sem = _asyncio.Semaphore(concurrency)

    async def _run_one(i: int, ctx: "_ObjectContext") -> dict:
        async with sem:
            return await _qe_process_one(
                ctx, vlm_urls[i % len(vlm_urls)], vlm_model, thresholds, force, log
            )

    return await _asyncio.gather(*[_run_one(i, c) for i, c in enumerate(ctxs)])


# ============================================================================
# Alignment gate runner (gate_text_align / gate_A)
# ============================================================================
# Implements the per-edit VLM alignment gate that runs after gen_edits.
# For each edit in parsed.json:
#   1. Rule checks (check_rules)
#   2. Pixel-count visibility check (count_part_pixels_in_overview)
#   3. Build 5x2 gate image (RGB top row + red/grey highlight bottom row)
#   4. Call VLM with SYSTEM_PROMPT_ALIGN_GATE
#   5. Write gate_a: pass/fail to edit_status.json
#
# Public entry point: run_gate_text_align(ctxs, *, vlm_urls, vlm_model, ...)
# ============================================================================

_RED_BGR  = (45, 45, 220)   # selected parts colour in gate image
_GREY_BGR = (65, 65, 65)    # non-selected parts colour in gate image

_LOG_GTA = logging.getLogger("pipeline_v3.gate_text_align")


def _gta_extract_cell(
    img: "np.ndarray", col: int, row: int,
    n_views: int, col_sep: int, row_sep: int,
) -> "np.ndarray":
    """Extract one cell from the 5x2 overview grid."""
    H_total, W_total = img.shape[:2]
    W_cell = (W_total - (n_views - 1) * col_sep) // n_views
    H_cell = (H_total - row_sep) // 2
    x0 = col * (W_cell + col_sep)
    y0 = row * (H_cell + row_sep)
    return img[y0: y0 + H_cell, x0: x0 + W_cell].copy()


def _gta_highlight_cell(
    cell: "np.ndarray",
    selected_part_ids: "set[int]",
    palette_bgr: list,
) -> "np.ndarray":
    """Recolour bottom-row palette cell: selected parts -> red, rest -> grey."""
    import numpy as np
    pal = np.array(palette_bgr, dtype=np.int32)
    flat = cell.reshape(-1, 3).astype(np.int32)
    diffs = np.linalg.norm(flat[:, None, :] - pal[None, :, :], axis=2)
    nearest = np.argmin(diffs, axis=1)
    is_bg = np.all(flat > 230, axis=1)
    n_pal = len(pal)
    sel_slots = {pid % n_pal for pid in selected_part_ids}
    is_sel = np.array([idx in sel_slots for idx in nearest])
    out = np.empty_like(flat)
    out[is_bg] = [255, 255, 255]
    out[~is_bg & is_sel] = list(_RED_BGR)
    out[~is_bg & ~is_sel] = list(_GREY_BGR)
    return out.reshape(cell.shape).astype(np.uint8)


def build_gate_image(
    ov_img: "np.ndarray",
    selected_part_ids: "list[int]",
    column_map: "list[int]",
) -> bytes:
    """Build the VLM gate image from an overview BGR image.

    Non-global edits (selected_part_ids non-empty):
        5x2 grid — top row = RGB photos, bottom row = highlight renders
        (selected parts RED, all others GREY, background WHITE).
    Global edits (selected_part_ids empty):
        5x1 strip — RGB photos only; VLM picks best view from photos alone.

    Parameters
    ----------
    ov_img:
        BGR overview image as returned by ``cv2.imdecode``.
    selected_part_ids:
        Part IDs that are the target of the edit.  Empty for global edits.
    column_map:
        List of 5 column indices into the overview (which overview view goes
        into each gate-image column).  col 0 = highest-visibility view.
    """
    import cv2
    import numpy as np
    from .qc_rules import _PALETTE_BGR, _N_VIEWS, _COL_SEP, _ROW_SEP

    sel_set = set(selected_part_ids)

    def _hstack(cells: list) -> "np.ndarray":
        row = cells[0]
        for c in cells[1:]:
            sep = np.full((c.shape[0], _COL_SEP, 3), 255, dtype=np.uint8)
            row = np.concatenate([row, sep, c], axis=1)
        return row

    top_cells = [
        _gta_extract_cell(ov_img, v, 0, _N_VIEWS, _COL_SEP, _ROW_SEP)
        for v in column_map
    ]

    if not sel_set:
        full = _hstack(top_cells)
    else:
        bot_cells = [
            _gta_highlight_cell(
                _gta_extract_cell(ov_img, v, 1, _N_VIEWS, _COL_SEP, _ROW_SEP),
                sel_set, _PALETTE_BGR,
            )
            for v in column_map
        ]
        total_w = sum(c.shape[1] for c in top_cells) + (_N_VIEWS - 1) * _COL_SEP
        sep_h = np.full((_ROW_SEP, total_w, 3), 255, dtype=np.uint8)
        full = np.concatenate([_hstack(top_cells), sep_h, _hstack(bot_cells)], axis=0)

    ok, buf = cv2.imencode(".png", full)
    assert ok, "cv2.imencode failed building gate image"
    return buf.tobytes()


async def _gta_process_one(
    ctx,
    vlm_url: str,
    vlm_model: str,
    force: bool,
    log: "logging.Logger",
) -> dict:
    """Run gate_text_align for one object.

    Reads phase1/parsed.json and overview.png, runs per-edit alignment gate,
    writes gate_a results to edit_status.json.
    """
    import cv2
    import numpy as np
    from .qc_rules import check_rules, count_part_pixels_in_overview, _N_VIEWS
    from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP, step_done
    from .qc_io import update_edit_gate
    from .edit_status_io import update_edit_stage
    from openai import AsyncOpenAI

    if not force and step_done(ctx, "sq1_qc_A"):
        return {"obj_id": ctx.obj_id, "skipped": True}

    # If phase1 was skipped (too many parts) — nothing to gate
    status_path = ctx.status_path
    if status_path.is_file():
        try:
            import json as _json
            _st = _json.loads(status_path.read_text())
            if _st.get("steps", {}).get("s1_phase1", {}).get("status") == "skip":
                update_step(ctx, "sq1_qc_A", status=STATUS_SKIP, reason="s1_skipped")
                return {"obj_id": ctx.obj_id, "skipped": True}
        except Exception:
            pass

    if not ctx.parsed_path.is_file():
        update_step(ctx, "sq1_qc_A", status=STATUS_FAIL, error="missing_parsed_json")
        return {"obj_id": ctx.obj_id, "error": "missing_parsed_json"}

    try:
        import json as _json
        raw = _json.loads(ctx.parsed_path.read_text())
    except Exception as exc:
        update_step(ctx, "sq1_qc_A", status=STATUS_FAIL,
                    error=f"corrupt_parsed_json: {exc}")
        return {"obj_id": ctx.obj_id, "error": "corrupt_parsed_json"}

    obj = (raw.get("parsed") or {}).get("object") or {}
    parts_by_id = {
        p["part_id"]: p
        for p in (obj.get("parts") or [])
        if isinstance(p, dict) and "part_id" in p
    }
    edits = (raw.get("parsed") or {}).get("edits") or []

    # Load overview BGR image (may be absent → auto-pass path)
    ov_img: "np.ndarray | None" = None
    if ctx.overview_path.is_file():
        buf = np.frombuffer(ctx.overview_path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is not None:
            ov_img = decoded

    from partcraft.edit_types import FLUX_TYPES as _FLUX_TYPES

    client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
    n_pass = n_fail = 0

    # Use the same flux_seq / del_seq counters as iter_all_specs in specs.py
    # so that gate_a is written to the canonical edit_id used by s4/s5.
    flux_seq = 0
    del_seq  = 0

    for idx, edit in enumerate(edits):
        et     = edit.get("edit_type", "unknown")

        # Mirror iter_all_specs: skip identity and addition edits; assign seq.
        if et in _FLUX_TYPES:
            seq = flux_seq; flux_seq += 1
        elif et == "deletion":
            seq = del_seq; del_seq += 1
        else:
            continue  # identity / addition / unknown — not gated here

        edit_id = ctx.edit_id(et, seq)  # canonical id matching specs.py
        prompt = edit.get("prompt", "")
        sel    = list(edit.get("selected_part_ids") or [])

        # ── Layer 1: rule checks ──────────────────────────────────────
        rule_fails = check_rules(edit, parts_by_id)
        rule_result = {"pass": not rule_fails, "checks": rule_fails}

        if rule_fails:
            update_edit_gate(ctx, edit_id, et, "A",
                             rule_result=rule_result)
            update_edit_stage(ctx, edit_id, et,
                              "gate_a", status="fail")
            n_fail += 1
            continue

        # ── No overview: auto-pass ────────────────────────────────────
        if ov_img is None:
            update_edit_gate(ctx, edit_id, et, "A",
                             rule_result=rule_result,
                             vlm_result={"pass": True, "score": 1.0,
                                         "reason": "no_overview_auto_pass",
                                         "best_view": 0})
            update_edit_stage(ctx, edit_id, et,
                              "gate_a", status="pass")
            n_pass += 1
            continue

        # ── Layer 2: pixel visibility (non-deletion, non-global) ──────
        px = [count_part_pixels_in_overview(ov_img, v, sel)
              for v in range(_N_VIEWS)]
        best_col_view = int(np.argmax(px)) if sel else 0
        column_map = [best_col_view, 0, 1, 2, 3]

        # Skip if zero pixels for non-deletion edits with parts
        if (sel and et != "deletion"
                and all(p <= 0 for p in px)):
            update_edit_gate(ctx, edit_id, et, "A",
                             rule_result=rule_result,
                             vlm_result={"pass": False, "score": 0.0,
                                         "reason": "zero_visible_pixels",
                                         "best_view": 0,
                                         "pixel_counts": px,
                                         "column_map": column_map})
            update_edit_stage(ctx, edit_id, et,
                              "gate_a", status="fail")
            n_fail += 1
            continue

        # ── Layer 3: VLM alignment gate ───────────────────────────────
        try:
            gate_img  = build_gate_image(ov_img, sel, column_map)
            gate_user = build_align_gate_user_prompt(et, prompt, sel)
            gate_raw  = await call_vlm_async(
                client, gate_img,
                SYSTEM_PROMPT_ALIGN_GATE, gate_user,
                vlm_model, max_tokens=256,
            )
            gate_out = parse_align_gate_output(gate_raw)
        except Exception as exc:
            log.warning("[gate_text_align] %s edit %d VLM error: %s",
                        ctx.obj_id, idx, exc)
            gate_out = None

        if gate_out is None:
            gate_out = {"aligned": False, "reason": "vlm_error", "best_view": 0}

        aligned       = bool(gate_out.get("aligned", False))
        bv_col        = gate_out.get("best_view", 0)
        if not (type(bv_col) is int and 0 <= bv_col < 5):
            bv_col = 0
        best_view_abs = column_map[bv_col]

        update_edit_gate(ctx, edit_id, et, "A",
                         rule_result=rule_result,
                         vlm_result={
                             "pass":         aligned,
                             "score":        1.0 if aligned else 0.0,
                             "reason":       gate_out.get("reason", ""),
                             "best_view":    best_view_abs,
                             "best_view_col": bv_col,
                             "pixel_counts": px,
                             "column_map":   column_map,
                         })
        update_edit_stage(ctx, edit_id, et,
                          "gate_a", status="pass" if aligned else "fail")
        if aligned:
            n_pass += 1
        else:
            n_fail += 1

    update_step(ctx, "sq1_qc_A", status=STATUS_OK,
                n_pass=n_pass, n_fail=n_fail)
    log.info("[gate_text_align] %s done: pass=%d fail=%d",
             ctx.obj_id, n_pass, n_fail)
    return {"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail}


async def run_gate_text_align(
    ctxs: "Iterable",
    *,
    vlm_urls: "list[str]",
    vlm_model: str,
    force: bool = False,
    concurrency: int = 8,
    logger: "logging.Logger | None" = None,
) -> "list[dict]":
    """Async entry point for the alignment gate (gate_text_align / gate_A).

    For each object reads phase1/parsed.json, runs rule checks + pixel
    visibility + VLM alignment gate, and writes gate_a results to
    edit_status.json.

    This is the replacement for the deleted ``sq1_qc_a.py``.

    Args:
        vlm_urls:    One URL per VLM server (round-robin across objects).
        concurrency: Max simultaneous objects being gated.
    """
    if not vlm_urls:
        raise ValueError("vlm_urls must not be empty")
    log = logger or _LOG_GTA
    ctxs = list(ctxs)
    sem = _asyncio.Semaphore(concurrency)

    async def _run_one(i: int, ctx) -> dict:
        async with sem:
            return await _gta_process_one(
                ctx, vlm_urls[i % len(vlm_urls)], vlm_model, force, log
            )

    return await _asyncio.gather(*[_run_one(i, c) for i, c in enumerate(ctxs)])

