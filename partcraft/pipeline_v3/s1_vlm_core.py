"""Core VLM helpers for Phase 1 v2 (one-shot edit generation).

Previously lived in ``scripts/standalone/run_phase1_v2.py`` and was
imported via a ``sys.path`` hack. Moved here so it can be imported as a
proper package module.

Exports used by :mod:`partcraft.pipeline_v2.s1_phase1_vlm`:
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, MAX_PARTS,
    build_part_menu, render_overview_png,
    call_vlm_async, extract_json_object, validate, quota_for
"""
from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from partcraft.render.overview import (
    VIEW_INDICES, _PALETTE, _PALETTE_NAMES,
    extract_parts, load_views_from_npz, run_blender, stitch_two_rows,
)

SYSTEM_PROMPT = """You are a 3D Spatial Reasoning Engine generating a JSON dataset for 3D part editing. \
You are given a 5x2 grid: TOP row = 5 RGB photos of one 3D object from 5 cameras; \
BOTTOM row = the same 5 cameras re-rendered with each editable part in a fixed \
palette color.

CRITICAL: maintain 3D OBJECT-SPACE consistency. An object's anatomical "left" is \
a fixed physical part — it does NOT change just because the camera moved it to \
the image-right. Whenever you write a directional word, you MUST first reason \
about which camera you are looking through and apply the mirror rule explicitly.

Output ONE valid JSON object — no prose, no markdown. Begin with '{' and end with '}'."""


USER_PROMPT_TEMPLATE = """[Image: 5×2 grid. TOP row = 5 RGB photos. BOTTOM row =
same 5 cameras re-rendered with each editable part in a fixed palette color
(same column = same camera). The palette colors are INTERNAL labels for you
only — they are NOT properties of the real object.]

# CAMERA GEOMETRY (fixed for every object — memorize this)

Views 0,1,2,3 form a horizontal ring around the object at 90° yaw increments,
all looking slightly downward (elev ≈ +30°):
  • view (k+2) mod 4  is the 180° OPPOSITE of view k  (back-to-back cameras)
  • view (k+1) mod 4  is the 90° rotation of view k   (perpendicular / profile)
View 4 is a low camera looking UP from below the object.

This geometry is intrinsic to the cameras, NOT to the object. The object itself
has an arbitrary world orientation — you must decide which camera happens to
face the object's front by LOOKING at the photos.

Parts (id, palette color in bottom row, cluster_size):
{part_menu}

# CORE LOGIC: THE 3D SPATIAL RULEBOOK

P1. ID-FIRST TRUTH. selected_part_ids are the absolute ground truth on which
    part is being edited. Your text MUST visually match the highlighted parts.
    If unsure what a part is, describe its shape/structure, do not guess a name.

P2. ANTI-VIEWPORT RULE. Bare directional words ("left", "right", "front",
    "back", "leftmost", "rightmost") are FORBIDDEN as standalone descriptors.
    Use either:
      (a) view-invariant cues — shape, size, function, or structural relation
          to another part ("the ear above the raised paw", "the wheel under
          the driver seat"), OR
      (b) the object-anatomical form defined in P3+P4 below.

P3. CANONICAL FRONT (mandatory field).
    object.canonical_front: ONE structural sentence describing what visually
        marks the object's intrinsic forward direction (e.g. "the side with
        the snout and eyes", "the side with the headlights and windshield",
        "the side where you sit on the chair"), OR null if the object has no
        unambiguous orientation (sphere, vase, symmetric drum, etc.).
    object.frontal_view_index: int in [0..4], the view_index whose camera
        most directly faces canonical_front (the camera you would describe
        as "looking the object in the face"). Set to null iff
        canonical_front is null.

    When canonical_front is null, NO directional words at all (P4 disabled).

P4. MIRROR RULE (only valid when canonical_front and frontal_view_index are set).
    Let F = frontal_view_index. Then because the camera and the object are
    facing each other:
      • In view F   : image-LEFT half = object's anatomical RIGHT side,
                       image-RIGHT half = object's anatomical LEFT side. (MIRROR)
      • In view (F+2) mod 4 (back view): image-side = object-side. (NO MIRROR)
      • In view (F±1) mod 4 (profile views): the object is sideways — you
        CANNOT read anatomical left/right from these views. Do NOT pick a
        profile view as view_index for any edit that uses left/right.
      • View 4 (bottom-up): also unreliable for left/right; use other cues.

    So any anatomical-left/right edit MUST have view_index ∈ {{F, (F+2) mod 4}}.

    EVERY use of an anatomical "left" / "right" in prompt or target_part_desc
    MUST be tagged with "(object's anatomical left/right)". The rationale
    field MUST cite the mirror reasoning explicitly, e.g.:
       "frontal_view_index=1; target visible in view 1 on the image-RIGHT
        half → mirror → object's anatomical LEFT ear."

P5. LEVEL HIERARCHY. The part menu shows ``level=N`` for each part (lower N
    = closer to the object root). Parts at the same level are siblings in the
    part hierarchy. Use level to reason about part relationships — e.g. parts
    at the same level may be symmetric instances (wheels, legs, wings).


P6. NO PALETTE COLORS. The palette names (red, orange, yellow, lime, green,
    teal, cyan, blue, navy, purple, magenta, pink, brown, tan, black, gray)
    are INTERNAL labels and MUST NOT appear in any output text field. To
    describe real color, use the appearance from the TOP row photos
    ("the dark wooden seat", "the chrome pipe").

# OUTPUT — one JSON object

object:
  full_desc            full English description of the object
  full_desc_stage1     geometry-only version (no colors/materials/finish words)
  full_desc_stage2     texture-only version (no shape/count/layout words)
  canonical_front      ONE structural sentence OR null  (see P3)
  frontal_view_index   int in [0..4] OR null            (see P3)
  parts                [{{part_id, color, name}}, ...] for every menu entry;
                       name = your short semantic label, "(artifact)" or
                       "(invisible)" for noise / unseen parts.

edits: EXACTLY {n_total} entries with these per-type counts
  - {n_deletion} deletion
  - {n_modification} modification
  - {n_scale} scale
  - {n_material} material
  - {n_color} color
  - {n_global} global       (selected_part_ids = [])

Each edit MUST list these fields IN THIS ORDER (the rationale comes FIRST so
you reason before you write the prompt):
  rationale           ONE sentence.
                      • If the edit uses anatomical left/right: MUST include
                        the literal value "frontal_view_index=N" and the
                        mirror calculation, e.g.:
                        "frontal_view_index=1; in view 1 the target ear is on
                         the image-RIGHT half → mirror → object's anatomical
                         LEFT ear (part_id 5)."
                      • For global edits: MUST name the source style category
                        (Rendering / Historical / Genre), e.g.:
                        "Choosing 'ukiyo-e woodblock print' from the
                         Historical category."
                      • For all other edits: a single short reason.
  edit_type           one of: deletion | modification | scale | material | color | global
                      DECISION: deletion=remove part · modification=shape/identity change ·
                      scale=size only · material=substance only · color=hue only · global=art style
  selected_part_ids   list of int part_ids; empty ONLY for global
  prompt              imperative starting with Remove/Delete/Add/Change/
                      Replace/Make/Scale/Resize. NO part_id numbers,
                      NO palette color names. Obeys P2/P4/P6. Any anatomical
                      left/right MUST be tagged "(object's anatomical L/R)".
  target_part_desc    short visual description of the target part(s) — same
                      forbidden-word rules as prompt.
  view_index          int in [0..4]: the view where the target is most
                      visible. If the edit uses anatomical left/right this
                      MUST equal frontal_view_index OR (frontal_view_index+2)
                      mod 4. (For global, pick the best overall view.)

# MODIFICATION EDITS — SHAPE MORPH OR FUNCTIONAL REPLACEMENT
  A modification either (a) changes a part's geometry while keeping its identity, OR
  (b) substitutes it with a completely different but logically equivalent object in
  the same structural slot. Both are valid and encouraged.

  (a) Shape morph — same functional identity, different geometry:
      • straight sword blade    →  curved saber blade
      • cylindrical barrel      →  hexagonal prism barrel
      • spherical head          →  cubic head
      • upright rabbit ears     →  floppy drooping ears
      • rectangular door panel  →  arched gothic door panel

  (b) Functional replacement — completely different object, same structural role:
      • sword blade             →  axe head
      • circular wheel          →  triangular wheel
      • vertical antenna        →  parabolic satellite dish
      • cylindrical chair leg   →  hairpin metal rod leg
      • vertical stabilizer     →  swept-back winglet
      • rectangular table top   →  circular table top

  new_part_desc MUST name the new object AND describe its key geometry, e.g.:
    "a broad wedge-shaped axe head" · "a flat triangular wheel" · "a parabolic dish"
    "hairpin-bent thin metal rod" · "a swept-back delta-shaped winglet"

  TYPE BOUNDARY — pick modification ONLY if the geometry or identity changes:
    • Changing ONLY colour?              → use "color"      (not modification)
    • Changing ONLY surface material?   → use "material"   (not modification)
    • Removing the part entirely?       → use "deletion"   (not modification)
    • Resizing without shape change?    → use "scale"      (not modification)

  STRICTLY FORBIDDEN in modification: changing only color, surface finish, or
  material. The new_part_desc MUST describe a geometry or identity change.

  edit_params         deletion: {{}}
                      modification: {{"new_part_desc": "..."}}
                      scale:        {{"factor": float in [0.3, 0.85]}}
                                    Shrink only. Prefer large/dominant parts (main body, primary limbs).
                                    Do NOT enlarge small decorative parts.
                      material:     {{"target_material": "..."}}
                                    Target must be a specific surface substance or finish, e.g.:
                                    "polished walnut wood", "brushed stainless steel",
                                    "frosted borosilicate glass", "hand-stitched leather",
                                    "poured concrete", "translucent amber resin".
                                    FORBIDDEN in target_material: style/aesthetic words
                                    (cartoon, vintage, futuristic, minimalist, steampunk,
                                    cyberpunk) — those belong in "global" edits.
                      color:        {{"target_color": "..."}}
                                    Target must be a specific, descriptive colour phrase, e.g.:
                                    "deep crimson red", "matte charcoal black", "cobalt blue",
                                    "ivory cream white", "forest green", "warm amber orange".
                                    FORBIDDEN: bare internal palette names (red, orange, lime, …)
                                    — always qualify: "vivid lime green", not "lime".
                                    Do NOT change the surface material or finish; use "material"
                                    for that.
                      global:       {{"target_style": "..."}}
  after_desc_full / after_desc_stage1 / after_desc_stage2
                      object after the edit. For deletion: ALL three null.
                      For others: all three filled, stage1 has no
                      colors/materials, stage2 has no shape changes.
  new_parts_desc / new_parts_desc_stage1 / new_parts_desc_stage2
                      modification only: describe the new replacement parts.
                      null for non-modification edits.
  confidence          "high" | "medium" | "low"

# COLOR EDITS — HUE AND SHADE CHANGES ONLY
  A color edit repaints one or more parts with a new hue or shade while keeping the
  surface material and geometry unchanged.
  Think: what color contrast or accent would improve the object?
  Examples:
    • beige seat → deep burgundy red seat
    • silver handle → matte charcoal black handle
    • white lamp shade → warm amber orange shade
  STRICTLY FORBIDDEN in color: changing surface material or finish (use "material"),
  changing geometry (use "modification"), or changing the whole object (use "global").
  Use descriptive colour phrases — never bare internal palette names.
  The new_part_desc is NOT required for color edits (there is no shape change).

# GLOBAL STYLE EDITS — ARTISTIC / RENDERING AESTHETIC ONLY

  A global edit transforms the ENTIRE object's artistic or rendering aesthetic.
  It must change how the object looks as a *visual artwork* — NOT what material
  it is made of.

  STRICTLY FORBIDDEN in global target_style:
    • Surface-material words: gold, silver, metal, wood, stone, clay, glass,
      ceramic, rubber, plastic, ice, crystal, fabric, concrete, leather.
      → Those belong in "material" edits.
    • Generic quality descriptors: "realistic", "detailed", "high quality".
    • Near-duplicate styles: "cartoon", "cartoonish", "toon" count as ONE choice.

  VALID target_style — for this object you MUST use ONLY the per-object style roster
  injected below (it is randomised per object to enforce diversity).

{global_roster}

  DIVERSITY RULE: For this object's {{n_global}} global edits, each target_style
  MUST come from a DIFFERENT category row of the roster above.
  The rationale for every global edit MUST name the source category, e.g.:
  "Choosing 'watercolour wash' from the Rendering category."

# HARD RULES (violations drop that edit)

R1. selected_part_ids ⊆ part menu ids; never target parts with cluster_size<30
    UNLESS the part appears to be the primary body of the object — infer
    valid semantic parts regardless of cluster_size);
    never target parts you cannot see in the bottom row.
R2. Each edit is distinct: no two with same edit_type AND same
    selected_part_ids.
R3. Never delete or extreme-scale a part that forms the structural body —
    the object should remain recognizable.
R4. prompt and target_part_desc must obey P2, P4, P5, P6.
R5. Non-deletion edits fill all three after_desc_*. Deletion edits set
    all three to null.
R6. view_index ∈ [0,4] and the target must be clearly visible in that view.
R7. If canonical_front is null, NO directional words anywhere; use group
    edits or structural anchors only.
R8. If an edit uses anatomical left/right, view_index ∈ {{F, (F+2) mod 4}}
    where F = frontal_view_index, and the rationale must cite the mirror
    reasoning explicitly.

# OUTPUT FORMAT

ONE JSON object. Begin with '{{', end with '}}'. No prose, no markdown."""



def build_image_semantic_menu(
    mesh_npz: Path,
    img_npz: Path,
    anno_obj_dir: "Path | None" = None,
) -> tuple[list[int], str]:
    """Semantic part menu — for image_semantic mode (Mode A).

    Columns: part_id | palette-colour | description
    The palette colour matches the colour-coded BOTTOM-row image overlay.

    Format per line:
        part_{id:<3d}   {colour}   "{description}"
    """
    z = np.load(img_npz, allow_pickle=True)
    sm = json.loads(bytes(z["split_mesh.json"]).decode())
    clusters = sm.get("valid_clusters", {})
    z2 = np.load(mesh_npz, allow_pickle=True)
    import re as _re

    def _parse_pid(k: str) -> int | None:
        m = _re.search(r"\d+", k)
        return int(m.group()) if m else None

    pids = sorted(
        pid
        for k in z2.files
        if k.startswith("part_") and (k.endswith(".glb") or k.endswith(".ply"))
        if (pid := _parse_pid(k)) is not None
    )

    # Load per-part captions from embedded part_captions.json
    part_captions: dict[int, list[str]] = {}
    if "part_captions.json" in z2.files:
        try:
            raw_caps: dict[str, list] = json.loads(bytes(z2["part_captions.json"]).decode())
            part_captions = {int(k): v for k, v in raw_caps.items()}
        except Exception:
            pass

    import re as _re2
    _ADET = _re2.compile(r'^(?:a |an |the )+', _re2.I)

    def _caption(pid: int) -> str:
        caps = part_captions.get(pid, [])
        if caps and isinstance(caps[0], str) and caps[0].strip():
            s = caps[0].strip().rstrip(".")
            return _ADET.sub("", s).strip() or f"part_{pid}"
        return f"part_{pid}"

    lines = []
    for pid in pids:
        color = _PALETTE_NAMES[pid % len(_PALETTE_NAMES)]
        desc = _caption(pid)
        base = f'  part_{pid:<3d}   {color:<8s}  "{desc}"'
        lines.append(base)
    return pids, "\n".join(lines)


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

def call_vlm(image_png: bytes, system: str, user: str,
             url: str, model: str, max_tokens: int = 4096) -> str:
    """Synchronous single-call (kept for non-async path)."""
    from openai import OpenAI
    client = OpenAI(base_url=url, api_key="EMPTY")
    return _do_call_sync(client, image_png, system, user, model, max_tokens)


def _do_call_sync(client, image_png, system, user, model, max_tokens):
    b64 = base64.b64encode(image_png).decode()
    resp = client.chat.completions.create(
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
        timeout=300,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


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
ALLOWED_VERBS = ("remove", "delete", "add", "change", "replace",
                 "make", "scale", "resize", "paint", "recolor")
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


def validate(parsed: dict, valid_pids: set[int], quota: dict | None = None) -> dict:
    """Lightweight check: returns {ok, errors, warnings, n_kept_edits}."""
    out = {"ok": False, "errors": [], "warnings": [], "n_kept_edits": 0}
    if not isinstance(parsed, dict):
        out["errors"].append("not a dict")
        return out
    if "object" not in parsed or "edits" not in parsed:
        out["errors"].append("missing object/edits keys")
        return out
    obj = parsed["object"]
    for k in ("full_desc", "full_desc_stage1", "full_desc_stage2", "parts"):
        if k not in obj:
            out["errors"].append(f"object missing {k}")
    edits = parsed["edits"]
    if not isinstance(edits, list):
        out["errors"].append("edits is not a list")
        return out
    type_count: dict[str, int] = {}
    kept = 0
    _invalid_indices: set[int] = set()
    for i, e in enumerate(edits):
        problems = []
        et = e.get("edit_type")
        if et not in EDIT_TYPES:
            problems.append(f"bad edit_type={et}")
        pids = e.get("selected_part_ids", [])
        if not isinstance(pids, list) or any(p not in valid_pids for p in pids):
            problems.append(f"invalid selected_part_ids={pids}")
        if et == "global" and pids:
            problems.append("global edit has selected_part_ids")
        if et != "global" and not pids:
            problems.append("non-global edit has empty selected_part_ids")
        if et != "deletion":
            for k in REQUIRED_AFTER_FIELDS:
                if not e.get(k):
                    problems.append(f"missing {k}")
        prompt = (e.get("prompt") or "").strip().lower()
        if not any(prompt.startswith(v) for v in ALLOWED_VERBS):
            problems.append("prompt missing imperative verb")
        if any(f"part_{p}" in prompt for p in valid_pids):
            problems.append("prompt mentions part_id")
        vi = e.get("view_index")
        if not isinstance(vi, int) or vi < 0 or vi >= N_VIEWS:
            problems.append(f"invalid view_index={vi}")
        if problems:
            out["warnings"].append({"edit_index": i, "problems": problems})
            _invalid_indices.add(i)
        else:
            kept += 1
            type_count[et] = type_count.get(et, 0) + 1
    out["n_kept_edits"] = kept
    # R2 cross-edit check: no two valid edits with same (edit_type, selected_part_ids).
    # Global edits are exempt: they always have selected_part_ids=[] and differ only by
    # target_style; multiple globals are allowed by quota for large objects.
    seen_signatures: set[tuple] = set()
    for i, e in enumerate(edits):
        et = e.get("edit_type")
        if et == "global":
            continue
        if i in _invalid_indices:
            continue
        pids = tuple(sorted(e.get("selected_part_ids", [])))
        sig = (et, pids)
        if sig in seen_signatures:
            out["warnings"].append({
                "edit_index": i,
                "problems": [f"R2 violation: duplicate (edit_type={et}, selected_part_ids={list(pids)})"],
            })
        else:
            seen_signatures.add(sig)
    out["type_counts"] = type_count
    out["expected_dist"] = quota or {}
    target = sum((quota or {}).values()) if quota else len(edits)
    # Allow 70% recovery as success threshold
    out["ok"] = kept >= max(1, int(target * 0.7)) and not out["errors"]
    return out


# ─────────────────────────────── main ───────────────────────────────────────


__all__ = [
    # ── original (image_menu full prompt) ──────────────────────────────
    "SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "MAX_PARTS",
    "EDIT_TYPES", "ALLOWED_VERBS", "REQUIRED_AFTER_FIELDS", "N_VIEWS",
    "build_image_semantic_menu", "build_image_only_menu", "render_overview_png",
    "call_vlm", "call_vlm_async",
    "extract_json_object", "validate", "quota_for",
    # ── simplified multi-mode (pipeline_v3) ────────────────────────────
    "SYSTEM_PROMPT_A", "SYSTEM_PROMPT_B", "SYSTEM_PROMPT_C",
    "USER_PROMPT_IMAGE_SEMANTIC", "USER_PROMPT_TEXT_SEMANTIC", "USER_PROMPT_IMAGE_ONLY",
    "PROMPT_MODES",
    "build_image_semantic_menu", "build_image_only_menu", "build_semantic_list",
    "call_vlm_text_async", "build_prompt_for_mode",
    "validate_simple",
    "_GLOBAL_STYLE_POOL", "_sample_global_note",
    # ── two-stage (Mode D) ──────────────────────────────────────────
    "SYSTEM_PROMPT_S1", "SYSTEM_PROMPT_S2",
    "build_s1_user_prompt", "build_s2_user_prompt", "parse_s1_output",
    # ── alignment gate (Mode E) ─────────────────────────────────────
    "SYSTEM_PROMPT_ALIGN_GATE",
    "build_align_gate_user_prompt",
    "parse_align_gate_output",
]


# ═══════════════════════════════════════════════════════════════════════
#  SIMPLIFIED PROMPT SYSTEM  (pipeline_v3)
#
#  Three prompt modes — each has its own static system prompt (KV-cacheable)
#  and a thin per-object user prompt (part menu + quota line + global roster).
#
#  Mode A  "image_semantic"  — 5×2 grid image + semantic menu
#  Mode B  "text_semantic"   — semantic part list only (no image)
#  Mode C  "image_only"      — 5×2 grid image + colour-only menu
#
#  Architecture:
#    _SYSTEM_CORE          shared invariant rules + schema (never has {} placeholders)
#    _PREAMBLE_{A,B,C}     mode-specific input declaration (2–8 lines)
#    SYSTEM_PROMPT_{A,B,C} = preamble + core  → KV-cached per mode
#    USER_PROMPT_*         only per-object variables: {part_menu}, {n_*}, {global_note}
# ═══════════════════════════════════════════════════════════════════════

# ── Shared core: rules, edit-type guidance, output schema ─────────────
# No {placeholders} here — purely static, eligible for KV-cache on all modes.
_SYSTEM_CORE = """
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
R1. selected_part_ids must only contain IDs from the part menu.
R2. No two edits may share the same edit_type AND selected_part_ids.
R3. global edits use selected_part_ids = [].
R4. view_index ∈ 0–4: the view where the target is most clearly visible (0 for global).

# SEMANTIC GROUPING — applies to BOTH deletion and modification
  Parts sharing the same functional role (all four chair legs, both wheels, all fin
  panels, all support struts) form ONE semantic group.  Identify groups by shared
  function: parts with similar names or descriptions are typically symmetric
  instances of the same group.

  For DELETION:
  • Treat the entire group as ONE edit — list ALL member part_ids in
    selected_part_ids.  NEVER generate separate deletions per group member.
  • Never target the primary structural body for deletion — the part whose
    removal makes the object unrecognisable (chair seat, car chassis, sword hilt,
    lamp base).  Ask: "would the object still be identifiable without this part?"
    If no → skip it.
  • Each deletion should be a plausible user action: "remove all four legs",
    "remove both armrests", "remove the decorative trim ring".

  For MODIFICATION:
  • When the same shape/identity change applies uniformly to a group (all legs →
    hairpin rods, both wheels → triangular), group all IDs into ONE edit.
  • Unlike deletion, the primary structural body CAN be a modification target.
  • Individual unique parts (seat, backrest, steering wheel) with no semantic
    siblings are naturally single-ID modification edits.

  RESULT: edit count reflects semantic groups, not raw part count — each edit
  is meaningful and non-redundant.

# DELETION EDITS
  A deletion removes one part or semantic group; the remaining object is still
  recognisable as the same object category.
  edit_params must be {}. after_desc must be null.

# MODIFICATION — two valid sub-types (both use edit_params.new_part_desc):
  (a) Shape morph — same functional identity, different geometry:
        cylindrical barrel → hexagonal prism barrel
        straight sword blade → curved saber blade
        spherical head → cubic head   ·   upright ears → floppy drooping ears
  (b) Functional replacement — completely different object in the same structural slot:
        sword blade → axe head   ·   circular wheel → triangular wheel
        round antenna → satellite dish   ·   vertical stabilizer → swept-back tail fin
  new_part_desc must name the new object/shape AND its key geometry, e.g.:
    "a broad wedge-shaped axe head" · "a triangular flat wheel" · "a parabolic dish"
  FORBIDDEN in modification: changing ONLY colour (→ color) or ONLY material (→ material).

# MATERIAL — target_material: specific surface substance, e.g.
  "polished walnut wood" · "brushed stainless steel" · "frosted borosilicate glass"
  · "hand-stitched leather" · "poured concrete" · "translucent amber resin"
  Forbidden: style/aesthetic words (cartoon, vintage, futuristic …)

# COLOR — target_color: descriptive colour phrase (hue/shade only), e.g.
  "deep crimson red" · "matte charcoal black" · "cobalt blue" · "ivory cream white"
  · "forest green" · "warm amber orange" · "pale lavender" · "glossy navy blue"
  Forbidden: bare palette names (red, orange, lime, …) — always qualify them.
  Do NOT change material/finish; use "material" for that.

# GLOBAL — target_style: a specific artistic or rendering aesthetic.
  The allowed styles for THIS object are listed under "GLOBAL STYLE ROSTER" in the
  user message — choose target_style values ONLY from that per-object list.
  Each global edit must use a style from a DIFFERENT category row of the roster.
  Forbidden: surface-material words (gold, wood, metal) — use "material" for those.

# OUTPUT — one JSON object
{
  "object": {
    "full_desc": "complete English description",
    "parts": [{"part_id": <int>, "name": "<semantic label>"}, ...]
  },
  "edits": [
    {
      "edit_type":         "deletion | modification | scale | material | color | global",
      "selected_part_ids": [<int>, ...],
      "prompt":            "<imperative verb phrase>",
      "target_part_desc":  "<visual description of target part(s)>",
      "view_index":        <int 0-4>,
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

# ── Mode-specific preambles (static, no placeholders) ─────────────────
# Actual part-menu column formats (no "level" column anywhere):
#   Mode A  part_id | palette-colour | description
#   Mode B  part_id | description
#   Mode C  part_id | palette-colour
# All three preambles share the same block order for easy comparison:
#   identity → INPUT (with column format) → PALETTE RULE* → VIEW RULE → PART ID RULE*
#   (* = absent in Mode B / only in Mode C respectively)

_PREAMBLE_A = """You are a 3D-object edit-set generator. Output ONE valid JSON object — no prose, no markdown.

INPUT (Mode A — image + semantic menu):
  You will receive:
    • A 5×2 grid image: TOP row = 5 RGB photos (view 0–4), BOTTOM row = same 5
      cameras re-rendered with parts colour-coded by palette ID (column = camera)
    • A semantic part menu — columns: part_id | palette-colour | description

PALETTE RULE: Palette colour names (red, orange, yellow, lime, green, teal, cyan,
  blue, navy, purple, magenta, pink, brown, tan, black, gray) are INTERNAL labels
  matching the BOTTOM-row highlights — do NOT use them in any output text field.
  Describe real colour/appearance using the TOP-row RGB photos.

VIEW RULE: Use the images to select view_index (0–4) where each target part is
  most clearly visible.
"""

_PREAMBLE_B = """You are a 3D-object edit-set generator. Output ONE valid JSON object — no prose, no markdown.

INPUT (Mode B — text-only semantic list, no image):
  You will receive:
    • A semantic part list — columns: part_id | description
    • NO image is provided.

VIEW RULE: Without an image, view_index is a structural best-estimate.
  Use 0 for parts typically facing front, 4 for parts on the bottom.
  Parts with similar names or descriptions are likely symmetric instances
  (legs, wheels, fins) — group them into one edit accordingly.
"""

_PREAMBLE_C = """You are a 3D-object edit-set generator. Output ONE valid JSON object — no prose, no markdown.

INPUT (Mode C — image + colour-only menu):
  You will receive:
    • A 5×2 grid image: TOP row = 5 RGB photos (view 0–4), BOTTOM row = same 5
      cameras re-rendered with parts colour-coded by palette ID (column = camera)
    • A colour-only part menu — columns: part_id | palette-colour
      (no text descriptions — identify parts by matching palette colour in the
       BOTTOM row to the corresponding region in the TOP-row photos)

PALETTE RULE: Palette colour names (red, orange, yellow, lime, green, teal, cyan,
  blue, navy, purple, magenta, pink, brown, tan, black, gray) are INTERNAL labels
  matching the BOTTOM-row highlights — do NOT use them in any output text field.
  Describe real colour/appearance using the TOP-row RGB photos.

VIEW RULE: Use the images to select view_index (0–4) where each target part is
  most clearly visible.

PART ID RULE: To identify what part_N is, locate its palette colour in the BOTTOM
  row and examine the matching region in the TOP-row RGB photos.
"""

# ── Assembled system prompts (preamble + core) — one per mode ─────────
# These are purely static strings: no {placeholders}. They are the same
# for every object within a mode, making them eligible for KV-cache reuse.
SYSTEM_PROMPT_A = _PREAMBLE_A + _SYSTEM_CORE   # image + semantic menu
SYSTEM_PROMPT_B = _PREAMBLE_B + _SYSTEM_CORE   # text-only semantic list
SYSTEM_PROMPT_C = _PREAMBLE_C + _SYSTEM_CORE   # image + colour-only menu

# ── User prompts: only the per-object variable parts ──────────────────
_QUOTA_LINE = "Generate EXACTLY {n_total} edits — {n_deletion} deletion · {n_modification} modification · {n_scale} scale · {n_material} material · {n_color} color · {n_global} global{global_note}"

# Mode A: image + semantic menu (palette colour + description)
USER_PROMPT_IMAGE_SEMANTIC = """[Image: 5 RGB photos (top row) + same 5 views re-rendered with parts colour-coded by ID (bottom row). Palette colours are INTERNAL labels — do NOT use them in output text.]

# PART MENU  (id · palette-colour · description)
{part_menu}

""" + _QUOTA_LINE

# Mode B: text-only semantic list, no image
USER_PROMPT_TEXT_SEMANTIC = """# PART LIST  (id · description)
{part_menu}

""" + _QUOTA_LINE

# Mode C: image + colour-only menu (no descriptions — VLM reasons from image)
USER_PROMPT_IMAGE_ONLY = """[Image: 5 RGB photos (top row) + same 5 views re-rendered with parts colour-coded by ID (bottom row). Palette colours are INTERNAL labels — do NOT use them in output text.]

# PART MENU  (id · palette-colour)
{part_menu}

""" + _QUOTA_LINE

PROMPT_MODES = ("image_semantic", "text_semantic", "image_only")


# ═══════════════════════════════════════════════════════════════════════
#  Menu builders for the simplified modes
# ═══════════════════════════════════════════════════════════════════════

def build_image_only_menu(
    mesh_npz: Path,
    img_npz: Path,
) -> tuple[list[int], str]:
    """Colour-only part menu — for image_only mode (Mode C).

    No semantic descriptions or level — the VLM must reason purely from
    the colour-coded image overlay.

    Format per line:
        part_{id:<3d}   {colour}
    """
    z2 = __import__('numpy').load(mesh_npz, allow_pickle=True)
    z = __import__('numpy').load(img_npz, allow_pickle=True)
    sm = json.loads(bytes(z["split_mesh.json"]).decode())
    clusters = sm.get("valid_clusters", {})
    import re as _re

    def _parse_pid(k: str) -> int | None:
        m = _re.search(r"\d+", k)
        return int(m.group()) if m else None

    pids = sorted(
        pid
        for k in z2.files
        if k.startswith("part_") and (k.endswith(".glb") or k.endswith(".ply"))
        if (pid := _parse_pid(k)) is not None
    )

    lines = []
    for pid in pids:
        color = _PALETTE_NAMES[pid % len(_PALETTE_NAMES)]
        base = f"  part_{pid:<3d}   {color}"
        lines.append(base)
    return pids, "\n".join(lines)


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


def build_prompt_for_mode(
    mode: str,
    pids: list[int],
    part_menu: str,
    quota: dict,
    *,
    variety_seed: int | None = None,
) -> tuple[str, str]:
    """Return (system, user) prompt strings for the given mode.

    Args:
        mode:           one of PROMPT_MODES:
                          "image_semantic" — image + semantic menu (Mode A)
                          "text_semantic"  — semantic menu only, no image (Mode B)
                          "image_only"     — image + colour-only menu (Mode C)
        pids:           list of valid part IDs
        part_menu:      pre-built menu string from the matching builder
        quota:          output of quota_for()
        variety_seed:   integer seed for per-object style randomisation.
                        Pass hash(obj_id) or any per-object integer. When
                        None, falls back to a seed derived from pids (less
                        diverse). Always provide this for production runs.
    """
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {mode!r}. Choose from {PROMPT_MODES}")

    n_global = quota.get("global", 0)
    seed = variety_seed if variety_seed is not None else hash(tuple(sorted(pids)))
    global_note = _sample_global_note(seed, n_global) if n_global > 0 else ""

    n_total = sum(quota.values())
    fmt = dict(
        part_menu=part_menu,
        n_total=n_total,
        n_deletion=quota.get("deletion", 0),
        n_modification=quota.get("modification", 0),
        n_scale=quota.get("scale", 0),
        n_material=quota.get("material", 0),
        n_color=quota.get("color", 0),
        n_global=n_global,
        global_note=global_note,
    )

    if mode == "image_semantic":
        user = USER_PROMPT_IMAGE_SEMANTIC.format(**fmt)
    elif mode == "text_semantic":
        user = USER_PROMPT_TEXT_SEMANTIC.format(**fmt)
    else:  # image_only
        user = USER_PROMPT_IMAGE_ONLY.format(**fmt)

    sys_map = {
        "image_semantic": SYSTEM_PROMPT_A,
        "text_semantic":  SYSTEM_PROMPT_B,
        "image_only":     SYSTEM_PROMPT_C,
    }
    return sys_map[mode], user


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

        prompt = (e.get("prompt") or "").strip().lower()
        if not any(prompt.startswith(v) for v in ALLOWED_VERBS):
            problems.append("prompt must start with an imperative verb")

        vi = e.get("view_index")
        if not isinstance(vi, int) or vi < 0 or vi >= N_VIEWS:
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


# ═══════════════════════════════════════════════════════════════════════
#  TWO-STAGE PIPELINE  (Mode D — image→semantics then text→edits)
#
#  Stage 1  call_vlm_async(image + colour menu)  → s1_parts JSON
#  Stage 2  call_vlm_text_async(s1_parts text)   → edit JSON   (no image)
#
#  Key invariant: the image is ONLY seen in Stage 1. Stage 2 is fully
#  text-driven so it cannot be misled by PartVerse caption noise.
# ═══════════════════════════════════════════════════════════════════════

# ── Stage 1: visual part-semantic reconstruction ─────────────────────

SYSTEM_PROMPT_S1 = """\
You are a 3D-part semantic labeller. Given a colour-coded overview image of a 3D object, assign a precise functional name to every visible part.

INPUT:
  • A 5×2 grid image:
      TOP row    = 5 RGB photos (views 0–4, same camera each column)
      BOTTOM row = same 5 views re-rendered with parts colour-coded by palette ID
  • A colour-only part menu: part_id | palette-colour

TASK:
  For each entry in the menu, locate its palette colour in the BOTTOM row,
  examine the matching region in the TOP-row RGB photos, and decide what
  structural role that region plays in the overall object.

OUTPUT: ONE valid JSON object — no prose, no markdown fences.
{
  "object_desc": "<one sentence describing the whole object>",
  "parts": [
    {
      "part_id": <int>,
      "name": "<concise functional label, 1-4 words, e.g. 'front left wheel', 'gun barrel', 'left arm'>",
      "view_index": <0-4, the column where this part is most clearly visible>,
      "appearance": "<brief visual description from the TOP-row RGB photos, 5-12 words>"
    }
  ]
}

RULES:
  R1. name is a SHORT FUNCTIONAL LABEL — not a visual description.
      Good: "rear left leg"  "trigger guard"  "left winglet"
      Bad:  "cylindrical green component"  "blue faceted object"
  R2. Symmetric/repeated instances must be individually named with a
      positional qualifier: "front left wheel" not just "wheel".
  R3. For each part, scan ALL 5 bottom-row columns for its palette colour.
      If you CANNOT clearly locate the colour in ANY of the 5 views:
        • Set "name" to null and "view_index" to -1.
        • Do NOT guess or invent a name from context or position.
      Occluded / invisible parts will be excluded from editing — accuracy
      matters more than completeness.
  R4. Do NOT use palette colour names (red, orange, yellow, lime …) in any
      output field — use the real appearance from the TOP row.
  R5. Output parts in ascending part_id order.
"""

# ── Stage 2: text-only edit generation ───────────────────────────────
# Reuses _PREAMBLE_B (text-only preamble) + _SYSTEM_CORE.
# SYSTEM_PROMPT_S2 is identical to SYSTEM_PROMPT_B in behaviour.
# We give it a distinct name to signal intent.
SYSTEM_PROMPT_S2 = _PREAMBLE_B + _SYSTEM_CORE  # same as SYSTEM_PROMPT_B


# ── User-prompt builders ──────────────────────────────────────────────

def build_s1_user_prompt(pids: list[int]) -> str:
    """Colour-only menu for Stage 1 (identical format to build_image_only_menu
    but takes pre-computed pids directly, no NPZ access needed)."""
    lines = [f"  part_{pid:<3d}   {_PALETTE_NAMES[pid % len(_PALETTE_NAMES)]}"
             for pid in sorted(pids)]
    return "Colour-only part menu:\n" + "\n".join(lines)


def build_s2_user_prompt(
    s1_parts: list[dict],
    object_desc: str,
    quota: dict,
    *,
    variety_seed: int | None = None,
) -> str:
    """Format Stage 1 visible-part output into a Stage 2 (text-only) user prompt.

    Accepts the contents of d["parts_visible"] — null-name parts are already
    filtered out by parse_s1_output and must NOT be passed here.

    Each part line:  part_{id}   "{name}"   [view: N]  — appearance
    """
    pids = [p["part_id"] for p in s1_parts]
    n_global = quota.get("global", 0)
    seed = variety_seed if variety_seed is not None else hash(tuple(sorted(pids)))
    global_note = _sample_global_note(seed, n_global) if n_global > 0 else ""

    part_lines = []
    for p in sorted(s1_parts, key=lambda x: x["part_id"]):
        pid  = p["part_id"]
        name = p.get("name", f"part_{pid}")
        vi   = p.get("view_index", 0)
        app  = p.get("appearance", "")
        vi_str = str(vi) if vi >= 0 else "hidden"
        line = f'  part_{pid:<3d}   "{name}"   [view: {vi_str}]'
        if app:
            line += f"  — {app}"
        part_lines.append(line)

    parts_block = "\n".join(part_lines)
    n_total = sum(quota.values())

    return (
        f"Object: {object_desc}\n\n"
        f"Parts:\n{parts_block}\n\n"
        f"Generate at most {n_total} edits total "
        f"({quota.get('deletion',0)} deletion, "
        f"{quota.get('modification',0)} modification, "
        f"{quota.get('scale',0)} scale, "
        f"{quota.get('material',0)} material, "
        f"{quota.get('color',0)} color, "
        f"{quota.get('global',0)} global)."
        + (f"\n\nGlobal style roster:\n{global_note}" if global_note else "")
    )


def parse_s1_output(raw: str) -> dict | None:
    """Extract and lightly validate the Stage 1 JSON.

    Returns the parsed dict on success, None on failure.
    Adds two keys to the dict:
      "parts_visible"  — parts whose name is non-null (will be sent to Stage 2)
      "parts_hidden"   — parts whose name is null (invisible, excluded from Stage 2)
    """
    d = extract_json_object(raw)   # already returns dict | None
    if not isinstance(d, dict):
        return None
    parts = d.get("parts")
    if not isinstance(parts, list) or not parts:
        return None
    if not all("part_id" in p for p in parts):
        return None
    visible = [p for p in parts if p.get("name") not in (None, "null", "")]
    hidden  = [p for p in parts if p.get("name") in (None, "null", "")]
    d["parts_visible"] = visible
    d["parts_hidden"]  = hidden
    # Require at least one visible part
    if not visible:
        return None
    return d


# ═══════════════════════════════════════════════════════════════════════
#  Alignment Gate (Mode E) — image+text call, judges edit↔part alignment
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_ALIGN_GATE = """\
You are a 3D-edit alignment judge.

INPUT:
  • A 5×2 grid image (same layout as a standard overview):
      TOP row    = 5 RGB photos (views 0–4, left to right)
      BOTTOM row = 5 part-coloured renders
      IMPORTANT: Column 0 bottom is special — selected (target) parts are RED,
                 all other parts are GREY.
                 Columns 1–4 bottom use normal palette colours (unmodified).
  • Edit instruction and type in text.

TASK:
  1. Find the RED region in column 0 (bottom row) — these are the target parts.
  2. Examine the matching region in column 0 top-row RGB for visual context.
  3. Use columns 1–4 for additional object context.
  4. Judge whether the instruction makes semantic sense for the red-highlighted parts.
  5. Choose which column (0–4) gives the clearest view of the target parts.

OUTPUT: ONE valid JSON object — no prose, no markdown fences.
{
  "aligned":   <true|false>,
  "reason":    "<1-2 sentences>",
  "best_view": <0-4, column index in THIS image where target parts are clearest>
}

RULES:
  R1. aligned=true  iff the red parts match the instruction's stated target AND
      the edit type is appropriate for those parts.
  R2. aligned=false if no red is visible in column 0 (parts fully occluded), or
      the highlighted parts clearly do not match the instruction's intent.
  R3. best_view = column where red coverage is largest AND top-row RGB shows the
      target parts most clearly for editing purposes.
  R4. For global edits the image will be all-grey (no red). Always output
      aligned=true, best_view=0 for global edits.
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
    if not isinstance(d.get("best_view"), int):
        d["best_view"] = 0   # safe default
    return d

