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

P5. PEER GROUPS → GROUP EDITS. The part menu marks ``peer_group=[i,j,...]``
    for parts that are semantically equivalent (same structural tier, similar
    size). When editing ANY part that belongs to a peer_group, you MUST
    produce a single group edit whose selected_part_ids includes EVERY member
    of that peer_group — never edit just one member in isolation.
    Example: if parts 2 and 3 share peer_group=[2,3], the edit must have
    selected_part_ids=[2,3], never [2] or [3] alone.
    Parts marked [STRUCTURAL BODY] must never appear in deletion or
    scale edits (R3 reinforcement). Mix group and single-part edits.

P5b. MANDATORY SEMANTIC GROUPING — applies even when NO peer_group is marked.
    GROUP AS MUCH AS POSSIBLE. Two categories of parts MUST always be combined
    into a single edit regardless of peer_group annotations:

    (a) Bilateral / repeated instances — parts whose names differ ONLY by
        side label ("Left"/"Right"), position, or index share the same
        functional role and MUST be a single edit:
          Left Arm + Right Arm → ONE edit   (selected_part_ids = [id_L, id_R])
          Left Leg + Right Leg → ONE edit
          Left Ear + Right Ear → ONE edit
          all four chair legs  → ONE edit
          both wheels          → ONE edit
        FORBIDDEN: separate deletion/modification for "Left Arm" and "Right Arm".

    (b) Semantic constituents — distinct parts that together form one coherent
        sub-object, where removing/changing only a subset produces an incomplete
        or broken result:
          face features (eyes + nose + mouth) → ONE deletion
          hand → fingers + palm → ONE edit
          engine block + pistons → ONE edit
        Clue: if you cannot describe the edit without mentioning multiple parts
        that are always perceived as a single unit, they MUST be grouped.

    SELF-CHECK before writing each edit: "Is this part the bilateral/repeated
    partner of another part I have already listed or am about to list?" If yes,
    merge them. NEVER produce N separate edits for N symmetric/repeated parts.

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
  edit_type           one of: deletion, modification, scale, material, global
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

# MODIFICATION EDITS — SHAPE, FORM AND FUNCTION ONLY
  A modification replaces the geometry, silhouette, or functional role of a part.
  Think creatively: what shape or form would be surprising yet meaningful?
  Examples:
    • straight sword blade  →  curved saber blade
    • cylindrical barrel    →  hexagonal prism barrel
    • spherical head        →  cubic head
    • upright rabbit ears   →  floppy drooping ears
  STRICTLY FORBIDDEN in modification: changing only color, surface finish, or
  material. Those belong exclusively to "material" or "global" edit types.
  The new_part_desc MUST describe a geometry or silhouette change.

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
                      global:       {{"target_style": "..."}}
  after_desc_full / after_desc_stage1 / after_desc_stage2
                      object after the edit. For deletion: ALL three null.
                      For others: all three filled, stage1 has no
                      colors/materials, stage2 has no shape changes.
  new_parts_desc / new_parts_desc_stage1 / new_parts_desc_stage2
                      modification only: describe the new replacement parts.
                      null for non-modification edits.
  confidence          "high" | "medium" | "low"

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

  VALID target_style — choose from DIFFERENT categories for each object:

    Rendering / shading aesthetics:
      cel-shading, flat-shading, wireframe-outline with colored faces,
      watercolor wash, oil-painting impasto, impressionist brushstroke,
      pointillist dots, charcoal sketch, ink wash (sumi-e), stained-glass mosaic,
      neon-glow bloom

    Historical / regional art movements:
      Art Nouveau organic lines, Art Deco geometric, ukiyo-e woodblock print,
      Bauhaus functional, brutalist concrete aesthetic, baroque gilded,
      gothic cathedral tracery, ancient terracotta figurine,
      Ming dynasty blue-and-white porcelain

    Genre / subculture aesthetics:
      cyberpunk neon-and-chrome, steampunk brass-and-gears, solarpunk organic-tech,
      vaporwave pastel grid, retro-1980s pixel-art, lo-fi cassette-tape grain,
      biomechanical flesh-and-machine, origami paper-fold geometry,
      LEGO brick construction, Islamic geometric mosaic tile

  DIVERSITY RULE: For this object's {n_global} global edits, each target_style
  MUST come from a DIFFERENT category above. Near-synonyms count as the same choice.
  The rationale for every global edit MUST name the source category, e.g.:
  "Choosing 'ukiyo-e woodblock print' from the Historical category."

# HARD RULES (violations drop that edit)

R1. selected_part_ids ⊆ part menu ids; never target parts with cluster_size<30
    UNLESS the part has a peer_group or [STRUCTURAL BODY] annotation (those are
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


def _load_anno_peer_groups(
    anno_obj_dir: "Path | None",
    pids: "list[int]",
    weight_ratio_thr: float = 0.75,
    mesh_npz: "Path | None" = None,
) -> "tuple[dict[int,list[int]], set[int], dict[int,int]]":
    """Load _info.json and compute peer groups and structural parts.

    Source priority:
      1. ``anno_obj_dir/{obj_id}_info.json``  (external anno_infos directory)
      2. ``mesh_npz["anno_info.json"]``        (embedded via repack_mesh_add_anno.py)

    Returns:
        peer_groups: {pid: [peer_pid, ...]} — empty list if no peers.
        structural_pids: set of pids that are structural body (level=0, high weight).
        pid_levels: {pid: level_value}
    """
    info_bytes: bytes | None = None

    if anno_obj_dir is not None and anno_obj_dir.is_dir():
        obj_id = anno_obj_dir.name
        info_path = anno_obj_dir / f"{obj_id}_info.json"
        if info_path.is_file():
            try:
                info_bytes = info_path.read_bytes()
            except Exception:
                info_bytes = None

    if info_bytes is None and mesh_npz is not None and mesh_npz.is_file():
        try:
            z = np.load(str(mesh_npz), allow_pickle=True)
            if "anno_info.json" in z.files:
                info_bytes = bytes(z["anno_info.json"])
        except Exception:
            info_bytes = None

    if info_bytes is None:
        return {pid: [] for pid in pids}, set(), {}

    try:
        info = json.loads(info_bytes.decode())
    except Exception:
        return {pid: [] for pid in pids}, set(), {}

    levels_list = info.get("ordered_part_level", [])
    weights_list = info.get("weights", [])

    pid_levels: dict[int, int] = {}
    pid_weights: dict[int, float] = {}
    for pid in pids:
        if pid < len(levels_list):
            pid_levels[pid] = levels_list[pid]
        if pid < len(weights_list):
            pid_weights[pid] = float(weights_list[pid])

    # Group pids by level
    from collections import defaultdict
    by_level: dict[int, list[int]] = defaultdict(list)
    for pid, lv in pid_levels.items():
        by_level[lv].append(pid)

    # Peer groups: same level + weight ratio >= threshold
    peer_groups: dict[int, list[int]] = {pid: [] for pid in pids}
    for lv, group in by_level.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pi, pj = group[i], group[j]
                wi = pid_weights.get(pi, 0.0)
                wj = pid_weights.get(pj, 0.0)
                if wi > 0 and wj > 0 and min(wi, wj) / max(wi, wj) >= weight_ratio_thr:
                    if pj not in peer_groups[pi]:
                        peer_groups[pi].append(pj)
                    if pi not in peer_groups[pj]:
                        peer_groups[pj].append(pi)

    # Structural pids: level == 0 AND weight > 2× median weight
    structural_pids: set[int] = set()
    if pid_weights:
        import statistics
        med = statistics.median(pid_weights.values())
        for pid, lv in pid_levels.items():
            if lv == 0 and pid_weights.get(pid, 0) > 2.0 * med:
                structural_pids.add(pid)

    return peer_groups, structural_pids, pid_levels


def build_part_menu(
    mesh_npz: Path,
    img_npz: Path,
    anno_obj_dir: "Path | None" = None,
) -> tuple[list[int], str]:
    """Return (part_ids, menu text). cluster_size from split_mesh.json.

    If *anno_obj_dir* is given and contains ``{obj_id}_info.json``, the menu
    is enriched with ``level=`` and ``peer_group=[...]`` annotations so the
    VLM can identify semantically equivalent parts without geometric reasoning.
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

    peer_groups, structural_pids, pid_levels = _load_anno_peer_groups(anno_obj_dir, pids, mesh_npz=mesh_npz)

    lines = []
    for pid in pids:
        cs = clusters.get(f"part_{pid}", {}).get("cluster_size", "?")
        color = _PALETTE_NAMES[pid % len(_PALETTE_NAMES)]
        base = f"  part_{pid:<3d} {color:<8s}  cluster_size={cs}"
        if pid in pid_levels:
            lv = pid_levels[pid]
            if pid in structural_pids:
                base += f"  level={lv}  [STRUCTURAL BODY — do not delete]"
            else:
                peers = peer_groups.get(pid, [])
                if peers:
                    all_members = sorted([pid] + peers)
                    base += f"  level={lv}  peer_group={all_members}"
                else:
                    base += f"  level={lv}"
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
                 "make", "scale", "resize")
EDIT_TYPES = ("deletion", "modification", "scale", "material", "global")
N_VIEWS = 5  # must match len(VIEW_INDICES) in render_part_overview
MAX_PARTS = 16  # objects with more parts are skipped


def quota_for(n_parts: int) -> dict:
    """Per-edit-type quotas based on number of (valid) parts. Deletion is the
    cheapest + most useful task, so it gets the largest share. Scale is capped
    at 1 per object to avoid redundancy; prefer shrinking large parts."""
    if n_parts <= 2:  return {"deletion":1,  "modification":1,  "scale":1, "material":1, "global":3}
    if n_parts == 3:  return {"deletion":3,  "modification":3,  "scale":1, "material":1, "global":3}
    if n_parts == 4:  return {"deletion":4,  "modification":4,  "scale":1, "material":2, "global":3}
    if n_parts == 5:  return {"deletion":5,  "modification":5,  "scale":1, "material":2, "global":3}
    if n_parts == 6:  return {"deletion":6,  "modification":6,  "scale":1, "material":2, "global":3}
    if n_parts <= 8:  return {"deletion":8,  "modification":8,  "scale":1, "material":3, "global":3}
    if n_parts <= 10: return {"deletion":10, "modification":10, "scale":1, "material":4, "global":3}
    if n_parts <= 12: return {"deletion":12, "modification":12, "scale":1, "material":4, "global":4}
    if n_parts <= 14: return {"deletion":14, "modification":14, "scale":1, "material":5, "global":4}
    return                   {"deletion":16, "modification":16, "scale":1, "material":5, "global":4}


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
    "SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "MAX_PARTS",
    "EDIT_TYPES", "ALLOWED_VERBS", "REQUIRED_AFTER_FIELDS", "N_VIEWS",
    "build_part_menu", "render_overview_png",
    "call_vlm", "call_vlm_async",
    "extract_json_object", "validate", "quota_for",
]
