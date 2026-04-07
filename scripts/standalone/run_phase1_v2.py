#!/usr/bin/env python3
"""Phase 1 v2 prototype: one-shot VLM call producing object meta + 9 edits.

For each obj_id:
  1. Render the 4×2 colored overview (top = 4 photos, bottom = colored parts)
  2. Build the part menu from the mesh + cluster_size hints
  3. Build the unified VLM prompt (system + user + image)
  4. Call the local Qwen3.5-VL server (OpenAI-compatible)
  5. Parse + lightly validate the JSON response
  6. Save raw + parsed to outputs/_debug/phase1_v2/{obj_id}.json

Usage:
    python scripts/standalone/run_phase1_v2.py \
        --obj-ids 112204b2a12c4e25bbbdcc0d196b1ad5 19e9f218f8c84af0af0b9ed8e930775e \
        --shard 01
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "tools"))
from render_part_overview import (  # noqa: E402
    VIEW_INDICES, _PALETTE, _PALETTE_NAMES,
    extract_parts, load_views_from_npz, run_blender, stitch_two_rows,
)


# ─────────────────────────── prompt construction ───────────────────────────

SYSTEM_PROMPT = """You are a structured-data generator for a 3D part-editing dataset. \
You look at multi-view renders of one 3D object plus a part menu, and produce \
a single JSON object describing the object and a diverse set of editing \
instructions.

The output is consumed directly by an automated pipeline. Output ONE valid \
JSON object — no prose, no markdown, no extra text. Begin with '{' and end \
with '}'."""


USER_PROMPT_TEMPLATE = """[The image shows a 5×2 grid. Top row = 5 photos of the same 3D object. \
The 5 columns are indexed 0..4 (left to right):
  views 0, 1, 2, 3 — 4 overhead viewpoints (camera above the object,
                    tilted ~30° downward), each from a different yaw
                    direction so together they cover all sides.
  view 4 — a steep upward viewpoint (camera below, looking up ~50°)
           that exposes the underside of the object.
Look at the images yourself to figure out which view shows which side. \
Bottom row = the same 5 viewpoints re-rendered with each editable part \
painted in a fixed color. Same column = same camera pose.]

Available parts (each line: id, color in the bottom-row image, cluster size):
{part_menu}

Notes:
- Use the IMAGE as the source of truth for what each part is. You will name
  each part yourself in the "object.parts" output below.
- Very small clusters (cluster_size < 30) are likely segmentation artifacts
  with no real geometry. Do not target them.
- A part may be invisible in the bottom row (occluded or degenerate). Do not
  target invisible parts.

## YOUR JOB

Produce a single JSON object with two top-level keys: "object" and "edits".

### object block

  full_desc        Natural English description of the whole object. Mention
                   all visible parts with shape, color, material, and spatial
                   relationships.

  full_desc_stage1 Geometry-only version of full_desc. Keep all shape, count,
                   layout and structural relationships. REMOVE all colors,
                   materials, textures, and finish words.

  full_desc_stage2 Texture-only version of full_desc. Keep all colors,
                   materials, finishes and textures. REMOVE shape modifiers,
                   counts, and layout words.

  parts            List of {{part_id, color, name}} for every part in the
                   menu. "name" is YOUR clean short semantic label based on
                   what you see. Use "(invisible)" or "(artifact)" for parts
                   you cannot see or that are noise.

### edits block

A list of EXACTLY {n_total} editing instructions with the following per-type
quotas (the total {n_total} is fixed for this object, scaled to its part count):
  - {n_deletion} deletion       (mix of single-part, multi-part bundles, and "remove all of a kind"; each must target a DIFFERENT part)
  - {n_modification} modification   (swap a part for a different object of similar role; vary the verb — Change/Replace/Make)
  - {n_scale} scale          (resize a part by a factor in [0.3, 2.5])
  - {n_material} material       (change the material/finish of one or more parts)
  - {n_global} global         (change the entire object's style; selected_part_ids = [])

Each edit:

  edit_type           One of: deletion, modification, scale, material, global
  prompt              Natural English imperative starting with one of:
                      Remove, Delete, Add, Change, Replace, Make, Scale, Resize.
                      Use the semantic part name you assigned in object.parts.
                      Do NOT mention part_id numbers.
  view_index          Integer 0..4. The single view that BEST shows the target
                      of this edit (clearest, least occluded angle on the
                      affected parts). For global edits, pick the view that
                      best shows the overall object.
  selected_part_ids   List of int part_ids targeted. Empty [] ONLY for global.
  target_part_desc    Short natural-language description of the target.
  after_desc_full     Complete description of the object AFTER this edit.
                      OMIT for deletion edits (set to null) — the object is
                      fully described by object.full_desc minus the removed
                      parts, no separate after description is needed.
  after_desc_stage1   Geometry-only version of after_desc_full. OMIT (null)
                      for deletion edits.
  after_desc_stage2   Texture-only version of after_desc_full. OMIT (null)
                      for deletion edits.
  new_parts_desc      For modification edits: rich description of the NEW parts
                      that replaced the old ones, with all visible detail.
                      Null for other edit types.
  new_parts_desc_stage1   Geometry-only version of new_parts_desc, or null.
  new_parts_desc_stage2   Texture-only version of new_parts_desc, or null.
  edit_params         Type-specific parameters (see below).
  rationale           One short sentence explaining why this edit makes sense.
  confidence          "high" | "medium" | "low"

edit_params per type:
  deletion:     {{}}
  modification: {{"new_part_desc": "<short label, e.g. 'a tall wooden stool'>"}}
  scale:        {{"factor": <float in [0.3, 2.5]>}}
  material:     {{"target_material": "<e.g. 'polished brass', 'frosted glass'>"}}
  global:       {{"target_style": "<e.g. 'wooden carved', 'industrial metal'>"}}

## HARD RULES (violations cause that edit to be discarded)

R1. selected_part_ids may only contain ids that appear in the part menu.
R2. Never target a part you cannot see in the bottom row of the image.
R3. Never target parts with cluster_size < 30 (they are noise).
R4. Never delete or extreme-scale the part forming the structural body of the
    object (the object should remain recognizable).
R5. prompt MUST start with an imperative verb: Remove, Delete, Add, Change,
    Replace, Make, Scale, or Resize. prompt MUST NOT mention part_id numbers.
    Color names, materials, and other natural descriptive language are allowed.
R6. Each edit must be DISTINCT — no two edits with the same edit_type AND
    same selected_part_ids.
R7. selected_part_ids must be empty for global edits and non-empty otherwise.
R8. For non-deletion edits, all three after_desc_* fields must be filled.
    Stage1 must omit colors/materials. Stage2 must omit shape changes.
    For deletion edits, set all three after_desc_* fields to null (the
    object minus the removed parts is self-explanatory from object.full_desc).
R9. view_index must be an integer in [0, 4] and must point to a view where
    the target parts are clearly visible (not occluded). For global edits,
    pick the view that gives the best overall look at the object.

## OUTPUT

ONE JSON object. Begin with '{{' and end with '}}'. No prose, no markdown."""


def build_part_menu(mesh_npz: Path, img_npz: Path) -> tuple[list[int], str]:
    """Return (part_ids, menu text). cluster_size from split_mesh.json."""
    z = np.load(img_npz, allow_pickle=True)
    sm = json.loads(bytes(z["split_mesh.json"]).decode())
    clusters = sm.get("valid_clusters", {})
    z2 = np.load(mesh_npz, allow_pickle=True)
    pids = sorted(int(k.replace("part_", "").replace(".ply", ""))
                  for k in z2.files if k.startswith("part_"))
    lines = []
    for pid in pids:
        cs = clusters.get(f"part_{pid}", {}).get("cluster_size", "?")
        color = _PALETTE_NAMES[pid % len(_PALETTE_NAMES)]
        lines.append(f"  part_{pid:<3d} {color:<8s}  cluster_size={cs}")
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
    cheapest + most useful task, so it gets the largest share."""
    if n_parts <= 2:  return {"deletion":1, "modification":1, "scale":1,"material":1,"global":1}
    if n_parts == 3:  return {"deletion":3, "modification":3, "scale":1,"material":1,"global":1}
    if n_parts == 4:  return {"deletion":4, "modification":4, "scale":2,"material":2,"global":1}
    if n_parts == 5:  return {"deletion":5, "modification":5, "scale":2,"material":2,"global":1}
    if n_parts == 6:  return {"deletion":6, "modification":6, "scale":2,"material":2,"global":1}
    if n_parts <= 8:  return {"deletion":8, "modification":8, "scale":3,"material":3,"global":1}
    if n_parts <= 10: return {"deletion":10,"modification":10,"scale":3,"material":4,"global":1}
    if n_parts <= 12: return {"deletion":12,"modification":12,"scale":4,"material":4,"global":2}
    if n_parts <= 14: return {"deletion":14,"modification":14,"scale":4,"material":5,"global":2}
    return                  {"deletion":16,"modification":16,"scale":5,"material":5,"global":2}  # 15-16


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
        else:
            kept += 1
            type_count[et] = type_count.get(et, 0) + 1
    out["n_kept_edits"] = kept
    out["type_counts"] = type_count
    out["expected_dist"] = quota or {}
    target = sum((quota or {}).values()) if quota else len(edits)
    # Allow 70% recovery as success threshold
    out["ok"] = kept >= max(1, int(target * 0.7)) and not out["errors"]
    return out


# ─────────────────────────────── main ───────────────────────────────────────

def process_one(obj_id: str, shard: str, mesh_root: Path, images_root: Path,
                blender: str, vlm_url: str, vlm_model: str, out_dir: Path):
    print(f"\n{'='*70}\n[{obj_id}]")
    mesh_npz = mesh_root / shard / f"{obj_id}.npz"
    img_npz = images_root / shard / f"{obj_id}.npz"

    print("  rendering 4×2 overview…")
    img_png = render_overview_png(mesh_npz, img_npz, blender)
    overview_path = out_dir / f"{obj_id}.overview.png"
    overview_path.write_bytes(img_png)
    print(f"    → {overview_path}")

    pids, menu = build_part_menu(mesh_npz, img_npz)
    print(f"  part menu ({len(pids)} parts):")
    for line in menu.splitlines():
        print(line)

    user_msg = USER_PROMPT_TEMPLATE.format(part_menu=menu)
    print(f"  calling VLM ({vlm_model})…")
    raw = call_vlm(img_png, SYSTEM_PROMPT, user_msg, vlm_url, vlm_model)
    raw_path = out_dir / f"{obj_id}.raw.txt"
    raw_path.write_text(raw)
    print(f"    raw len={len(raw)}  → {raw_path}")

    parsed = extract_json_object(raw)
    if parsed is None:
        print("  [ERR] failed to parse JSON from VLM output")
        return
    valid_pids = set(pids)
    rep = validate(parsed, valid_pids)
    out = {"obj_id": obj_id, "shard": shard,
           "validation": rep, "parsed": parsed}
    parsed_path = out_dir / f"{obj_id}.parsed.json"
    parsed_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  parsed: ok={rep['ok']} kept={rep['n_kept_edits']}/{len(parsed.get('edits', []))} "
          f"types={rep.get('type_counts')}")
    if rep["errors"]:
        print(f"  ERRORS: {rep['errors']}")
    if rep["warnings"]:
        print(f"  WARNINGS ({len(rep['warnings'])} edits with issues):")
        for w in rep["warnings"][:5]:
            print(f"    {w}")
    print(f"    → {parsed_path}")


def prerender_one(obj_id: str, shard: str, mesh_root: Path, images_root: Path,
                  blender: str, out_dir: Path):
    """Render overview + build menu. Returns (obj_id, png, user_prompt, pids, quota) or None."""
    try:
        mesh_npz = mesh_root / shard / f"{obj_id}.npz"
        img_npz = images_root / shard / f"{obj_id}.npz"
        pids, menu = build_part_menu(mesh_npz, img_npz)
        if len(pids) > MAX_PARTS:
            print(f"  [SKIP] {obj_id}: {len(pids)} parts > MAX_PARTS={MAX_PARTS}")
            return None
        quota = quota_for(len(pids))
        png = render_overview_png(mesh_npz, img_npz, blender)
        (out_dir / f"{obj_id}.overview.png").write_bytes(png)
        user_msg = USER_PROMPT_TEMPLATE.format(
            part_menu=menu, n_total=sum(quota.values()),
            n_deletion=quota["deletion"], n_modification=quota["modification"],
            n_scale=quota["scale"], n_material=quota["material"],
            n_global=quota["global"])
        return obj_id, png, user_msg, pids, quota
    except Exception as e:
        print(f"  [PRERENDER FAIL] {obj_id}: {e}")
        return None


async def vlm_one(client, obj_id, png, user_msg, valid_pids, quota, model, out_dir, sem):
    async with sem:
        t0 = time.time()
        try:
            raw = await call_vlm_async(client, png, SYSTEM_PROMPT, user_msg, model,
                                       max_tokens=12288)
        except Exception as e:
            print(f"  [VLM FAIL] {obj_id}: {e}")
            return obj_id, None
        dt = time.time() - t0
        (out_dir / f"{obj_id}.raw.txt").write_text(raw)
        parsed = extract_json_object(raw)
        if parsed is None:
            print(f"  [{obj_id}] {dt:.1f}s  PARSE_ERROR  raw_len={len(raw)}")
            return obj_id, None
        rep = validate(parsed, set(valid_pids), quota=quota)
        out = {"obj_id": obj_id, "validation": rep, "parsed": parsed}
        (out_dir / f"{obj_id}.parsed.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False))
        target = sum(quota.values())
        print(f"  [{obj_id}] {dt:.1f}s  N={len(valid_pids)}  ok={rep['ok']}  "
              f"kept={rep['n_kept_edits']}/{target}  types={rep.get('type_counts')}")
        return obj_id, rep


async def run_async(args, jobs):
    from openai import AsyncOpenAI
    urls = [u.strip() for u in args.vlm_url.split(",") if u.strip()]
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in urls]
    # One semaphore per server → each server stays sequential (avoid KV thrash)
    sems = [asyncio.Semaphore(1) for _ in clients]
    print(f"  using {len(clients)} VLM server(s): {urls}")
    tasks = []
    for i, (oid, png, msg, pids, quota) in enumerate(jobs):
        idx = i % len(clients)
        tasks.append(vlm_one(clients[idx], oid, png, msg, pids, quota,
                             args.vlm_model, args.out_dir, sems[idx]))
    results = await asyncio.gather(*tasks)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-ids", nargs="+", required=True)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--mesh-root", default="data/partverse/mesh", type=Path)
    ap.add_argument("--images-root", default="data/partverse/images", type=Path)
    ap.add_argument("--blender",
                    default="/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender")
    ap.add_argument("--vlm-url", default="http://localhost:8002/v1")
    ap.add_argument("--vlm-model", default="Qwen3.5-27B")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/_debug/phase1_v2"))
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Max concurrent VLM requests. Default 1 (sequential). "
                         "Higher values cause KV-cache thrashing on this 27B model — "
                         "use multi-GPU servers for parallelism instead.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: pre-render all overviews + build menus (sequential, fast with Workbench)
    print(f"[A] Pre-rendering {len(args.obj_ids)} overviews…")
    t0 = time.time()
    jobs = []
    for oid in args.obj_ids:
        r = prerender_one(oid, args.shard, args.mesh_root, args.images_root,
                          args.blender, args.out_dir)
        if r is not None:
            jobs.append(r)
    print(f"[A] Done in {time.time() - t0:.1f}s ({len(jobs)}/{len(args.obj_ids)} ok)")

    # Phase B: concurrent VLM calls
    print(f"[B] Calling VLM with concurrency={args.concurrency}…")
    t0 = time.time()
    results = asyncio.run(run_async(args, jobs))
    dt = time.time() - t0
    n_ok = sum(1 for _, r in results if r and r.get("ok"))
    n_total = len(results)
    print(f"[B] Done in {dt:.1f}s — {n_ok}/{n_total} ok  "
          f"({dt/max(1,n_total):.1f}s/obj wall, "
          f"{dt*args.concurrency/max(1,n_total):.1f}s/obj cpu-equiv)")


if __name__ == "__main__":
    main()
