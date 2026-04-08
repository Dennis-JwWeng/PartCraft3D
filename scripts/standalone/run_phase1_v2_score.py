#!/usr/bin/env python3
"""Phase 1.5 — VLM-based scoring of phase1_v2 edits.

Reads phase1_v2 outputs (one *.parsed.json per object), renders a per-edit
highlight image, asks a Qwen3.5-VL server to score the edit (selection
correctness + plausibility + prompt clarity), and writes a side-car
*.scored.json next to the input. Original files are not modified.

Pipeline:
  Phase A — render single-view highlight per edit (Blender Workbench, fast)
            cached at  <in_dir>/_hl/{obj_id}__e{edit_idx}.png
  Phase B — for each edit, build a 2-image stitch (ORIGINAL | HIGHLIGHT) +
            judge prompt → VLM → parse → tier
  Phase C — write {obj_id}.scored.json (full parsed copy + per-edit `score`
            field + top-level `scoring` summary), and a global _summary.json

Multi-GPU: pass --vlm-url with comma-separated server URLs (one per GPU).
Each server runs sequentially via its own asyncio.Semaphore(1) — same
discipline as run_phase1_v2.py to avoid SGLang KV-cache thrashing.

Usage:
    python scripts/standalone/run_phase1_v2_score.py \
        --in-dir outputs/_debug/phase1_v2_5view_morequota \
        --shard 01 \
        --vlm-url http://localhost:8002/v1,http://localhost:8003/v1
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import datetime as _dt
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "tools"))

from render_part_highlight import render_highlight, stitch_pair  # noqa: E402
from render_part_overview import VIEW_INDICES, load_views_from_npz  # noqa: E402

from partcraft.scoring.judge_prompts import (  # noqa: E402
    SYSTEM_PROMPT, build_user_prompt, parse_score,
)


# ─────────────────────────── small helpers ──────────────────────────────────

def _png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def _extract_json_object(text: str) -> dict | None:
    """Robust outermost-{...} parse (same as run_phase1_v2)."""
    text = (text or "").strip()
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
            esc = False; continue
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


# ─────────────────────────── Phase A: rendering ─────────────────────────────

def _hl_path(out_dir: Path, obj_id: str, edit_idx: int) -> Path:
    return out_dir / "_hl" / f"{obj_id}__e{edit_idx:02d}.png"


def render_one_edit(mesh_npz: Path, img_npz: Path, edit: dict,
                    blender: str, cache_path: Path,
                    orig_cache: dict[int, np.ndarray]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Render (or load from cache) the highlight image for ONE edit. Returns
    (original_bgr, highlight_bgr) for the edit's chosen view_index."""
    vi = int(edit["view_index"])
    pids = list(edit.get("selected_part_ids") or [])

    # Original view (cache per obj across all its edits)
    if vi not in orig_cache:
        top_imgs, _frames = load_views_from_npz(img_npz, VIEW_INDICES)
        for k, im in enumerate(top_imgs):
            orig_cache[k] = im
    orig = orig_cache[vi]

    if cache_path.is_file():
        hl = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
        if hl is not None:
            return orig, hl

    # Global edits have no selection — there's no useful highlight to render.
    # We just write the original photo as the "highlight" so downstream code
    # is uniform; the VLM prompt tells the model to ignore the right panel.
    if not pids:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache_path), orig)
        return orig, orig.copy()

    _orig, hl = render_highlight(mesh_npz, img_npz, vi, pids, blender)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), hl)
    return orig, hl


# ─────────────────────────── Phase B: VLM call ──────────────────────────────

async def _call_vlm(client, png: bytes, user_msg: str, model: str,
                    max_tokens: int = 1024) -> str:
    b64 = base64.b64encode(png).decode()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": [
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{b64}"}},
                 {"type": "text", "text": user_msg},
             ]},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        timeout=300,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


async def score_one_edit(client, png: bytes, edit: dict, model: str,
                         sem: asyncio.Semaphore) -> dict:
    user_msg = build_user_prompt(edit)
    async with sem:
        try:
            raw = await _call_vlm(client, png, user_msg, model)
        except Exception as e:
            return parse_score(None) | {"_err": f"vlm_call_failed: {e}"}
    obj = _extract_json_object(raw)
    return parse_score(obj)


# ─────────────────────────── per-object orchestration ───────────────────────

async def process_obj(parsed_path: Path, args, clients, sems, model: str
                      ) -> dict | None:
    obj_id = parsed_path.stem.replace(".parsed", "")
    out_path = parsed_path.with_suffix("").with_suffix(".scored.json")
    if out_path.is_file() and not args.force:
        return {"obj_id": obj_id, "skipped": True}

    j = json.loads(parsed_path.read_text())
    parsed = j.get("parsed") or {}
    edits = parsed.get("edits") or []
    if not edits:
        return {"obj_id": obj_id, "skipped": True, "reason": "no_edits"}

    mesh = args.mesh_root / args.shard / f"{obj_id}.npz"
    img = args.images_root / args.shard / f"{obj_id}.npz"
    for p in (mesh, img):
        if not p.is_file():
            print(f"  [SKIP] {obj_id}: missing {p.name}")
            return None

    # ── Phase A (sync, but cheap): render every highlight ──
    orig_cache: dict[int, np.ndarray] = {}
    rendered: list[tuple[int, dict, np.ndarray, np.ndarray]] = []
    t0 = time.time()
    for idx, e in enumerate(edits):
        if args.edit_types and e.get("edit_type") not in args.edit_types:
            continue
        try:
            cache = _hl_path(args.in_dir, obj_id, idx)
            orig, hl = render_one_edit(mesh, img, e, args.blender, cache,
                                       orig_cache)
            rendered.append((idx, e, orig, hl))
        except Exception as ex:
            print(f"  [HL FAIL] {obj_id} edit {idx}: {ex}")
    dt_a = time.time() - t0

    # ── Phase B (async per server): VLM scoring ──
    t0 = time.time()
    tasks = []
    for k, (idx, e, orig, hl) in enumerate(rendered):
        panel = stitch_pair(orig, hl, edit_type=e.get("edit_type", ""),
                            prompt=(e.get("prompt") or "")[:160])
        png = _png_bytes(panel)
        client = clients[k % len(clients)]
        sem = sems[k % len(sems)]
        tasks.append(score_one_edit(client, png, e, model, sem))
    scores = await asyncio.gather(*tasks)
    dt_b = time.time() - t0

    # ── Phase C: assemble + write ──
    by_idx = {idx: scores[k] for k, (idx, *_rest) in enumerate(rendered)}
    new_edits = []
    for idx, e in enumerate(edits):
        if idx in by_idx:
            new_edits.append({**e, "score": by_idx[idx]})
        else:
            new_edits.append(e)

    tier_count: Counter = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    for e in new_edits:
        s = e.get("score")
        if not s:
            continue
        t = s.get("tier", "low")
        tier_count[t] += 1
        by_type[e.get("edit_type", "?")][t] += 1

    out = {
        "obj_id": obj_id,
        "scoring": {
            "model": model,
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "n_total": sum(tier_count.values()),
            "n_high": tier_count["high"],
            "n_medium": tier_count["medium"],
            "n_low": tier_count["low"],
            "n_reject": tier_count["reject"],
            "by_type": {k: dict(v) for k, v in by_type.items()},
            "phase_a_seconds": round(dt_a, 2),
            "phase_b_seconds": round(dt_b, 2),
        },
        "validation": j.get("validation"),
        "parsed": {**parsed, "edits": new_edits},
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    summary = out["scoring"]
    print(f"  [{obj_id}] A={dt_a:.1f}s B={dt_b:.1f}s  "
          f"n={summary['n_total']}  high={summary['n_high']} "
          f"med={summary['n_medium']} low={summary['n_low']} "
          f"rej={summary['n_reject']}")
    return out


# ─────────────────────────── main ───────────────────────────────────────────

async def run(args):
    from openai import AsyncOpenAI

    in_files = sorted(args.in_dir.glob("*.parsed.json"))
    if args.obj_ids:
        keep = set(args.obj_ids)
        in_files = [p for p in in_files
                    if p.stem.replace(".parsed", "") in keep]
    print(f"[in] {len(in_files)} objects in {args.in_dir}")

    urls = [u.strip() for u in args.vlm_url.split(",") if u.strip()]
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in urls]
    sems = [asyncio.Semaphore(1) for _ in clients]
    print(f"[vlm] {len(clients)} server(s): {urls}")

    summaries = []
    for i, p in enumerate(in_files):
        print(f"[{i + 1}/{len(in_files)}] {p.name}")
        res = await process_obj(p, args, clients, sems, args.vlm_model)
        if res:
            summaries.append(res)

    # global summary
    g_tier = Counter()
    g_by_type: dict[str, Counter] = defaultdict(Counter)
    for s in summaries:
        sc = (s or {}).get("scoring") or {}
        for k in ("high", "medium", "low", "reject"):
            g_tier[k] += sc.get(f"n_{k}", 0)
        for et, d in (sc.get("by_type") or {}).items():
            for tier, n in d.items():
                g_by_type[et][tier] += n
    summary_path = args.in_dir / "_scoring_summary.json"
    summary_path.write_text(json.dumps({
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_objs": len([s for s in summaries if (s or {}).get("scoring")]),
        "n_skipped": len([s for s in summaries if (s or {}).get("skipped")]),
        "tiers": dict(g_tier),
        "by_type": {k: dict(v) for k, v in g_by_type.items()},
    }, indent=2, ensure_ascii=False))
    total = sum(g_tier.values())
    print(f"\n[done] {total} edits scored "
          f"high={g_tier['high']} med={g_tier['medium']} "
          f"low={g_tier['low']} rej={g_tier['reject']}")
    print(f"       summary → {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path,
                    help="phase1_v2 output dir containing *.parsed.json")
    ap.add_argument("--shard", default="01")
    ap.add_argument("--mesh-root", default="data/partverse/mesh", type=Path)
    ap.add_argument("--images-root", default="data/partverse/images", type=Path)
    ap.add_argument("--blender",
                    default="/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender")
    ap.add_argument("--vlm-url", default="http://localhost:8002/v1",
                    help="comma-separated server URLs (one per GPU)")
    ap.add_argument("--vlm-model", default="Qwen3.5-27B")
    ap.add_argument("--obj-ids", nargs="*", default=None,
                    help="restrict to these obj ids (default: all in --in-dir)")
    ap.add_argument("--edit-types", nargs="*", default=None,
                    help="only score edits of these types")
    ap.add_argument("--force", action="store_true",
                    help="re-score even if .scored.json already exists")
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
