#!/usr/bin/env python3
"""Mode-comparison test for bench_shard08 (20 objects).

Runs one prompt mode against all 20 benchmark objects and writes results to
a mode-specific output folder.

Usage (launch two modes in parallel on 2 servers each):

    # Terminal 1 — Mode A (image + semantic menu), GPUs 4,5
    python scripts/tools/run_mode_test_shard08.py \
        --mode image_semantic \
        --vlm-urls http://localhost:8142/v1,http://localhost:8143/v1

    # Terminal 2 — Mode C (image + colour-only menu), GPUs 6,7
    python scripts/tools/run_mode_test_shard08.py \
        --mode image_only \
        --vlm-urls http://localhost:8144/v1,http://localhost:8145/v1

Output per object:
    outputs/partverse/bench_shard08/mode_{a,c}_{label}/objects/08/{obj_id}/phase1/
        overview.png   <- symlink to pre-rendered bench_shard08_overviews
        parsed.json    <- {obj_id, mode, validation, parsed:{object,edits}}
        raw.txt        <- raw VLM completion

Resume: objects with a valid parsed.json are skipped automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mode_test")

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
OBJ_IDS_FILE  = REPO_ROOT / "configs" / "shard08_test_obj_ids.txt"
OVERVIEWS_ROOT = REPO_ROOT / "outputs" / "partverse" / "bench_shard08_overviews"
MESH_ROOT     = Path("/mnt/zsn/data/partverse/bench/inputs/mesh")
IMAGES_ROOT   = Path("/mnt/zsn/data/partverse/bench/inputs/images")
SHARD         = "08"


# ── path helpers ──────────────────────────────────────────────────────────────

def load_obj_ids() -> list[str]:
    return [l.strip() for l in OBJ_IDS_FILE.read_text().splitlines()
            if l.strip() and not l.startswith("#")]

def phase1_dir(output_root: Path, obj_id: str) -> Path:
    return output_root / "objects" / SHARD / obj_id / "phase1"

def src_overview(obj_id: str) -> Path:
    return OVERVIEWS_ROOT / "objects" / SHARD / obj_id / "phase1" / "overview.png"


# ── prerender worker (CPU, runs in ProcessPoolExecutor) ───────────────────────

def _prerender(obj_id: str, mode: str) -> tuple | None:
    """Build menu + quota + formatted prompts for one object. Returns None on skip."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from partcraft.pipeline_v3.s1_vlm_core import (
        build_image_semantic_menu, build_image_only_menu, build_semantic_list,
        build_prompt_for_mode, quota_for,
    )
    mnpz = MESH_ROOT   / SHARD / f"{obj_id}.npz"
    inpz = IMAGES_ROOT / SHARD / f"{obj_id}.npz"
    ov   = src_overview(obj_id)
    if not mnpz.is_file() or not inpz.is_file() or not ov.is_file():
        return None

    if mode == "image_semantic":
        pids, menu = build_image_semantic_menu(mnpz, inpz)
    elif mode == "image_only":
        pids, menu = build_image_only_menu(mnpz, inpz)
    else:
        pids, menu = build_semantic_list(mnpz, inpz)

    quota = quota_for(len(pids))
    sys_p, usr_p = build_prompt_for_mode(mode, pids, menu, quota, variety_seed=hash(obj_id))
    return ov.read_bytes(), sys_p, usr_p, pids, quota


# ── single VLM call ───────────────────────────────────────────────────────────

async def _call_one(client, sem, obj_id, mode, png, sys_p, usr_p,
                    pids, quota, vlm_model, out_dir) -> dict:
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from partcraft.pipeline_v3.s1_vlm_core import (
        call_vlm_async, call_vlm_text_async, extract_json_object, validate_simple,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_link = out_dir / "overview.png"
    if not ov_link.exists():
        try:
            ov_link.symlink_to(src_overview(obj_id).resolve())
        except Exception:
            pass

    async with sem:
        t0 = time.time()
        try:
            if mode == "text_semantic":
                raw = await call_vlm_text_async(client, sys_p, usr_p, vlm_model, max_tokens=8192)
            else:
                raw = await call_vlm_async(client, png, sys_p, usr_p, vlm_model, max_tokens=8192)
        except Exception as e:
            log.warning("%s [%s] VLM error: %s", obj_id[:12], mode, e)
            return {"obj_id": obj_id, "mode": mode, "ok": False, "error": str(e)}

        elapsed = round(time.time() - t0, 1)
        (out_dir / "raw.txt").write_text(raw)

        parsed = extract_json_object(raw)
        if parsed is None:
            log.warning("%s [%s] parse_error  raw_len=%d", obj_id[:12], mode, len(raw))
            rec = {"obj_id": obj_id, "mode": mode, "ok": False,
                   "error": "parse_error", "raw_len": len(raw), "wall_s": elapsed}
            (out_dir / "parsed.json").write_text(json.dumps(rec, indent=2))
            return rec

        rep = validate_simple(parsed, set(pids), quota=quota)
        out = {"obj_id": obj_id, "mode": mode, "validation": rep, "parsed": parsed}
        (out_dir / "parsed.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log.info("%s [%s] %s  kept=%d/%d  %.1fs",
                 obj_id[:12], mode, "OK" if rep["ok"] else "PARTIAL",
                 rep["n_kept_edits"], sum(quota.values()), elapsed)
        return {"obj_id": obj_id, "mode": mode, "ok": rep["ok"],
                "n_kept": rep["n_kept_edits"], "n_total": sum(quota.values()),
                "type_counts": rep.get("type_counts"), "wall_s": elapsed}


# ── streaming runner ──────────────────────────────────────────────────────────

async def run(mode, vlm_urls, vlm_model, output_root, obj_ids, force) -> list[dict]:
    from openai import AsyncOpenAI
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in vlm_urls]
    sems    = [asyncio.Semaphore(1) for _ in clients]

    todo, results = [], []
    for obj_id in obj_ids:
        p = phase1_dir(output_root, obj_id) / "parsed.json"
        if not force and p.is_file():
            try:
                d = json.loads(p.read_text())
                if d.get("parsed", {}).get("edits") is not None:
                    log.info("%s [%s] skip (done)", obj_id[:12], mode)
                    results.append({"obj_id": obj_id, "mode": mode, "ok": True, "resumed": True})
                    continue
            except Exception:
                pass
        todo.append(obj_id)

    log.info("mode=%s  todo=%d  resume=%d  servers=%d",
             mode, len(todo), len(results), len(vlm_urls))
    if not todo:
        return results

    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=min(8, len(todo)))
    queue: asyncio.Queue = asyncio.Queue(maxsize=2 * len(clients))

    async def producer():
        pre_sem = asyncio.Semaphore(16)
        async def _one(obj_id):
            async with pre_sem:
                try:
                    pre = await loop.run_in_executor(pool, _prerender, obj_id, mode)
                except Exception as e:
                    log.warning("%s prerender failed: %s", obj_id[:12], e); return
                if pre is None:
                    log.warning("%s missing inputs, skipped", obj_id[:12]); return
                await queue.put((obj_id, pre))
        await asyncio.gather(*[_one(o) for o in todo])
        for _ in range(len(clients)):
            await queue.put(None)

    async def consumer(idx):
        client, sem = clients[idx], sems[idx]
        while True:
            item = await queue.get()
            if item is None:
                return
            obj_id, (png, sys_p, usr_p, pids, quota) = item
            res = await _call_one(client, sem, obj_id, mode,
                                  png, sys_p, usr_p, pids, quota, vlm_model,
                                  phase1_dir(output_root, obj_id))
            results.append(res)

    try:
        await asyncio.gather(producer(), *[consumer(i) for i in range(len(clients))])
    finally:
        pool.shutdown(wait=False)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=["image_semantic", "image_only", "text_semantic"])
    ap.add_argument("--vlm-urls", required=True,
                    help="Comma-separated VLM URLs, e.g. http://localhost:8142/v1,...")
    ap.add_argument("--vlm-model", default="/mnt/zsn/ckpts/Qwen3.5-27B")
    ap.add_argument("--output-dir",
                    help="Override output root (default: outputs/partverse/bench_shard08/mode_X)")
    ap.add_argument("--force", action="store_true", help="Re-run even if parsed.json exists")
    args = ap.parse_args()

    mode = args.mode
    tag  = {"image_semantic": "a_image_semantic",
             "image_only":    "c_image_only",
             "text_semantic": "b_text_semantic"}[mode]
    out_root = (Path(args.output_dir) if args.output_dir
                else REPO_ROOT / "outputs" / "partverse" / "bench_shard08" / f"mode_{tag}")
    out_root.mkdir(parents=True, exist_ok=True)

    urls    = [u.strip() for u in args.vlm_urls.split(",") if u.strip()]
    obj_ids = load_obj_ids()

    log.info("=== bench_shard08 mode test: %s ===", mode)
    log.info("Objects=%d  Servers=%d  Output=%s", len(obj_ids), len(urls), out_root)

    t0 = time.time()
    results = asyncio.run(run(mode, urls, args.vlm_model, out_root, obj_ids, args.force))
    elapsed = time.time() - t0

    n_ok = sum(1 for r in results if r.get("ok"))
    log.info("=== DONE  ok=%d/%d  %.0fs ===", n_ok, len(results), elapsed)

    print(f"\n{'obj_id':<36}  {'':>2} {'kept':>5} {'tot':>4}  type_counts")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["obj_id"]):
        tc  = "  ".join(f"{k}={v}" for k, v in sorted((r.get("type_counts") or {}).items()))
        err = r.get("error", "")
        print(f"{r['obj_id']:<36}  {'OK' if r.get('ok') else 'FAIL':>2}"
              f" {str(r.get('n_kept', '-')):>5} {str(r.get('n_total', '-')):>4}  {tc or err}")

    summary = out_root / "_summary.json"
    summary.write_text(json.dumps({"mode": mode, "n_objects": len(results),
        "n_ok": n_ok, "total_wall_s": round(elapsed, 1), "results": results},
        indent=2, ensure_ascii=False))
    log.info("Summary -> %s", summary)


if __name__ == "__main__":
    main()
