#!/usr/bin/env python3
"""Two-stage (Mode D) benchmark on bench_shard08 (20 objects).

Stage 1: image + colour-only menu  →  s1_parts JSON  (visual grounding)
Stage 2: s1_parts text             →  edit JSON       (text-only, no image)

Usage:
    python scripts/tools/run_two_stage_test.py \
        --vlm-urls http://localhost:8142/v1,http://localhost:8143/v1 \
        --vlm-model <model-name> \
        --output-dir outputs/partverse/bench_shard08/mode_d_two_stage
"""
from __future__ import annotations

import argparse, asyncio, json, logging, time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("two_stage")

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
OBJ_IDS_FILE   = REPO_ROOT / "configs" / "shard08_test_obj_ids.txt"
OVERVIEWS_ROOT = REPO_ROOT / "outputs" / "partverse" / "bench_shard08_overviews"
MESH_ROOT      = Path("/mnt/zsn/data/partverse/bench/inputs/mesh")
IMAGES_ROOT    = Path("/mnt/zsn/data/partverse/bench/inputs/images")
SHARD          = "08"

import sys
sys.path.insert(0, str(REPO_ROOT))
from partcraft.pipeline_v3.vlm_core import (
    SYSTEM_PROMPT_S1, SYSTEM_PROMPT_S2,
    build_image_only_menu,     # Stage 1 uses colour-only menu
    build_s1_user_prompt,
    build_s2_user_prompt,
    parse_s1_output,
    call_vlm_image_async,            # Stage 1: image call
    call_vlm_text_async,       # Stage 2: text-only call
    extract_json_object,
    validate_edit_json,
    compute_edit_quota,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_obj_ids() -> list[str]:
    return [l.strip() for l in OBJ_IDS_FILE.read_text().splitlines()
            if l.strip() and not l.startswith("#")]

def phase1_dir(output_root: Path, obj_id: str) -> Path:
    return output_root / "objects" / SHARD / obj_id / "phase1"

def src_overview(obj_id: str) -> Path:
    return OVERVIEWS_ROOT / "objects" / SHARD / obj_id / "phase1" / "overview.png"

def mesh_npz(obj_id: str) -> Path:
    return MESH_ROOT / SHARD / f"{obj_id}.npz"

def img_npz(obj_id: str) -> Path:
    return IMAGES_ROOT / SHARD / f"{obj_id}.npz"


# ── per-object worker ─────────────────────────────────────────────────────────

async def process_one(
    obj_id: str,
    output_root: Path,
    vlm_urls: list[str],
    vlm_model: str,
    worker_idx: int,
    force: bool = False,
) -> dict:
    """Run Stage 1 then Stage 2 for one object. Returns a result dict."""
    out_dir = phase1_dir(output_root, obj_id)
    result_path = out_dir / "parsed.json"

    if not force and result_path.exists():
        try:
            d = json.loads(result_path.read_text())
            if d.get("validation", {}).get("ok"):
                log.info(f"[skip] {obj_id}  (valid parsed.json exists)")
                return {"obj_id": obj_id, "status": "skipped",
                        "s1_ok": True, "s2_ok": True,
                        "n_edits": len(d.get("parsed",{}).get("edits",[]))}
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)

    # symlink overview
    ov_src = src_overview(obj_id)
    ov_dst = out_dir / "overview.png"
    if not ov_dst.exists() and ov_src.exists():
        ov_dst.symlink_to(ov_src)

    url   = vlm_urls[worker_idx % len(vlm_urls)]
    t0    = time.perf_counter()

    # ── Stage 1: image → part semantics ──────────────────────────────────────
    try:
        pids, _ = build_image_only_menu(mesh_npz(obj_id), img_npz(obj_id))
    except Exception as e:
        log.warning(f"[{obj_id}] menu build failed: {e}")
        return {"obj_id": obj_id, "status": "error", "error": f"menu: {e}"}

    s1_user = build_s1_user_prompt(pids)
    overview_bytes = ov_src.read_bytes() if ov_src.exists() else None
    if overview_bytes is None:
        return {"obj_id": obj_id, "status": "error", "error": "no overview.png"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=url, api_key="EMPTY")

    try:
        s1_raw = await call_vlm_image_async(client, overview_bytes,
                                       SYSTEM_PROMPT_S1, s1_user,
                                       vlm_model, max_tokens=1024)
    except Exception as e:
        log.warning(f"[{obj_id}] Stage 1 API error: {e}")
        return {"obj_id": obj_id, "status": "error", "error": f"s1_api: {e}"}

    (out_dir / "s1_raw.txt").write_text(s1_raw, encoding="utf-8")

    s1_parsed = parse_s1_output(s1_raw)
    s1_ok = s1_parsed is not None
    if not s1_ok:
        log.warning(f"[{obj_id}] Stage 1 parse failed")
        (out_dir / "parsed.json").write_text(json.dumps({
            "obj_id": obj_id, "mode": "two_stage",
            "validation": {"ok": False, "errors": ["s1_parse_failed"]},
            "s1_raw": s1_raw,
        }, indent=2), encoding="utf-8")
        return {"obj_id": obj_id, "status": "s1_fail", "s1_ok": False, "s2_ok": False}

    (out_dir / "s1_parsed.json").write_text(
        json.dumps(s1_parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    t_s1 = time.perf_counter() - t0
    n_vis = len(s1_parsed["parts_visible"])
    n_hid = len(s1_parsed["parts_hidden"])
    log.info(f"[{obj_id}] S1 done in {t_s1:.1f}s  "
             f"({n_vis} visible, {n_hid} null/hidden)")

    # ── Stage 2: text → edits (visible parts only) ───────────────────────────
    quota   = compute_edit_quota(n_vis)   # quota based on visible parts count
    s2_user = build_s2_user_prompt(
        s1_parsed["parts_visible"],
        s1_parsed.get("object_desc", ""),
        quota,
        variety_seed=hash(obj_id),
    )

    try:
        s2_raw = await call_vlm_text_async(client, SYSTEM_PROMPT_S2, s2_user,
                                            vlm_model, max_tokens=4096)
    except Exception as e:
        log.warning(f"[{obj_id}] Stage 2 API error: {e}")
        return {"obj_id": obj_id, "status": "error", "s1_ok": True,
                "error": f"s2_api: {e}"}

    (out_dir / "raw.txt").write_text(s2_raw, encoding="utf-8")

    # parse + validate Stage 2
    s2_parsed = extract_json_object(s2_raw)  # already returns dict | None
    validation = {"ok": False, "errors": ["no_json"]}
    if s2_parsed is not None:
        validation = validate_edit_json(s2_parsed, set(pids), quota)

    # Merge s1 part names back into the parsed object for provenance
    all_parts = s1_parsed["parts_visible"] + s1_parsed["parts_hidden"]
    parts_lookup = {p["part_id"]: p for p in all_parts}
    if s2_parsed and "edits" in s2_parsed:
        # attach s1 part info to parsed output
        s2_parsed["object"] = {
            "full_desc": s1_parsed.get("object_desc", ""),
            "parts": [
                {"part_id": p["part_id"], "name": p.get("name")}
                for p in sorted(all_parts, key=lambda x: x["part_id"])
            ],
        }

    elapsed = time.perf_counter() - t0
    result_doc = {
        "obj_id":        obj_id,
        "mode":          "two_stage",
        "validation":    validation,
        "parsed":        s2_parsed or {},
        "s1_visible":    s1_parsed["parts_visible"],
        "s1_hidden":     s1_parsed["parts_hidden"],
    }
    result_path.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    n_edits = len((s2_parsed or {}).get("edits", []))
    status  = "ok" if validation["ok"] else "fail"
    log.info(f"[{obj_id}] S2 {status}  {n_edits} edits  total {elapsed:.1f}s")

    return {
        "obj_id":  obj_id, "status": status,
        "s1_ok":   s1_ok,  "s2_ok": validation["ok"],
        "n_edits": n_edits, "elapsed": elapsed,
        "n_errors": len(validation.get("errors", [])),
        "n_warnings": len(validation.get("warnings", [])),
    }


# ── main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    obj_ids  = load_obj_ids()
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",")]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Two-stage test: {len(obj_ids)} objects, {len(vlm_urls)} VLM server(s)")
    log.info(f"Output: {output_root}")

    # Dispatch: round-robin across VLM servers with limited concurrency
    sem  = asyncio.Semaphore(len(vlm_urls) * 2)
    wall = time.perf_counter()

    async def _guarded(obj_id: str, idx: int) -> dict:
        async with sem:
            return await process_one(obj_id, output_root, vlm_urls,
                                     args.vlm_model, idx, args.force)

    tasks   = [_guarded(oid, i) for i, oid in enumerate(obj_ids)]
    results = await asyncio.gather(*tasks)

    # Summary
    elapsed_total = time.perf_counter() - wall
    ok      = sum(1 for r in results if r.get("s2_ok"))
    s1_fail = sum(1 for r in results if not r.get("s1_ok"))
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    summary = {
        "mode": "two_stage",
        "n_objects":  len(obj_ids),
        "n_s1_fail":  s1_fail,
        "n_s2_pass":  ok,
        "n_skipped":  skipped,
        "pass_rate":  round(ok / len(obj_ids), 3),
        "wall_sec":   round(elapsed_total, 1),
        "objects":    results,
    }
    summary_path = output_root / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info(f"S1 parse fail : {s1_fail}/{len(obj_ids)}")
    log.info(f"S2 pass       : {ok}/{len(obj_ids)}  ({summary['pass_rate']:.0%})")
    log.info(f"Wall time     : {elapsed_total:.0f}s")
    log.info(f"Summary       : {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm-urls",   required=True,
                    help="Comma-separated VLM server URLs")
    ap.add_argument("--vlm-model",  required=True,
                    help="Model name (as registered in the VLM server)")
    ap.add_argument("--output-dir", default="outputs/partverse/bench_shard08/mode_d_two_stage",
                    help="Output root directory")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if parsed.json already exists")
    asyncio.run(main(ap.parse_args()))
