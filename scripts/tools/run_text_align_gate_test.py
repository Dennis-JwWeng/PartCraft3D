#!/usr/bin/env python3
"""Mode E: Text-only edit generation (Mode B) + per-edit VLM alignment gate.

Phase 1 (text): build_semantic_list -> call_vlm_text_async(SYSTEM_PROMPT_B)
Phase 2 (image): for each edit, build 5x2 gate image -> call_vlm_async(SYSTEM_PROMPT_ALIGN_GATE)

Usage:
    python scripts/tools/run_text_align_gate_test.py \
        --vlm-urls http://localhost:8142/v1,http://localhost:8143/v1 \
        --vlm-model <model-name> \
        [--output-dir <path>]  [--concurrency 4]
"""
from __future__ import annotations

import argparse, asyncio, json, logging, time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("text_align_gate")

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
OBJ_IDS_FILE   = REPO_ROOT / "configs" / "shard08_test_obj_ids.txt"
MESH_ROOT      = Path("/mnt/zsn/data/partverse/bench/inputs/mesh")
IMAGES_ROOT    = Path("/mnt/zsn/data/partverse/bench/inputs/images")
OVERVIEWS_ROOT = REPO_ROOT / "outputs" / "partverse" / "bench_shard08_overviews"
SHARD          = "08"

import sys
sys.path.insert(0, str(REPO_ROOT))

from partcraft.pipeline_v3.s1_vlm_core import (
    SYSTEM_PROMPT_B,
    SYSTEM_PROMPT_ALIGN_GATE,
    build_semantic_list,
    build_align_gate_user_prompt,
    parse_align_gate_output,
    call_vlm_async,
    call_vlm_text_async,
    extract_json_object,
    validate_simple,
    quota_for,
)
from partcraft.pipeline_v3.qc_rules import (
    check_rules,
    count_part_pixels_in_overview,
    _PALETTE_BGR, _N_VIEWS, _COL_SEP, _ROW_SEP,
)

# ── gate image constants ──────────────────────────────────────────────────────
_RED  = (45, 45, 220)    # BGR
_GREY = (65, 65, 65)


def _extract_cell(img: np.ndarray, col: int, row: int) -> np.ndarray:
    H_total, W_total = img.shape[:2]
    W_cell = (W_total - (_N_VIEWS - 1) * _COL_SEP) // _N_VIEWS
    H_cell = (H_total - _ROW_SEP) // 2
    x0 = col * (W_cell + _COL_SEP)
    y0 = row * (H_cell + _ROW_SEP)
    return img[y0: y0 + H_cell, x0: x0 + W_cell].copy()


def _highlight_cell(cell: np.ndarray, selected_part_ids: set[int]) -> np.ndarray:
    """Recolour bottom-row palette cell: selected → red, others → grey."""
    palette = np.array(_PALETTE_BGR, dtype=np.int32)
    flat    = cell.reshape(-1, 3).astype(np.int32)
    diffs   = np.linalg.norm(flat[:, None, :] - palette[None, :, :], axis=2)
    nearest = np.argmin(diffs, axis=1)
    is_bg   = np.all(flat > 230, axis=1)
    is_sel  = np.array([pid in selected_part_ids for pid in nearest])
    out_flat = np.empty_like(flat)
    out_flat[is_bg]            = [255, 255, 255]
    out_flat[~is_bg & is_sel]  = list(_RED)
    out_flat[~is_bg & ~is_sel] = list(_GREY)
    return out_flat.reshape(cell.shape).astype(np.uint8)


def build_gate_image(
    ov_img: np.ndarray,
    selected_part_ids: list[int],
    column_map: list[int],
) -> bytes:
    """5×2 gate image: col 0 = highlighted best view, cols 1-4 = normal context."""
    sel_set = set(selected_part_ids)
    top_cells, bot_cells = [], []
    for col_idx, orig_view in enumerate(column_map):
        top = _extract_cell(ov_img, orig_view, 0)
        bot = _extract_cell(ov_img, orig_view, 1)
        if col_idx == 0 and sel_set:
            bot = _highlight_cell(bot, sel_set)
        top_cells.append(top)
        bot_cells.append(bot)

    sep_v = np.full((top_cells[0].shape[0], _COL_SEP, 3), 255, dtype=np.uint8)
    sep_h = np.full(
        (_ROW_SEP,
         sum(c.shape[1] for c in top_cells) + (_N_VIEWS - 1) * _COL_SEP, 3),
        255, dtype=np.uint8,
    )

    def hstack(cells):
        row = cells[0]
        for c in cells[1:]:
            sv = np.full((c.shape[0], _COL_SEP, 3), 255, dtype=np.uint8)
            row = np.concatenate([row, sv, c], axis=1)
        return row

    full = np.concatenate([hstack(top_cells), sep_h, hstack(bot_cells)], axis=0)
    ok, buf = cv2.imencode(".png", full)
    assert ok, "cv2.imencode failed"
    return buf.tobytes()


# ── edit_status helpers ───────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _edit_id(obj_id: str, edit_type: str, idx: int) -> str:
    prefix = {"deletion": "del", "modification": "mod", "scale": "scl",
              "material": "mat", "color": "col", "global": "glb"}.get(edit_type, "unk")
    return f"{prefix}_{obj_id}_{idx:03d}"


def _write_edit_status(path: Path, obj_id: str, edits_status: dict) -> None:
    doc = {
        "obj_id": obj_id,
        "mode": "text_align",
        "schema_version": 1,
        "updated": _ts(),
        "edits": edits_status,
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


# ── per-object processing ─────────────────────────────────────────────────────

async def process_one(
    obj_id: str,
    client: AsyncOpenAI,
    vlm_model: str,
    out_base: Path,
) -> dict:
    t0 = time.perf_counter()
    shard = SHARD
    mesh_npz  = MESH_ROOT   / shard / f"{obj_id}.npz"
    img_npz   = IMAGES_ROOT / shard / f"{obj_id}.npz"
    ov_path   = OVERVIEWS_ROOT / shard / obj_id / "overview.png"
    out_dir   = out_base / "objects" / shard / obj_id / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "edit_status.json"

    # ── Phase 1: text edit generation ──────────────────────────────────────
    try:
        pids, sem_list = build_semantic_list(mesh_npz, img_npz)
    except Exception as e:
        log.warning("[%s] build_semantic_list failed: %s", obj_id, e)
        return {"obj_id": obj_id, "status": "p1_fail", "error": f"sem_list: {e}"}

    quota   = quota_for(len(pids))
    n_total = sum(quota.values())
    quota_line = (
        f"Generate EXACTLY {n_total} edits — "
        f"{quota.get('deletion',0)} deletion · {quota.get('modification',0)} modification · "
        f"{quota.get('scale',0)} scale · {quota.get('material',0)} material · "
        f"{quota.get('color',0)} color · {quota.get('global',0)} global"
    )
    user_p1 = sem_list + "\n\n" + quota_line

    try:
        p1_raw = await call_vlm_text_async(
            client, SYSTEM_PROMPT_B, user_p1, vlm_model, max_tokens=4096)
    except Exception as e:
        log.warning("[%s] Phase 1 VLM error: %s", obj_id, e)
        return {"obj_id": obj_id, "status": "p1_fail", "error": f"vlm: {e}"}

    (out_dir / "raw.txt").write_text(p1_raw, encoding="utf-8")

    p1_parsed = extract_json_object(p1_raw)
    validation = {"ok": False, "errors": ["no_json"]}
    if p1_parsed is not None:
        validation = validate_simple(p1_parsed, set(pids), quota)

    (out_dir / "parsed.json").write_text(
        json.dumps({"obj_id": obj_id, "mode": "text_align",
                    "validation": validation, "parsed": p1_parsed or {}},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    if not validation["ok"] or not p1_parsed:
        log.warning("[%s] Phase 1 validation failed: %s", obj_id, validation.get("errors"))
        return {"obj_id": obj_id, "status": "p1_fail", "validation": validation}

    edits = p1_parsed.get("edits", [])
    t_p1 = time.perf_counter() - t0
    log.info("[%s] Phase 1 ok in %.1fs — %d edits", obj_id, t_p1, len(edits))

    # ── Load overview for gate ──────────────────────────────────────────────
    ov_img: np.ndarray | None = None
    if ov_path.is_file():
        buf = np.frombuffer(ov_path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is not None:
            ov_img = decoded

    # parts_by_id for check_rules — built from pids
    parts_by_id = {p: {"part_id": p} for p in pids}

    # ── Phase 2: alignment gate per edit ───────────────────────────────────
    edits_status: dict[str, dict] = {}
    n_pass = n_fail = 0

    for idx, edit in enumerate(edits):
        et     = edit.get("edit_type", "unknown")
        prompt = edit.get("prompt", "")
        sel    = list(edit.get("selected_part_ids") or [])
        eid    = _edit_id(obj_id, et, idx)

        gate_record: dict = {}

        # Layer 1: rule check
        rule_fails = check_rules(edit, parts_by_id)
        gate_record["rule"] = {"pass": not rule_fails, "checks": rule_fails}

        if rule_fails:
            ts = _ts()
            edits_status[eid] = {
                "edit_type": et,
                "stages": {"gate_a": {"status": "fail", "ts": ts}},
                "gates": {"A": {**gate_record, "vlm": None}},
            }
            n_fail += 1
            log.debug("[%s] %s rule_fail: %s", obj_id, eid, rule_fails)
            continue

        # Global edit: auto-pass, no image call
        if et == "global" or not sel:
            ts = _ts()
            edits_status[eid] = {
                "edit_type": et,
                "stages": {"gate_a": {"status": "pass", "ts": ts}},
                "gates": {"A": {**gate_record,
                                "vlm": {"pass": True, "score": 1.0,
                                        "reason": "global edit auto-pass",
                                        "best_view": 0, "best_view_col": 0,
                                        "pixel_counts": [0]*5,
                                        "column_map": list(range(5))}}},
            }
            n_pass += 1
            continue

        # No overview: auto-pass all with best_view=0
        if ov_img is None:
            ts = _ts()
            edits_status[eid] = {
                "edit_type": et,
                "stages": {"gate_a": {"status": "pass", "ts": ts}},
                "gates": {"A": {**gate_record,
                                "vlm": {"pass": True, "score": 1.0,
                                        "reason": "no overview, auto-pass",
                                        "best_view": 0, "best_view_col": 0,
                                        "pixel_counts": [-1]*5,
                                        "column_map": list(range(5))}}},
            }
            n_pass += 1
            continue

        # Layer 2: compute pixel counts → column_map
        px = [count_part_pixels_in_overview(ov_img, v, sel) for v in range(_N_VIEWS)]
        best_col_view = int(np.argmax(px))
        other_views   = [v for v in range(_N_VIEWS) if v != best_col_view]
        column_map    = [best_col_view] + other_views

        # Layer 3: VLM alignment gate
        gate_img = build_gate_image(ov_img, sel, column_map)
        gate_user = build_align_gate_user_prompt(et, prompt, sel)

        try:
            gate_raw = await call_vlm_async(
                client, gate_img, SYSTEM_PROMPT_ALIGN_GATE, gate_user,
                vlm_model, max_tokens=256)
            gate_out = parse_align_gate_output(gate_raw)
        except Exception as e:
            log.warning("[%s] %s gate VLM error: %s", obj_id, eid, e)
            gate_out = None

        if gate_out is None:
            gate_out = {"aligned": False, "reason": "vlm_error", "best_view": 0}

        aligned        = bool(gate_out.get("aligned", False))
        best_view_col  = gate_out.get("best_view", 0)
        if not (type(best_view_col) is int) or not (0 <= best_view_col < 5):
            best_view_col = 0
        best_view_orig = column_map[best_view_col]

        ts = _ts()
        edits_status[eid] = {
            "edit_type": et,
            "stages": {"gate_a": {"status": "pass" if aligned else "fail", "ts": ts}},
            "gates": {"A": {
                **gate_record,
                "vlm": {
                    "pass":          aligned,
                    "score":         1.0 if aligned else 0.0,
                    "reason":        gate_out.get("reason", ""),
                    "best_view":     best_view_orig,
                    "best_view_col": best_view_col,
                    "pixel_counts":  px,
                    "column_map":    column_map,
                },
            }},
        }
        if aligned:
            n_pass += 1
        else:
            n_fail += 1

    _write_edit_status(result_path, obj_id, edits_status)

    elapsed = time.perf_counter() - t0
    log.info("[%s] done %.1fs — %d pass / %d fail (yield %.0f%%)",
             obj_id, elapsed, n_pass, n_fail,
             100 * n_pass / max(1, n_pass + n_fail))

    return {
        "obj_id": obj_id, "status": "ok",
        "n_edits": len(edits), "n_pass": n_pass, "n_fail": n_fail,
        "elapsed": round(elapsed, 1),
    }


# ── main ─────────────────────────────────────────────────────────────────────

async def main(args) -> None:
    vlm_urls  = [u.strip() for u in args.vlm_urls.split(",") if u.strip()]
    out_base  = Path(args.output_dir)
    obj_ids   = [l.strip() for l in Path(OBJ_IDS_FILE).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    log.info("Running Mode E on %d objects → %s", len(obj_ids), out_base)

    sem = asyncio.Semaphore(args.concurrency)

    async def _one(obj_id: str, url: str) -> dict:
        async with sem:
            client = AsyncOpenAI(base_url=url, api_key="EMPTY")
            return await process_one(obj_id, client, args.vlm_model, out_base)

    tasks = [_one(oid, vlm_urls[i % len(vlm_urls)])
             for i, oid in enumerate(obj_ids)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── summary ──────────────────────────────────────────────────────────
    by_type: dict[str, dict] = {}
    n_p1_fail = 0
    n_total = n_pass_total = 0

    for r in results:
        if isinstance(r, Exception):
            n_p1_fail += 1
            continue
        if r.get("status") != "ok":
            n_p1_fail += 1
            continue
        n_total      += r["n_edits"]
        n_pass_total += r["n_pass"]

        oid    = r["obj_id"]
        esjson = out_base / "objects" / SHARD / oid / "phase1" / "edit_status.json"
        if esjson.is_file():
            es = json.loads(esjson.read_text())
            for ev in es.get("edits", {}).values():
                et     = ev.get("edit_type", "unknown")
                passed = ev.get("stages", {}).get("gate_a", {}).get("status") == "pass"
                bt     = by_type.setdefault(et, {"total": 0, "pass": 0})
                bt["total"] += 1
                if passed:
                    bt["pass"] += 1

    summary = {
        "n_objects":     len(obj_ids),
        "n_p1_fail":     n_p1_fail,
        "n_edits_total": n_total,
        "n_gate_pass":   n_pass_total,
        "n_gate_fail":   n_total - n_pass_total,
        "yield_rate":    round(n_pass_total / max(1, n_total), 3),
        "by_type":       by_type,
    }
    (out_base / "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    log.info("=== SUMMARY ===")
    log.info("Objects: %d  |  P1 fail: %d", len(obj_ids), n_p1_fail)
    log.info("Edits total: %d  |  gate pass: %d  |  yield: %.0f%%",
             n_total, n_pass_total, 100 * summary["yield_rate"])
    for et, v in sorted(by_type.items()):
        log.info("  %-14s %d/%d", et, v["pass"], v["total"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-urls",    required=True,
                    help="Comma-separated VLM base URLs")
    ap.add_argument("--vlm-model",   required=True)
    ap.add_argument("--output-dir",
                    default=str(REPO_ROOT / "data" / "partverse" / "outputs" /
                                "partverse" / "bench_shard08" / "mode_e_text_align"))
    ap.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(ap.parse_args()))
