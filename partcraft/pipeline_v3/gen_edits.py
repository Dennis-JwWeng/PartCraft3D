"""Step s1 — Generate edit instructions via VLM (text-semantic mode).

Uses pipeline_v3 text-semantic mode (Mode B): part captions feed SYSTEM_PROMPT_B.
No Blender image rendering — the VLM reasons purely from text descriptions.

Writes into ``ObjectContext.phase1_dir``:

    ctx.phase1_dir/
        parsed.json    ← {obj_id, validation, parsed:{object,edits}}
        raw.txt        ← raw VLM completion text

Three entrypoints:

* :func:`gen_edits_for_one` — synchronous, single object (best for debug / tests).
* :func:`gen_edits_async` — async multi-server fan-out (kept for compat).
* :func:`gen_edits_streaming` — producer-consumer pipeline: a process
  pool builds semantic lists in parallel and feeds an asyncio queue
  consumed by N VLM clients (one per server, semaphore=1 each).

Both write the per-object ``status.json`` step entry ``s1_phase1`` on
success and rebuild nothing globally — the orchestrator calls
``rebuild_manifest`` after a batch.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from partcraft.pipeline_v3.vlm_core import (  # noqa: E402
    SYSTEM_PROMPT_B, USER_PROMPT_TEXT_SEMANTIC,
    build_semantic_list,
    call_vlm_text_async, extract_json_object, validate_edit_json, compute_edit_quota,
    MAX_PARTS,
)

from .paths import ObjectContext
from .status import update_step, step_done, STATUS_OK, STATUS_FAIL, STATUS_SKIP


@dataclass
class Phase1Result:
    obj_id: str
    ok: bool
    n_kept: int = 0
    n_total: int = 0
    type_counts: dict | None = None
    error: str | None = None


# ─────────────────── build prompt (text-only, no image) ──────────────────────

def prepare_edit_menu(
    ctx: ObjectContext,
    blender: str,  # kept for API compat — not used in text-semantic mode
    anno_dir: "Path | None" = None,
) -> tuple[str, list[int], dict, str] | None:
    """Build semantic text list + quota. Returns ``None`` if the object
    exceeds ``MAX_PARTS``.

    Returns ``(user_msg, pids, quota, menu)`` — no image bytes.
    No Blender rendering: v3 text-semantic mode uses part captions only.

    Side effect: creates ``ctx.phase1_dir`` if needed.
    """
    if ctx.mesh_npz is None or ctx.image_npz is None:
        raise ValueError(f"{ctx} missing mesh_npz/image_npz")
    _anno = (anno_dir / ctx.obj_id) if anno_dir else None
    pids, menu = build_semantic_list(ctx.mesh_npz, ctx.image_npz, anno_obj_dir=_anno)
    if len(pids) > MAX_PARTS:
        return None
    quota = compute_edit_quota(len(pids))
    ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
    from partcraft.pipeline_v3.vlm_core import _pick_global_edit_note
    _n_global = quota.get("global", 0)
    _roster = _pick_global_edit_note(hash(ctx.obj_id), _n_global) if _n_global > 0 else ""
    user_msg = USER_PROMPT_TEXT_SEMANTIC.format(
        part_menu=menu,
        n_total=sum(quota.values()),
        n_deletion=quota["deletion"],
        n_modification=quota["modification"],
        n_scale=quota["scale"],
        n_material=quota.get("material", 0),
        n_color=quota.get("color", 0),
        n_global=quota.get("global", 0),
        global_note=_roster,
    )
    return user_msg, pids, quota, menu


# ─────────────────── VLM quota halving for retry ──────────────────────

def _halve_edit_quota(quota: dict) -> dict:
    """Return a reduced quota for the retry attempt.

    Halving produces a shorter response (less truncation risk) while still
    covering all edit types.  Deletion and modification get at least 1 each;
    scale/material/color/global get at least 1 only when the original had >= 2.
    """
    out: dict = {}
    for k, v in quota.items():
        halved = max(1, v // 2) if k in ("deletion", "modification") else max(0, v // 2)
        out[k] = halved
    return out


def _format_retry_prompt(menu: str, quota: dict) -> str:
    """Rebuild the user prompt with a new quota (used on retry).

    The roster seed is salted so the second attempt sees a *different* shuffle
    of the global-style pool, avoiding the VLM re-locking onto the same
    preferences that caused the first attempt to fail.
    """
    from partcraft.pipeline_v3.vlm_core import _pick_global_edit_note
    _n_global = quota.get("global", 0)
    _roster = (
        _pick_global_edit_note(hash(menu[:32]) ^ 0xA11CE, _n_global)
        if _n_global > 0 else ""
    )
    return USER_PROMPT_TEXT_SEMANTIC.format(
        part_menu=menu,
        n_total=sum(quota.values()),
        n_deletion=quota.get("deletion", 0),
        n_modification=quota.get("modification", 0),
        n_scale=quota.get("scale", 0),
        n_material=quota.get("material", 0),
        n_color=quota.get("color", 0),
        n_global=quota.get("global", 0),
        global_note=_roster,
    )


# ─────────────────── single-object VLM call ───────────────────────────

async def _call_vlm_for_one(client, ctx: ObjectContext, user_msg: str,
                    valid_pids: list[int], quota: dict, model: str,
                    sem: asyncio.Semaphore,
                    part_menu: str = "") -> Phase1Result:
    """Make up to 2 VLM attempts for one object (text-only, no image).

    Attempt 1: full quota, full user_msg.
    Attempt 2 (on exception or JSON parse failure only): halved quota,
        rebuilt user_msg — shorter response reduces truncation risk.
    If attempt 2 also fails, status is written as FAIL.
    If attempt 1 succeeds but validation ok=False, the partial result is
    saved and returned as-is (downstream uses whatever edits passed).
    """
    async with sem:
        t0 = time.time()
        last_error: str = ""

        for attempt in range(2):
            eff_quota = quota if attempt == 0 else _halve_edit_quota(quota)
            eff_msg = user_msg if attempt == 0 else _format_retry_prompt(part_menu, eff_quota)

            try:
                raw = await call_vlm_text_async(
                    client, SYSTEM_PROMPT_B, eff_msg, model, max_tokens=12288,
                )
            except Exception as e:
                last_error = str(e)
                if attempt == 0:
                    continue  # retry
                update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                            error=last_error, attempts=2)
                return Phase1Result(ctx.obj_id, ok=False, error=last_error)

            ctx.raw_response_path.write_text(raw)
            parsed = extract_json_object(raw)
            if parsed is None:
                last_error = "parse_error"
                if attempt == 0:
                    continue  # retry with halved quota
                update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                            error=last_error, raw_len=len(raw), attempts=2)
                return Phase1Result(ctx.obj_id, ok=False, error=last_error)

            # Normalize: VLM sometimes nests edits inside object rather than
            # as a top-level sibling.  Hoist to the expected schema.
            if "edits" not in parsed and isinstance(
                (parsed.get("object") or {}).get("edits"), list
            ):
                parsed["edits"] = parsed["object"].pop("edits")

            # Parsed successfully — save result regardless of validation score.
            dt = time.time() - t0
            rep = validate_edit_json(parsed, set(valid_pids), quota=eff_quota)
            out = {
                "obj_id": ctx.obj_id,
                "shard": ctx.shard,
                "validation": rep,
                "parsed": parsed,
            }
            ctx.parsed_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            update_step(
                ctx, "s1_phase1",
                status=STATUS_OK if rep["ok"] else STATUS_FAIL,
                n_edits=len(parsed.get("edits") or []),
                n_kept=rep["n_kept_edits"],
                type_counts=rep.get("type_counts"),
                wall_s=round(dt, 2),
                attempts=attempt + 1,
            )
            return Phase1Result(
                ctx.obj_id, ok=rep["ok"], n_kept=rep["n_kept_edits"],
                n_total=sum(eff_quota.values()),
                type_counts=rep.get("type_counts"),
            )

        # Should never reach here (loop covers both attempts), but be safe.
        update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                    error=last_error or "unknown", attempts=2)
        return Phase1Result(ctx.obj_id, ok=False, error=last_error)


# ─────────────────── public entrypoints ────────────────────────────

def gen_edits_for_one(
    ctx: ObjectContext,
    *,
    blender: str,
    vlm_url: str,
    vlm_model: str,
) -> Phase1Result:
    """Synchronous one-object run (single VLM server)."""
    from openai import AsyncOpenAI

    pre = prepare_edit_menu(ctx, blender)
    if pre is None:
        update_step(ctx, "s1_phase1", status=STATUS_SKIP, reason="too_many_parts")
        return Phase1Result(ctx.obj_id, ok=False, error="too_many_parts")
    user_msg, pids, quota, menu = pre

    async def _go():
        client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
        sem = asyncio.Semaphore(1)
        return await _call_vlm_for_one(client, ctx, user_msg, pids, quota,
                               vlm_model, sem, part_menu=menu)

    return asyncio.run(_go())


async def gen_edits_async(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    vlm_urls: list[str],
    vlm_model: str,
    force: bool = False,
) -> list[Phase1Result]:
    """Build semantic lists + dispatch many objects across multiple VLM servers.

    Round-robins one job per server, semaphore=1 per server.
    """
    from openai import AsyncOpenAI
    from .status import step_done

    ctxs = list(ctxs)
    pending: list[tuple] = []
    results: list[Phase1Result] = []

    for ctx in ctxs:
        if not force and ctx.parsed_path.is_file():
            try:
                _j = json.loads(ctx.parsed_path.read_text())
                _p = _j.get("parsed") or {}
                if "edits" not in _p and isinstance((_p.get("object") or {}).get("edits"), list):
                    _p["edits"] = _p["object"].pop("edits")
                    _j["parsed"] = _p
                    ctx.parsed_path.write_text(json.dumps(_j, indent=2, ensure_ascii=False))
                if (_j.get("parsed") or {}).get("edits") is not None:
                    if not step_done(ctx, "s1_phase1"):
                        from .status import update_step, STATUS_OK
                        update_step(ctx, "s1_phase1", status=STATUS_OK,
                                    n_edits=len(_j["parsed"].get("edits") or []),
                                    resumed=True)
                    results.append(Phase1Result(ctx.obj_id, ok=True))
                    continue
            except Exception:
                pass
        if not force and step_done(ctx, "s1_phase1"):
            results.append(Phase1Result(ctx.obj_id, ok=True))
            continue
        try:
            pre = prepare_edit_menu(ctx, blender)
        except Exception as e:
            update_step(ctx, "s1_phase1", status=STATUS_FAIL, error=str(e))
            results.append(Phase1Result(ctx.obj_id, ok=False, error=str(e)))
            continue
        if pre is None:
            update_step(ctx, "s1_phase1", status=STATUS_SKIP,
                        reason="too_many_parts")
            results.append(Phase1Result(ctx.obj_id, ok=False,
                                        error="too_many_parts"))
            continue
        pending.append((ctx, *pre))

    if not pending:
        return results

    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in vlm_urls]
    sems = [asyncio.Semaphore(1) for _ in clients]
    tasks = []
    for i, (ctx, user_msg, pids, quota, menu) in enumerate(pending):
        idx = i % len(clients)
        tasks.append(_call_one(
            clients[idx], ctx, user_msg, pids, quota,
            vlm_model, sems[idx], part_menu=menu,
        ))
    results.extend(await asyncio.gather(*tasks))
    return results


# ─────────────────── overview.png backfill (CPU/Blender pool) ─────────
#
# Gate A (`gate_text_align`) needs the 5×2 overview PNG to:
#   1. compute pixel-visibility per view → pick best_view (npz_view)
#   2. crop a 5×2 sub-image to feed the VLM image+text judge
# When overview.png is absent the gate falls back to "no_overview_auto_pass"
# and `best_view=0`, which silently passes every edit and forces the FLUX
# stage to use the wrong frame.  v3 text-semantic mode skipped Blender for
# speed but never wired the render back in — this restores the documented
# `gen_edits` contract: parsed.json + overview.png per object.


def _render_overview_worker(args: tuple) -> tuple[str, str, str]:
    """Top-level pickleable worker: render one overview.png to disk.

    args = (obj_id, mesh_npz, image_npz, blender, out_path, force)
    Returns (obj_id, status, error)
        status ∈ {"ok", "skip", "err"}
    """
    obj_id, mesh_npz, image_npz, blender, out_path, force = args
    from pathlib import Path as _P
    out_p = _P(out_path)
    if not force and out_p.is_file() and out_p.stat().st_size > 1000:
        return obj_id, "skip", ""
    try:
        from partcraft.pipeline_v3.vlm_core import render_overview_png  # noqa: E402
        png = render_overview_png(_P(mesh_npz), _P(image_npz), blender)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # write atomically so a crash mid-Blender never leaves a torn file
        tmp = out_p.with_suffix(".png.tmp")
        tmp.write_bytes(png)
        tmp.replace(out_p)
        return obj_id, "ok", ""
    except Exception as e:
        return obj_id, "err", str(e)


def backfill_overviews(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    n_workers: int = 8,
    force: bool = False,
    log_every: int = 50,
    log: "object | None" = None,
) -> dict:
    """Render `phase1/overview.png` for every ctx that is missing it.

    Idempotent: skips objects whose overview.png already exists and is
    >1 KB unless ``force=True``.  Designed to run as a synchronous
    pre-pass to ``gen_edits_streaming`` (and as a standalone CLI step).

    Returns ``{"ok": int, "skip": int, "err": int, "errors": [..]}``.
    """
    import logging
    log = log or logging.getLogger("pipeline_v3.s1.overview")

    tasks: list[tuple] = []
    for ctx in ctxs:
        if ctx.mesh_npz is None or ctx.image_npz is None:
            continue
        if not ctx.mesh_npz.is_file() or not ctx.image_npz.is_file():
            continue
        ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((
            ctx.obj_id, str(ctx.mesh_npz), str(ctx.image_npz),
            blender, str(ctx.overview_path), force,
        ))

    n_total = len(tasks)
    if n_total == 0:
        return {"ok": 0, "skip": 0, "err": 0, "errors": []}

    log.info("overview backfill: total=%d  workers=%d  force=%s",
             n_total, n_workers, force)

    n_ok = n_skip = n_err = n_done = 0
    errors: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_render_overview_worker, t) for t in tasks]
        for fut in futs:
            try:
                oid, status, err = fut.result()
            except Exception as e:
                n_err += 1
                errors.append(("?", str(e)))
                oid, status = "?", "err"
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                errors.append((oid, err))
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                log.info("overview backfill: %d/%d  ok=%d skip=%d err=%d",
                         n_done, n_total, n_ok, n_skip, n_err)

    if errors:
        log.warning("overview backfill: %d errors (showing first 5):", len(errors))
        for oid, err in errors[:5]:
            log.warning("  [%s] %s", oid[:12], err[:200])

    return {"ok": n_ok, "skip": n_skip, "err": n_err, "errors": errors}


# ─────────────────── streaming pipeline (mp pool + N VLM consumers) ──

def _prepare_edit_menu_worker(args: tuple) -> tuple | None:
    """Top-level pickleable worker for ProcessPoolExecutor.

    Builds a text semantic list for one object using v3 build_semantic_list.
    Returns ``(user_msg, pids, quota, menu)`` or ``None`` if the object
    exceeds ``MAX_PARTS``. No Blender rendering needed.
    """
    mesh_npz, image_npz, _blender, _unused, anno_obj_dir_str = args
    from pathlib import Path as _P
    from partcraft.pipeline_v3.vlm_core import (  # noqa: E402
        build_semantic_list as _bsl,
        USER_PROMPT_TEXT_SEMANTIC as _U,
        compute_edit_quota as _qf,
        MAX_PARTS as _MAX,
        _pick_global_edit_note,
    )
    mesh_p = _P(mesh_npz); img_p = _P(image_npz)
    _anno_p = _P(anno_obj_dir_str) if anno_obj_dir_str else None
    pids, menu = _bsl(mesh_p, img_p, anno_obj_dir=_anno_p)
    if len(pids) > _MAX:
        return None
    quota = _qf(len(pids))
    _obj_id = _P(mesh_npz).stem
    _n_global = quota.get("global", 0)
    _roster = _pick_global_edit_note(hash(_obj_id), _n_global) if _n_global > 0 else ""
    user_msg = _U.format(
        part_menu=menu,
        n_total=sum(quota.values()),
        n_deletion=quota.get("deletion", 0),
        n_modification=quota.get("modification", 0),
        n_scale=quota.get("scale", 0),
        n_material=quota.get("material", 0),
        n_color=quota.get("color", 0),
        n_global=quota.get("global", 0),
        global_note=_roster,
    )
    return user_msg, pids, quota, menu


async def gen_edits_streaming(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    vlm_urls: list[str],
    vlm_model: str,
    n_prerender_workers: int = 8,
    force: bool = False,
    log_every: int = 20,
    post_object_fn=None,
    anno_dir: "Path | None" = None,
) -> list[Phase1Result]:
    """Producer-consumer streaming s1: ``n_prerender_workers`` processes
    build semantic lists in parallel and feed an asyncio queue consumed by
    ``len(vlm_urls)`` VLM clients. No Blender rendering required.

    Resume rule: any obj that already has ``parsed.json`` on disk is
    skipped. The orchestrator just calls this after a crash and we pick
    up exactly where we left off.
    """
    from openai import AsyncOpenAI
    import logging
    log = logging.getLogger("pipeline_v3.s1.stream")

    ctxs = list(ctxs)
    todo: list[ObjectContext] = []
    results: list[Phase1Result] = []

    for ctx in ctxs:
        if not force and ctx.parsed_path.is_file():
            try:
                _j = json.loads(ctx.parsed_path.read_text())
                _p = _j.get("parsed") or {}
                if "edits" not in _p and isinstance((_p.get("object") or {}).get("edits"), list):
                    _p["edits"] = _p["object"].pop("edits")
                    _j["parsed"] = _p
                    ctx.parsed_path.write_text(json.dumps(_j, indent=2, ensure_ascii=False))
                if (_j.get("parsed") or {}).get("edits") is not None:
                    if not step_done(ctx, "s1_phase1"):
                        update_step(
                            ctx, "s1_phase1", status=STATUS_OK,
                            n_edits=len(_j["parsed"].get("edits") or []),
                            resumed=True,
                        )
                    results.append(Phase1Result(ctx.obj_id, ok=True))
                    continue
            except Exception:
                pass
        if not force and step_done(ctx, "s1_phase1"):
            results.append(Phase1Result(ctx.obj_id, ok=True))
            continue
        if ctx.mesh_npz is None or ctx.image_npz is None:
            results.append(Phase1Result(ctx.obj_id, ok=False, error="no_input"))
            continue
        todo.append(ctx)

    log.info("s1 streaming: todo=%d resume=%d  vlm_servers=%d  workers=%d",
             len(todo), len(results), len(vlm_urls), n_prerender_workers)

    # ── Overview backfill pre-pass ────────────────────────────────────
    # Render phase1/overview.png for *every* ctx (resume + todo) before
    # any VLM call fires.  This guarantees that the per-object Gate A
    # hook (post_object_fn → run_gate_text_align) can read overview.png
    # and do real pixel-visibility + image judging instead of falling
    # back to "no_overview_auto_pass".  Idempotent: skips files that
    # already exist on disk.  Runs synchronously since Gate A correctness
    # depends on overviews being present before the consumer fires.
    try:
        ov_stats = backfill_overviews(
            ctxs, blender=blender, n_workers=n_prerender_workers,
            force=False, log=log,
        )
        log.info("s1 streaming: overview backfill done — ok=%d skip=%d err=%d",
                 ov_stats["ok"], ov_stats["skip"], ov_stats["err"])
    except Exception as exc:
        log.warning("overview backfill failed: %s — Gate A will auto-pass", exc)

    if not todo:
        return results

    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=n_prerender_workers)
    queue: asyncio.Queue = asyncio.Queue(maxsize=2 * len(vlm_urls))

    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in vlm_urls]
    sems = [asyncio.Semaphore(1) for _ in clients]

    n_done = 0
    n_total = len(todo)

    async def build_one(ctx: ObjectContext):
        try:
            _anno_dir = (anno_dir / ctx.obj_id) if anno_dir else None
            pre = await loop.run_in_executor(
                pool, _prepare_edit_menu_worker,
                (str(ctx.mesh_npz), str(ctx.image_npz),
                 blender, "",  # unused slot kept for worker signature compat
                 str(_anno_dir) if _anno_dir else ""),
            )
        except Exception as e:
            log.warning("build_list %s: %s", ctx.obj_id[:12], e)
            update_step(ctx, "s1_phase1", status=STATUS_FAIL, error=str(e))
            return
        if pre is None:
            update_step(ctx, "s1_phase1", status=STATUS_SKIP,
                        reason="too_many_parts")
            return
        ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
        await queue.put((ctx, pre))

    async def producer():
        sem = asyncio.Semaphore(n_prerender_workers * 2)

        async def _wrap(c):
            async with sem:
                await build_one(c)

        await asyncio.gather(*[_wrap(c) for c in todo])
        for _ in range(len(clients)):
            await queue.put(None)

    async def consumer(idx: int):
        nonlocal n_done
        client = clients[idx]
        sem = sems[idx]
        while True:
            item = await queue.get()
            if item is None:
                return
            ctx, pre = item
            user_msg, pids, quota, menu = pre
            res = await _call_vlm_for_one(client, ctx, user_msg, pids, quota,
                                  vlm_model, sem, part_menu=menu)
            results.append(res)
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                log.info("s1 stream: %d/%d  ok_so_far=%d",
                         n_done, n_total,
                         sum(1 for r in results if r.ok))
            if post_object_fn is not None and res.error != "too_many_parts":
                try:
                    await post_object_fn(ctx, vlm_urls[idx])
                except Exception as _hook_exc:
                    log.warning("post_object_fn %s: %s", ctx.obj_id[:12], _hook_exc)

    try:
        await asyncio.gather(producer(), *[consumer(i) for i in range(len(clients))])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return results


__all__ = [
    "Phase1Result", "prepare_edit_menu", "gen_edits_for_one",
    "gen_edits_async", "gen_edits_streaming",
    "backfill_overviews",
]
