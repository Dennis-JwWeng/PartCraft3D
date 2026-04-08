"""Step s1 — Phase 1 v2 VLM edit generator (object-centric).

Reuses the prompt template, validators, quota table, overview renderer,
and VLM call from :mod:`scripts.standalone.run_phase1_v2` but writes
exclusively into ``ObjectContext.phase1_dir``:

    ctx.phase1_dir/
        overview.png   ← 5×2 colored grid (VLM input image)
        parsed.json    ← {obj_id, validation, parsed:{object,edits}}
        raw.txt        ← raw VLM completion text

Three entrypoints:

* :func:`run_one` — synchronous, single object (best for debug / tests).
* :func:`run_many_async` — async multi-server fan-out, prerender first
  then dispatch (kept for compatibility / single-server runs).
* :func:`run_many_streaming` — producer-consumer pipeline: a process
  pool runs blender prerenders in parallel and feeds an asyncio queue
  consumed by N VLM clients (one per server, semaphore=1 each). The
  first VLM call fires as soon as the first prerender completes, so a
  4-server config keeps all GPUs working from second 1.

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

# Pull legacy helpers in-place (single source of truth for the prompt + render).
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "standalone"))
from run_phase1_v2 import (  # noqa: E402
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    build_part_menu, render_overview_png,
    call_vlm_async, extract_json_object, validate, quota_for,
    MAX_PARTS,
)

from .paths import ObjectContext
from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP


@dataclass
class Phase1Result:
    obj_id: str
    ok: bool
    n_kept: int = 0
    n_total: int = 0
    type_counts: dict | None = None
    error: str | None = None


# ─────────────────── render-only (cheap, sequential) ──────────────────

def prerender(ctx: ObjectContext, blender: str) -> tuple[bytes, str, list[int], dict] | None:
    """Render overview + build menu + quota. Returns ``None`` if the
    object exceeds ``MAX_PARTS``.

    Side effect: writes ``ctx.overview_path``.
    """
    if ctx.mesh_npz is None or ctx.image_npz is None:
        raise ValueError(f"{ctx} missing mesh_npz/image_npz")
    pids, menu = build_part_menu(ctx.mesh_npz, ctx.image_npz)
    if len(pids) > MAX_PARTS:
        return None
    quota = quota_for(len(pids))
    ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
    # Reuse a cached overview.png if it already exists (parsed.json may
    # be missing because the previous run died at the VLM phase).
    if ctx.overview_path.is_file() and ctx.overview_path.stat().st_size > 1000:
        png = ctx.overview_path.read_bytes()
    else:
        png = render_overview_png(ctx.mesh_npz, ctx.image_npz, blender)
        ctx.overview_path.write_bytes(png)
    user_msg = USER_PROMPT_TEMPLATE.format(
        part_menu=menu, n_total=sum(quota.values()),
        n_deletion=quota["deletion"], n_modification=quota["modification"],
        n_scale=quota["scale"], n_material=quota["material"],
        n_global=quota["global"],
    )
    return png, user_msg, pids, quota


# ─────────────────── single-object VLM call ───────────────────────────

async def _call_one(client, ctx: ObjectContext, png: bytes, user_msg: str,
                    valid_pids: list[int], quota: dict, model: str,
                    sem: asyncio.Semaphore) -> Phase1Result:
    async with sem:
        t0 = time.time()
        try:
            raw = await call_vlm_async(
                client, png, SYSTEM_PROMPT, user_msg, model, max_tokens=12288,
            )
        except Exception as e:
            update_step(ctx, "s1_phase1", status=STATUS_FAIL, error=str(e))
            return Phase1Result(ctx.obj_id, ok=False, error=str(e))
        dt = time.time() - t0
        ctx.raw_response_path.write_text(raw)
        parsed = extract_json_object(raw)
        if parsed is None:
            update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                        error="parse_error", raw_len=len(raw))
            return Phase1Result(ctx.obj_id, ok=False, error="parse_error")
        rep = validate(parsed, set(valid_pids), quota=quota)
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
        )
        return Phase1Result(
            ctx.obj_id, ok=rep["ok"], n_kept=rep["n_kept_edits"],
            n_total=sum(quota.values()), type_counts=rep.get("type_counts"),
        )


# ─────────────────── public entrypoints ──────────────────────────────

def run_one(
    ctx: ObjectContext,
    *,
    blender: str,
    vlm_url: str,
    vlm_model: str,
) -> Phase1Result:
    """Synchronous one-object run (single VLM server)."""
    from openai import AsyncOpenAI

    pre = prerender(ctx, blender)
    if pre is None:
        update_step(ctx, "s1_phase1", status=STATUS_SKIP, reason="too_many_parts")
        return Phase1Result(ctx.obj_id, ok=False, error="too_many_parts")
    png, user_msg, pids, quota = pre

    async def _go():
        client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
        sem = asyncio.Semaphore(1)
        return await _call_one(client, ctx, png, user_msg, pids, quota,
                               vlm_model, sem)

    return asyncio.run(_go())


async def run_many_async(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    vlm_urls: list[str],
    vlm_model: str,
    force: bool = False,
) -> list[Phase1Result]:
    """Render + dispatch many objects across multiple VLM servers.

    Round-robins one job per server, semaphore=1 per server (matches the
    legacy KV-friendly mode that avoids 27B cache thrash).
    """
    from openai import AsyncOpenAI
    from .status import step_done

    ctxs = list(ctxs)
    pending: list[tuple[ObjectContext, bytes, str, list[int], dict]] = []
    results: list[Phase1Result] = []

    # Phase A: prerender (sequential, cheap)
    for ctx in ctxs:
        # Resume short-circuit: if parsed.json exists on disk and parses,
        # skip this obj entirely (no prerender, no VLM call). This is the
        # only way to make a crashed run cheap to resume — status.json
        # alone is not enough because it isn't written until after the
        # VLM reply lands.
        if not force and ctx.parsed_path.is_file():
            try:
                _j = json.loads(ctx.parsed_path.read_text())
                if (_j.get("parsed") or {}).get("edits") is not None:
                    # backfill status if it's missing
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
            pre = prerender(ctx, blender)
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

    # Phase B: dispatch
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in vlm_urls]
    sems = [asyncio.Semaphore(1) for _ in clients]
    tasks = []
    for i, (ctx, png, user_msg, pids, quota) in enumerate(pending):
        idx = i % len(clients)
        tasks.append(_call_one(
            clients[idx], ctx, png, user_msg, pids, quota,
            vlm_model, sems[idx],
        ))
    results.extend(await asyncio.gather(*tasks))
    return results


# ─────────────────── streaming pipeline (mp pool + N VLM consumers) ──

def _prerender_worker(args: tuple) -> tuple | None:
    """Top-level pickleable worker for ProcessPoolExecutor.

    Returns ``(png_bytes, user_msg, pids, quota)`` or ``None`` if the
    object exceeds ``MAX_PARTS``. Side effect: writes
    ``overview_path`` if it doesn't already exist.
    """
    mesh_npz, image_npz, blender, overview_path = args
    from pathlib import Path as _P
    import sys as _sys
    _here = _P(__file__).resolve().parents[2] / "scripts" / "standalone"
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    from run_phase1_v2 import (  # type: ignore
        build_part_menu as _bpm,
        render_overview_png as _rov,
        USER_PROMPT_TEMPLATE as _U,
        quota_for as _qf,
        MAX_PARTS as _MAX,
    )
    mesh_p = _P(mesh_npz); img_p = _P(image_npz); ov_p = _P(overview_path)
    pids, menu = _bpm(mesh_p, img_p)
    if len(pids) > _MAX:
        return None
    quota = _qf(len(pids))
    if ov_p.is_file() and ov_p.stat().st_size > 1000:
        png = ov_p.read_bytes()
    else:
        png = _rov(mesh_p, img_p, blender)
        ov_p.parent.mkdir(parents=True, exist_ok=True)
        ov_p.write_bytes(png)
    user_msg = _U.format(
        part_menu=menu, n_total=sum(quota.values()),
        n_deletion=quota["deletion"], n_modification=quota["modification"],
        n_scale=quota["scale"], n_material=quota["material"],
        n_global=quota["global"],
    )
    return png, user_msg, pids, quota


async def run_many_streaming(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    vlm_urls: list[str],
    vlm_model: str,
    n_prerender_workers: int = 8,
    force: bool = False,
    log_every: int = 20,
) -> list[Phase1Result]:
    """Producer-consumer streaming s1: ``n_prerender_workers`` blender
    processes feed an asyncio queue consumed by ``len(vlm_urls)`` VLM
    clients in parallel. The first VLM call fires within ~1s of start.

    Resume rule: any obj that already has ``parsed.json`` on disk is
    skipped (no prerender, no VLM). The orchestrator just calls this
    after a crash and we pick up exactly where we left off.
    """
    from openai import AsyncOpenAI
    import logging
    log = logging.getLogger("pipeline_v2.s1.stream")

    ctxs = list(ctxs)
    todo: list[ObjectContext] = []
    results: list[Phase1Result] = []

    # Filter resume cases up-front so the queue only carries fresh work.
    for ctx in ctxs:
        if not force and ctx.parsed_path.is_file():
            try:
                _j = json.loads(ctx.parsed_path.read_text())
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

    log.info("s1 streaming: todo=%d resume=%d  vlm_servers=%d  prerender_workers=%d",
             len(todo), len(results), len(vlm_urls), n_prerender_workers)

    if not todo:
        return results

    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=n_prerender_workers)
    queue: asyncio.Queue = asyncio.Queue(maxsize=2 * len(vlm_urls))

    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in vlm_urls]
    sems = [asyncio.Semaphore(1) for _ in clients]

    n_done = 0
    n_total = len(todo)

    async def render_one(ctx: ObjectContext):
        try:
            pre = await loop.run_in_executor(
                pool, _prerender_worker,
                (str(ctx.mesh_npz), str(ctx.image_npz),
                 blender, str(ctx.overview_path)),
            )
        except Exception as e:
            log.warning("prerender %s: %s", ctx.obj_id[:12], e)
            update_step(ctx, "s1_phase1", status=STATUS_FAIL, error=str(e))
            return
        if pre is None:
            update_step(ctx, "s1_phase1", status=STATUS_SKIP,
                        reason="too_many_parts")
            return
        await queue.put((ctx, pre))

    async def producer():
        # Cap concurrent prerender submissions to keep memory bounded.
        sem = asyncio.Semaphore(n_prerender_workers * 2)

        async def _wrap(c):
            async with sem:
                await render_one(c)

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
            png, user_msg, pids, quota = pre
            res = await _call_one(client, ctx, png, user_msg, pids, quota,
                                  vlm_model, sem)
            results.append(res)
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                log.info("s1 stream: %d/%d  ok_so_far=%d",
                         n_done, n_total,
                         sum(1 for r in results if r.ok))

    try:
        await asyncio.gather(producer(), *[consumer(i) for i in range(len(clients))])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return results


__all__ = [
    "Phase1Result", "prerender", "run_one",
    "run_many_async", "run_many_streaming",
]
