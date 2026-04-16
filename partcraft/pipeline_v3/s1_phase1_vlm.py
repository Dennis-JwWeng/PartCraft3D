"""Step s1 — Phase 1 v3 VLM edit generator (text-semantic mode).

Uses pipeline_v3 text-semantic mode (Mode B): part captions feed SYSTEM_PROMPT_B.
No Blender image rendering — the VLM reasons purely from text descriptions.

Writes into ``ObjectContext.phase1_dir``:

    ctx.phase1_dir/
        parsed.json    ← {obj_id, validation, parsed:{object,edits}}
        raw.txt        ← raw VLM completion text

Three entrypoints:

* :func:`run_one` — synchronous, single object (best for debug / tests).
* :func:`run_many_async` — async multi-server fan-out (kept for compat).
* :func:`run_many_streaming` — producer-consumer pipeline: a process
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

from partcraft.pipeline_v3.s1_vlm_core import (  # noqa: E402
    SYSTEM_PROMPT_B, USER_PROMPT_TEXT_SEMANTIC,
    build_semantic_list,
    call_vlm_text_async, extract_json_object, validate_simple, quota_for,
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

def prerender(
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
    quota = quota_for(len(pids))
    ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
    from partcraft.pipeline_v3.s1_vlm_core import _sample_global_note as _sgn
    _roster = _sgn(hash(ctx.obj_id), quota.get("global", 0))
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

def _halve_quota(quota: dict) -> dict:
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


def _rebuild_user_msg(menu: str, quota: dict) -> str:
    """Rebuild the user prompt with a new quota (used on retry)."""
    from partcraft.pipeline_v3.s1_vlm_core import _sample_global_note as _sgn
    _roster = _sgn(hash(menu[:32]), quota.get("global", 0))
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

async def _call_one(client, ctx: ObjectContext, user_msg: str,
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
            eff_quota = quota if attempt == 0 else _halve_quota(quota)
            eff_msg = user_msg if attempt == 0 else _rebuild_user_msg(part_menu, eff_quota)

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
            rep = validate_simple(parsed, set(valid_pids), quota=eff_quota)
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
    user_msg, pids, quota, menu = pre

    async def _go():
        client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
        sem = asyncio.Semaphore(1)
        return await _call_one(client, ctx, user_msg, pids, quota,
                               vlm_model, sem, part_menu=menu)

    return asyncio.run(_go())


async def run_many_async(
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


# ─────────────────── streaming pipeline (mp pool + N VLM consumers) ──

def _prerender_worker(args: tuple) -> tuple | None:
    """Top-level pickleable worker for ProcessPoolExecutor.

    Builds a text semantic list for one object using v3 build_semantic_list.
    Returns ``(user_msg, pids, quota, menu)`` or ``None`` if the object
    exceeds ``MAX_PARTS``. No Blender rendering needed.
    """
    mesh_npz, image_npz, _blender, _unused, anno_obj_dir_str = args
    from pathlib import Path as _P
    from partcraft.pipeline_v3.s1_vlm_core import (  # noqa: E402
        build_semantic_list as _bsl,
        USER_PROMPT_TEXT_SEMANTIC as _U,
        quota_for as _qf,
        MAX_PARTS as _MAX,
        _sample_global_note as _sgn,
    )
    mesh_p = _P(mesh_npz); img_p = _P(image_npz)
    _anno_p = _P(anno_obj_dir_str) if anno_obj_dir_str else None
    pids, menu = _bsl(mesh_p, img_p, anno_obj_dir=_anno_p)
    if len(pids) > _MAX:
        return None
    quota = _qf(len(pids))
    _obj_id = _P(mesh_npz).stem
    _roster = _sgn(hash(_obj_id), quota.get("global", 0))
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


async def run_many_streaming(
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
                pool, _prerender_worker,
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
            res = await _call_one(client, ctx, user_msg, pids, quota,
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
    "Phase1Result", "prerender", "run_one",
    "run_many_async", "run_many_streaming",
]
