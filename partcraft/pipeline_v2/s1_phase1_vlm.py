"""Step s1 — Phase 1 v2 VLM edit generator (object-centric).

Reuses the prompt template, validators, quota table, overview renderer,
and VLM call from :mod:`scripts.standalone.run_phase1_v2` but writes
exclusively into ``ObjectContext.phase1_dir``:

    ctx.phase1_dir/
        overview.png   ← 5×2 colored grid (VLM input image)
        parsed.json    ← {obj_id, validation, parsed:{object,edits}}
        raw.txt        ← raw VLM completion text

Two entrypoints:

* :func:`run_one` — synchronous, single object (best for debug / tests).
* :func:`run_many_async` — async multi-server fan-out, one object per
  request, semaphore = 1 per server (matches legacy KV-friendly mode).

Both write the per-object ``status.json`` step entry ``s1_phase1`` on
success and rebuild nothing globally — the orchestrator calls
``rebuild_manifest`` after a batch.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
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
    png = render_overview_png(ctx.mesh_npz, ctx.image_npz, blender)
    ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
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


__all__ = ["Phase1Result", "prerender", "run_one", "run_many_async"]
