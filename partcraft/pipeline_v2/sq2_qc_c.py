from __future__ import annotations
import asyncio, base64, logging
from typing import Iterable
import cv2, numpy as np
from openai import AsyncOpenAI
from .paths import ObjectContext
from .specs import iter_flux_specs
from .status import update_step, STATUS_OK, step_done
from .qc_io import update_edit_gate, is_edit_qc_failed
from .s1_vlm_core import extract_json_object

LOG = logging.getLogger("pipeline_v2.sq2")
_SYS = "You are a visual quality judge. Output only valid JSON."
_USR = ("LEFT=highlighted region (magenta=target part). RIGHT=2D edit result. "
        "Did the edit primarily affect the magenta region? "
        "Reply ONLY: {\"region_match\":true/false,\"reason\":\"one short phrase\"}")

def _stitch(a, b):
    h = min(a.shape[0], b.shape[0], 512)
    def _r(img): s = h / img.shape[0]; return cv2.resize(img, (int(img.shape[1] * s), h))
    ok, buf = cv2.imencode(".png", np.hstack([_r(a), _r(b)]))
    if not ok: raise RuntimeError("imencode failed")
    return buf.tobytes()

async def _vlm_one(client, model, png):
    b64 = base64.b64encode(png).decode()
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                          {"type": "text", "text": _USR}]}],
            temperature=0.1, max_tokens=128, timeout=60,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        p = extract_json_object(r.choices[0].message.content or "")
        if p and "region_match" in p: return p
    except Exception as e: LOG.warning("sq2 vlm: %s", e)
    return {"region_match": False, "reason": "vlm_error"}

async def _process_one(ctx, vlm_url, vlm_model, force):
    if not force and step_done(ctx, "sq2_qc_C"):
        return {"obj_id": ctx.obj_id, "skipped": True}
    client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
    n_pass = n_fail = n_skip = 0

    async def _check(spec):
        nonlocal n_pass, n_fail, n_skip
        if is_edit_qc_failed(ctx, spec.edit_id): n_skip += 1; return
        hl = ctx.highlight_path(spec.edit_idx); ed = ctx.edit_2d_output(spec.edit_id)
        if not hl.is_file() or not ed.is_file():
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "C",
                             vlm_result={"pass": False, "score": 0.0, "reason": "missing_artifact"})
            n_fail += 1; return
        hi = cv2.imread(str(hl)); ei = cv2.imread(str(ed))
        if hi is None or ei is None:
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "C",
                             vlm_result={"pass": False, "score": 0.0, "reason": "unreadable_image"})
            n_fail += 1; return
        raw = await _vlm_one(client, vlm_model, _stitch(hi, ei))
        ok = bool(raw.get("region_match"))
        update_edit_gate(ctx, spec.edit_id, spec.edit_type, "C",
                         vlm_result={"pass": ok, "score": 1.0 if ok else 0.0,
                                     "reason": raw.get("reason", "")})
        if ok: n_pass += 1
        else: n_fail += 1

    await asyncio.gather(*[_check(s) for s in iter_flux_specs(ctx)])
    update_step(ctx, "sq2_qc_C", status=STATUS_OK,
                n_pass=n_pass, n_fail=n_fail, n_skip=n_skip)
    return {"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail}

async def run(ctxs, *, vlm_urls, vlm_model, force=False, concurrency=4):
    if not vlm_urls:
        raise ValueError("vlm_urls must not be empty")
    ctxs = list(ctxs); sem = asyncio.Semaphore(concurrency)
    async def _o(i, c):
        async with sem: return await _process_one(c, vlm_urls[i % len(vlm_urls)], vlm_model, force)
    return await asyncio.gather(*[_o(i, c) for i, c in enumerate(ctxs)])

__all__ = ["run"]
