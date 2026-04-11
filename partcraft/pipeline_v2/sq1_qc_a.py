from __future__ import annotations
import asyncio, base64, cv2, json, logging
from typing import Iterable

import numpy as np
from openai import AsyncOpenAI

from .paths import ObjectContext
from .specs import iter_all_specs
from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP, step_done
from .qc_rules import check_rules, count_part_pixels_in_overview
from .qc_io import update_edit_gate
from .s1_vlm_core import extract_json_object
from .validators import _phase1_skipped

LOG = logging.getLogger("pipeline_v2.sq1")
_SYS = "You evaluate 3D part-editing instructions. Output only valid JSON."
_TPL = ("5x2 overview (top=photos, bottom=colored part renders).\n"
        "Edit type: {edit_type}\nPrompt: \"{prompt}\"\n"
        "Target: \"{target_part_desc}\"\nParts: {part_labels}\n"
        "Reply ONLY: {{\"instruction_clear\":true/false,\"part_identifiable\":true/false,"
        "\"type_consistent\":true/false,\"reason\":\"one sentence\"}}")


async def _vlm_one(client, model, png, et, prompt, tpd, labels):
    b64 = base64.b64encode(png).decode()
    user = _TPL.format(edit_type=et, prompt=prompt, target_part_desc=tpd or "",
                       part_labels=", ".join(labels) or "(none)")
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                          {"type": "text", "text": user}]}],
            temperature=0.2, max_tokens=256, timeout=120,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        p = extract_json_object(r.choices[0].message.content or "")
        if p and "instruction_clear" in p: return p
    except Exception as e: LOG.warning("sq1 vlm: %s", e)
    return {"instruction_clear": False, "part_identifiable": False,
            "type_consistent": False, "reason": "vlm_error"}


async def _process_one(ctx, vlm_url, vlm_model, force):
    if not force and step_done(ctx, "sq1_qc_A"):
        return {"obj_id": ctx.obj_id, "skipped": True}
    # s1 was skipped (too_many_parts) → nothing to QC; propagate skip
    if _phase1_skipped(ctx):
        update_step(ctx, "sq1_qc_A", status=STATUS_SKIP, reason="s1_skipped")
        return {"obj_id": ctx.obj_id, "skipped": True}
    if not ctx.parsed_path.is_file():
        update_step(ctx, "sq1_qc_A", status=STATUS_FAIL, error="missing_parsed_json")
        return {"obj_id": ctx.obj_id, "error": "missing_parsed_json"}
    try:
        raw = json.loads(ctx.parsed_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        update_step(ctx, "sq1_qc_A", status=STATUS_FAIL, error=f"corrupt_parsed_json: {exc}")
        return {"obj_id": ctx.obj_id, "error": "corrupt_parsed_json"}
    obj = (raw.get("parsed") or {}).get("object") or {}
    parts_by_id = {p["part_id"]: p for p in (obj.get("parts") or [])
                   if isinstance(p, dict) and "part_id" in p}
    edits = (raw.get("parsed") or {}).get("edits") or []
    ov = ctx.overview_path.read_bytes() if ctx.overview_path.is_file() else None

    # Decode overview once for fast per-edit visibility checks (no Blender needed).
    # The bottom row of overview.png already encodes per-part palette renders.
    ov_img: np.ndarray | None = None
    if ov is not None:
        buf = np.frombuffer(ov, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is not None:
            ov_img = decoded

    client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
    n_pass = n_fail = 0; vlm_q = []
    for spec in iter_all_specs(ctx):
        e = edits[spec.edit_idx] if spec.edit_idx < len(edits) else {}
        fails = check_rules(e, parts_by_id)

        # Visibility check: if selected parts have zero pixels in the chosen
        # view of the overview bottom-row, the edit is geometrically invalid.
        # This replaces the former s2_highlights Blender re-render.
        if not fails and spec.selected_part_ids and ov_img is not None:
            vi = int(e.get("view_index", 0))
            n_px = count_part_pixels_in_overview(ov_img, vi, list(spec.selected_part_ids))
            if n_px == 0:
                fails["zero_visible_pixels"] = True
                LOG.debug("[sq1] %s e%d: zero pixels at view %d → Gate A fail",
                          ctx.obj_id, spec.edit_idx, vi)

        rr = {"pass": not fails, "checks": fails}
        if fails:
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "A", rule_result=rr)
            n_fail += 1
        else:
            vlm_q.append((spec, rr))
    if vlm_q and ov:
        async def _c(spec, rr):
            v = await _vlm_one(client, vlm_model, ov, spec.edit_type,
                               spec.prompt, spec.target_part_desc, spec.part_labels)
            ok = bool(v.get("instruction_clear") and v.get("part_identifiable")
                      and v.get("type_consistent"))
            sc = sum(map(bool, [v.get("instruction_clear"), v.get("part_identifiable"),
                                v.get("type_consistent")])) / 3.0
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "A", rule_result=rr,
                             vlm_result={"pass": ok, "score": round(sc, 3),
                                         "reason": v.get("reason", "")})
            return ok
        res = await asyncio.gather(*[_c(s, r) for s, r in vlm_q])
        n_pass += sum(res); n_fail += len(res) - sum(res)
    else:
        for spec, rr in vlm_q:
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "A", rule_result=rr,
                             vlm_result={"pass": True, "score": 1.0, "reason": "no_overview_skip"})
            n_pass += 1
    update_step(ctx, "sq1_qc_A", status=STATUS_OK, n_pass=n_pass, n_fail=n_fail)
    return {"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail}


async def run(ctxs: Iterable[ObjectContext], *, vlm_urls, vlm_model, force=False, concurrency=4):
    if not vlm_urls:
        raise ValueError("vlm_urls must not be empty")
    ctxs = list(ctxs); sem = asyncio.Semaphore(concurrency)
    async def _o(i, c):
        async with sem: return await _process_one(c, vlm_urls[i % len(vlm_urls)], vlm_model, force)
    return await asyncio.gather(*[_o(i, c) for i, c in enumerate(ctxs)])

__all__ = ["run"]
