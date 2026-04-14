"""Step sq3 / E_qc — VLM final quality gate (all edit types).

For every non-identity edit that has survived gate A, we:
  1. Load 5 before-state views from image_npz (zero render cost).
  2. Load 5 after-state preview_{0..4}.png rendered by s6p.
  3. Build a 2-row × 5-col collage (top row = before, bottom = after).
  4. Call the VLM judge and record the result in qc.json.

Coverage now includes *addition* edits (discovered via edits_3d/add_*/meta.json).
Step key: sq3_qc_E.

Concurrency model (mirrors sq1/sq2):
  - ``run()`` is async; callers must ``asyncio.run(run(...))``.
  - ``vlm_urls`` (list) are distributed round-robin across objects.
  - Up to ``concurrency`` objects are judged simultaneously via
    ``asyncio.Semaphore``; edits within each object are sequential
    (safe for qc.json file updates, which are synchronous).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Iterable

import cv2
import numpy as np
from openai import AsyncOpenAI

from .paths import ObjectContext
from .specs import iter_all_specs, VIEW_INDICES
from .status import update_step, STATUS_OK, step_done
from .qc_io import update_edit_gate, is_edit_qc_failed
from .edit_status_io import update_edit_stage


_DEFS = {
    "deletion":     {"min_visual_quality": 3, "require_preserve_other": True},
    "modification": {"min_visual_quality": 3, "require_preserve_other": True},
    "scale":        {"min_visual_quality": 3, "require_preserve_other": True},
    "material":     {"min_visual_quality": 3, "require_preserve_other": True},
    "global":       {"min_visual_quality": 3, "require_preserve_other": True},
    "addition":     {"min_visual_quality": 3, "require_preserve_other": True},
}

LOG = logging.getLogger("pipeline_v2.sq3")


def _passes(j, et, thr):
    t = {**_DEFS.get(et, {}), **(thr.get(et) or {})}
    if not j.get("edit_executed", False):
        return False
    vq = j.get("visual_quality", 0)
    try:
        vq = int(vq)
    except (TypeError, ValueError):
        vq = 0
    if vq < t.get("min_visual_quality", 3):
        return False
    # correct_region is required for ALL edit types.
    # For global edits the prompt redefines it as "style applied consistently
    # across the whole object"; for all others it means the right part changed.
    if not j.get("correct_region", False):
        return False
    if t.get("require_preserve_other") and not j.get("preserve_other", False):
        return False
    return True


def _load_before_imgs(ctx: ObjectContext) -> list[np.ndarray] | None:
    """Load 5 before-state BGR images from image_npz at VIEW_INDICES."""
    if ctx.image_npz is None or not ctx.image_npz.is_file():
        return None
    try:
        from partcraft.render.overview import load_views_from_npz
        imgs, _ = load_views_from_npz(ctx.image_npz, VIEW_INDICES)
        return imgs
    except Exception:
        return None


def _load_after_previews(edit_dir) -> list[np.ndarray] | None:
    """Load preview_0.png … preview_4.png. Returns None on any missing."""
    imgs = []
    for i in range(5):
        p = edit_dir / f"preview_{i}.png"
        if not p.is_file():
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        imgs.append(img)
    return imgs


def _make_collage(before_imgs: list[np.ndarray], after_imgs: list[np.ndarray]) -> bytes | None:
    """Build a 2-row × 5-col PNG collage (top = before, bottom = after)."""
    h = 256

    def _r(x):
        s = h / x.shape[0]
        return cv2.resize(x, (int(x.shape[1] * s), h))

    try:
        row_b = np.hstack([_r(img) for img in before_imgs])
        row_a = np.hstack([_r(img) for img in after_imgs])
        wb, wa = row_b.shape[1], row_a.shape[1]
        w = max(wb, wa)
        if wb < w:
            row_b = np.pad(row_b, ((0, 0), (0, w - wb), (0, 0)))
        if wa < w:
            row_a = np.pad(row_a, ((0, 0), (0, w - wa), (0, 0)))
        ok, buf = cv2.imencode(".png", np.vstack([row_b, row_a]))
        return buf.tobytes() if ok else None
    except Exception:
        return None


def _iter_add_edits(ctx: ObjectContext):
    """Yield (edit_id, meta_dict) for addition edits from edits_3d/*/meta.json."""
    if not ctx.edits_3d_dir.is_dir():
        return
    for add_dir in sorted(ctx.edits_3d_dir.iterdir()):
        if not add_dir.is_dir():
            continue
        meta_path = add_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("edit_type") == "addition":
            yield add_dir.name, meta


async def _call_vlm_judge_async(
    client: AsyncOpenAI,
    model: str,
    img_bytes: bytes,
    edit_prompt: str,
    edit_type: str,
    object_desc: str,
    part_label: str,
    target_part_desc: str = "",
    edit_params: dict | None = None,
    max_retries: int = 4,
    max_tokens: int = 1024,
) -> dict | None:
    """Async version of call_vlm_judge (partcraft.cleaning.vlm_filter)."""
    from partcraft.cleaning.vlm_filter import build_judge_prompt, _extract_json_from_vlm  # type: ignore

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    base_text = build_judge_prompt(
        edit_prompt, edit_type, object_desc, part_label,
        target_part_desc=target_part_desc,
        edit_params=edit_params or {},
    )
    strict_suffix = (
        "\n\nIf you already wrote analysis above, IGNORE it for the parser: "
        "output ONE new line that is ONLY the JSON object, starting with { "
        "and ending with }."
    )

    for attempt in range(max_retries + 1):
        text = base_text + (strict_suffix if attempt > 0 else "")
        sys_msg = (
            "You output only valid JSON for machine parsing. "
            "Never write explanations, headings, or markdown."
        )
        if attempt > 0:
            sys_msg += " Your reply must be a single JSON object; no chain-of-thought."
        try:
            create_kw: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": text},
                        ],
                    },
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "timeout": 120,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            try:
                resp = await client.chat.completions.create(**create_kw)
            except TypeError:
                create_kw.pop("extra_body", None)
                resp = await client.chat.completions.create(**create_kw)

            content = resp.choices[0].message.content
            if not content:
                LOG.warning("[sq3] VLM empty response (attempt %d/%d)",
                            attempt + 1, max_retries + 1)
                if attempt < max_retries:
                    await asyncio.sleep(2 * (attempt + 1))
                continue

            result = _extract_json_from_vlm(content)
            if result is not None:
                return result

            LOG.warning("[sq3] VLM JSON parse failed (attempt %d/%d): %s",
                        attempt + 1, max_retries + 1, content[:200])

        except Exception as e:
            LOG.warning("[sq3] VLM call error (attempt %d/%d): %s",
                        attempt + 1, max_retries + 1, e)

        if attempt < max_retries:
            await asyncio.sleep(2 * (attempt + 1))

    return None


async def _judge_one_async(
    client: AsyncOpenAI,
    vlm_model: str,
    edit_id: str,
    edit_type: str,
    prompt: str,
    obj_desc: str,
    part_label: str,
    before_imgs: list[np.ndarray],
    edit_dir,
    thr: dict,
    ctx: ObjectContext,
    target_part_desc: str = "",
    edit_params: dict | None = None,
    swap_collage: bool = False,
) -> tuple[bool, int, int]:
    """Async VLM judge for one edit. Returns (ok, n_pass_delta, n_fail_delta).

    ``swap_collage=True`` is used for addition edits: the preview stores the
    before-addition state (object without part), while image_npz before_imgs
    represents the after-addition target (original complete object), so we
    swap to build the collage in the correct before→after direction.
    """
    if is_edit_qc_failed(ctx, edit_id):
        return False, 0, 0

    after_imgs = _load_after_previews(edit_dir)
    if after_imgs is None:
        update_edit_gate(ctx, edit_id, edit_type, "E",
                         vlm_result={"pass": False, "score": 0.0,
                                     "reason": "missing_previews"})
        update_edit_stage(ctx, edit_id, edit_type, "gate_e", status="fail")
        return False, 0, 1

    # For addition edits the preview is the before-addition state (object minus
    # part) and image_npz frames are the after-addition target (original with
    # part). Swap so the collage shows the correct before→after direction.
    if swap_collage:
        coll = _make_collage(after_imgs, before_imgs)
    else:
        coll = _make_collage(before_imgs, after_imgs)
    if coll is None:
        update_edit_gate(ctx, edit_id, edit_type, "E",
                         vlm_result={"pass": False, "score": 0.0,
                                     "reason": "collage_failed"})
        update_edit_stage(ctx, edit_id, edit_type, "gate_e", status="fail")
        return False, 0, 1

    j = await _call_vlm_judge_async(
        client, vlm_model, coll,
        edit_prompt=prompt, edit_type=edit_type,
        object_desc=obj_desc, part_label=part_label,
        target_part_desc=target_part_desc,
        edit_params=edit_params,
    )
    if j is None:
        update_edit_gate(ctx, edit_id, edit_type, "E",
                         vlm_result={"pass": False, "score": 0.0,
                                     "reason": "vlm_no_response"})
        update_edit_stage(ctx, edit_id, edit_type, "gate_e", status="fail")
        return False, 0, 1

    ok = _passes(j, edit_type, thr)
    update_edit_gate(ctx, edit_id, edit_type, "E",
                     vlm_result={"pass": ok,
                                 "score": round(j.get("visual_quality", 0) / 5.0, 2),
                                 "reason": j.get("reason", "")})
    update_edit_stage(ctx, edit_id, edit_type, "gate_e",
                      status="pass" if ok else "fail")
    return ok, (1 if ok else 0), (0 if ok else 1)


async def _process_one(
    ctx: ObjectContext,
    vlm_url: str,
    vlm_model: str,
    thr: dict,
    force: bool,
    log: logging.Logger,
) -> dict:
    """Judge all edits for one object using the given VLM URL."""
    if not force and step_done(ctx, "sq3_qc_E"):
        return {"obj_id": ctx.obj_id, "skipped": True}

    n_pass = n_fail = n_skip = 0

    before_imgs = _load_before_imgs(ctx)
    if before_imgs is None:
        log.warning("[sq3] %s: cannot load before images from image_npz", ctx.obj_id)
        update_step(ctx, "sq3_qc_E", status=STATUS_OK,
                    n_pass=0, n_fail=0, n_skip=0,
                    reason="missing_image_npz")
        return {"obj_id": ctx.obj_id, "n_pass": 0, "n_fail": 0}

    client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")

    # spec-based edits (del, mod, scl, mat, glb) — sequential to avoid
    # concurrent qc.json writes from the same object
    for spec in iter_all_specs(ctx):
        if is_edit_qc_failed(ctx, spec.edit_id):
            n_skip += 1
            continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        _, dp, df = await _judge_one_async(
            client, vlm_model,
            spec.edit_id, spec.edit_type,
            spec.prompt, spec.object_desc,
            ", ".join(spec.part_labels),
            before_imgs, edit_dir, thr, ctx,
            target_part_desc=spec.target_part_desc,
            edit_params=spec.edit_params,
        )
        n_pass += dp
        n_fail += df

    # addition edits — swap_collage=True because add_dir/preview_*.png stores
    # the before-addition state (object minus part), while before_imgs from
    # image_npz represents the after-addition target (original complete object).
    for add_id, meta in _iter_add_edits(ctx):
        if is_edit_qc_failed(ctx, add_id):
            n_skip += 1
            continue
        edit_dir = ctx.edit_3d_dir(add_id)
        _, dp, df = await _judge_one_async(
            client, vlm_model,
            add_id, "addition",
            meta.get("prompt", ""),
            meta.get("object_desc", ""),
            ", ".join(meta.get("part_labels", [])),
            before_imgs, edit_dir, thr, ctx,
            target_part_desc=meta.get("target_part_desc", ""),
            edit_params=meta.get("edit_params", {}),
            swap_collage=True,
        )
        n_pass += dp
        n_fail += df

    update_step(ctx, "sq3_qc_E", status=STATUS_OK,
                n_pass=n_pass, n_fail=n_fail, n_skip=n_skip)
    log.info("[sq3] %s done: pass=%d fail=%d skip=%d",
             ctx.obj_id, n_pass, n_fail, n_skip)
    return {"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail}


async def run(
    ctxs: Iterable[ObjectContext],
    *,
    vlm_urls: list[str],
    vlm_model: str,
    cfg: dict,
    force: bool = False,
    concurrency: int = 8,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Async entry point — distribute objects across vlm_urls round-robin.

    Args:
        vlm_urls:    List of VLM base URLs (one per GPU server).
        concurrency: Max number of objects judged simultaneously.
    """
    if not vlm_urls:
        raise ValueError("vlm_urls must not be empty")
    log = logger or LOG
    thr = (cfg.get("qc") or {}).get("thresholds_by_type") or _DEFS
    ctxs = list(ctxs)
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(i: int, ctx: ObjectContext) -> dict:
        url = vlm_urls[i % len(vlm_urls)]
        async with sem:
            return await _process_one(ctx, url, vlm_model, thr, force, log)

    return await asyncio.gather(*[_run_one(i, c) for i, c in enumerate(ctxs)])


__all__ = ["run"]
