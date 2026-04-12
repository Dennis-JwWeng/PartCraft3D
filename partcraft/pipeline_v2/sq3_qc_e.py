"""Step sq3 / E_qc — VLM final quality gate (all edit types).

For every non-identity edit that has survived gate A, we:
  1. Load 5 before-state views from image_npz (zero render cost).
  2. Load 5 after-state preview_{0..4}.png rendered by s6p.
  3. Build a 2-row × 5-col collage (top row = before, bottom = after).
  4. Call the VLM judge and record the result in qc.json.

Coverage now includes *addition* edits (discovered via edits_3d/add_*/meta.json).
Step key: sq3_qc_E.
"""
from __future__ import annotations

import json
import logging

import cv2
import numpy as np
from openai import OpenAI

from .paths import ObjectContext
from .specs import iter_all_specs, VIEW_INDICES
from .status import update_step, STATUS_OK, step_done
from .qc_io import update_edit_gate, is_edit_qc_failed


_DEFS = {
    "deletion":     {"min_visual_quality": 3, "require_preserve_other": False},
    "modification": {"min_visual_quality": 3, "require_preserve_other": True},
    "scale":        {"min_visual_quality": 3, "require_preserve_other": True},
    "material":     {"min_visual_quality": 3, "require_preserve_other": False},
    "global":       {"min_visual_quality": 3, "require_preserve_other": False},
    "addition":     {"min_visual_quality": 3, "require_preserve_other": False},
}


def _passes(j, et, thr):
    t = {**_DEFS.get(et, {}), **(thr.get(et) or {})}
    if not j.get("edit_executed", False): return False
    vq = j.get("visual_quality", 0)
    try: vq = int(vq)
    except (TypeError, ValueError): vq = 0
    if vq < t.get("min_visual_quality", 3): return False
    if et == "deletion" and not j.get("correct_region", False): return False
    if t.get("require_preserve_other") and not j.get("preserve_other", False): return False
    return True


def _load_before_imgs(ctx: ObjectContext) -> list[np.ndarray] | None:
    """Load 5 before-state BGR images from image_npz at VIEW_INDICES."""
    if ctx.image_npz is None or not ctx.image_npz.is_file():
        return None
    try:
        from partcraft.render.overview import load_views_from_npz
        imgs, _ = load_views_from_npz(ctx.image_npz, VIEW_INDICES)
        return imgs  # list of BGR np.ndarray
    except Exception:
        return None


def _load_after_previews(edit_dir) -> list[np.ndarray] | None:
    """Load preview_0.png … preview_4.png from edit_dir. Returns None on any missing."""
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
    h = 256  # per-image height
    def _r(x):
        s = h / x.shape[0]
        return cv2.resize(x, (int(x.shape[1] * s), h))
    try:
        row_b = np.hstack([_r(img) for img in before_imgs])
        row_a = np.hstack([_r(img) for img in after_imgs])
        # Pad to same width
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


def _judge_one(
    client,
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
    log: logging.Logger,
) -> tuple[bool, int, int]:
    """Run VLM judge for one edit. Returns (ok, n_pass_delta, n_fail_delta)."""
    from partcraft.cleaning.vlm_filter import call_vlm_judge

    if is_edit_qc_failed(ctx, edit_id):
        return False, 0, 0  # skip

    after_imgs = _load_after_previews(edit_dir)
    if after_imgs is None:
        update_edit_gate(ctx, edit_id, edit_type, "E",
                         vlm_result={"pass": False, "score": 0.0,
                                     "reason": "missing_previews"})
        return False, 0, 1

    coll = _make_collage(before_imgs, after_imgs)
    if coll is None:
        update_edit_gate(ctx, edit_id, edit_type, "E",
                         vlm_result={"pass": False, "score": 0.0,
                                     "reason": "collage_failed"})
        return False, 0, 1

    j = call_vlm_judge(client, vlm_model, coll,
                       edit_prompt=prompt, edit_type=edit_type,
                       object_desc=obj_desc, part_label=part_label)
    if j is None:
        update_edit_gate(ctx, edit_id, edit_type, "E",
                         vlm_result={"pass": False, "score": 0.0,
                                     "reason": "vlm_no_response"})
        return False, 0, 1

    ok = _passes(j, edit_type, thr)
    update_edit_gate(ctx, edit_id, edit_type, "E",
                     vlm_result={"pass": ok,
                                 "score": round(j.get("visual_quality", 0) / 5.0, 2),
                                 "reason": j.get("reason", "")})
    return ok, (1 if ok else 0), (0 if ok else 1)


def run(ctxs, *, vlm_url, vlm_model, cfg, force=False, logger=None):
    log = logger or logging.getLogger("pipeline_v2.sq3")
    thr = (cfg.get("qc") or {}).get("thresholds_by_type") or _DEFS
    client = OpenAI(base_url=vlm_url, api_key="EMPTY")
    out = []

    for ctx in ctxs:
        if not force and step_done(ctx, "sq3_qc_E"):
            out.append({"obj_id": ctx.obj_id, "skipped": True})
            continue

        n_pass = n_fail = n_skip = 0

        before_imgs = _load_before_imgs(ctx)
        if before_imgs is None:
            log.warning("[sq3] %s: cannot load before images from image_npz", ctx.obj_id)
            update_step(ctx, "sq3_qc_E", status=STATUS_OK,
                        n_pass=0, n_fail=0, n_skip=0,
                        reason="missing_image_npz")
            out.append({"obj_id": ctx.obj_id, "n_pass": 0, "n_fail": 0})
            continue

        # --- spec-based edits (del, mod, scl, mat, glb) ---
        for spec in iter_all_specs(ctx):
            if is_edit_qc_failed(ctx, spec.edit_id):
                n_skip += 1
                continue
            edit_dir = ctx.edit_3d_dir(spec.edit_id)
            _, dp, df = _judge_one(
                client, vlm_model,
                spec.edit_id, spec.edit_type,
                spec.prompt, spec.object_desc,
                ", ".join(spec.part_labels),
                before_imgs, edit_dir, thr, ctx, log,
            )
            n_pass += dp
            n_fail += df

        # --- addition edits (from edits_3d/add_*/meta.json) ---
        for add_id, meta in _iter_add_edits(ctx):
            if is_edit_qc_failed(ctx, add_id):
                n_skip += 1
                continue
            edit_dir = ctx.edit_3d_dir(add_id)
            _, dp, df = _judge_one(
                client, vlm_model,
                add_id, "addition",
                meta.get("prompt", ""),
                meta.get("object_desc", ""),
                ", ".join(meta.get("part_labels", [])),
                before_imgs, edit_dir, thr, ctx, log,
            )
            n_pass += dp
            n_fail += df

        update_step(ctx, "sq3_qc_E", status=STATUS_OK,
                    n_pass=n_pass, n_fail=n_fail, n_skip=n_skip)
        out.append({"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail})

    return out


__all__ = ["run"]
