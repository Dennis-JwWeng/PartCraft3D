from __future__ import annotations
import logging
import cv2, numpy as np
from openai import OpenAI
from .paths import ObjectContext
from .specs import iter_all_specs
from .status import update_step, STATUS_OK, step_done
from .qc_io import update_edit_gate, is_edit_qc_failed

LOG = logging.getLogger("pipeline_v2.sq3")
_DEFS = {
    "deletion":     {"min_visual_quality": 3, "require_preserve_other": False},
    "modification": {"min_visual_quality": 3, "require_preserve_other": True},
    "scale":        {"min_visual_quality": 3, "require_preserve_other": True},
    "material":     {"min_visual_quality": 3, "require_preserve_other": False},
    "global":       {"min_visual_quality": 3, "require_preserve_other": False},
    "addition":     {"min_visual_quality": 3, "require_preserve_other": False},
}

def _collage(b, a):
    bi = cv2.imread(str(b)); ai = cv2.imread(str(a))
    if bi is None or ai is None: return None
    h = 512
    def _r(x): s = h / x.shape[0]; return cv2.resize(x, (int(x.shape[1] * s), h))
    ok, buf = cv2.imencode(".png", np.vstack([_r(bi), _r(ai)]))
    return buf.tobytes() if ok else None

def _passes(j, et, thr):
    t = thr.get(et, _DEFS.get(et, {}))
    if not j.get("edit_executed", False): return False
    if j.get("visual_quality", 0) < t.get("min_visual_quality", 3): return False
    if et == "deletion" and not j.get("correct_region", False): return False
    if t.get("require_preserve_other") and not j.get("preserve_other", False): return False
    return True

def run(ctxs, *, vlm_url, vlm_model, cfg, force=False):
    from partcraft.cleaning.vlm_filter import call_vlm_judge
    thr = (cfg.get("qc") or {}).get("thresholds_by_type") or _DEFS
    client = OpenAI(base_url=vlm_url, api_key="EMPTY")
    out = []
    for ctx in ctxs:
        if not force and step_done(ctx, "sq3_qc_E"):
            out.append({"obj_id": ctx.obj_id, "skipped": True}); continue
        n_pass = n_fail = n_skip = 0
        for spec in iter_all_specs(ctx):
            if is_edit_qc_failed(ctx, spec.edit_id): n_skip += 1; continue
            bp = ctx.edit_3d_png(spec.edit_id, "before"); ap = ctx.edit_3d_png(spec.edit_id, "after")
            if not bp.is_file() or not ap.is_file():
                update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                                 vlm_result={"pass": False, "score": 0.0, "reason": "missing_3d_renders"})
                n_fail += 1; continue
            coll = _collage(bp, ap)
            if coll is None:
                update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                                 vlm_result={"pass": False, "score": 0.0, "reason": "collage_failed"})
                n_fail += 1; continue
            j = call_vlm_judge(client, vlm_model, coll, edit_prompt=spec.prompt,
                               edit_type=spec.edit_type, object_desc=spec.object_desc,
                               part_label=", ".join(spec.part_labels))
            if j is None:
                update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                                 vlm_result={"pass": False, "score": 0.0, "reason": "vlm_no_response"})
                n_fail += 1; continue
            ok = _passes(j, spec.edit_type, thr)
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                             vlm_result={"pass": ok,
                                         "score": round(j.get("visual_quality", 0) / 5.0, 2),
                                         "reason": j.get("reason", "")})
            if ok: n_pass += 1
            else: n_fail += 1
        update_step(ctx, "sq3_qc_E", status=STATUS_OK,
                    n_pass=n_pass, n_fail=n_fail, n_skip=n_skip)
        out.append({"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail})
    return out

__all__ = ["run"]
