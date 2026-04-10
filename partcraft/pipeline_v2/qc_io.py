from __future__ import annotations
import json, os, tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from .paths import ObjectContext

def _now(): return datetime.now().isoformat(timespec="seconds")

def load_qc(ctx: ObjectContext) -> dict[str, Any]:
    if ctx.qc_path.is_file():
        try: return json.loads(ctx.qc_path.read_text())
        except json.JSONDecodeError: pass
    return {"obj_id": ctx.obj_id, "shard": ctx.shard, "updated": None, "edits": {}}

def save_qc(ctx: ObjectContext, qc: dict[str, Any]) -> None:
    qc.update({"obj_id": ctx.obj_id, "shard": ctx.shard, "updated": _now()})
    ctx.dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".qc.", suffix=".tmp", dir=str(ctx.dir))
    try:
        with os.fdopen(fd, "w") as f: json.dump(qc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ctx.qc_path)
    except Exception: Path(tmp).unlink(missing_ok=True); raise

def update_edit_gate(
    ctx: ObjectContext, edit_id: str, edit_type: str, gate: str,
    *, rule_result: dict | None = None, vlm_result: dict | None = None,
) -> None:
    qc = load_qc(ctx)
    entry = qc.setdefault("edits", {}).setdefault(edit_id, {
        "edit_type": edit_type, "gates": {"A": None, "C": None, "E": None}, "final_pass": False})
    gd: dict[str, Any] = {}
    if rule_result is not None: gd["rule"] = rule_result
    if vlm_result is not None:  gd["vlm"] = vlm_result
    entry["gates"][gate] = gd if gd else None
    entry["final_pass"] = all(_gp(entry["gates"][g]) for g in ("A", "C", "E"))
    if not entry["final_pass"]:
        for g in ("A", "C", "E"):
            gd2 = entry["gates"][g]
            if gd2 is not None and not _gp(gd2):
                entry["fail_gate"] = g
                r = gd2.get("rule"); v = gd2.get("vlm")
                if r and not r.get("pass", True):
                    entry["fail_reason"] = next(iter(r.get("checks") or {}), "rule_fail")
                elif v and not v.get("pass", True):
                    entry["fail_reason"] = (v.get("reason") or "vlm_fail")[:80]
                break
    else:
        entry.pop("fail_gate", None); entry.pop("fail_reason", None)
    save_qc(ctx, qc)

def _gp(gd: dict | None) -> bool:
    if gd is None: return True
    r = gd.get("rule"); v = gd.get("vlm")
    if r is not None and not r.get("pass", True): return False
    if v is not None and not v.get("pass", True): return False
    return True

def is_edit_qc_failed(ctx: ObjectContext, edit_id: str) -> bool:
    if not ctx.qc_path.is_file(): return False
    e = load_qc(ctx).get("edits", {}).get(edit_id)
    return e is not None and e.get("final_pass", True) is False

def is_gate_a_failed(ctx: ObjectContext, edit_id: str) -> bool:
    """Block only on Gate A failures (upstream part-selection errors).

    Gate C (2D image region check) is intentionally excluded: spatial
    localisation is handled by the 3D voxel mask, so a globally-edited
    2D reference image is still valid as appearance guidance.
    Gate E is a post-s5 quality signal and never blocks s5 input.
    """
    if not ctx.qc_path.is_file(): return False
    e = load_qc(ctx).get("edits", {}).get(edit_id)
    if e is None: return False
    gate_a = (e.get("gates") or {}).get("A")
    return not _gp(gate_a)

__all__ = ["load_qc", "save_qc", "update_edit_gate", "is_edit_qc_failed", "is_gate_a_failed"]
