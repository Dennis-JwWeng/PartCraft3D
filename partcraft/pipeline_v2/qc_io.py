"""Gate QC helpers stored in edit_status.json.

Legacy callers still use the qc_io API, but data is persisted in
``edit_status.json`` under each edit entry (no qc.json writes).
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ObjectContext


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def _es_lock(ctx: ObjectContext):
    import threading

    lock_path = ctx.dir / "edit_status.json.lock"
    ctx.dir.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    if not hasattr(_es_lock, "_thread_mutexes"):
        _es_lock._thread_mutexes = {}
        _es_lock._thread_mutexes_guard = threading.Lock()
    with _es_lock._thread_mutexes_guard:
        if key not in _es_lock._thread_mutexes:
            _es_lock._thread_mutexes[key] = threading.Lock()
        thread_mtx = _es_lock._thread_mutexes[key]
    with thread_mtx:
        with open(lock_path, "a") as lf:
            fcntl.lockf(lf, fcntl.LOCK_EX)
            yield


def _load_es(ctx: ObjectContext) -> dict[str, Any]:
    p = ctx.dir / "edit_status.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "schema_version": 2,
        "updated": None,
        "edits": {},
        "steps": {},
    }


def _save_es(ctx: ObjectContext, es: dict[str, Any]) -> None:
    es["obj_id"] = ctx.obj_id
    es["shard"] = ctx.shard
    es.setdefault("schema_version", 2)
    es["updated"] = _now()
    ctx.dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".es.", suffix=".tmp", dir=str(ctx.dir))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(es, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ctx.dir / "edit_status.json")
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _gp(gd: dict | None) -> bool:
    if gd is None:
        return True
    r = gd.get("rule")
    v = gd.get("vlm")
    if r is not None and not r.get("pass", True):
        return False
    if v is not None and not v.get("pass", True):
        return False
    return True


def _sync_fail_fields(entry: dict[str, Any]) -> None:
    gates = entry.setdefault("gates", {"A": None, "C": None, "E": None})
    entry["final_pass"] = all(_gp(gates[g]) for g in ("A", "C", "E"))
    if not entry["final_pass"]:
        for g in ("A", "C", "E"):
            gd2 = gates.get(g)
            if gd2 is not None and not _gp(gd2):
                entry["fail_gate"] = g
                r = gd2.get("rule") if isinstance(gd2, dict) else None
                v = gd2.get("vlm") if isinstance(gd2, dict) else None
                if r and not r.get("pass", True):
                    entry["fail_reason"] = next(iter(r.get("checks") or {}), "rule_fail")
                elif v and not v.get("pass", True):
                    entry["fail_reason"] = (v.get("reason") or "vlm_fail")[:80]
                break
    else:
        entry.pop("fail_gate", None)
        entry.pop("fail_reason", None)


def _build_qc_view_from_es(ctx: ObjectContext, es: dict[str, Any]) -> dict[str, Any]:
    edits_out: dict[str, Any] = {}
    for edit_id, e in (es.get("edits") or {}).items():
        if not isinstance(e, dict):
            continue
        gates = dict((e.get("gates") or {}))
        gates.setdefault("A", None)
        gates.setdefault("C", None)
        gates.setdefault("E", None)

        # Backfill from stage statuses when explicit gate payloads are missing.
        stages = e.get("stages") or {}
        for k, g in (("gate_a", "A"), ("gate_c", "C"), ("gate_e", "E")):
            if gates.get(g) is None and k in stages:
                st = (stages.get(k) or {}).get("status")
                if st in ("pass", "fail"):
                    gates[g] = {"vlm": {"pass": st == "pass", "reason": "from_stage_status"}}

        out = {
            "edit_type": e.get("edit_type", ""),
            "gates": gates,
            "final_pass": bool(e.get("final_pass", True)),
        }
        if "fail_gate" in e:
            out["fail_gate"] = e.get("fail_gate")
        if "fail_reason" in e:
            out["fail_reason"] = e.get("fail_reason")
        edits_out[edit_id] = out

    return {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "updated": es.get("updated"),
        "edits": edits_out,
    }


def load_qc(ctx: ObjectContext) -> dict[str, Any]:
    es = _load_es(ctx)
    return _build_qc_view_from_es(ctx, es)


def save_qc(ctx: ObjectContext, qc: dict[str, Any]) -> None:
    with _es_lock(ctx):
        es = _load_es(ctx)
        edits = es.setdefault("edits", {})
        for edit_id, qentry in (qc.get("edits") or {}).items():
            if not isinstance(qentry, dict):
                continue
            e = edits.setdefault(edit_id, {
                "edit_type": qentry.get("edit_type", ""),
                "stages": {},
            })
            e["edit_type"] = qentry.get("edit_type", e.get("edit_type", ""))
            e["gates"] = dict(qentry.get("gates") or {"A": None, "C": None, "E": None})
            _sync_fail_fields(e)
        _save_es(ctx, es)


def update_edit_gate(
    ctx: ObjectContext,
    edit_id: str,
    edit_type: str,
    gate: str,
    *,
    rule_result: dict | None = None,
    vlm_result: dict | None = None,
) -> None:
    with _es_lock(ctx):
        es = _load_es(ctx)
        entry = es.setdefault("edits", {}).setdefault(edit_id, {
            "edit_type": edit_type,
            "stages": {},
            "gates": {"A": None, "C": None, "E": None},
            "final_pass": False,
        })
        entry["edit_type"] = edit_type
        gates = entry.setdefault("gates", {"A": None, "C": None, "E": None})
        gd: dict[str, Any] = {}
        if rule_result is not None:
            gd["rule"] = rule_result
        if vlm_result is not None:
            gd["vlm"] = vlm_result
        gates[gate] = gd if gd else None
        _sync_fail_fields(entry)
        _save_es(ctx, es)


def is_edit_qc_failed(ctx: ObjectContext, edit_id: str) -> bool:
    e = load_qc(ctx).get("edits", {}).get(edit_id)
    return e is not None and e.get("final_pass", True) is False


def is_gate_a_failed(ctx: ObjectContext, edit_id: str) -> bool:
    """Block only on Gate A failures (upstream part-selection errors)."""
    e = load_qc(ctx).get("edits", {}).get(edit_id)
    if e is None:
        return False
    gate_a = (e.get("gates") or {}).get("A")
    return not _gp(gate_a)


__all__ = ["load_qc", "save_qc", "update_edit_gate", "is_edit_qc_failed", "is_gate_a_failed"]
