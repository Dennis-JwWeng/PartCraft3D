"""Per-edit operational state tracking (edit_status.json).

``edit_status.json`` records which pipeline stages each edit has reached
and whether they succeeded.  It complements (does not replace)
``status.json`` (step-level aggregates) and ``qc.json`` (gate quality
signals).

Schema::

    {
      "obj_id": "...",
      "shard": "06",
      "schema_version": 1,
      "updated": "2026-04-14T07:00:00",
      "edits": {
        "mod_..._001": {
          "edit_type": "modification",
          "stages": {
            "gate_a": {"status": "pass", "ts": "..."},
            "s4":     {"status": "done", "ts": "..."},
            ...
          }
        }
      }
    }

Status values:
  - ``"pass"`` / ``"fail"`` for gate stages
  - ``"done"`` / ``"error"`` for processing stages
  - absent key = not yet reached
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ObjectContext

SCHEMA_VERSION = 1

_thread_mutexes: dict[str, threading.Lock] = {}
_thread_guard = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def _edit_status_lock(ctx: ObjectContext):
    """Per-object exclusive lock for edit_status.json read-modify-write.

    Same pattern as ``status._status_lock``: fcntl.lockf (NFS-safe) +
    per-key threading.Lock for in-process safety.
    """
    lock_path = ctx.dir / "edit_status.json.lock"
    ctx.dir.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    with _thread_guard:
        if key not in _thread_mutexes:
            _thread_mutexes[key] = threading.Lock()
        thread_mtx = _thread_mutexes[key]
    with thread_mtx:
        with open(lock_path, "a") as lf:
            fcntl.lockf(lf, fcntl.LOCK_EX)
            yield


# --- I/O ---

def _es_path(ctx: ObjectContext) -> Path:
    return ctx.dir / "edit_status.json"


def load_edit_status(ctx: ObjectContext) -> dict[str, Any]:
    p = _es_path(ctx)
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "schema_version": SCHEMA_VERSION,
        "updated": None,
        "edits": {},
    }


def save_edit_status(ctx: ObjectContext, data: dict[str, Any]) -> None:
    data["updated"] = _now()
    p = _es_path(ctx)
    ctx.dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".es.", suffix=".tmp", dir=str(ctx.dir))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


# --- read-modify-write ---

def update_edit_stage(
    ctx: ObjectContext,
    edit_id: str,
    edit_type: str,
    stage_key: str,
    *,
    status: str,
    reason: str | None = None,
) -> None:
    """Atomically set the status for one stage of one edit.

    Process-safe via lockf + threading.Lock.
    """
    with _edit_status_lock(ctx):
        es = load_edit_status(ctx)
        edit_entry = es.setdefault("edits", {}).setdefault(edit_id, {
            "edit_type": edit_type,
            "stages": {},
        })
        stage_entry: dict[str, Any] = {"status": status, "ts": _now()}
        if reason is not None:
            stage_entry["reason"] = reason
        edit_entry.setdefault("stages", {})[stage_key] = stage_entry
        save_edit_status(ctx, es)


# --- resume logic ---

def edit_needs_step(
    ctx: ObjectContext,
    edit_id: str,
    stage_key: str,
    prereq_map: dict[str, str | None],
    *,
    force: bool = False,
) -> bool:
    """Single authoritative resume function for all processing steps.

    Returns True iff the edit should be (re-)processed for *stage_key*:
      1. Prerequisite gate must have status ``"pass"`` -- always enforced,
         even when *force* is True.  Additions are exempt from ``gate_a``
         (D5: guaranteed by construction).
      2. Without *force*: own stage absent -> run; ``"error"`` -> retry;
         ``"done"``/``"pass"`` -> skip.
      3. With *force*: always run (gate permitting).
    """
    stages = (load_edit_status(ctx)
              .get("edits", {})
              .get(edit_id, {})
              .get("stages", {}))

    prereq = prereq_map.get(stage_key)
    if prereq:
        if edit_id.startswith("add_") and prereq == "gate_a":
            pass  # D5: additions exempt from gate_a
        else:
            gate_status = stages.get(prereq, {}).get("status")
            if gate_status != "pass":
                return False

    if force:
        return True

    own = stages.get(stage_key)
    if own is None:
        return True
    return own.get("status") == "error"


def gate_already_done(
    ctx: ObjectContext,
    edit_id: str,
    gate_key: str,
) -> bool:
    """True iff the gate has already been evaluated (pass or fail)."""
    stages = (load_edit_status(ctx)
              .get("edits", {})
              .get(edit_id, {})
              .get("stages", {}))
    entry = stages.get(gate_key)
    return entry is not None and entry.get("status") in ("pass", "fail")


# --- config-derived prerequisites ---

def build_prereq_map(cfg: dict) -> dict[str, str | None]:
    """Derive stage prerequisites from active config steps.

    Only gates whose QC step is present in the config become
    prerequisites.  D4: gate_c is optional (only when sq2 is active).
    """
    active = {step for stage in cfg.get("pipeline", {}).get("stages", [])
              for step in stage.get("steps", [])}
    return {
        "s4":  "gate_a" if "sq1" in active else None,
        "s5b": "gate_a" if "sq1" in active else None,
        "s5":  ("gate_c" if "sq2" in active else
                "gate_a" if "sq1" in active else None),
        "s6p": "gate_a" if "sq1" in active else None,
        "s6":  "gate_e" if "sq3" in active else None,
        "s6b": "gate_e" if "sq3" in active else None,
    }


__all__ = [
    "load_edit_status", "save_edit_status",
    "update_edit_stage", "edit_needs_step", "gate_already_done",
    "build_prereq_map",
]
