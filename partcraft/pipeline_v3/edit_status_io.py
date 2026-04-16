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
    _empty: dict[str, Any] = {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "schema_version": SCHEMA_VERSION,
        "updated": None,
        "edits": {},
    }
    for _attempt in range(3):
        try:
            if not p.is_file():
                return _empty
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
            return _empty
        except json.JSONDecodeError:
            return _empty
        except OSError:
            import time as _t
            _t.sleep(0.5 * (_attempt + 1))
    return _empty


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


def obj_needs_stage(
    ctx: ObjectContext,
    edit_ids: list[str],
    stage_key: str,
    prereq_map: dict[str, str | None],
    *,
    force: bool = False,
) -> bool:
    """Return True iff any edit in *edit_ids* needs *stage_key* processed.

    Loads ``edit_status.json`` exactly once for the object, making this
    efficient for per-object filtering before dispatching to a step runner.
    Replicates the gate + force + own-status logic of :func:`edit_needs_step`.
    """
    es_edits = load_edit_status(ctx).get("edits", {})
    prereq = prereq_map.get(stage_key)
    for edit_id in edit_ids:
        stages = es_edits.get(edit_id, {}).get("stages", {})
        if prereq:
            if not (edit_id.startswith("add_") and prereq == "gate_a"):
                if stages.get(prereq, {}).get("status") != "pass":
                    continue  # gate not passed → skip this edit
        if force:
            return True
        own = stages.get(stage_key)
        if own is None or own.get("status") == "error":
            return True
    return False


# --- config-derived prerequisites ---

def build_prereq_map(cfg: dict) -> dict[str, str | None]:
    """Return the prerequisite gate key for each pipeline step.

    Each entry maps a step identifier (as used in ALL_STEPS / --steps) to
    the ``edit_status.json`` stage key that must have ``status == "pass"``
    before that step is allowed to run on a given edit.

    Gate prerequisite rules (Mode E):
      * ``del_mesh`` and ``preview_del`` require gate_a (text-align gate).
      * ``gate_quality`` (Gate E) has no hard file prerequisite here;
        it is gated implicitly by the presence of preview_{0..4}.png.
      * Inactive steps (flux_2d, trellis_3d, etc.) are commented out.

    The ``active`` set is derived from ``pipeline.stages[].steps`` in the
    YAML config, so enabling a QC gate in config automatically wires it as
    a prerequisite without touching this function.
    """
    # Read active step names from config stages to decide whether each gate
    # should act as a prerequisite.  Both the functional names (new CLI names)
    # and the internal stage keys are accepted so that either config style works.
    active = {step for stage in cfg.get("pipeline", {}).get("stages", [])
              for step in stage.get("steps", [])}
    # gate_a is a prerequisite when gate_text_align (or its legacy alias sq1)
    # is declared in the active stages.
    gate_a_active = "gate_text_align" in active or "sq1" in active
    gate_e_active = "gate_quality" in active or "sq3" in active
    prereq_a = "gate_a" if gate_a_active else None
    prereq_e = "gate_e" if gate_e_active else None

    # Keys here are the INTERNAL stage keys used by step runners when they
    # call edit_needs_step(ctx, edit_id, stage_key, prereq_map).
    # Do NOT change these to functional/CLI names — the step runner code
    # (s5b_deletion.py, s6_preview.py, etc.) passes these strings directly.
    return {
        # Internal stage keys used by step runners in edit_needs_step() calls.
        # These MUST match the strings passed in s4/s5/s5b/s6/s6p/*.py code.
        "s4":  prereq_a,   # flux_2d step runner (s4_flux_2d.py)
        "s5":  prereq_a,   # trellis_3d step runner (s5_trellis_3d.py)
        "s5b": prereq_a,   # del_mesh step runner (s5b_deletion.py)
        "s6p": prereq_a,   # preview_del / preview_flux step runner (s6_preview.py)
        "s6":  prereq_e,   # render_3d step runner (s6_render_3d.py)
        "s6b": prereq_e,   # reencode_del step runner (s5b_deletion.py run_reencode)
    }


__all__ = [
    "load_edit_status", "save_edit_status", "obj_needs_stage",
    "update_edit_stage", "edit_needs_step", "gate_already_done",
    "build_prereq_map",
]
