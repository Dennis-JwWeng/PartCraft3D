"""Per-object status.json + global manifest rebuild.

``status.json`` lives at ``ObjectContext.status_path`` and tracks which
pipeline steps have run for that object. It is the single source of
truth for resume / skip logic — orchestration never relies on a separate
cache directory.

Schema (all keys optional, missing = not run)::

    {
      "obj_id": "...",
      "shard": "01",
      "steps": {
        "s1_phase1":     {"status": "ok", "n_edits": 17, "ts": "2026-04-07T..."},
        "s2_highlights": {"status": "ok", "n": 17},
        "s4_flux_2d":    {"status": "ok", "n": 11, "fail": 0},
        "s5_trellis":    {"status": "ok", "n": 11, "fail": 0},
        "s6_render_3d":  {"status": "ok", "n": 22},
        "s5b_del_mesh":  {"status": "pending"},
        ...
      },
      "updated": "2026-04-07T..."
    }

Atomic writes: write to ``status.json.tmp`` then ``os.replace`` so a
crash mid-write never leaves a half-file.

The global manifest at ``PipelineRoot.manifest_path`` is a derived view:
one line per object, regenerated from all ``status.json``s on demand.
It is never updated incrementally — always rebuilt with
:func:`rebuild_manifest`.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ObjectContext, PipelineRoot, normalize_shard

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"
STATUS_PENDING = "pending"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_status(ctx: ObjectContext) -> dict[str, Any]:
    """Read status.json. Returns empty skeleton if absent."""
    if ctx.status_path.is_file():
        try:
            return json.loads(ctx.status_path.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "steps": {},
        "updated": None,
    }


def save_status(ctx: ObjectContext, status: dict[str, Any]) -> None:
    """Atomic write to status.json."""
    status["obj_id"] = ctx.obj_id
    status["shard"] = ctx.shard
    status["updated"] = _now()
    ctx.dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".status.", suffix=".tmp", dir=str(ctx.dir)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ctx.status_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_step(
    ctx: ObjectContext,
    step: str,
    *,
    status: str = STATUS_OK,
    **fields: Any,
) -> dict[str, Any]:
    """Read-modify-write a single step entry."""
    s = load_status(ctx)
    s.setdefault("steps", {})[step] = {
        "status": status, "ts": _now(), **fields,
    }
    save_status(ctx, s)
    return s


def step_done(ctx: ObjectContext, step: str) -> bool:
    """True iff status.json marks ``step`` as ``ok``."""
    s = load_status(ctx)
    return (s.get("steps") or {}).get(step, {}).get("status") == STATUS_OK


def needs_step(ctx: ObjectContext, step: str, *, force: bool = False) -> bool:
    return force or not step_done(ctx, step)


# ─────────────────── global manifest (derived) ────────────────────────

def rebuild_manifest(root: PipelineRoot) -> Path:
    """Scan all ``objects/<shard>/<obj>/status.json`` and rewrite the
    global manifest at ``_global/manifest.jsonl``.

    Each line: ``{"shard": ..., "obj_id": ..., "steps": {...}}``.
    """
    root.ensure()
    lines: list[str] = []
    if root.objects_root.is_dir():
        for shard_dir in sorted(root.objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            shard = shard_dir.name
            for od in sorted(shard_dir.iterdir()):
                if not od.is_dir():
                    continue
                ctx = root.context(shard, od.name)
                s = load_status(ctx)
                lines.append(json.dumps({
                    "shard": shard,
                    "obj_id": od.name,
                    "steps": s.get("steps") or {},
                    "updated": s.get("updated"),
                }, ensure_ascii=False))
    tmp = root.manifest_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, root.manifest_path)
    return root.manifest_path


def manifest_summary(root: PipelineRoot) -> dict[str, Any]:
    """Cheap aggregate over the manifest (does not rebuild)."""
    if not root.manifest_path.is_file():
        return {"objects": 0, "steps": {}}
    objs = 0
    step_counts: dict[str, dict[str, int]] = {}
    with open(root.manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            objs += 1
            for k, v in (r.get("steps") or {}).items():
                bucket = step_counts.setdefault(k, {})
                bucket[v.get("status", "?")] = bucket.get(
                    v.get("status", "?"), 0) + 1
    return {"objects": objs, "steps": step_counts}


__all__ = [
    "STATUS_OK", "STATUS_FAIL", "STATUS_SKIP", "STATUS_PENDING",
    "load_status", "save_status", "update_step",
    "step_done", "needs_step",
    "rebuild_manifest", "manifest_summary",
]
