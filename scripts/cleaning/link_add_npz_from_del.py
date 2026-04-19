#!/usr/bin/env python
"""Backfill addition edits' per-edit ``before.npz`` / ``after.npz`` in v1 layout.

In the v1 layout, every promoted edit owns a per-edit ``before.npz`` and
``after.npz`` with the combined SLAT+SS schema (keys: ``slat_feats``,
``slat_coords``, ``ss``).  Addition edits cannot be materialised at promote
time, because they are derived from the corresponding deletion's encoded
output:

    add.before.npz  <-  del.after.npz   (= state with the part removed)
    add.after.npz   <-  original object  (= same as obj-level shared cache)

This script runs *after* ``encode_del_latent`` has produced
``edits/{del_id}/after.npz`` for the source deletion.  It is **idempotent**:
re-running it on a fully-linked dataset is a no-op.

Per-edit ``add.after.npz`` is hardlinked to a per-object cache
(``before/_combined.npz``) so we materialise the original-object combined npz
at most once per object.  The cache is built on demand from the SLAT/SS .pt
files in ``--slat-root`` (same layout used by ``_materialize_before``).

Pending markers
---------------
When the source deletion has not yet been encoded, this script writes a
``_before_pending.json`` marker into the addition edit dir so subsequent
``v1_status`` runs can report exactly which addition edits are blocked and on
which deletions.  When the link succeeds, the marker is removed and the
addition's ``qc.json.passes`` gains ``add_npz_link: pass``.

Usage
-----
    python -m scripts.cleaning.link_add_npz_from_del \
        --v1-root data/partverse_edit_v1 \
        --slat-root data/partverse_slat_v0 \
        [--shard 08] [--obj-id <oid>] \
        [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partcraft.cleaning.v1.layout import V1Layout  # noqa: E402

LOG = logging.getLogger("link_add_npz_from_del")


def _now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LinkStats:
    scanned: int = 0
    linked_before: int = 0
    linked_after: int = 0
    deferred: int = 0
    already_linked: int = 0
    no_source_field: int = 0
    bad_qc: int = 0
    errors: list[str] = field(default_factory=list)

    def to_summary(self) -> str:
        return (
            f"scanned={self.scanned}  "
            f"linked_before={self.linked_before}  "
            f"linked_after={self.linked_after}  "
            f"deferred={self.deferred}  "
            f"already_linked={self.already_linked}  "
            f"no_source_field={self.no_source_field}  "
            f"bad_qc={self.bad_qc}  "
            f"errors={len(self.errors)}"
        )


@dataclass(frozen=True)
class AddEdit:
    shard: str
    obj_id: str
    edit_id: str
    suffix: str
    edit_dir: Path


def _hardlink_or_copy(src: Path, dst: Path, *, force: bool) -> str:
    if dst.exists():
        if not force:
            return "exists"
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(str(src), str(dst))
        return "fellback_copy"


def _write_combined_cache(
    *, shard: str, obj_id: str, slat_root: Path, layout: V1Layout, force: bool
) -> Optional[Path]:
    cache = layout.before_dir(shard, obj_id) / "_combined.npz"
    if cache.is_file() and not force:
        return cache

    feats_pt = slat_root / shard / f"{obj_id}_feats.pt"
    coords_pt = slat_root / shard / f"{obj_id}_coords.pt"
    ss_pt = slat_root / shard / f"{obj_id}_ss.pt"
    if not feats_pt.is_file() or not coords_pt.is_file() or not ss_pt.is_file():
        return None

    import numpy as np
    import torch

    def _to_np(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        if hasattr(t, "numpy"):
            return t.numpy()
        return t

    feats = _to_np(torch.load(feats_pt, map_location="cpu", weights_only=False))
    coords = _to_np(torch.load(coords_pt, map_location="cpu", weights_only=False))
    ss = _to_np(torch.load(ss_pt, map_location="cpu", weights_only=False))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, slat_feats=feats, slat_coords=coords.astype("int32"), ss=ss)
    return cache


def _read_qc(edit_dir: Path) -> dict | None:
    p = edit_dir / "qc.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        LOG.warning("bad qc.json %s: %s", p, e)
        return None


def _source_del_id(qc: dict) -> str | None:
    gta = qc.get("passes", {}).get("gate_text_align") or {}
    extra = gta.get("extra") or {}
    return extra.get("inherited_from") or None


def _iter_add_edits(
    layout: V1Layout, *, shard: str | None, obj_id: str | None
) -> Iterator[AddEdit]:
    objects_root = layout.root / "objects"
    if not objects_root.is_dir():
        return
    shard_iter = (
        [objects_root / shard] if shard else sorted(objects_root.iterdir())
    )
    for shard_dir in shard_iter:
        if not shard_dir.is_dir():
            continue
        s = shard_dir.name
        obj_iter = (
            [shard_dir / obj_id] if obj_id else sorted(shard_dir.iterdir())
        )
        for obj_dir in obj_iter:
            if not obj_dir.is_dir():
                continue
            edits_root = obj_dir / "edits"
            if not edits_root.is_dir():
                continue
            for ed in sorted(edits_root.iterdir()):
                if not ed.is_dir():
                    continue
                name = ed.name
                if "__r" in name:
                    base, _, n = name.rpartition("__r")
                    if n.isdigit():
                        eid, suf = base, f"__r{n}"
                    else:
                        eid, suf = name, ""
                else:
                    eid, suf = name, ""
                if not eid.startswith("add_"):
                    continue
                yield AddEdit(s, obj_dir.name, eid, suf, ed)


def _write_pending_marker(edit_dir: Path, *, source_del_id: str) -> None:
    (edit_dir / "_before_pending.json").write_text(
        json.dumps({"source_del_id": source_del_id, "ts": _now_z()})
    )


def _clear_pending_marker(edit_dir: Path) -> None:
    p = edit_dir / "_before_pending.json"
    if p.is_file():
        p.unlink()


def _record_qc_pass(edit_dir: Path, *, reason: str) -> None:
    qc_p = edit_dir / "qc.json"
    if not qc_p.is_file():
        return
    qc = json.loads(qc_p.read_text())
    qc.setdefault("passes", {})["add_npz_link"] = {
        "pass": True,
        "score": None,
        "producer": "link_add_npz_from_del.py@1.0.0",
        "reason": reason,
        "ts": _now_z(),
    }
    qc_p.write_text(json.dumps(qc, indent=2))


def link_one(
    add: AddEdit, *, layout: V1Layout, slat_root: Path,
    stats: LinkStats, dry_run: bool, force: bool,
) -> None:
    qc = _read_qc(add.edit_dir)
    if qc is None:
        stats.bad_qc += 1
        stats.errors.append(f"{add.edit_id}: missing/bad qc.json")
        return

    src_del = _source_del_id(qc)
    if not src_del:
        stats.no_source_field += 1
        stats.errors.append(
            f"{add.edit_id}: passes.gate_text_align.extra.inherited_from missing"
        )
        return

    del_after = layout.after_npz_path(add.shard, add.obj_id, src_del)
    add_before = layout.before_npz_path(
        add.shard, add.obj_id, add.edit_id, suffix=add.suffix
    )
    add_after = layout.after_npz_path(
        add.shard, add.obj_id, add.edit_id, suffix=add.suffix
    )

    has_before = add_before.is_file() and not force
    has_after = add_after.is_file() and not force
    if has_before and has_after:
        _clear_pending_marker(add.edit_dir)
        stats.already_linked += 1
        return

    if not has_before:
        if not del_after.is_file():
            if not dry_run:
                _write_pending_marker(add.edit_dir, source_del_id=src_del)
            stats.deferred += 1
            return
        if dry_run:
            stats.linked_before += 1
        else:
            res = _hardlink_or_copy(del_after, add_before, force=force)
            if res in ("linked", "fellback_copy"):
                stats.linked_before += 1

    if not has_after:
        if dry_run:
            stats.linked_after += 1
        else:
            cache = _write_combined_cache(
                shard=add.shard, obj_id=add.obj_id,
                slat_root=slat_root, layout=layout, force=False,
            )
            if cache is None:
                stats.errors.append(
                    f"{add.edit_id}: cannot build _combined.npz "
                    f"(missing SLAT/SS .pt under {slat_root}/{add.shard}/)"
                )
                return
            res = _hardlink_or_copy(cache, add_after, force=force)
            if res in ("linked", "fellback_copy"):
                stats.linked_after += 1

    if not dry_run:
        _clear_pending_marker(add.edit_dir)
        _record_qc_pass(add.edit_dir, reason=f"linked_from_del:{src_del}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v1-root", type=Path, required=True)
    ap.add_argument(
        "--slat-root", type=Path, required=True,
        help="Directory with {shard}/{obj_id}_{feats,coords,ss}.pt files "
             "(same as promoter's --slat-root).",
    )
    ap.add_argument("--shard", default=None)
    ap.add_argument("--obj-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing add.before.npz / add.after.npz hardlinks.",
    )
    ap.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v=INFO, -vv=DEBUG).",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    layout = V1Layout(root=args.v1_root)
    stats = LinkStats()

    for add in _iter_add_edits(layout, shard=args.shard, obj_id=args.obj_id):
        stats.scanned += 1
        try:
            link_one(
                add, layout=layout, slat_root=args.slat_root,
                stats=stats, dry_run=args.dry_run, force=args.force,
            )
        except Exception as e:
            stats.errors.append(f"{add.edit_id}: {type(e).__name__}: {e}")
            LOG.exception("failed on %s", add.edit_id)

    LOG.info("DONE  %s", stats.to_summary())
    if stats.errors:
        LOG.warning("first 10 errors:")
        for line in stats.errors[:10]:
            LOG.warning("  %s", line)
    return 0 if not stats.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
