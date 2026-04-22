#!/usr/bin/env python3
"""Deduplicate per-shard-per-type manifest jsonls in a local H3D_v1 tree.

Usage::

    python -m scripts.cleaning.h3d_v1.dedup_manifests \
        --dataset-root data/H3D_v1 --dry-run
    python -m scripts.cleaning.h3d_v1.dedup_manifests \
        --dataset-root data/H3D_v1 --rebuild-aggregate

Walks ``manifests/<edit_type>/<shard>.jsonl`` for every edit_type and
for each file keeps only **one** record per ``edit_id``, rewriting the
file atomically. A timestamped backup copy is saved as
``<path>.bak.<YYYYMMDD-HHMMSS>`` unless ``--no-backup`` is given.

Duplicate shapes observed in the current pipeline output:

* byte-identical lines (upstream fragment appended twice by a
  re-running Phase-F worker) -- either keep-policy produces the same
  record here.
* lines that differ only in ``views.best_view_index`` (the same edit
  was rendered multiple times; each re-render overwrites
  ``before.png`` / ``after.png`` and ``meta.json`` on disk AND appends
  a new manifest line, so earlier lines become stale pointers to a
  view that is no longer what is actually rendered on disk).

**Default keep-policy is ``last``** -- a full-shard cross-check on
``global/08`` confirmed that the *last* manifest line's
``best_view_index`` matches ``meta.json`` on disk for 1923/1923
duplicated edits, while the *first* line matches 0/1923. Keeping the
last line therefore makes the manifest consistent with the
actually-rendered PNGs.

On-disk edit directories are never touched. Dedup is a manifest-only
operation.

With ``--rebuild-aggregate``, runs ``build_h3d_v1_index`` afterwards
so ``manifests/all.jsonl`` reflects the cleaned per-type files.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from partcraft.cleaning.h3d_v1.layout import EDIT_TYPES_ALL, H3DLayout
from partcraft.cleaning.h3d_v1.manifest import rewrite_jsonl

LOG = logging.getLogger("h3d_v1.dedup_manifests")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--shard", action="append", default=None,
                    help="Limit to these shards (repeatable). "
                         "Default: all shards found on disk.")
    ap.add_argument("--edit-type", action="append", default=None,
                    choices=list(EDIT_TYPES_ALL),
                    help="Limit to these edit types (repeatable). "
                         "Default: all.")
    ap.add_argument("--keep", choices=("first", "last"), default="last",
                    help="Which occurrence to keep when an edit_id "
                         "appears more than once (default: last, "
                         "matches on-disk meta.json).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report duplicates without modifying anything.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip writing <path>.bak.<ts> backup copies.")
    ap.add_argument("--rebuild-aggregate", action="store_true",
                    help="After cleaning, rerun build_h3d_v1_index to "
                         "refresh manifests/all.jsonl.")
    ap.add_argument("--log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return ap.parse_args()


def _dedup_file(
    path: Path,
    keep: str,
    dry_run: bool,
    no_backup: bool,
    stamp: str,
) -> tuple[int, int, int, int]:
    """Return (n_lines, n_unique, n_byte_identical_drops, n_differing_drops)."""
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    raw_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            raw_lines.append(ln)
            rec = json.loads(ln)
            buckets[rec["edit_id"]].append((ln, rec))

    n_lines = len(raw_lines)
    n_unique = len(buckets)

    kept: list[dict[str, Any]] = []
    identical_drops = 0
    differing_drops = 0
    for items in buckets.values():
        pick_idx = -1 if keep == "last" else 0
        kept.append(items[pick_idx][1])
        if len(items) == 1:
            continue
        extras = len(items) - 1
        all_same = all(it[0] == items[0][0] for it in items[1:])
        if all_same:
            identical_drops += extras
        else:
            differing_drops += extras

    if dry_run or n_lines == n_unique:
        return n_lines, n_unique, identical_drops, differing_drops

    if not no_backup:
        backup = path.with_name(f"{path.name}.bak.{stamp}")
        shutil.copy2(path, backup)
        LOG.debug("backup -> %s", backup)

    rewrite_jsonl(path, kept)
    return n_lines, n_unique, identical_drops, differing_drops


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    layout = H3DLayout(root=args.dataset_root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    edit_types = tuple(args.edit_type) if args.edit_type else EDIT_TYPES_ALL

    if args.shard:
        shards = sorted(set(args.shard))
    else:
        shards_seen: set[str] = set()
        for et in edit_types:
            d = layout.manifest_dir(et)
            if not d.is_dir():
                continue
            for p in d.glob("*.jsonl"):
                shards_seen.add(p.stem)
        shards = sorted(shards_seen)

    if not shards:
        LOG.warning("no per-type jsonls found under %s",
                    args.dataset_root / "manifests")
        return 0

    LOG.info("mode: %s  keep=%s%s",
             "DRY-RUN" if args.dry_run else "REWRITE",
             args.keep,
             "" if args.no_backup or args.dry_run
             else f"  (backup tag .{stamp})")
    LOG.info("edit_types: %s", ", ".join(edit_types))
    LOG.info("shards:     %s", ", ".join(shards))

    totals = Counter()
    per_shard: dict[str, Counter] = defaultdict(Counter)
    touched_files = 0
    for et in edit_types:
        for shard in shards:
            p = layout.manifest_path(et, shard)
            if not p.is_file():
                continue
            n, uniq, ident, diff = _dedup_file(
                p, args.keep, args.dry_run, args.no_backup, stamp,
            )
            extras = n - uniq
            totals["lines"] += n
            totals["unique"] += uniq
            totals["extras_identical"] += ident
            totals["extras_differing"] += diff
            per_shard[shard]["lines"] += n
            per_shard[shard]["unique"] += uniq
            per_shard[shard]["extras"] += extras
            if extras:
                touched_files += 0 if args.dry_run else 1
                LOG.info(
                    "  %s/%s.jsonl: %d -> %d "
                    "(drop %d = %d identical + %d views-only)",
                    et, shard, n, uniq, extras, ident, diff,
                )

    LOG.info("")
    LOG.info("per-shard summary:")
    for shard in shards:
        c = per_shard[shard]
        LOG.info("  %s: %d lines -> %d unique (drop %d)",
                 shard, c["lines"], c["unique"], c["extras"])
    LOG.info("")
    LOG.info(
        "TOTAL: %d lines -> %d unique "
        "(drop %d: %d identical + %d views-only)",
        totals["lines"], totals["unique"],
        totals["extras_identical"] + totals["extras_differing"],
        totals["extras_identical"], totals["extras_differing"],
    )
    if args.dry_run:
        LOG.info("dry-run: no files modified.")
        return 0
    LOG.info("rewrote %d per-type jsonl file(s).", touched_files)

    if args.rebuild_aggregate:
        LOG.info("rebuilding manifests/all.jsonl via build_h3d_v1_index")
        cmd = [
            sys.executable, "-m",
            "scripts.cleaning.h3d_v1.build_h3d_v1_index",
            "--dataset-root", str(args.dataset_root),
        ]
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            LOG.error("build_h3d_v1_index failed (rc=%d)", r.returncode)
            return r.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
