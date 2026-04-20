#!/usr/bin/env python3
"""Aggregate H3D_v1 per-shard manifests into ``manifests/all.jsonl``.

Usage::

    python -m scripts.cleaning.h3d_v1.build_h3d_v1_index \
        --dataset-root data/H3D_v1 \
        [--validate]

Walks ``manifests/<edit_type>/*.jsonl`` for every known type, then
writes the concatenation atomically to ``manifests/all.jsonl``.

With ``--validate``, additionally checks for each record that:

* the per-edit ``meta.json`` exists,
* the four ``before.npz`` / ``after.npz`` / ``before.png`` /
  ``after.png`` paths exist on disk,
* every ``*.npz`` is loadable via ``numpy.load`` and has the expected
  keys (``slat_feats``, ``slat_coords``, ``ss``).

Validation is read-only and does not modify the manifest. Exit code is
``0`` only if every record passes (or ``--validate`` is off).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from partcraft.cleaning.h3d_v1.layout import (
    EDIT_TYPES_ALL,
    H3DLayout,
)
from partcraft.cleaning.h3d_v1.manifest import read_jsonl, rewrite_jsonl

LOG = logging.getLogger("h3d_v1.build_index")
EXPECTED_NPZ_KEYS = {"slat_feats", "slat_coords", "ss"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--validate", action="store_true",
                    help="Walk every record and verify files exist + npz keys.")
    ap.add_argument("--log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return ap.parse_args()


def _iter_records(layout: H3DLayout) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Concatenate per-shard jsonls in (type, shard) order."""
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for edit_type in EDIT_TYPES_ALL:
        type_dir = layout.manifest_dir(edit_type)
        if not type_dir.is_dir():
            continue
        for shard_jsonl in sorted(type_dir.glob("*.jsonl")):
            shard = shard_jsonl.stem
            n_before = len(records)
            for rec in read_jsonl(shard_jsonl):
                records.append(rec)
                counts[edit_type] += 1
            LOG.info("loaded %d records from %s/%s.jsonl",
                     len(records) - n_before, edit_type, shard)
    return records, dict(counts)


def _validate_record(layout: H3DLayout, rec: dict[str, Any]) -> list[str]:
    """Return a list of problems for one record (empty if clean)."""
    problems: list[str] = []
    et = rec.get("edit_type")
    shard = rec.get("shard")
    obj = rec.get("obj_id")
    eid = rec.get("edit_id")
    if not all([et, shard, obj, eid]):
        return [f"missing required fields in {rec!r}"]

    edit_dir = layout.edit_dir(et, shard, obj, eid)
    if not edit_dir.is_dir():
        return [f"edit_dir missing: {edit_dir}"]

    expected_files = [
        layout.meta_json(et, shard, obj, eid),
        layout.before_npz(et, shard, obj, eid),
        layout.after_npz(et, shard, obj, eid),
        layout.before_image(et, shard, obj, eid),
        layout.after_image(et, shard, obj, eid),
    ]
    for fp in expected_files:
        if not fp.is_file():
            problems.append(f"missing file: {fp}")

    for npz_path in (layout.before_npz(et, shard, obj, eid),
                      layout.after_npz(et, shard, obj, eid)):
        if not npz_path.is_file():
            continue
        try:
            with np.load(npz_path) as data:
                missing_keys = EXPECTED_NPZ_KEYS - set(data.files)
                if missing_keys:
                    problems.append(f"{npz_path}: missing npz keys {sorted(missing_keys)}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{npz_path}: load failed ({exc})")
    return problems


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    layout = H3DLayout(root=args.dataset_root)

    LOG.info("aggregating manifests under %s", args.dataset_root)
    records, counts = _iter_records(layout)
    LOG.info("total records: %d (%s)", len(records),
              ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    out = layout.aggregated_manifest()
    n_written = rewrite_jsonl(out, records)
    LOG.info("wrote %d records to %s", n_written, out)

    if args.validate:
        LOG.info("validating %d records...", len(records))
        bad = 0
        for i, rec in enumerate(records, 1):
            problems = _validate_record(layout, rec)
            if problems:
                bad += 1
                for p in problems:
                    LOG.warning("record %s/%s/%s/%s: %s",
                                rec.get("edit_type"), rec.get("shard"),
                                rec.get("obj_id"), rec.get("edit_id"), p)
            if i % 1000 == 0:
                LOG.info("validated %d/%d (%d bad so far)", i, len(records), bad)
        LOG.info("validation: %d/%d records ok, %d bad", len(records) - bad, len(records), bad)
        return 0 if bad == 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
