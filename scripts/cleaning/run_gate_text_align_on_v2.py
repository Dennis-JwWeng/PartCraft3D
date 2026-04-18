#!/usr/bin/env python3
"""Run pipeline_v3 ``gate_text_align`` (Gate A) over a pipeline_v2 run directory.

Constructs ObjectContext instances pointing at v2's
``<run>/objects/<NN>/<obj_id>/`` and invokes
``partcraft.pipeline_v3.vlm_core.run_gate_text_align`` directly.  Verdicts are
written back to v2's ``edit_status.json`` under ``gates.A`` — **overwriting
v2's stale Gate-A results** (which are known-bad and the whole point of this
script).  v2 module code is **not** modified.

Notes:
  * v2 and v3 share the on-disk object layout (``objects/<NN>/<obj_id>/``,
    ``phase1/parsed.json``, ``phase1/overview.png``, ``edit_status.json``),
    so a vanilla ``PipelineRoot(root=v2_run)`` is a drop-in adapter.
  * ``run_gate_text_align`` is async — wrapped with ``asyncio.run`` here.
  * The gate writes its step under ``status.json`` as ``sq1_qc_A`` (v3 step
    name).  v2's edit_status step names are unaffected; only the
    per-edit ``gates.A`` field is overwritten.
  * ``--force`` is passed through to bypass any ``sq1_qc_A`` skip on rerun.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from partcraft.pipeline_v3.paths import PipelineRoot, ObjectContext, normalize_shard
from partcraft.pipeline_v3.vlm_core import run_gate_text_align

LOG = logging.getLogger("run_gate_text_align_on_v2")


def _build_ctx(
    pr: PipelineRoot, shard: str, obj_id: str, *, image_npz_root: Path,
) -> ObjectContext:
    s = normalize_shard(shard)
    image_npz = image_npz_root / s / f"{obj_id}.npz"
    return pr.context(s, obj_id, image_npz=image_npz)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-run", type=Path, required=True,
                    help="path like outputs/partverse/pipeline_v2_shard05")
    ap.add_argument("--v3-config", type=Path, required=True,
                    help="a v3 config YAML (used for services.vlm)")
    ap.add_argument("--shards", nargs="+", default=None)
    ap.add_argument("--obj-ids", type=Path, default=None,
                    help="optional file with one obj_id per line")
    ap.add_argument("--image-npz-root", type=Path,
                    default=Path("data/partverse/images"),
                    help="dir holding <shard>/<obj_id>.npz (kept for parity with run_gate_quality_on_v2)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if sq1_qc_A is already marked done")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.v3_config.read_text())

    pr = PipelineRoot(root=args.v2_run)
    if not pr.objects_root.is_dir():
        LOG.error("not a v2 run dir (no objects/ under %s)", args.v2_run)
        return 2

    obj_filter: set[str] | None = None
    if args.obj_ids:
        obj_filter = {ln.strip() for ln in args.obj_ids.read_text().splitlines()
                      if ln.strip() and not ln.startswith("#")}

    ctxs: list[ObjectContext] = []
    for shard_dir in sorted(pr.objects_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        if args.shards and shard_dir.name not in args.shards:
            continue
        for obj_dir in sorted(shard_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            if obj_filter and obj_dir.name not in obj_filter:
                continue
            ctxs.append(_build_ctx(pr, shard_dir.name, obj_dir.name,
                                    image_npz_root=args.image_npz_root))
    if args.limit > 0:
        ctxs = ctxs[: args.limit]
    LOG.info("running gate_text_align on %d objects from %s", len(ctxs), args.v2_run)

    services = cfg.get("services") or {}
    vlm = services.get("vlm") or {}
    vlm_urls = list(vlm.get("urls") or [])
    vlm_model = vlm.get("model", "")
    if not vlm_urls:
        LOG.error("no services.vlm.urls in %s", args.v3_config)
        return 2

    asyncio.run(run_gate_text_align(
        ctxs, vlm_urls=vlm_urls, vlm_model=vlm_model,
        force=args.force, concurrency=args.concurrency,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
