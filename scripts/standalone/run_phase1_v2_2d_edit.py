#!/usr/bin/env python3
"""Run FLUX 2D edits on a phase1_v2 output dir, multi-GPU.

Reads:
    <in-dir>/edit_specs.jsonl   (created by parsed_to_edit_specs.py)
Writes:
    <in-dir>/2d_edits/{edit_id}_input.png
    <in-dir>/2d_edits/{edit_id}_edited.png
    <in-dir>/2d_edits/manifest.jsonl

Multi-server: pass --edit-urls comma-separated, one per FLUX server.
Round-robin assignment via ThreadPoolExecutor (same as run_pipeline step3).

Usage:
    python scripts/standalone/run_phase1_v2_2d_edit.py \
        --in-dir outputs/_debug/phase1_v2_mirror5 \
        --shard 01 \
        --edit-urls http://localhost:8004,http://localhost:8005,http://localhost:8006,http://localhost:8007
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.phase1_planning.planner import EditSpec  # noqa: E402
from scripts.run_2d_edit import (  # noqa: E402
    process_one, check_edit_server,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--mesh-root", default="data/partverse/mesh", type=Path)
    ap.add_argument("--images-root", default="data/partverse/images", type=Path)
    ap.add_argument("--specs", default=None, type=Path)
    ap.add_argument("--edit-urls", required=True,
                    help="comma-separated FLUX server urls")
    ap.add_argument("--workers-per-server", type=int, default=2)
    ap.add_argument("--types", default="modification,scale,material,global")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logger = logging.getLogger("phase1v2_2d")

    urls = [u.strip() for u in args.edit_urls.split(",") if u.strip()]
    live = [u for u in urls if check_edit_server(u)]
    if not live:
        raise SystemExit(f"no live FLUX servers in {urls}")
    logger.info(f"FLUX servers ({len(live)}): {live}")
    workers = max(len(live), len(live) * args.workers_per_server)

    types = set(args.types.split(","))
    specs_path = args.specs or (args.in_dir / "edit_specs.jsonl")
    specs = []
    with open(specs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if spec.edit_type in types:
                specs.append(spec)
    logger.info(f"loaded {len(specs)} specs from {specs_path}")

    # Build a tiny dataset shim (HY3DPartDataset wants the standard partverse layout)
    from partcraft.io.hy3d_loader import HY3DPartDataset
    dataset = HY3DPartDataset(
        str(args.images_root), str(args.mesh_root), [args.shard])

    out_dir = args.in_dir / "2d_edits"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Resume
    done_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except Exception:
                    pass
    pending = [s for s in specs if s.edit_id not in done_ids]
    logger.info(f"pending={len(pending)} done={len(done_ids)} workers={workers}")
    if not pending:
        return

    success = fail = 0
    with open(manifest_path, "a") as fp, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for spec in pending:
            url = live[len(futures) % len(live)]
            fut = pool.submit(process_one, spec, dataset, None, out_dir,
                              "flux", logger, edit_server_url=url)
            futures[fut] = spec
        for i, fut in enumerate(as_completed(futures)):
            spec = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {"edit_id": spec.edit_id, "status": "failed",
                          "reason": str(e)}
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            fp.flush()
            if result.get("status") == "success":
                success += 1
            else:
                fail += 1
            if (i + 1) % 5 == 0:
                logger.info(f"  {i+1}/{len(pending)} ok={success} fail={fail}")
    logger.info(f"done: {success} ok, {fail} fail → {out_dir}")


if __name__ == "__main__":
    main()
