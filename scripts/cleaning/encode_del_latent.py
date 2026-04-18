#!/usr/bin/env python3
"""Encode deletion edits' ``after_new.glb`` into ``after.npz`` (SS + SLAT + DINOv2).

Drives ``scripts/tools/migrate_slat_to_npz._render_and_full_encode`` for each
entry in ``data/partverse_edit_v1/_pending/del_latent.txt``. The Phase-5 helper
returns a ``dict[str, np.ndarray]`` (SLAT feats/coords + SS latent +
``dino_voxel_mean``); we ``np.savez`` it into the per-edit ``after.npz``.

Multi-GPU is via subprocess fan-out: ``--num-gpus N`` spawns N children, each
with ``CUDA_VISIBLE_DEVICES`` set, processing a round-robin slice of the
pending list.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from partcraft.cleaning.v1.layout import V1Layout
from partcraft.cleaning.v1.pending import DelLatentPending, PendingEntry

LOG = logging.getLogger("encode_del_latent")


def _now_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _process_one(
    entry: PendingEntry,
    *,
    layout: V1Layout,
    ss_encoder,
    device: str,
    work_dir: Path,
    dino_views: int,
    blender_path: str | None,
) -> bool:
    marker_p = layout.after_pending_marker(
        entry.shard, entry.obj_id, entry.edit_id, suffix=entry.suffix,
    )
    if not marker_p.is_file():
        LOG.warning("no marker for %s; skipping", entry.edit_id)
        return False
    marker = json.loads(marker_p.read_text())
    glb = Path(marker["after_glb"]) if marker.get("after_glb") else None
    if glb is None or not glb.is_file():
        LOG.warning("missing after_new.glb for %s (%s)", entry.edit_id, glb)
        return False

    out_npz = layout.after_npz_path(
        entry.shard, entry.obj_id, entry.edit_id, suffix=entry.suffix,
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from scripts.tools.migrate_slat_to_npz import _render_and_full_encode

    name = f"{entry.shard}_{entry.obj_id}_{entry.edit_id}{entry.suffix}"
    result = _render_and_full_encode(
        glb, name, work_dir, ss_encoder, device,
        num_views=dino_views, blender_path=blender_path,
    )
    np.savez(out_npz, **result)

    if not out_npz.is_file():
        LOG.error("encode produced no output for %s", entry.edit_id)
        return False

    qc_p = layout.qc_json(
        entry.shard, entry.obj_id, entry.edit_id, suffix=entry.suffix,
    )
    qc = json.loads(qc_p.read_text())
    qc.setdefault("passes", {})["del_latent_encode"] = {
        "pass": True,
        "score": None,
        "producer": "encode_del_latent.py@1.0.0",
        "reason": "",
        "ts": _now_z(),
    }
    qc_p.write_text(json.dumps(qc, indent=2))
    marker_p.unlink(missing_ok=True)
    return True


def _run_single(
    entries: list[PendingEntry],
    *,
    layout: V1Layout,
    pending: DelLatentPending,
    ss_encoder,
    device: str,
    work_dir: Path,
    dino_views: int,
    blender_path: str | None,
) -> tuple[int, int]:
    ok = fail = 0
    for e in entries:
        try:
            if _process_one(
                e, layout=layout, ss_encoder=ss_encoder, device=device,
                work_dir=work_dir, dino_views=dino_views, blender_path=blender_path,
            ):
                pending.remove(e)
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            LOG.exception("encode failed for %s: %s", e.edit_id, exc)
            fail += 1
    return ok, fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", type=Path,
                    default=Path("configs/cleaning/promote_v1.yaml"))
    ap.add_argument("--v1-root", type=Path, default=None)
    ap.add_argument("--ckpt-root", type=str, default=None,
                    help="Trellis SS encoder checkpoint root (required unless dry-run).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--gpu-shard", type=str, default="",
                    help="internal: e.g. '1/4' to process slice 1 of 4")
    ap.add_argument("--dino-views", type=int, default=40)
    ap.add_argument("--dino-work-dir", type=Path, default=None,
                    help="Scratch dir for Blender renders (default: <v1_root>/_render_tmp).")
    ap.add_argument("--blender-path", type=str, default=None,
                    help="Override BLENDER_PATH env var.")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    rules = yaml.safe_load(args.rules.read_text())
    v1_root = args.v1_root or Path(rules["v1_root"])
    layout = V1Layout(root=v1_root)
    pending = DelLatentPending(layout.pending_del_latent_file())
    entries = list(pending.iter_entries())
    if not entries:
        LOG.info("nothing pending")
        return 0

    if args.gpu_shard:
        i, n = (int(x) for x in args.gpu_shard.split("/"))
        entries = [e for j, e in enumerate(entries) if j % n == i]
        LOG.info("shard %d/%d: %d entries", i, n, len(entries))

    if args.num_gpus <= 1 or args.gpu_shard:
        if not args.ckpt_root:
            LOG.error("--ckpt-root is required for actual encoding")
            return 2
        from scripts.tools.migrate_slat_to_npz import _load_ss_encoder
        ss_encoder = _load_ss_encoder(Path(args.ckpt_root), args.device)
        work_dir = args.dino_work_dir or (v1_root / "_render_tmp")
        work_dir.mkdir(parents=True, exist_ok=True)
        blender = args.blender_path or os.environ.get("BLENDER_PATH")
        ok, fail = _run_single(
            entries, layout=layout, pending=pending,
            ss_encoder=ss_encoder, device=args.device,
            work_dir=work_dir, dino_views=args.dino_views, blender_path=blender,
        )
        LOG.info("ok=%d fail=%d", ok, fail)
        return 0 if fail == 0 else 1

    children: list[subprocess.Popen] = []
    for i in range(args.num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [
            sys.executable, "-m", "scripts.cleaning.encode_del_latent",
            "--rules", str(args.rules),
            "--v1-root", str(v1_root),
            "--device", args.device,
            "--gpu-shard", f"{i}/{args.num_gpus}",
            "--dino-views", str(args.dino_views),
        ]
        if args.ckpt_root:
            cmd += ["--ckpt-root", args.ckpt_root]
        if args.dino_work_dir:
            cmd += ["--dino-work-dir", str(args.dino_work_dir)]
        if args.blender_path:
            cmd += ["--blender-path", args.blender_path]
        LOG.info("spawning GPU %d: %s", i, " ".join(cmd))
        children.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for ch in children:
        rc = max(rc, ch.wait())
    return rc


if __name__ == "__main__":
    sys.exit(main())
