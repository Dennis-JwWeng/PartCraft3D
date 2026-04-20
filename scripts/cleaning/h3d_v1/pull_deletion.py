#!/usr/bin/env python3
"""Promote deletion edits from a pipeline_v3 shard into ``H3D_v1``.

Usage::

    python -m scripts.cleaning.h3d_v1.pull_deletion \
        --pipeline-cfg configs/pipeline_v3_shard08.yaml \
        --shard 08 \
        --dataset-root data/H3D_v1 \
        --gpu-ids 0,1,2,3,4,5,6,7 \
        --blender /usr/local/bin/blender \
        --workers 8

End-to-end per shard:

1. Walk ``<output_dir>/objects/<NN>/<obj_id>/edits_3d/del_*/``.
2. Filter via ``filter.accept_deletion`` (gate_a status == "pass").
3. **Encode** any accepted edit whose ``after.npz`` is missing — this
   is pipeline_v3's s6b step (Blender → DINOv2 → SLAT + SS encoder),
   inlined here because pipeline_v3 currently does not run s6b
   automatically. Encode jobs are **partitioned across ``--gpu-ids``**
   (default ``0..7``). On each GPU we run **two phases**: (a) Blender
   multi-view render + voxelize for **all** edits assigned to that GPU
   (no torch yet), then (b) load ``ss_encoder`` once and run DINO/SLAT/SS
   for those staged renders. This keeps Cycles and PyTorch from
   alternating every edit on the same device (spawn workers; CUDA-safe).
4. ``promote_deletion`` for each (now-ready) edit; append manifest.

``--skip-encode`` skips step 3 — useful when running with ``--dry-run``
or on a CPU-only machine to count what's pending.

Legacy: ``--device cuda:N`` forces a **single** GPU ``N`` (same as
``--gpu-ids N``) and prints a deprecation warning.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from partcraft.cleaning.h3d_v1.filter import accept_deletion
from partcraft.cleaning.h3d_v1.layout import H3DLayout
from partcraft.cleaning.h3d_v1.pipeline_io import PipelineEdit, load_edit_status
from partcraft.cleaning.h3d_v1.promoter import promote_deletion
from scripts.cleaning.h3d_v1._common import (
    add_common_args,
    build_promote_context,
    collect_edits,
    print_summary,
    resolve_obj_filter,
    run_promote_pool,
    setup_logging,
)

LOG = logging.getLogger("h3d_v1.pull_deletion")


def _add_encode_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--device", default=None,
                    help='Deprecated: use --gpu-ids. If set (e.g. "cuda:3"), '
                         'forces a single GPU index.')
    ap.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7",
                    help="Comma-separated physical GPU indices for parallel "
                         "s6b encode (Blender render + SLAT encode).")
    ap.add_argument("--blender",
                    default=os.environ.get("BLENDER_PATH", "blender"),
                    help='Blender 3.5 binary path (defaults to $BLENDER_PATH or "blender").')
    ap.add_argument("--ckpt-root", type=Path,
                    default=Path(os.environ.get("PARTCRAFT_CKPT_ROOT",
                                                Path.cwd() / "checkpoints")),
                    help="Trellis checkpoints root for ss_encoder.")
    ap.add_argument("--num-views", type=int, default=40,
                    help="DINOv2 multi-view count for s6b (default 40, matches s6b).")
    ap.add_argument("--encode-work-dir", type=Path,
                    default=Path("outputs/h3d_v1_encode"),
                    help="Scratch dir for Blender renders during encode.")
    ap.add_argument("--skip-encode", action="store_true",
                    help="Don't run s6b for missing after.npz; skip those edits instead.")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    _add_encode_args(ap)
    return ap.parse_args()


def _parse_gpu_ids(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--gpu-ids must list at least one integer")
    return [int(p) for p in parts]


def _gpu_index_from_device(device: str) -> list[int]:
    """Map cuda:N -> [N]; cuda -> [0]."""
    if device == "cuda" or device == "cuda:":
        return [0]
    if device.startswith("cuda:"):
        idx = device.split(":", 1)[1].strip()
        return [int(idx)]
    raise ValueError(f'expected device like "cuda:0", got {device!r}')


def _normalize_device_env(device: str) -> str:
    """Map ``cuda:N`` → ``cuda`` after pinning ``CUDA_VISIBLE_DEVICES=N``.

    The downstream loaders (``trellis``,
    ``encode_asset.encode_into_SLAT._get_slat_encoder``) check only for
    ``device == "cuda"``; passing ``cuda:0`` causes
    ``partcraft.io.npz_utils.load_ss_encoder`` to silently fall through
    to CPU and the encode step then fails with a torch type mismatch
    on the first conv. Pin the visible device and call everything
    below with the bare ``"cuda"`` token.
    """
    if device == "cuda" or not device.startswith("cuda:"):
        return device
    idx = device.split(":", 1)[1]
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        LOG.info("pinned CUDA_VISIBLE_DEVICES=%s (from --device %s)", idx, device)
    elif existing != idx:
        LOG.warning("--device %s but CUDA_VISIBLE_DEVICES=%s already set; "
                    "trusting existing env", device, existing)
    return "cuda"


def _resolve_gpu_ids(args: argparse.Namespace) -> list[int]:
    if args.device:
        LOG.warning("--device is deprecated for pull_deletion; use --gpu-ids")
        _normalize_device_env(args.device)
        return _gpu_index_from_device(args.device)
    return _parse_gpu_ids(args.gpu_ids)


def _partition_round_robin(items: list[PipelineEdit], n: int) -> list[list[PipelineEdit]]:
    if n <= 0:
        raise ValueError("partition count must be positive")
    buckets: list[list[PipelineEdit]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def _maybe_load_encoder(ckpt_root: Path, device: str):
    """Lazy-load Trellis ss_encoder; deferred so CPU-only paths skip torch."""
    from scripts.tools.migrate_slat_to_npz import _load_ss_encoder  # noqa: PLC0415
    LOG.info("loading ss_encoder from ckpt_root=%s on %s", ckpt_root, device)
    return _load_ss_encoder(ckpt_root, device)


def _encode_after_npz(edit: PipelineEdit, *, ss_encoder, device: str,
                      work_dir: Path, num_views: int, blender_path: str) -> bool:
    """Run s6b inline for one deletion edit. Returns True on success.

    Always **Blender render first**, then DINO/SLAT/SS (``_encode_from_render_dir``)
    so Cycles and torch are not interleaved at the wrong granularity.
    """
    import numpy as np  # noqa: PLC0415
    from scripts.tools.migrate_slat_to_npz import (  # noqa: PLC0415
        _encode_from_render_dir,
        _render_ply_views,
    )

    glb = edit.edit_dir / "after_new.glb"
    if not glb.is_file():
        LOG.warning("[%s] no after_new.glb; cannot encode", edit.edit_id)
        return False
    out_npz = edit.edit_dir / "after.npz"
    work_dir.mkdir(parents=True, exist_ok=True)
    name = f"{edit.shard}_{edit.obj_id}_{edit.edit_id}"
    try:
        render_out = _render_ply_views(
            glb, name, work_dir, num_views, blender_path,
        )
        payload = _encode_from_render_dir(
            render_out, ss_encoder, device, name, num_views,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[%s] encode failed: %s", edit.edit_id, exc)
        return False
    np.savez(out_npz, **payload)
    return True


def _encode_worker_entrypoint(
    gpu_id: int,
    edits: list[PipelineEdit],
    ckpt_root: Path,
    work_root: Path,
    num_views: int,
    blender_path: str,
) -> tuple[int, int]:
    """Child process: pin one GPU; **two phases** per partition.

    1) **Blender-only**: run ``_render_ply_views`` for every edit in this GPU's
       bucket (no ``ss_encoder`` / torch init yet) — avoids Cycles vs torch
       ping-pong on the same device.
    2) **Encode-only**: load ``ss_encoder`` once, then ``_encode_from_render_dir``
       + ``np.savez`` for each staged edit.

    Uses ``spawn`` from the parent Pool (CUDA-safe).

    Returns (success_count, failure_count).
    """
    import logging as _logging

    import numpy as np  # noqa: PLC0415

    log = _logging.getLogger(f"h3d_v1.pull_deletion.gpu{gpu_id}")
    if not edits:
        return (0, 0)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    work_dir = Path(work_root) / f"gpu{gpu_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    from scripts.tools.migrate_slat_to_npz import (  # noqa: PLC0415
        _encode_from_render_dir,
        _render_ply_views,
    )

    # ── Phase 1: Blender + voxelize for all edits on this GPU ─────────────
    staged: list[tuple[PipelineEdit, Path, str]] = []  # edit, render_out, name
    render_fail = 0
    t_r0 = time.time()
    for j, edit in enumerate(edits, 1):
        glb = edit.edit_dir / "after_new.glb"
        if not glb.is_file():
            log.warning("[gpu%s R %d/%d] %s skip: no after_new.glb", gpu_id, j, len(edits), edit.edit_id)
            render_fail += 1
            continue
        name = f"{edit.shard}_{edit.obj_id}_{edit.edit_id}"
        t0 = time.time()
        try:
            render_out = _render_ply_views(
                glb, name, work_dir, num_views, blender_path,
            )
            staged.append((edit, render_out, name))
            log.info(
                "[gpu%s R %d/%d] %s render ok (%.1fs)",
                gpu_id, j, len(edits), edit.edit_id, time.time() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            render_fail += 1
            log.warning(
                "[gpu%s R %d/%d] %s render FAILED: %s",
                gpu_id, j, len(edits), edit.edit_id, exc,
            )
    log.info(
        "gpu %s phase1 (Blender) done: staged=%d render_fail=%d in %.1fs",
        gpu_id, len(staged), render_fail, time.time() - t_r0,
    )

    if not staged:
        return (0, render_fail)

    # ── Phase 2: DINO / SLAT / SS for all staged edits ─────────────────────
    device = "cuda"
    t_load = time.time()
    ss_encoder = _maybe_load_encoder(Path(ckpt_root), device)
    log.info(
        "gpu %s: ss_encoder ready in %.1fs; encoding %d staged edits",
        gpu_id, time.time() - t_load, len(staged),
    )

    ok_c = 0
    enc_fail = 0
    for j, (edit, render_out, name) in enumerate(staged, 1):
        out_npz = edit.edit_dir / "after.npz"
        t0 = time.time()
        try:
            payload = _encode_from_render_dir(
                render_out, ss_encoder, device, name, num_views,
            )
            np.savez(out_npz, **payload)
            ok_c += 1
            log.info(
                "[gpu%s E %d/%d] %s encode ok (%.1fs)",
                gpu_id, j, len(staged), edit.edit_id, time.time() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            enc_fail += 1
            log.warning(
                "[gpu%s E %d/%d] %s encode FAILED: %s",
                gpu_id, j, len(staged), edit.edit_id, exc,
            )

    log.info(
        "gpu %s done: encode_ok=%d encode_fail=%d render_fail=%d",
        gpu_id, ok_c, enc_fail, render_fail,
    )
    return (ok_c, render_fail + enc_fail)


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    layout = H3DLayout(root=args.dataset_root)
    obj_filter = resolve_obj_filter(args)

    LOG.info("scanning shard=%s for deletion edits", args.shard)
    edits, stats, paths = collect_edits(
        args.pipeline_cfg, args.shard, ("deletion",), obj_filter,
        accept_fn=accept_deletion, load_status=load_edit_status,
    )
    if args.limit:
        edits = edits[: args.limit]
    LOG.info("after filter+limit: %d deletion edits to consider", len(edits))

    if args.dry_run:
        n_missing = sum(1 for e in edits if not (e.edit_dir / "after.npz").is_file())
        gpu_ids = _resolve_gpu_ids(args)
        LOG.info("dry-run: %d edits already have after.npz, %d need s6b; "
                 "would use %d encode workers on GPU ids %s",
                 len(edits) - n_missing, n_missing, len(gpu_ids), gpu_ids)
        print_summary(stats, "pull_deletion (dry-run)")
        return 0

    needs_encode = [e for e in edits if not (e.edit_dir / "after.npz").is_file()]
    ready = [e for e in edits if (e.edit_dir / "after.npz").is_file()]
    LOG.info("ready=%d needs_encode=%d", len(ready), len(needs_encode))

    if needs_encode and not args.skip_encode:
        gpu_ids = _resolve_gpu_ids(args)
        if not gpu_ids:
            LOG.error("no GPU ids resolved; pass --gpu-ids or --device")
            return 2
        n_workers = len(gpu_ids)
        buckets = _partition_round_robin(needs_encode, n_workers)
        pairs = [(gpu_ids[i], buckets[i]) for i in range(n_workers) if buckets[i]]
        LOG.info(
            "encode: %d edits across %d parallel workers (GPU ids %s, %d non-empty partitions)",
            len(needs_encode), n_workers, gpu_ids, len(pairs),
        )
        t0 = time.time()
        mp_ctx = mp.get_context("spawn")
        tasks = [
            (gid, bucket, args.ckpt_root, args.encode_work_dir,
             args.num_views, args.blender)
            for gid, bucket in pairs
        ]
        with mp_ctx.Pool(processes=len(pairs)) as pool:
            results = pool.starmap(_encode_worker_entrypoint, tasks)
        total_ok = sum(r[0] for r in results)
        total_fail = sum(r[1] for r in results)
        for _ in range(total_fail):
            stats.add_skip("encode_failed")
        ready = [e for e in edits if (e.edit_dir / "after.npz").is_file()]
        LOG.info(
            "encode phase: ok=%d fail=%d across %d workers in %.1fs; ready for promote=%d",
            total_ok, total_fail, len(pairs), time.time() - t0, len(ready),
        )
    elif needs_encode and args.skip_encode:
        LOG.warning("--skip-encode set; skipping %d edits with missing after.npz", len(needs_encode))
        for e in needs_encode:
            stats.add_skip("missing_after_npz_skip_encode")

    ctx = build_promote_context(paths, pipeline_cfg_path=args.pipeline_cfg)
    run_promote_pool(
        ready, layout, ctx,
        promote_fn=lambda e, l: promote_deletion(e, l, ctx=ctx),
        workers=args.workers, stats=stats,
    )
    print_summary(stats, "pull_deletion")
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
