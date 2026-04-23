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
   is pipeline_v3's s6b step (Blender → DINOv2 → SLAT + SS encoder).
   Encode is split into two pools:

   a. **render pool** (``--phase render`` or ``both``): Blender Cycles
      multi-view render + voxelize for all pending edits, writing render
      artifacts to ``<encode-work-dir>/<name>/`` and a ``render.done``
      marker.  No torch/trellis is loaded, so this can co-exist with
      other GPU workloads.

   b. **encode pool** (``--phase encode`` or ``both``): load
      ``ss_encoder`` once per GPU, then DINOv2 → SLAT → SS for every
      staged edit, writing ``after.npz`` to the pipeline edit dir.
      Render artifacts under ``<encode-work-dir>/<name>/`` are always
      left in place after a successful encode (delete manually if you
      need disk).

   Both pools use ``fork`` (CUDA-safe because the parent never inits
   CUDA), which avoids the numpy-in-spawn race seen on earlier runs.

4. ``promote_deletion`` for each ready edit; append to manifest.

Phase flags:

- ``--phase both`` (default): run render, then encode, then promote.
- ``--phase render``: run render pool only; exit without encoding or
  promoting.  Useful to pre-stage render artifacts while GPU compute is
  busy.
- ``--phase encode``: skip render; only encode already-staged edits
  (those with ``render.done``), then promote.

``--skip-encode`` skips step 3 entirely (CPU-only accounting).
``--skip-encode`` and ``--phase`` are mutually exclusive.

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

# Cap per-worker BLAS thread counts so 8 workers * N_blas_threads stays
# well under system limits (previous runs hit OpenBLAS pthread_create
# failures when OPENBLAS_NUM_THREADS defaulted to nproc=192).
_BLAS_THREAD_ENV: dict[str, str] = {
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}


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
                    help="Don't run s6b for missing after.npz; skip those edits instead. "
                         "Mutually exclusive with --phase.")
    ap.add_argument("--phase", choices=("both", "render", "encode"), default="both",
                    help=(
                        "Control which encode sub-phase(s) to run. "
                        "'render': Blender only, writes render.done markers, exits. "
                        "'encode': SLAT/SS encode from already-staged renders, then promote. "
                        "'both' (default): render → encode → promote."
                    ))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    _add_encode_args(ap)
    args = ap.parse_args()
    if args.skip_encode and args.phase != "both":
        ap.error("--skip-encode and --phase are mutually exclusive; "
                 "use --phase both --skip-encode or just --phase encode/render")
    return args


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


def _edit_name(edit: PipelineEdit) -> str:
    """Unique staging dir name for one edit; stable across render and encode."""
    return f"{edit.shard}_{edit.obj_id}_{edit.edit_id}"


# ── Worker: render-only (Blender + voxelize, no torch/trellis) ──────────────

def _render_worker(
    gpu_id: int,
    edits: list[PipelineEdit],
    work_root: Path,
    num_views: int,
    blender_path: str,
) -> tuple[int, int]:
    """Fork child: Blender render + voxelize for one GPU's edit bucket.

    Caps BLAS thread counts and pins ``CUDA_VISIBLE_DEVICES`` for Blender.
    Does **not** import torch or trellis.

    Returns (render_ok, render_fail).
    """
    import logging as _logging  # noqa: PLC0415

    for k, v in _BLAS_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log = _logging.getLogger(f"h3d_v1.pull_deletion.gpu{gpu_id}")
    if not edits:
        return (0, 0)

    from scripts.tools.migrate_slat_to_npz import _render_ply_views  # noqa: PLC0415

    ok_c = 0
    fail_c = 0
    t_r0 = time.time()
    for j, edit in enumerate(edits, 1):
        glb = edit.edit_dir / "after_new.glb"
        name = _edit_name(edit)
        work_dir = Path(work_root) / name
        done_marker = work_dir / "render.done"

        if not glb.is_file():
            log.warning("[gpu%s R %d/%d] %s skip: no after_new.glb",
                        gpu_id, j, len(edits), edit.edit_id)
            fail_c += 1
            continue
        if done_marker.is_file():
            log.info("[gpu%s R %d/%d] %s already staged; skip re-render",
                     gpu_id, j, len(edits), edit.edit_id)
            ok_c += 1
            continue
        t0 = time.time()
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            # _render_ply_views places output in work_root/<name>/
            _render_ply_views(glb, name, Path(work_root), num_views, blender_path)
            done_marker.touch()
            ok_c += 1
            log.info("[gpu%s R %d/%d] %s render ok (%.1fs)",
                     gpu_id, j, len(edits), edit.edit_id, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            fail_c += 1
            log.warning("[gpu%s R %d/%d] %s render FAILED: %s",
                        gpu_id, j, len(edits), edit.edit_id, exc)

    log.info("gpu %s render done: ok=%d fail=%d in %.1fs",
             gpu_id, ok_c, fail_c, time.time() - t_r0)
    return (ok_c, fail_c)


# ── Worker: encode-only (DINOv2 + SLAT + SS, requires CUDA) ─────────────────

def _encode_worker(
    gpu_id: int,
    edits: list[PipelineEdit],
    ckpt_root: Path,
    work_root: Path,
    num_views: int,
) -> tuple[int, int]:
    """Fork child: DINO/SLAT/SS encode from pre-staged render artifacts.

    Expects ``<work_root>/<name>/render.done`` for each edit.
    Loads ``ss_encoder`` once, streams the bucket, writes ``after.npz``.
    Staging dirs under ``work_root`` are left intact after success.

    Returns (encode_ok, encode_fail).
    """
    import logging as _logging  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    for k, v in _BLAS_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log = _logging.getLogger(f"h3d_v1.pull_deletion.gpu{gpu_id}")
    if not edits:
        return (0, 0)

    from scripts.tools.migrate_slat_to_npz import _encode_from_render_dir  # noqa: PLC0415

    device = "cuda"
    t_load = time.time()
    ss_encoder = _maybe_load_encoder(Path(ckpt_root), device)
    log.info("gpu %s: ss_encoder ready in %.1fs; encoding %d staged edits",
             gpu_id, time.time() - t_load, len(edits))

    ok_c = 0
    fail_c = 0
    for j, edit in enumerate(edits, 1):
        name = _edit_name(edit)
        work_dir = Path(work_root) / name
        done_marker = work_dir / "render.done"
        out_npz = edit.edit_dir / "after.npz"
        t0 = time.time()
        if not done_marker.is_file():
            log.warning("[gpu%s E %d/%d] %s skip: render.done missing",
                        gpu_id, j, len(edits), edit.edit_id)
            fail_c += 1
            continue
        try:
            payload = _encode_from_render_dir(work_dir, ss_encoder, device, name, num_views)
            np.savez(out_npz, **payload)
            ok_c += 1
            log.info("[gpu%s E %d/%d] %s encode ok (%.1fs)",
                     gpu_id, j, len(edits), edit.edit_id, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            fail_c += 1
            log.warning("[gpu%s E %d/%d] %s encode FAILED: %s",
                        gpu_id, j, len(edits), edit.edit_id, exc)

    log.info("gpu %s encode done: ok=%d fail=%d", gpu_id, ok_c, fail_c)
    return (ok_c, fail_c)


# ── Pool helpers ─────────────────────────────────────────────────────────────

def _run_render_pool(
    needs_encode: list[PipelineEdit],
    gpu_ids: list[int],
    work_dir: Path,
    num_views: int,
    blender_path: str,
    stats,
) -> int:
    """Launch fork pool for render phase. Returns render_fail count."""
    n_workers = len(gpu_ids)
    buckets = _partition_round_robin(needs_encode, n_workers)
    pairs = [(gpu_ids[i], buckets[i]) for i in range(n_workers) if buckets[i]]
    LOG.info("render: %d edits across %d workers (GPU ids %s, %d non-empty partitions)",
             len(needs_encode), n_workers, gpu_ids, len(pairs))
    t0 = time.time()
    tasks = [(gid, bucket, work_dir, num_views, blender_path) for gid, bucket in pairs]
    mp_ctx = mp.get_context("fork")
    with mp_ctx.Pool(processes=len(pairs)) as pool:
        results = pool.starmap(_render_worker, tasks)
    total_ok = sum(r[0] for r in results)
    total_fail = sum(r[1] for r in results)
    LOG.info("render phase: ok=%d fail=%d in %.1fs", total_ok, total_fail, time.time() - t0)
    for _ in range(total_fail):
        stats.add_skip("render_failed")
    return total_fail


def _run_encode_pool(
    staged: list[PipelineEdit],
    gpu_ids: list[int],
    ckpt_root: Path,
    work_dir: Path,
    num_views: int,
    stats,
) -> None:
    """Launch fork pool for encode phase. Updates stats in place."""
    n_workers = len(gpu_ids)
    buckets = _partition_round_robin(staged, n_workers)
    pairs = [(gpu_ids[i], buckets[i]) for i in range(n_workers) if buckets[i]]
    LOG.info("encode: %d edits across %d workers (GPU ids %s, %d non-empty partitions)",
             len(staged), n_workers, gpu_ids, len(pairs))
    t0 = time.time()
    tasks = [(gid, bucket, ckpt_root, work_dir, num_views)
             for gid, bucket in pairs]
    mp_ctx = mp.get_context("fork")
    with mp_ctx.Pool(processes=len(pairs)) as pool:
        results = pool.starmap(_encode_worker, tasks)
    total_ok = sum(r[0] for r in results)
    total_fail = sum(r[1] for r in results)
    LOG.info("encode phase: ok=%d fail=%d in %.1fs", total_ok, total_fail, time.time() - t0)
    for _ in range(total_fail):
        stats.add_skip("encode_failed")


# ── main ─────────────────────────────────────────────────────────────────────

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

    needs_encode = [e for e in edits if not (e.edit_dir / "after.npz").is_file()]
    ready = [e for e in edits if (e.edit_dir / "after.npz").is_file()]
    LOG.info("ready=%d needs_encode=%d", len(ready), len(needs_encode))

    if args.dry_run:
        n_staged = sum(
            1 for e in needs_encode
            if (args.encode_work_dir / _edit_name(e) / "render.done").is_file()
        )
        gpu_ids = _resolve_gpu_ids(args)
        LOG.info(
            "dry-run: phase=%s | ready=%d needs_encode=%d render_staged=%d; "
            "would use %d encode workers on GPU ids %s",
            args.phase, len(ready), len(needs_encode), n_staged, len(gpu_ids), gpu_ids,
        )
        print_summary(stats, "pull_deletion (dry-run)")
        return 0

    # ── Legacy --skip-encode bypass ──────────────────────────────────────────
    if args.skip_encode:
        LOG.warning("--skip-encode set; skipping %d edits with missing after.npz",
                    len(needs_encode))
        for e in needs_encode:
            stats.add_skip("missing_after_npz_skip_encode")
        needs_encode = []

    # ── Phase: render ────────────────────────────────────────────────────────
    if needs_encode and args.phase in ("render", "both"):
        gpu_ids = _resolve_gpu_ids(args)
        if not gpu_ids:
            LOG.error("no GPU ids resolved; pass --gpu-ids or --device")
            return 2
        _run_render_pool(
            needs_encode, gpu_ids, args.encode_work_dir,
            args.num_views, args.blender, stats,
        )

    if args.phase == "render":
        print_summary(stats, "pull_deletion (render)")
        return 0 if stats.skipped == 0 else 1

    # ── Phase: encode ────────────────────────────────────────────────────────
    if needs_encode and args.phase in ("encode", "both"):
        gpu_ids = _resolve_gpu_ids(args)
        if not gpu_ids:
            LOG.error("no GPU ids resolved; pass --gpu-ids or --device")
            return 2

        staged = [
            e for e in needs_encode
            if (args.encode_work_dir / _edit_name(e) / "render.done").is_file()
        ]
        missing_staged = [e for e in needs_encode if e not in set(staged)]
        if missing_staged:
            LOG.warning(
                "--phase %s: %d edits lack render.done staging "
                "(run --phase render first); recording as missing_staged_render",
                args.phase, len(missing_staged),
            )
            for e in missing_staged:
                stats.add_skip("missing_staged_render")

        if staged:
            _run_encode_pool(
                staged, gpu_ids, args.ckpt_root, args.encode_work_dir,
                args.num_views, stats,
            )

        ready = [e for e in edits if (e.edit_dir / "after.npz").is_file()]
        LOG.info("after encode: ready for promote=%d", len(ready))

    # ── Phase: promote ───────────────────────────────────────────────────────
    ctx = build_promote_context(paths, pipeline_cfg_path=args.pipeline_cfg)
    run_promote_pool(
        ready, layout, ctx,
        promote_fn=lambda e, l: promote_deletion(e, l, ctx=ctx),
        workers=args.workers, stats=stats,
    )
    print_summary(stats, f"pull_deletion ({args.phase})")
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
