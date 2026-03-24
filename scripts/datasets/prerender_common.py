"""Shared utilities for dataset-specific prerender scripts.

Provides GPU discovery, Blender render orchestration, and SLAT encoding that
are common across PartObjaverse-Tiny, PartVerse, and future datasets.

Dataset-specific scripts (datasets/partobjaverse/prerender.py, etc.) call
these functions after providing their own path constants and GLB accessors.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

def get_available_gpus() -> list[int]:
    """Return GPU indices from CUDA_VISIBLE_DEVICES, or all GPUs via nvidia-smi."""
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_env:
        return [int(x.strip()) for x in cuda_env.split(",") if x.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split("\n")
                    if x.strip()]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# third_party/outputs symlink
# ---------------------------------------------------------------------------

def ensure_outputs_symlink(third_party_dir: Path, data_dir: Path,
                            logger: logging.Logger | None = None):
    """Symlink third_party/outputs → data_dir.

    encode_asset scripts use relative paths like outputs/img_Enc/ and
    outputs/slat/ when cwd = third_party/.  This symlink redirects those
    writes to the correct dataset data directory.
    """
    log = logger or logging.getLogger(__name__)
    link = third_party_dir / "outputs"
    target = data_dir

    if link.is_symlink():
        current = link.resolve()
        if current == target.resolve():
            return
        log.warning(f"third_party/outputs points to {current}, re-linking → {target}")
        link.unlink()
    elif link.exists():
        log.warning("third_party/outputs is a real directory — skipping symlink creation")
        return

    link.symlink_to(target)
    log.info(f"Created symlink: third_party/outputs → {target}")


# ---------------------------------------------------------------------------
# Phase A: Blender render + Open3D voxelize
# ---------------------------------------------------------------------------

def run_render(
    obj_ids: list[str],
    glb_getter: Callable[[str], Path],
    img_enc_dir: Path,
    third_party_dir: Path,
    force: bool,
    render_workers: int,
    script_path: Path,
    logger: logging.Logger,
    extra_worker_args: list[str] | None = None,
):
    """Render pending objects. Dispatches to parallel or sequential mode.

    Args:
        obj_ids:          All object IDs to process.
        glb_getter:       Callable(obj_id) -> Path to the source GLB.
        img_enc_dir:      Output root for rendered views (img_Enc/).
        third_party_dir:  Project's third_party/ directory.
        force:            Re-render even if cached.
        render_workers:   Number of parallel Blender workers (each on 1 GPU).
        script_path:      Path to the calling dataset script (used for subprocesses).
        logger:           Logger instance.
        extra_worker_args: Extra CLI args forwarded to parallel worker subprocesses
                          (e.g. ["--shard", "00", "--num-shards", "10"]).
    """
    pending = [
        oid for oid in obj_ids
        if force or not (img_enc_dir / oid / "voxels.ply").exists()
    ]
    cached = len(obj_ids) - len(pending)
    logger.info(f"Render: {len(pending)} pending, {cached} cached")
    if not pending:
        return

    if render_workers > 1:
        _render_parallel(pending, script_path, render_workers, force, logger,
                         extra_worker_args or [])
    else:
        gpus = get_available_gpus()
        gpu_id = gpus[0] if gpus else None
        _render_sequential(pending, glb_getter, third_party_dir, gpu_id, logger)


def _render_sequential(
    obj_ids: list[str],
    glb_getter: Callable[[str], Path],
    third_party_dir: Path,
    gpu_id: int | None,
    logger: logging.Logger,
):
    """Render objects one-by-one in the current process."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Blender rendering on GPU {gpu_id}")

    sys.path.insert(0, str(third_party_dir))
    os.chdir(str(third_party_dir))
    os.makedirs("outputs/img_Enc", exist_ok=True)

    from encode_asset.render_img_for_enc import renderImg_voxelize

    for i, oid in enumerate(obj_ids):
        glb = glb_getter(oid)
        if not glb.exists():
            logger.warning(f"[render {i+1}/{len(obj_ids)}] {oid}: GLB not found, skip")
            continue
        logger.info(f"[render {i+1}/{len(obj_ids)}] {oid}")
        try:
            renderImg_voxelize(str(glb))
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()


def _render_parallel(
    obj_ids: list[str],
    script_path: Path,
    num_workers: int,
    force: bool,
    logger: logging.Logger,
    extra_worker_args: list[str] | None = None,
):
    """Launch num_workers subprocesses, each assigned to a dedicated GPU.

    Workers are distributed round-robin across available GPUs. Blender reads
    CUDA_VISIBLE_DEVICES and uses all visible devices, so restricting each
    worker to one GPU avoids contention.
    """
    gpus = get_available_gpus()
    if not gpus:
        logger.warning("No GPUs found — Blender will attempt GPU rendering without "
                       "CUDA_VISIBLE_DEVICES restriction.")
        gpus = list(range(num_workers))

    if num_workers > len(gpus):
        logger.warning(
            f"{num_workers} workers requested but only {len(gpus)} GPUs available. "
            "Multiple workers will share GPUs — may cause OOM.")

    shards = [[] for _ in range(num_workers)]
    for i, oid in enumerate(obj_ids):
        shards[i % num_workers].append(oid)

    worker_gpu = [gpus[i % len(gpus)] for i in range(num_workers)]
    logger.info(f"Launching {num_workers} render workers across GPUs {gpus}")

    processes = []
    for idx, (shard, gpu_id) in enumerate(zip(shards, worker_gpu)):
        if not shard:
            continue
        cmd = [
            sys.executable, str(script_path),
            "--render-only", "--render-workers", "1",
            "--obj-ids", *shard,
            *(extra_worker_args or []),
        ]
        if force:
            cmd.append("--force")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"[worker {idx} -> GPU {gpu_id}] {len(shard)} objects")
        processes.append((idx, gpu_id, subprocess.Popen(cmd, env=env)))

    failed = []
    for idx, gpu_id, p in processes:
        if p.wait() != 0:
            failed.append(idx)
            logger.error(f"[worker {idx} GPU {gpu_id}] FAILED (exit {p.returncode})")
        else:
            logger.info(f"[worker {idx} GPU {gpu_id}] done")
    if failed:
        logger.warning(f"Workers {failed} failed — re-run to retry")


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------

def get_shard_ids(num_shards: int) -> list[str]:
    """Return zero-padded shard name strings, e.g. ["00","01","02"] for 3 shards."""
    width = max(2, len(str(num_shards - 1)))
    return [str(i).zfill(width) for i in range(num_shards)]


def select_shard(all_obj_ids: list[str], shard: str, num_shards: int) -> list[str]:
    """Return the subset of obj_ids belonging to this shard (range-based).

    Objects are sorted alphabetically first so the partition is deterministic.
    Example: 12030 objects, 10 shards → each shard gets ~1203 contiguous objects.

    Args:
        all_obj_ids:  Full sorted list of object IDs.
        shard:        Zero-padded shard string, e.g. "03".
        num_shards:   Total number of shards.
    """
    shard_idx = int(shard)
    if shard_idx >= num_shards:
        raise ValueError(f"shard {shard!r} out of range for num_shards={num_shards}")
    n = len(all_obj_ids)
    chunk = (n + num_shards - 1) // num_shards   # ceiling division
    start = shard_idx * chunk
    end   = min(start + chunk, n)
    return all_obj_ids[start:end]


# ---------------------------------------------------------------------------
# Phase B: DINOv2 + SLAT encode
# ---------------------------------------------------------------------------

def run_encode(
    obj_ids: list[str],
    img_enc_dir: Path,
    slat_dir: Path,
    third_party_dir: Path,
    force: bool,
    logger: logging.Logger,
):
    """Encode pending objects into SLAT features. Requires GPU.

    Args:
        obj_ids:        All object IDs to process.
        img_enc_dir:    Directory containing per-object render outputs (voxels.ply).
        slat_dir:       Output directory for {obj_id}_feats.pt / _coords.pt.
        third_party_dir: Project's third_party/ directory.
        force:          Re-encode even if cached.
        logger:         Logger instance.
    """
    pending = []
    for oid in obj_ids:
        feats  = slat_dir / f"{oid}_feats.pt"
        coords = slat_dir / f"{oid}_coords.pt"
        voxels = img_enc_dir / oid / "voxels.ply"
        if not force and feats.exists() and coords.exists():
            logger.debug(f"[encode] skip {oid} (cached)")
        elif not voxels.exists():
            logger.warning(f"[encode] skip {oid} (no renders)")
        else:
            pending.append(oid)

    cached = len(obj_ids) - len(pending)
    logger.info(f"Encode: {len(pending)} pending, {cached} cached/skipped")
    if not pending:
        return

    sys.path.insert(0, str(third_party_dir))
    os.chdir(str(third_party_dir))
    os.makedirs("outputs/slat", exist_ok=True)

    from encode_asset.encode_into_SLAT import encode_into_SLAT

    for i, oid in enumerate(pending):
        logger.info(f"[encode {i+1}/{len(pending)}] {oid}")
        try:
            encode_into_SLAT(oid)
            feats = slat_dir / f"{oid}_feats.pt"
            if feats.exists():
                logger.info(f"  -> SLAT encoded ({feats.stat().st_size // 1024} KB)")
                # Remove rendered PNGs and transforms.json — they are packed into NPZ
                # already; only voxels.ply (cache sentinel) is kept.
                obj_dir = img_enc_dir / oid
                for f in obj_dir.iterdir():
                    if f.suffix in (".png", ".jpg") or f.name == "transforms.json":
                        f.unlink()
            else:
                logger.error(f"  -> feats.pt not found after encoding")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Multi-GPU encode launcher
# ---------------------------------------------------------------------------

def launch_multi_gpu_encode(
    obj_ids: list[str],
    slat_dir: Path,
    script_path: Path,
    num_gpus: int,
    force: bool,
    logger: logging.Logger,
):
    """Launch one encode subprocess per GPU, each processing a shard."""
    pending = [
        oid for oid in obj_ids
        if force or not (slat_dir / f"{oid}_feats.pt").exists()
    ]
    if not pending:
        logger.info("All objects already encoded")
        return

    logger.info(f"Multi-GPU encode: {len(pending)} objects across {num_gpus} GPUs")

    shards = [[] for _ in range(num_gpus)]
    for i, oid in enumerate(pending):
        shards[i % num_gpus].append(oid)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_gpus = [g.strip() for g in visible.split(",") if g.strip()] if visible else []

    processes = []
    for idx, shard in enumerate(shards):
        if not shard:
            continue
        physical_gpu = visible_gpus[idx] if idx < len(visible_gpus) else str(idx)
        cmd = [
            sys.executable, str(script_path),
            "--encode-only",
            "--obj-ids", *shard,
        ]
        if force:
            cmd.append("--force")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        logger.info(f"[GPU {physical_gpu}] {len(shard)} objects")
        processes.append((physical_gpu, subprocess.Popen(cmd, env=env)))

    for physical_gpu, p in processes:
        ret = p.wait()
        status = "done" if ret == 0 else f"FAILED (exit {ret})"
        logger.info(f"[GPU {physical_gpu}] {status}")


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def print_summary(obj_ids: list[str], img_enc_dir: Path, slat_dir: Path,
                  logger: logging.Logger):
    rendered = sum(1 for oid in obj_ids if (img_enc_dir / oid / "voxels.ply").exists())
    encoded  = sum(1 for oid in obj_ids if (slat_dir / f"{oid}_feats.pt").exists())
    logger.info(f"Summary: {rendered}/{len(obj_ids)} rendered, "
                f"{encoded}/{len(obj_ids)} SLAT encoded")
