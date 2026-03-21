#!/usr/bin/env python3
"""Pre-render all objects: Blender 150 views + Open3D voxelization + SLAT encoding.

This decouples rendering from editing, so Phase 0 can use high-quality views
and Phase 2.5 can skip inline rendering entirely.

Outputs per object:
  data/partobjaverse_tiny/img_Enc/{obj_id}/
    ├── 000.png .. 149.png   # 150 Blender-rendered views (512x512)
    ├── mesh.ply             # Normalized mesh in [-0.5, 0.5]^3
    ├── voxels.ply           # Open3D voxelized mesh (64^3)
    └── transforms.json      # Camera parameters per view
  data/partobjaverse_tiny/slat/
    ├── {obj_id}_feats.pt    # SLAT features
    └── {obj_id}_coords.pt   # SLAT coordinates

Usage:
    # 4 GPUs, each running 1 Blender worker (4 workers total)
    CUDA_VISIBLE_DEVICES=0,2,4,6 python scripts/prerender.py \
        --config configs/partobjaverse.yaml --render-only --render-workers 4

    # 4 GPUs, each running 4 Blender workers (16 workers total)
    CUDA_VISIBLE_DEVICES=0,2,4,6 python scripts/prerender.py \
        --config configs/partobjaverse.yaml --render-only --render-workers 16

    # 1 GPU, 4 workers on that GPU
    CUDA_VISIBLE_DEVICES=2 python scripts/prerender.py \
        --config configs/partobjaverse.yaml --render-only --render-workers 4

    # Render + encode all objects (single GPU)
    CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/prerender.py \
        --config configs/partobjaverse.yaml

    # Multi-GPU encode only
    ATTN_BACKEND=xformers python scripts/prerender.py \
        --config configs/partobjaverse.yaml --encode-only --num-gpus 8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
_DATA_DIR = _PROJECT_ROOT / "data" / "partobjaverse_tiny"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIRD_PARTY))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging


def get_available_gpus() -> list[int]:
    """Parse CUDA_VISIBLE_DEVICES env var to get available GPU indices.

    Returns list of GPU indices. If env var is not set, returns all GPUs
    detected by nvidia-smi. If no GPUs found, returns empty list.
    """
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_env:
        return [int(x.strip()) for x in cuda_env.split(",") if x.strip()]
    # Fallback: detect via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        pass
    return []


def extract_glb(mesh_zip_path: str, obj_id: str, out_dir: str) -> str:
    """Extract a single GLB from source/mesh.zip."""
    with zipfile.ZipFile(mesh_zip_path) as zf:
        matches = [n for n in zf.namelist() if obj_id in n and n.endswith('.glb')]
        if not matches:
            raise FileNotFoundError(f"GLB for {obj_id} not found in {mesh_zip_path}")
        out_path = os.path.join(out_dir, f"{obj_id}.glb")
        with zf.open(matches[0]) as src, open(out_path, 'wb') as dst:
            dst.write(src.read())
        return out_path


def get_all_obj_ids(cfg: dict) -> list[str]:
    """Get all object IDs from the dataset (mesh NPZ directory)."""
    mesh_dir = Path(cfg["data"]["mesh_npz_dir"])
    shards = cfg["data"].get("shards", ["00"])
    obj_ids = []
    for shard in shards:
        shard_dir = mesh_dir / shard
        if not shard_dir.exists():
            continue
        for f in sorted(shard_dir.iterdir()):
            if f.suffix == ".npz":
                obj_ids.append(f.stem)
    return obj_ids


def launch_multi_gpu(args, obj_ids: list[str], num_gpus: int):
    """Launch one subprocess per GPU, each handling a shard of objects.

    Uses GPUs from CUDA_VISIBLE_DEVICES if set, otherwise uses 0..num_gpus-1.
    """
    script_path = Path(__file__).resolve()
    gpus = get_available_gpus()
    if not gpus:
        gpus = list(range(num_gpus))
    else:
        gpus = gpus[:num_gpus]  # Use at most num_gpus from available

    # Split objects evenly across GPUs
    shards = [[] for _ in range(len(gpus))]
    for i, oid in enumerate(obj_ids):
        shards[i % len(gpus)].append(oid)

    processes = []
    for idx, gpu_id in enumerate(gpus):
        if not shards[idx]:
            continue

        cmd = [
            sys.executable, str(script_path),
            "--config", args.config or "",
            "--obj-ids", *shards[idx],
        ]
        if args.render_only:
            cmd.append("--render-only")
        if args.encode_only:
            cmd.append("--encode-only")
        if args.force:
            cmd.append("--force")
        if args.render_workers > 1:
            cmd.extend(["--render-workers", str(args.render_workers)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[GPU {gpu_id}] Launching with {len(shards[idx])} objects")
        p = subprocess.Popen(cmd, env=env)
        processes.append((gpu_id, p))

    # Wait for all
    failed = []
    for gpu_id, p in processes:
        ret = p.wait()
        if ret != 0:
            failed.append(gpu_id)
            print(f"[GPU {gpu_id}] FAILED (exit code {ret})")
        else:
            print(f"[GPU {gpu_id}] Done")

    if failed:
        print(f"\nWARNING: GPUs {failed} had failures. Re-run to retry.")
    else:
        print(f"\nAll {num_gpus} GPUs completed successfully.")


def launch_render_workers(args, obj_ids: list[str], gpus: list[int],
                          num_workers: int, logger):
    """Launch render subprocesses, round-robin assigned to available GPUs.

    Multiple workers can share the same GPU (Blender uses ~3-5GB per process,
    A800 80GB can handle many). Each subprocess gets CUDA_VISIBLE_DEVICES
    set to its assigned GPU.

    Args:
        num_workers: Total number of parallel Blender workers to launch.
        gpus: Available GPU indices to distribute workers across.
    """
    script_path = Path(__file__).resolve()

    # Split objects evenly across workers
    shards = [[] for _ in range(num_workers)]
    for i, oid in enumerate(obj_ids):
        shards[i % num_workers].append(oid)

    # Assign GPUs round-robin
    worker_gpu = [gpus[i % len(gpus)] for i in range(num_workers)]

    # Log GPU allocation
    from collections import Counter
    gpu_counts = Counter(worker_gpu)
    for gpu_id, count in sorted(gpu_counts.items()):
        logger.info(f"  GPU {gpu_id}: {count} worker(s)")

    processes = []
    for idx in range(num_workers):
        if not shards[idx]:
            continue

        gpu_id = worker_gpu[idx]
        cmd = [
            sys.executable, str(script_path),
            "--config", args.config or "",
            "--render-only",
            "--render-workers", "1",  # Each subprocess renders sequentially
            "--obj-ids", *shards[idx],
        ]
        if args.force:
            cmd.append("--force")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger.info(f"[worker {idx} -> GPU {gpu_id}] Launching with "
                    f"{len(shards[idx])} objects")
        p = subprocess.Popen(cmd, env=env)
        processes.append((idx, gpu_id, p))

    # Wait for all
    failed = []
    for idx, gpu_id, p in processes:
        ret = p.wait()
        if ret != 0:
            failed.append((idx, gpu_id))
            logger.error(f"[worker {idx} GPU {gpu_id}] FAILED (exit {ret})")
        else:
            logger.info(f"[worker {idx} GPU {gpu_id}] Done")

    if failed:
        logger.warning(f"Workers {[(i, g) for i, g in failed]} had failures. "
                       f"Re-run to retry.")
    else:
        logger.info(f"All {num_workers} render workers completed successfully.")


def run_render(obj_ids: list[str], mesh_zip: str,
               force: bool, render_workers: int, logger, args=None):
    """Phase A: Blender 150 views + Open3D voxelize.

    When render_workers > 1 and GPUs are available, launches one subprocess
    per GPU (each with its own CUDA_VISIBLE_DEVICES). Otherwise runs
    sequentially in the current process.
    """
    # Resolve to absolute path before chdir
    mesh_zip = os.path.abspath(mesh_zip)
    render_pending = []
    for oid in obj_ids:
        voxels_path = str(_DATA_DIR / "img_Enc" / oid / "voxels.ply")
        if not force and os.path.exists(voxels_path):
            logger.info(f"[render] Skipping {oid} (cached)")
        else:
            render_pending.append(oid)

    logger.info(f"Rendering: {len(render_pending)} objects "
                f"({len(obj_ids) - len(render_pending)} cached)")

    if not render_pending:
        return

    # Multi-worker parallel render: launch subprocesses
    gpus = get_available_gpus()
    if render_workers > 1 and gpus and args is not None:
        logger.info(f"Parallel render: {render_workers} workers across "
                    f"{len(gpus)} GPUs {gpus}, {len(render_pending)} objects")
        launch_render_workers(args, render_pending, gpus, render_workers, logger)
        return

    # Single-process sequential rendering (1 worker or 1 GPU)
    if gpus:
        logger.info(f"Rendering on GPU {gpus[0]}")
    else:
        logger.info(f"No GPUs detected, Blender will use CPU rendering")

    original_cwd = os.getcwd()
    # encode_asset scripts use relative paths like outputs/img_Enc/
    # third_party/outputs/ symlinks to data/ for compatibility
    os.chdir(str(_THIRD_PARTY))
    os.makedirs("outputs/img_Enc", exist_ok=True)

    from encode_asset.render_img_for_enc import renderImg_voxelize

    with tempfile.TemporaryDirectory(prefix="partcraft_glb_") as tmp_dir:
        for i, oid in enumerate(render_pending):
            logger.info(f"[render {i+1}/{len(render_pending)}] {oid}")
            try:
                glb_path = extract_glb(mesh_zip, oid, tmp_dir)
                renderImg_voxelize(glb_path)
                if os.path.exists(f"outputs/img_Enc/{oid}/voxels.ply"):
                    n_imgs = len([f for f in os.listdir(f"outputs/img_Enc/{oid}")
                                 if f.endswith('.png')])
                    logger.info(f"  -> {n_imgs} views + voxels.ply + mesh.ply")
                else:
                    logger.error(f"  -> voxels.ply not created!")
            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
                import traceback
                traceback.print_exc()

    os.chdir(original_cwd)


def run_encode(obj_ids: list[str], force: bool, logger):
    """Phase B: DINOv2 + SLAT encoding (requires GPU)."""
    encode_pending = []
    for oid in obj_ids:
        feats_path = str(_DATA_DIR / "slat" / f"{oid}_feats.pt")
        coords_path = str(_DATA_DIR / "slat" / f"{oid}_coords.pt")
        voxels_path = str(_DATA_DIR / "img_Enc" / oid / "voxels.ply")
        if not force and os.path.exists(feats_path) and os.path.exists(coords_path):
            logger.info(f"[encode] Skipping {oid} (cached)")
        elif not os.path.exists(voxels_path):
            logger.warning(f"[encode] Skipping {oid} (no renders)")
        else:
            encode_pending.append(oid)

    logger.info(f"Encoding SLAT: {len(encode_pending)} objects "
                f"({len(obj_ids) - len(encode_pending)} cached/skipped)")

    if not encode_pending:
        return

    original_cwd = os.getcwd()
    # encode_asset scripts use relative paths like outputs/slat/
    os.chdir(str(_THIRD_PARTY))
    os.makedirs("outputs/slat", exist_ok=True)

    from encode_asset.encode_into_SLAT import encode_into_SLAT

    for i, oid in enumerate(encode_pending):
        logger.info(f"[encode {i+1}/{len(encode_pending)}] {oid}")
        try:
            encode_into_SLAT(oid)
            if os.path.exists(f"outputs/slat/{oid}_feats.pt"):
                logger.info(f"  -> SLAT encoded")
            else:
                logger.error(f"  -> feats.pt not created!")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            import traceback
            traceback.print_exc()

    os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-render all objects: Blender 150 views + SLAT encoding")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--obj-ids", nargs="*", default=None,
                        help="Specific object IDs (default: all)")
    parser.add_argument("--render-only", action="store_true",
                        help="Only render views + voxelize (no SLAT encoding, no GPU)")
    parser.add_argument("--encode-only", action="store_true",
                        help="Only encode SLAT (assumes renders exist)")
    parser.add_argument("--force", action="store_true",
                        help="Re-render/encode even if cached")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs for parallel encoding. "
                             "Launches one subprocess per GPU with auto-sharding. "
                             "0 = single process (default)")
    parser.add_argument("--render-workers", type=int, default=1,
                        help="Number of parallel Blender render workers (CPU). "
                             "Only used within a single process.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "prerender")
    p25_cfg = cfg.get("phase2_5", {})

    # Paths
    data_dir = Path(cfg["data"].get("data_dir", "data/partobjaverse_tiny"))
    mesh_zip = data_dir / "source" / "mesh.zip"
    if not mesh_zip.exists():
        img_dir = Path(cfg["data"]["image_npz_dir"])
        data_dir = img_dir.parent
        mesh_zip = data_dir / "source" / "mesh.zip"

    if not mesh_zip.exists():
        logger.error(f"source/mesh.zip not found at {mesh_zip}")
        sys.exit(1)

    # Determine object IDs
    if args.obj_ids:
        obj_ids = args.obj_ids
    else:
        obj_ids = get_all_obj_ids(cfg)

    if not obj_ids:
        logger.error("No objects found in dataset")
        sys.exit(1)

    logger.info(f"Total objects: {len(obj_ids)}")

    # --- Multi-GPU mode: launch subprocesses ---
    if args.num_gpus > 1:
        # Filter to pending objects first
        pending = []
        for oid in obj_ids:
            feats = str(_DATA_DIR / "slat" / f"{oid}_feats.pt")
            voxels = str(_DATA_DIR / "img_Enc" / oid / "voxels.ply")
            if args.render_only:
                if args.force or not os.path.exists(voxels):
                    pending.append(oid)
            elif args.encode_only:
                if args.force or not os.path.exists(feats):
                    pending.append(oid)
            else:
                if args.force or not os.path.exists(feats):
                    pending.append(oid)

        if not pending:
            logger.info("All objects already processed")
            return

        logger.info(f"Launching {args.num_gpus} GPU workers for "
                    f"{len(pending)} pending objects")
        # Override obj_ids for subprocess launch
        args_copy = argparse.Namespace(**vars(args))
        args_copy.obj_ids = None  # will be set per-GPU
        launch_multi_gpu(args, pending, args.num_gpus)
        return

    # --- Single process mode ---
    # Phase A: Render
    if not args.encode_only:
        run_render(obj_ids, str(mesh_zip),
                   args.force, args.render_workers, logger, args=args)

    # Phase B: Encode
    if not args.render_only:
        run_encode(obj_ids, args.force, logger)

    # Summary
    rendered = sum(1 for oid in obj_ids
                   if (_DATA_DIR / "img_Enc" / oid / "voxels.ply").exists())
    encoded = sum(1 for oid in obj_ids
                  if (_DATA_DIR / "slat" / f"{oid}_feats.pt").exists())
    logger.info(f"\nSummary: {rendered}/{len(obj_ids)} rendered, "
                f"{encoded}/{len(obj_ids)} SLAT encoded")


if __name__ == "__main__":
    main()
