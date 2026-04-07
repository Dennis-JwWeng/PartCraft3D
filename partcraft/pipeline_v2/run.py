"""Pipeline v2 orchestrator + CLI.

Single entrypoint that selects which steps to run on which objects::

    python -m partcraft.pipeline_v2.run \\
        --config configs/pipeline_v2_mirror5.yaml \\
        --shard 01 \\
        --steps s1,s2,s4,s5,s6,s5b,s6b,s7 \\
        --obj-ids ABCDE...                # or --all (use existing object dirs)

Per-step shape::

    s1 / s2 / s4   single process, multi-server fan-out where applicable
    s5 / s6 / s6b  multi-GPU subprocess pool (CUDA_VISIBLE_DEVICES splits)
    s5b / s7       single process, CPU only

The orchestrator:
* loads YAML config + CLI overrides;
* resolves object list (CLI ids OR all dirs under ``objects/<shard>/``);
* for each requested step, dispatches the matching runner;
* after each step, calls :func:`status.rebuild_manifest` so the global
  manifest stays in sync.

For multi-GPU steps, the parent process spawns one child per GPU using
``subprocess.Popen`` with ``CUDA_VISIBLE_DEVICES`` set, each child
re-invokes ``python -m partcraft.pipeline_v2.run`` with the same flags
plus ``--gpu-shard <i>/<n>`` and ``--single-gpu``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml

from .paths import PipelineRoot, ObjectContext, normalize_shard
from .status import rebuild_manifest, manifest_summary, step_done
from .validators import apply_check

LOG = logging.getLogger("pipeline_v2")

ALL_STEPS = ("s1", "s2", "s4", "s5", "s5b", "s6", "s6b", "s7")
GPU_STEPS = frozenset({"s5", "s6", "s6b"})


# ─────────────────── config + ctx resolution ─────────────────────────

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("data", {})
    cfg.setdefault("phase2_5", {})
    return cfg


def resolve_root(cfg: dict) -> PipelineRoot:
    out = cfg["data"].get("output_dir") or cfg["data"].get("pipeline_v2_root")
    if not out:
        raise SystemExit("[CONFIG] data.output_dir is required")
    root = PipelineRoot(Path(out))
    root.ensure()
    return root


def resolve_ctxs(
    root: PipelineRoot,
    cfg: dict,
    *,
    shard: str,
    obj_ids: list[str] | None,
    all_objs: bool,
) -> list[ObjectContext]:
    shard = normalize_shard(shard)
    mesh_root = Path(cfg["data"].get("mesh_root", "data/partverse/mesh"))
    images_root = Path(cfg["data"].get("images_root", "data/partverse/images"))

    if obj_ids:
        ids = obj_ids
    elif all_objs:
        ids = [d.name for d in sorted(root.shard_dir(shard).iterdir())
               if d.is_dir()] if root.shard_dir(shard).is_dir() else []
    else:
        raise SystemExit("[CLI] one of --obj-ids or --all is required")

    ctxs: list[ObjectContext] = []
    for oid in ids:
        ctxs.append(root.context(
            shard, oid,
            mesh_npz=mesh_root / shard / f"{oid}.npz",
            image_npz=images_root / shard / f"{oid}.npz",
        ))
    return ctxs


def slice_for_gpu(ctxs: list[ObjectContext], i: int, n: int) -> list[ObjectContext]:
    return [c for k, c in enumerate(ctxs) if k % n == i]


# ─────────────────── step dispatch ───────────────────────────────────

def run_step(
    step: str,
    ctxs: list[ObjectContext],
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    log = LOG.getChild(step)
    log.info("=" * 60)
    log.info("STEP %s on %d objects", step, len(ctxs))
    log.info("=" * 60)
    if not ctxs:
        return

    images_root = Path(cfg["data"].get("images_root", "data/partverse/images"))
    mesh_root = Path(cfg["data"].get("mesh_root", "data/partverse/mesh"))
    shard = ctxs[0].shard

    if step == "s1":
        from .s1_phase1_vlm import run_many_async
        import asyncio
        urls = (cfg.get("phase0") or {}).get("vlm_base_urls") or [
            (cfg.get("phase0") or {}).get("vlm_base_url",
                                          "http://localhost:8002/v1")
        ]
        model = (cfg.get("phase0") or {}).get("vlm_model", "Qwen3.5-27B")
        blender = cfg.get("blender",
                          "/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender")
        asyncio.run(run_many_async(
            ctxs, blender=blender, vlm_urls=urls,
            vlm_model=model, force=args.force,
        ))

    elif step == "s2":
        from .s2_highlights import run_many
        blender = cfg.get("blender",
                          "/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender")
        run_many(ctxs, blender=blender, force=args.force)

    elif step == "s4":
        from .s4_flux_2d import run as s4_run
        urls = (cfg.get("phase2_5") or {}).get("image_edit_base_urls") or []
        if not urls:
            single = (cfg.get("phase2_5") or {}).get("image_edit_base_url")
            if single:
                urls = [single]
        if not urls:
            raise SystemExit("[CONFIG] phase2_5.image_edit_base_urls required for s4")
        s4_run(ctxs, edit_urls=urls,
               workers_per_server=cfg.get("phase2_5", {}).get("workers_per_server", 2),
               images_root=images_root, mesh_root=mesh_root, shard=shard,
               force=args.force, logger=log)

    elif step == "s5":
        from .s5_trellis_3d import run as s5_run
        s5_run(ctxs, cfg=cfg, images_root=images_root, mesh_root=mesh_root,
               shard=shard, force=args.force, logger=log)

    elif step == "s5b":
        from .s5b_deletion import run_mesh_delete
        run_mesh_delete(ctxs, cfg=cfg, images_root=images_root,
                        mesh_root=mesh_root, shard=shard,
                        force=args.force, logger=log)

    elif step == "s6":
        from .s6_render_3d import run as s6_run
        ckpt = (cfg.get("phase2_5") or {}).get(
            "trellis_text_ckpt", "checkpoints/TRELLIS-text-xlarge")
        s6_run(ctxs, ckpt=ckpt, force=args.force, logger=log)

    elif step == "s6b":
        from .s5b_deletion import run_reencode
        blender = cfg.get("blender",
                          "/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender")
        run_reencode(ctxs, cfg=cfg, blender_path=blender,
                     num_views=cfg.get("phase5", {}).get("num_views", 40),
                     force=args.force, logger=log)

    elif step == "s7":
        from .s7_addition_backfill import run as s7_run
        s7_run(ctxs, force=args.force, logger=log)

    else:
        raise SystemExit(f"unknown step: {step}")


# ─────────────────── multi-GPU dispatch ──────────────────────────────

def dispatch_gpus(
    step: str,
    cfg_path: Path,
    args: argparse.Namespace,
) -> int:
    """Spawn one child per GPU. Each child re-invokes this CLI with
    ``--single-gpu --gpu-shard i/n`` and a single-step ``--steps``."""
    gpus = [g.strip() for g in (args.gpus or "").split(",") if g.strip()]
    n = len(gpus)
    if n <= 1:
        return run_single_gpu(step, cfg_path, args)

    LOG.info("[%s] dispatching across GPUs %s", step, gpus)
    procs = []
    for i, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("ATTN_BACKEND", "xformers")
        cmd = [
            sys.executable, "-m", "partcraft.pipeline_v2.run",
            "--config", str(cfg_path),
            "--shard", args.shard,
            "--steps", step,
            "--single-gpu",
            "--gpu-shard", f"{i}/{n}",
        ]
        if args.obj_ids:
            cmd += ["--obj-ids", *args.obj_ids]
        if args.all:
            cmd += ["--all"]
        if args.force:
            cmd += ["--force"]
        LOG.info("  GPU %s: %s", gpu, " ".join(cmd[-6:]))
        procs.append((gpu, subprocess.Popen(cmd, env=env)))

    rc = 0
    for gpu, p in procs:
        r = p.wait()
        LOG.info("[%s] GPU %s exit=%d", step, gpu, r)
        if r != 0:
            rc = r
    return rc


def run_single_gpu(
    step: str,
    cfg_path: Path,
    args: argparse.Namespace,
) -> int:
    cfg = load_config(cfg_path)
    root = resolve_root(cfg)
    ctxs = resolve_ctxs(root, cfg, shard=args.shard,
                        obj_ids=args.obj_ids, all_objs=args.all)
    if args.gpu_shard:
        i, n = (int(x) for x in args.gpu_shard.split("/"))
        ctxs = slice_for_gpu(ctxs, i, n)
        LOG.info("[%s] gpu shard %d/%d -> %d objects", step, i, n, len(ctxs))
    run_step(step, ctxs, cfg, args)
    rebuild_manifest(root)
    return 0


# ─────────────────── CLI ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(prog="pipeline_v2")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--shard", default="01")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--obj-ids", nargs="+")
    grp.add_argument("--all", action="store_true")
    ap.add_argument("--steps", default=",".join(ALL_STEPS),
                    help=f"comma list, any of: {','.join(ALL_STEPS)}")
    ap.add_argument("--gpus", default=None,
                    help="comma list, e.g. 4,5,6,7 (only used for "
                         "GPU steps; ignored otherwise)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    # Internal: child workers set these.
    ap.add_argument("--single-gpu", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--gpu-shard", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    cfg = load_config(args.config)
    root = resolve_root(cfg)
    ctxs = resolve_ctxs(root, cfg, shard=args.shard,
                        obj_ids=args.obj_ids, all_objs=args.all)
    LOG.info("root=%s shard=%s objects=%d", root.root, args.shard, len(ctxs))

    if args.dry_run:
        for c in ctxs:
            done = {s for s in ALL_STEPS if step_done(c, _step_to_status_key(s))}
            print(f"  {c.obj_id}  done={sorted(done) or '-'}")
        print(json.dumps(manifest_summary(root), indent=2))
        return

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        raise SystemExit(f"unknown steps: {bad}")

    for step in steps:
        if step in GPU_STEPS and args.gpus and not args.single_gpu:
            dispatch_gpus(step, args.config, args)
        else:
            run_step(step, ctxs, cfg, args)
        # post-step validation: rewrite status to reflect product reality
        n_pass = n_fail = 0
        for c in ctxs:
            rep = apply_check(c, step)
            if rep.ok:
                n_pass += 1
            else:
                n_fail += 1
                LOG.warning("[%s] %s incomplete: %d/%d (missing: %s)",
                            step, c.obj_id, rep.found, rep.expected,
                            rep.missing[:3])
        LOG.info("[%s] validate: pass=%d fail=%d", step, n_pass, n_fail)
        rebuild_manifest(root)

    LOG.info("\n%s", json.dumps(manifest_summary(root),
                                 indent=2, ensure_ascii=False))


_STATUS_KEYS = {
    "s1": "s1_phase1", "s2": "s2_highlights", "s4": "s4_flux_2d",
    "s5": "s5_trellis", "s5b": "s5b_del_mesh", "s6": "s6_render_3d",
    "s6b": "s6b_del_reencode", "s7": "s7_add_backfill",
}


def _step_to_status_key(step: str) -> str:
    return _STATUS_KEYS.get(step, step)


if __name__ == "__main__":
    main()
