"""Pipeline v2 orchestrator + CLI.

Single entrypoint that selects which steps to run on which objects::

    python -m partcraft.pipeline_v2.run \\
        --config configs/pipeline_v2_mirror5.yaml \\
        --shard 01 \\
        --steps s1,s2,s4,s5,s6,s5b,s6b,s7 \\
        --obj-ids ABCDE...                # or --all (use existing object dirs)

Stages (``pipeline.stages``) map to steps via ``--stage``.

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

from .paths import (DatasetRoots, PipelineRoot, ObjectContext, normalize_shard,
                      resolve_blender_executable)
from .status import rebuild_manifest, manifest_summary, step_done
from .validators import apply_check
from . import scheduler as sched
from . import services_cfg as psvc

LOG = logging.getLogger("pipeline_v2")

ALL_STEPS = ("s1", "s2", "sq1", "s4", "s5", "s5b", "sq2", "s6", "s6b", "s7", "sq3")
GPU_STEPS = frozenset({"s5", "s6", "s6b"})


# ─────────────────── config + ctx resolution ─────────────────────────

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("data", {})
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
    roots = DatasetRoots.from_pipeline_cfg(cfg)

    if obj_ids:
        ids = obj_ids
    elif all_objs:
        # First try existing object dirs (resume mode); if empty, fall
        # back to discovering from input npz under mesh_root/<shard>/.
        ids = []
        if root.shard_dir(shard).is_dir():
            ids = [d.name for d in sorted(root.shard_dir(shard).iterdir())
                   if d.is_dir()]
        if not ids:
            mesh_shard = roots.mesh_root / shard
            if mesh_shard.is_dir():
                ids = sorted(p.stem for p in mesh_shard.glob("*.npz"))
    else:
        raise SystemExit("[CLI] one of --obj-ids or --all is required")

    ctxs: list[ObjectContext] = []
    for oid in ids:
        mesh_npz, image_npz = roots.input_npz_paths(shard, oid)
        ctxs.append(root.context(
            shard, oid,
            mesh_npz=mesh_npz,
            image_npz=image_npz,
        ))
    return ctxs


def slice_for_gpu(ctxs: list[ObjectContext], i: int, n: int) -> list[ObjectContext]:
    return [c for k, c in enumerate(ctxs) if k % n == i]


def _apply_obj_limit(ctxs: list[ObjectContext]) -> list[ObjectContext]:
    """Trim object list using env ``LIMIT`` (positive integer).

    Documented in ``docs/ARCH.md``. Applied after ``--gpu-shard`` slicing.
    """
    raw = os.environ.get("LIMIT", "").strip()
    if not raw:
        return ctxs
    try:
        n = int(raw)
    except ValueError:
        LOG.warning("LIMIT=%r is not an integer — ignoring", raw)
        return ctxs
    if n <= 0:
        return ctxs
    if len(ctxs) > n:
        LOG.info("LIMIT=%s → using first %d of %d objects", n, n, len(ctxs))
        return ctxs[:n]
    return ctxs


# ─────────────────── step dispatch ───────────────────────────────────

def run_step(
    step: str,
    ctxs: list[ObjectContext],
    cfg: dict,
    args: argparse.Namespace,
    post_object_fn=None,
) -> None:
    log = LOG.getChild(step)
    log.info("=" * 60)
    log.info("STEP %s on %d objects", step, len(ctxs))
    log.info("=" * 60)
    if not ctxs:
        return

    roots = DatasetRoots.from_pipeline_cfg(cfg)
    images_root = roots.images_root
    mesh_root = roots.mesh_root
    shard = ctxs[0].shard

    if step == "s1":
        from .s1_phase1_vlm import run_many_streaming
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None)
                else sched.vlm_urls_for(cfg))
        model = psvc.vlm_model_name(cfg)
        blender = resolve_blender_executable(cfg)
        n_pre = int((cfg.get("pipeline") or {}).get("prerender_workers", 8))
        asyncio.run(run_many_streaming(
            ctxs, blender=blender, vlm_urls=urls,
            vlm_model=model, n_prerender_workers=n_pre,
            force=args.force,
            post_object_fn=post_object_fn,
        ))

    elif step == "s2":
        from .s2_highlights import run_many
        blender = resolve_blender_executable(cfg)
        run_many(ctxs, blender=blender, force=args.force)

    elif step == "s4":
        from .s4_flux_2d import run as s4_run
        urls = ([u.strip() for u in args.flux_url.split(",") if u.strip()]
                if getattr(args, "flux_url", None)
                else sched.flux_urls_for(cfg))
        if not urls:
            raise SystemExit("[CONFIG] no FLUX urls (set pipeline.gpus or services.image_edit.base_urls)")
        s4_run(ctxs, edit_urls=urls,
               workers_per_server=psvc.image_edit_service(cfg).get("workers_per_server", 2),
               images_root=images_root, mesh_root=mesh_root, shard=shard,
               force=args.force, logger=log)

    elif step == "s5":
        from .s5_trellis_3d import run as s5_run
        s5_run(ctxs, cfg=cfg, images_root=images_root, mesh_root=mesh_root,
               shard=shard, force=args.force, logger=log)

    elif step == "s5b":
        from .s5b_deletion import run_mesh_delete
        # use_refiner=False keeps s5b CPU-only: just trimesh-direct
        # delete to before/after.ply. The proper after.npz is produced
        # by s6b later (Blender 40 views → DINOv2 → SLAT enc → SS enc).
        run_mesh_delete(ctxs, cfg=cfg, images_root=images_root,
                        mesh_root=mesh_root, shard=shard,
                        force=args.force, use_refiner=False, logger=log)

    elif step == "s6":
        from .s6_render_3d import run as s6_run
        ckpt = psvc.image_edit_service(cfg).get(
            "trellis_text_ckpt", "checkpoints/TRELLIS-text-xlarge")
        s6_run(ctxs, ckpt=ckpt, force=args.force, logger=log)

    elif step == "s6b":
        from .s5b_deletion import run_reencode
        blender = resolve_blender_executable(cfg)
        run_reencode(ctxs, cfg=cfg, blender_path=blender,
                     num_views=psvc.step_params_for(cfg, "s5").get("num_views", 40),
                     force=args.force, logger=log)

    elif step == "s7":
        from .s7_addition_backfill import run as s7_run
        s7_run(ctxs, force=args.force, logger=log)

    elif step == "sq1":
        from .sq1_qc_a import run as sq1_run
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        asyncio.run(sq1_run(ctxs, vlm_urls=urls,
                            vlm_model=psvc.vlm_model_name(cfg), force=args.force))

    elif step == "sq2":
        from .sq2_qc_c import run as sq2_run
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        asyncio.run(sq2_run(ctxs, vlm_urls=urls,
                            vlm_model=psvc.vlm_model_name(cfg), force=args.force))

    elif step == "sq3":
        from .sq3_qc_e import run as sq3_run
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        if not urls:
            raise SystemExit("[CONFIG] no VLM urls for sq3 (set pipeline.gpus or services.vlm.base_urls)")
        sq3_run(ctxs, vlm_url=urls[0], vlm_model=psvc.vlm_model_name(cfg),
                cfg=cfg, force=args.force)

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
        env.setdefault("ATTN_BACKEND", "flash_attn")
        # Children always receive --steps (single step) regardless of
        # whether the parent was launched with --stage, so no special-case.
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
    ctxs = _apply_obj_limit(ctxs)
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
    ap.add_argument("--steps", default=None,
                    help=f"comma list, any of: {','.join(ALL_STEPS)} "
                         "(mutually exclusive with --stage)")
    ap.add_argument("--stage", default=None,
                    help="run a single pipeline stage by name (e.g. A,C,D) "
                         "using pipeline.stages from the config")
    ap.add_argument("--gpus", default=None,
                    help="comma list e.g. 4,5,6,7. If omitted, falls back "
                         "to pipeline.gpus from the config when needed.")
    ap.add_argument("--vlm-url", dest="vlm_url", default=None,
                    help="Override VLM URL(s), comma-separated. Single URL "
                         "used by per-GPU workers in parallel mode.")
    ap.add_argument("--flux-url", dest="flux_url", default=None,
                    help="Override FLUX URL(s), comma-separated. Single URL "
                         "used by per-GPU workers in parallel mode.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-pending", action="store_true",
                    help="Print count of objects with pending work for the given stage/steps and exit")

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
    if args.gpu_shard:
        _i, _n = (int(x) for x in args.gpu_shard.split("/"))
        ctxs = slice_for_gpu(ctxs, _i, _n)
        LOG.info("gpu-shard %d/%d → %d objects", _i, _n, len(ctxs))
    ctxs = _apply_obj_limit(ctxs)
    LOG.info("root=%s shard=%s objects=%d", root.root, args.shard, len(ctxs))

    if args.dry_run:
        for c in ctxs:
            done = {s for s in ALL_STEPS if step_done(c, _step_to_status_key(s))}
            print(f"  {c.obj_id}  done={sorted(done) or '-'}")
        print(json.dumps(manifest_summary(root), indent=2))
        return

    # Resolve steps + use_gpus from --stage or --steps
    run_stage = args.stage

    if args.count_pending:
        # Resolve steps for the given stage/steps flags (same logic as below)
        _cp_steps: list[str] = []
        if run_stage:
            _cp_ph = sched.get_stage(cfg, run_stage)
            _cp_steps = list(_cp_ph.steps)
        elif args.steps:
            _cp_steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        else:
            _cp_steps = list(ALL_STEPS)
        _done_statuses = {"ok", "skip"}
        _pending = 0
        for _c in ctxs:
            _sf = _c.dir / "status.json"
            if not _sf.exists():
                _pending += 1
                continue
            _step_data = json.loads(_sf.read_text()).get("steps", {})
            if any(_step_data.get(_step_to_status_key(_s), {}).get("status")
                   not in _done_statuses for _s in _cp_steps):
                _pending += 1
        print(_pending)
        return

    phase_use_gpus = False
    if run_stage:
        ph = sched.get_stage(cfg, run_stage)
        steps = list(ph.steps)
        phase_use_gpus = ph.use_gpus
        LOG.info("stage %s (%s): steps=%s use_gpus=%s",
                 ph.name, ph.desc, steps, phase_use_gpus)
    elif args.steps:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    else:
        steps = list(ALL_STEPS)
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        raise SystemExit(f"unknown steps: {bad}")

    # Resolve gpu list: explicit --gpus first, then pipeline.gpus
    if args.gpus is None and (phase_use_gpus or run_stage):
        try:
            gpus_list = sched.gpus_for(cfg)
            args.gpus = ",".join(str(g) for g in gpus_list)
        except Exception:
            pass

    exit_rc = 0
    for step in steps:
        # look-ahead: inject sq1 as per-object post-hook when s1 and sq1
        # share the same VLM stage, so sq1 runs immediately after each
        # object's s1 completes without a VLM restart.
        _post_fn = None
        if step == "s1" and "sq1" in steps:
            from .sq1_qc_a import _process_one as _sq1_process_one
            _vlm_model = psvc.vlm_model_name(cfg)
            _force = args.force
            async def _sq1_hook(ctx, vlm_url,
                                _m=_vlm_model, _f=_force):
                await _sq1_process_one(ctx, vlm_url, _m, _f)
            _post_fn = _sq1_hook

        # GPU dispatch only when this step is GPU-bound AND the stage
        # asked for it (or the user passed --gpus explicitly).
        wants_dispatch = (step in GPU_STEPS
                          and args.gpus
                          and (phase_use_gpus or not run_stage)
                          and not args.single_gpu)
        if wants_dispatch:
            rc = dispatch_gpus(step, args.config, args)
            if rc != 0:
                LOG.error("[%s] dispatch_gpus returned rc=%d — aborting", step, rc)
                exit_rc = rc
        else:
            run_step(step, ctxs, cfg, args, post_object_fn=_post_fn)
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
        if exit_rc != 0:
            break

    LOG.info("\n%s", json.dumps(manifest_summary(root),
                                 indent=2, ensure_ascii=False))
    if exit_rc != 0:
        raise SystemExit(exit_rc)


_STATUS_KEYS = {
    "s1": "s1_phase1", "s2": "s2_highlights", "s4": "s4_flux_2d",
    "s5": "s5_trellis", "s5b": "s5b_del_mesh", "s6": "s6_render_3d",
    "s6b": "s6b_del_reencode", "s7": "s7_add_backfill",
    "sq1": "sq1_qc_A", "sq2": "sq2_qc_C", "sq3": "sq3_qc_E",
}


def _step_to_status_key(step: str) -> str:
    return _STATUS_KEYS.get(step, step)


if __name__ == "__main__":
    main()
