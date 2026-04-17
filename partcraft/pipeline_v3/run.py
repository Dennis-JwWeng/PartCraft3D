"""Pipeline v3 orchestrator + CLI — Mode E (text-driven part editing).

Mode E uses raw text captions + part lists as the sole input; no image
encoding is required at the prompt-generation stage.  The decided pipeline
for Mode E is::

    gen_edits → gate_text_align → del_mesh → preview_del → gate_quality

Typical invocation::

    python -m partcraft.pipeline_v3.run \\
        --config configs/pipeline_v3_shard08_test.yaml \\
        --shard 08 \\
        --steps gen_edits,gate_text_align,del_mesh,preview_del,gate_quality \\
        --all

Active steps (decided for Mode E):

    gen_edits        VLM generates edit instructions from text captions + part list.
                     Single process, multi-VLM-server fan-out.
    gate_text_align  Gate A: VLM checks that each instruction is unambiguous and
                     the target part is identifiable in the overview image.
                     Runs immediately after gen_edits (post_object_fn hook).
    del_mesh         CPU-only: KD-tree face-centroid masking on normalized GLB
                     → produces edits_3d/<edit_id>/after_new.glb for deletion edits.
    preview_del      Blender renders 5-view preview_{0..4}.png from after_new.glb.
                     These are consumed by gate_quality.
    gate_quality     Gate E: VLM compares before (image_npz) vs after (preview_del)
                     as a 2×5 collage and scores visual_quality / correct_region /
                     preserve_other.

Commented-out steps (not yet decided for Mode E, kept for reference):

    flux_2d          FLUX 2D image edit for mod/scl/mat/glb edits.
    trellis_3d       Trellis 3D latent edit (GPU).  Requires flux_2d output.
    preview_flux     5-view preview decoded from Trellis after.npz (GPU).
    render_3d        Full 40-view 3D render (GPU).
    reencode_del     GPU re-encode: Blender 40-view → DINOv2 → SLAT → after.npz.

Orchestration:
* loads YAML config + CLI overrides;
* resolves object list (CLI ids OR all dirs under ``objects/<shard>/``);
* for each requested step, dispatches the matching runner;
* after each step, calls :func:`status.rebuild_manifest` so the global
  manifest stays in sync.

For GPU-bound steps the parent process spawns one child per GPU via
``subprocess.Popen`` with ``CUDA_VISIBLE_DEVICES`` set; each child
re-invokes this CLI with ``--single-gpu --gpu-shard i/n``.
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
from .status import rebuild_manifest, manifest_summary, step_done, load_status
from .edit_status_io import build_prereq_map
from .validators import apply_check
from . import scheduler as sched
from . import services_cfg as psvc

LOG = logging.getLogger("pipeline_v3")

# ── Active steps (decided for Mode E) ────────────────────────────────
ALL_STEPS = (
    # ── Text generation + Gate A ──────────────────────────────────────
    "gen_edits",       # Phase 1: VLM generates edit instructions from text
    "gate_text_align", # Gate A:  text-image alignment validation
    # ── Deletion branch (CPU) ────────────────────────────────────────
    "del_mesh",        # Step 5b: CPU deletion mesh masking → after_new.glb
    "preview_del",     # Step 6p: Blender 5-view preview for deletions
    # ── Flux branch (mod/scl/mat/glb/color, GPU) ─────────────────────
    "flux_2d",         # Step 4:  FLUX 2D image edit → edits_2d/_edited.png
    "trellis_3d",      # Step 5:  Trellis 3D latent edit → after.npz  (GPU)
    "preview_flux",    # Step 6p: Trellis-decoded 5-view preview        (GPU)
    "render_3d",       # Step 6:  Full 40-view 3D render (optional)     (GPU)
    # ── Final quality gate (all types) ───────────────────────────────
    "gate_quality",    # Gate E:  final VLM visual quality check
    # ── Inactive ─────────────────────────────────────────────────────
    # "reencode_del",  # Step 6b: Blender 40-view → SLAT enc → after.npz (GPU)
)

# Steps that require multi-GPU subprocess dispatch via dispatch_gpus().
# preview_del uses Blender (CPU-bound per call, parallelised by n_workers)
# so it is NOT in GPU_STEPS.
GPU_STEPS: frozenset[str] = frozenset({"trellis_3d", "preview_flux", "render_3d"})


# ─────────────────── config + ctx resolution ─────────────────────────

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("data", {})
    return cfg


def resolve_root(cfg: dict) -> PipelineRoot:
    out = cfg["data"].get("output_dir")
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
        # Always use mesh_root as the authoritative full set of objects for
        # this shard, then union with any existing output dirs so that
        # in-progress objects are never dropped on resume.
        mesh_shard = roots.mesh_root / shard
        mesh_ids = (
            {p.stem for p in mesh_shard.glob("*.npz")}
            if mesh_shard.is_dir() else set()
        )
        existing_ids = (
            {d.name for d in root.shard_dir(shard).iterdir() if d.is_dir()}
            if root.shard_dir(shard).is_dir() else set()
        )
        ids = sorted(mesh_ids | existing_ids)
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


# ─────────────────── input pre-flight check ──────────────────────────

def check_inputs(
    ctxs: list[ObjectContext],
    roots: DatasetRoots,
    shard: str,
) -> None:
    """Verify all required input files exist before any step runs.

    Checks per object:
      * mesh_npz  — {mesh_root}/{shard}/{obj_id}.npz
      * image_npz — {images_root}/{shard}/{obj_id}.npz
      * slat coords — {slat_dir}/{shard}/{obj_id}_coords.pt  (if slat_dir configured)
      * slat feats  — {slat_dir}/{shard}/{obj_id}_feats.pt   (if slat_dir configured)

    Prints a summary table and raises SystemExit if any file is missing.
    """
    log = LOG.getChild("check_inputs")
    log.info("=" * 60)
    log.info("INPUT PRE-FLIGHT CHECK  shard=%s  objects=%d", shard, len(ctxs))
    log.info("  mesh_root   : %s", roots.mesh_root)
    log.info("  images_root : %s", roots.images_root)
    log.info("  slat_dir    : %s", roots.slat_dir or "(not configured)")
    log.info("=" * 60)

    missing_by_obj: dict[str, list[str]] = {}

    for ctx in ctxs:
        missing: list[str] = []

        if ctx.mesh_npz is None or not ctx.mesh_npz.is_file():
            missing.append(f"mesh_npz ({ctx.mesh_npz})")

        if ctx.image_npz is None or not ctx.image_npz.is_file():
            missing.append(f"image_npz ({ctx.image_npz})")

        if roots.slat_dir is not None:
            try:
                coords_pt, feats_pt = roots.slat_pt_paths(shard, ctx.obj_id)
                if not coords_pt.is_file():
                    missing.append(f"slat_coords ({coords_pt})")
                if not feats_pt.is_file():
                    missing.append(f"slat_feats ({feats_pt})")
            except Exception as exc:
                missing.append(f"slat_error ({exc})")

        if missing:
            missing_by_obj[ctx.obj_id] = missing

    n_ok = len(ctxs) - len(missing_by_obj)
    log.info("INPUT CHECK  ok=%d/%d  missing_objects=%d",
             n_ok, len(ctxs), len(missing_by_obj))

    if missing_by_obj:
        for oid, ms in missing_by_obj.items():
            for m in ms:
                log.error("  MISSING  %s  %s", oid, m)
        raise SystemExit(
            f"[INPUT CHECK FAILED] {len(missing_by_obj)}/{len(ctxs)} objects "
            f"have missing input files — aborting before any step runs."
        )

    log.info("INPUT CHECK PASSED — all %d objects have required inputs", len(ctxs))


# ─────────────────── step dispatch ───────────────────────────────────

def run_step(
    step: str,
    ctxs: list[ObjectContext],
    cfg: dict,
    args: argparse.Namespace,
    post_object_fn=None,
) -> None:
    """Dispatch one pipeline step across the given object contexts.

    Each branch is self-contained: it imports its runner, resolves its
    service URLs / paths from cfg, and delegates to the runner's ``run()``
    or ``gen_edits_streaming()`` function.

    The ``post_object_fn`` hook (only used by ``gen_edits``) allows
    ``gate_text_align`` to run inline immediately after each object's
    Phase 1 VLM call completes, so Gate A never lags behind.
    """
    log = LOG.getChild(step)
    log.info("=" * 60)
    log.info("STEP %s on %d objects", step, len(ctxs))
    log.info("=" * 60)
    if not ctxs:
        return

    prereq_map = build_prereq_map(cfg)
    roots = DatasetRoots.from_pipeline_cfg(cfg)
    images_root = roots.images_root
    mesh_root = roots.mesh_root
    shard = ctxs[0].shard

    # ── gen_edits ─────────────────────────────────────────────────────
    # Phase 1: VLM reads text captions + part list and proposes a batch
    # of edit instructions (deletion / modification / material / etc.).
    # Output: phase1/parsed.json + phase1/overview.png per object.
    if step == "gen_edits":
        from .gen_edits import gen_edits_streaming
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None)
                else sched.vlm_urls_for(cfg))
        model = psvc.vlm_model_name(cfg)
        blender = resolve_blender_executable(cfg)
        n_pre = int((cfg.get("pipeline") or {}).get("prerender_workers", 8))
        asyncio.run(gen_edits_streaming(
            ctxs, blender=blender, vlm_urls=urls,
            vlm_model=model, n_prerender_workers=n_pre,
            force=args.force,
            post_object_fn=post_object_fn,
            anno_dir=roots.anno_dir,
        ))

    # ── gate_text_align ───────────────────────────────────────────────
    # Gate A: for each proposed edit, the VLM judges whether the instruction
    # is unambiguous, the target part is identifiable in the 5×2 overview
    # image, and the edit type is consistent with the prompt.
    # Writes gate_a: pass/fail to edit_status.json for every edit.
    elif step == "gate_text_align":
        from .vlm_core import run_gate_text_align
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        if not urls:
            raise SystemExit("[CONFIG] no VLM urls for gate_text_align")
        asyncio.run(run_gate_text_align(
            ctxs, vlm_urls=urls,
            vlm_model=psvc.vlm_model_name(cfg), force=args.force
        ))

    # ── del_mesh ──────────────────────────────────────────────────────
    # CPU-only deletion step: for every deletion edit that passed gate_a,
    # removes the target faces from the normalized GLB using KD-tree
    # face-centroid matching.  Produces edits_3d/<edit_id>/after_new.glb.
    # Also backfills a paired add_* meta.json for the inverse addition.
    elif step == "del_mesh":
        from .mesh_deletion import run_deletion_batch
        run_deletion_batch(ctxs, cfg=cfg, images_root=images_root,
                        mesh_root=mesh_root, shard=shard,
                        normalized_glb_dir=roots.normalized_glb_dir,
                        anno_dir=roots.anno_dir,
                        prereq_map=prereq_map,
                        force=args.force, logger=log)

    # ── preview_del ───────────────────────────────────────────────────
    # Blender renders 5 views (VIEW_INDICES cameras) of after_new.glb for
    # every deletion edit.  Output: edits_3d/<edit_id>/preview_{0..4}.png.
    # These previews are the "after" row of the Gate E collage.
    elif step == "preview_del":
        from .preview_render import render_del_previews_batch as s6p_del_run
        blender = resolve_blender_executable(cfg)
        try:
            _gpus = sched.gpus_for(cfg)
        except (ValueError, KeyError):
            _gpus = []
        _n_workers = int((cfg.get("pipeline") or {}).get("s6p_del_workers", 4))
        s6p_del_run(ctxs, blender=blender, n_workers=_n_workers,
                    gpus=_gpus if _gpus else None,
                    prereq_map=prereq_map,
                    force=args.force, logger=log)

    # ── gate_quality ──────────────────────────────────────────────────
    # Gate E: the VLM receives a 2-row × 5-col collage (top = before from
    # image_npz, bottom = after from preview_del) and scores the edit on
    # edit_executed / visual_quality / correct_region / preserve_other.
    # Writes gate_e: pass/fail to edit_status.json.
    elif step == "gate_quality":
        from .vlm_core import run_gate_quality as sq3_run
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        if not urls:
            raise SystemExit("[CONFIG] no VLM urls for gate_quality "
                             "(set pipeline.gpus or services.vlm.base_urls)")
        asyncio.run(sq3_run(ctxs, vlm_urls=urls, vlm_model=psvc.vlm_model_name(cfg),
                            cfg=cfg, force=args.force))

    # ── flux_2d ───────────────────────────────────────────────────────────
    # FLUX 2D image edit for mod/scl/mat/glb/color edits.
    # Calls the FLUX image-edit service pool (HTTP endpoint); writes
    # edits_2d/<edit_id>_input.png + edits_2d/<edit_id>_edited.png.
    # Requires gate_a == pass. Use --flux-url or services.image_edit.base_urls.
    elif step == "flux_2d":
        from .flux_2d import run as s4_run
        urls = ([u.strip() for u in args.flux_url.split(",") if u.strip()]
                if getattr(args, "flux_url", None)
                else sched.flux_urls_for(cfg))
        if not urls:
            raise SystemExit("[CONFIG] no FLUX urls (set services.image_edit.base_urls or --flux-url)")
        s4_run(ctxs, edit_urls=urls,
               workers_per_server=psvc.image_edit_service(cfg).get("workers_per_server", 2),
               images_root=images_root, mesh_root=mesh_root, shard=shard,
               prereq_map=prereq_map, force=args.force, logger=log)

    # ── trellis_3d ───────────────────────────────────────────────────────
    # Trellis 3D latent edit (GPU). Consumes the FLUX-edited 2D image and
    # the original mesh.npz SLAT to produce edits_3d/<edit_id>/after.npz.
    # Multi-GPU via dispatch_gpus(). Requires gate_a == pass.
    elif step == "trellis_3d":
        from .trellis_3d import run as s5_run
        s5_run(ctxs, cfg=cfg, images_root=images_root, mesh_root=mesh_root,
               shard=shard, prereq_map=prereq_map, force=args.force, logger=log)

    # ── preview_flux ─────────────────────────────────────────────────────
    # Decode Trellis after.npz and render 5 views (GPU).
    # Output: edits_3d/<edit_id>/preview_{0..4}.png for flux edits.
    # Consumed by gate_quality as the "after" collage row.
    elif step == "preview_flux":
        from .preview_render import render_flux_previews_batch as s6p_flux_run
        ckpt = psvc.image_edit_service(cfg).get(
            "trellis_text_ckpt", "checkpoints/TRELLIS-text-xlarge")
        s6p_flux_run(ctxs, ckpt=ckpt, prereq_map=prereq_map,
                     force=args.force, logger=log)

    # ── render_3d ────────────────────────────────────────────────────────
    # Full 40-view 3D render (GPU) for final report generation.
    # Requires gate_e == pass. Optional for Mode E; can be omitted if
    # only the 5-view preview is needed.
    elif step == "render_3d":
        from .render_3d import run as s6_run
        ckpt = psvc.image_edit_service(cfg).get(
            "trellis_text_ckpt", "checkpoints/TRELLIS-text-xlarge")
        s6_run(ctxs, ckpt=ckpt, prereq_map=prereq_map,
               force=args.force, logger=log)

    # ── reencode_del (inactive) ───────────────────────────────────────────
    # GPU re-encode for deletion edits: Blender 40-view → DINOv2 →
    # SLAT encoder → SS encoder → after.npz.
    # Requires gate_e == pass. Not yet in default flow; uncomment to enable.
    # elif step == "reencode_del":
    #     from .mesh_deletion import link_slat_assets_batch
    #     blender = resolve_blender_executable(cfg)
    #     link_slat_assets_batch(ctxs, cfg=cfg, blender_path=blender,
    #                  num_views=psvc.step_params_for(cfg, "s5").get("num_views", 40),
    #                  prereq_map=prereq_map, force=args.force, logger=log)

    else:
        raise SystemExit(f"unknown step: {step!r}  (active steps: {', '.join(ALL_STEPS)})")


# ─────────────────── multi-GPU dispatch ──────────────────────────────

def dispatch_gpus(
    step: str,
    cfg_path: Path,
    args: argparse.Namespace,
) -> int:
    """Spawn ``K * N`` children where ``N = #GPUs`` and ``K`` is the
    per-GPU worker count for this step.

    K is read from ``services.image_edit.trellis_workers_per_gpu`` (env
    override ``TRELLIS_WORKERS_PER_GPU``). Only ``trellis_3d`` honors
    K > 1 — the other GPU steps (``preview_flux``, ``render_3d``) keep
    K = 1 because they are compute-bound, not I/O-bound.

    Each child receives ``CUDA_VISIBLE_DEVICES=<gpu>`` and a global
    ``--gpu-shard k/(K*N)`` so ``slice_for_gpu`` partitions the edit
    list across all workers without overlap.
    """
    gpus = [g.strip() for g in (args.gpus or "").split(",") if g.strip()]
    n = len(gpus)
    if n == 0:
        return run_single_gpu(step, cfg_path, args)

    k = 1
    if step == "trellis_3d":
        cfg = load_config(cfg_path)
        k = psvc.trellis_workers_per_gpu(cfg)

    if n == 1 and k == 1:
        return run_single_gpu(step, cfg_path, args)

    total = n * k
    LOG.info(
        "[%s] dispatching: gpus=%s workers_per_gpu=%d total_workers=%d",
        step, gpus, k, total,
    )

    procs: list[tuple[str, int, subprocess.Popen]] = []
    for gpu_idx, gpu in enumerate(gpus):
        for w in range(k):
            shard_id = gpu_idx * k + w
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("ATTN_BACKEND", "flash_attn")
            cmd = [
                sys.executable, "-m", "partcraft.pipeline_v3.run",
                "--config", str(cfg_path),
                "--shard", args.shard,
                "--steps", step,
                "--single-gpu",
                "--gpu-shard", f"{shard_id}/{total}",
            ]
            if args.obj_ids:
                cmd += ["--obj-ids", *args.obj_ids]
            elif getattr(args, "obj_ids_file", None) and args.obj_ids_file:
                cmd += ["--obj-ids-file", str(args.obj_ids_file)]
            if args.all:
                cmd += ["--all"]
            if args.force:
                cmd += ["--force"]
            LOG.info(
                "  GPU %s worker %d/%d (shard %d/%d): %s",
                gpu, w + 1, k, shard_id, total, " ".join(cmd[-6:]),
            )
            procs.append((gpu, w, subprocess.Popen(cmd, env=env)))

    rc = 0
    for gpu, w, p in procs:
        r = p.wait()
        LOG.info("[%s] GPU %s worker %d exit=%d", step, gpu, w, r)
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
    # Resolve --obj-ids-file into a list before passing to resolve_ctxs
    _obj_ids = args.obj_ids
    if getattr(args, "obj_ids_file", None) and args.obj_ids_file:
        _obj_ids = [
            line.strip() for line in args.obj_ids_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        LOG.info("--obj-ids-file: loaded %d obj_ids from %s", len(_obj_ids), args.obj_ids_file)
    ctxs = resolve_ctxs(root, cfg, shard=args.shard,
                        obj_ids=_obj_ids, all_objs=args.all)
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
    ap = argparse.ArgumentParser(prog="pipeline_v3")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--shard", default="01")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--obj-ids", nargs="+")
    grp.add_argument("--obj-ids-file", type=Path, metavar="FILE",
                     help="text file with one obj_id per line (# comments ok)")
    grp.add_argument("--all", action="store_true")
    ap.add_argument("--steps", default=None,
                    help=f"comma list of active steps: {','.join(ALL_STEPS)} "
                         "(mutually exclusive with --stage)")
    ap.add_argument("--stage", default=None,
                    help="run a pipeline stage by name (e.g. A,E) "
                         "using pipeline.stages from the config")
    ap.add_argument("--gpus", default=None,
                    help="comma list e.g. 4,5,6,7. If omitted, falls back "
                         "to pipeline.gpus from the config when needed.")
    ap.add_argument("--vlm-url", dest="vlm_url", default=None,
                    help="Override VLM URL(s), comma-separated.")
    ap.add_argument("--flux-url", dest="flux_url", default=None,
                    help="Override FLUX URL(s), comma-separated (reserved, not active).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-pending", action="store_true",
                    help="Print count of objects with pending work for the "
                         "given stage/steps and exit")

    # Internal: child GPU workers set these.
    ap.add_argument("--single-gpu", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--gpu-shard", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    cfg = load_config(args.config)
    root = resolve_root(cfg)
    # Resolve --obj-ids-file into a list before passing to resolve_ctxs
    _obj_ids = args.obj_ids
    if getattr(args, "obj_ids_file", None) and args.obj_ids_file:
        _obj_ids = [
            line.strip() for line in args.obj_ids_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        LOG.info("--obj-ids-file: loaded %d obj_ids from %s", len(_obj_ids), args.obj_ids_file)
    ctxs = resolve_ctxs(root, cfg, shard=args.shard,
                        obj_ids=_obj_ids, all_objs=args.all)
    if args.gpu_shard:
        _i, _n = (int(x) for x in args.gpu_shard.split("/"))
        ctxs = slice_for_gpu(ctxs, _i, _n)
        LOG.info("gpu-shard %d/%d → %d objects", _i, _n, len(ctxs))
    ctxs = _apply_obj_limit(ctxs)
    LOG.info("root=%s shard=%s objects=%d", root.root, args.shard, len(ctxs))

    # Run input pre-flight check unless we are a GPU child worker
    # (children share the same inputs already validated by the parent).
    if not args.single_gpu:
        roots_for_check = DatasetRoots.from_pipeline_cfg(cfg)
        check_inputs(ctxs, roots_for_check, args.shard)

    if args.dry_run:
        for c in ctxs:
            done = {s for s in ALL_STEPS if step_done(c, _step_to_status_key(s))}
            print(f"  {c.obj_id}  done={sorted(done) or '-'}")
        print(json.dumps(manifest_summary(root), indent=2))
        return

    # Resolve steps + use_gpus from --stage or --steps
    run_stage = args.stage

    if args.count_pending:
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
            _step_data = (load_status(_c).get("steps") or {})
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
        raise SystemExit(f"unknown steps: {bad}  (active steps: {', '.join(ALL_STEPS)})")

    # Resolve GPU list: explicit --gpus first, then pipeline.gpus from config.
    if args.gpus is None and (phase_use_gpus or run_stage):
        try:
            gpus_list = sched.gpus_for(cfg)
            args.gpus = ",".join(str(g) for g in gpus_list)
        except Exception:
            pass

    exit_rc = 0
    for step in steps:
        _post_fn = None
        # Inline Gate A hook: when gen_edits and gate_text_align run together,
        # attach gate_text_align as a per-object callback so it executes
        # immediately after each object's VLM call (no extra pass needed).
        # Inline hook: when gen_edits and gate_text_align run together,
        # attach gate_text_align as a per-object callback so it fires
        # immediately after each object's Phase 1 VLM call completes.
        if step == "gen_edits" and "gate_text_align" in steps:
            from .vlm_core import run_gate_text_align as _run_gta
            _vlm_model = psvc.vlm_model_name(cfg)
            _force = args.force
            _gta_urls = sched.vlm_urls_for(cfg)
            async def _gta_hook(ctx, vlm_url,
                                _m=_vlm_model, _f=_force, _urls=_gta_urls):
                await _run_gta([ctx], vlm_urls=[vlm_url], vlm_model=_m,
                               force=_f, concurrency=1)
            _post_fn = _gta_hook

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

        # Post-step validation: rewrite status.json to reflect filesystem reality.
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


# ── Step-name → status.json key mapping ──────────────────────────────
# The status keys written to edit_status.json are intentionally stable
# (backwards compatible with existing data); only the step identifiers
# used by the CLI and run_step() have been renamed to functional names.
_STATUS_KEYS: dict[str, str] = {
    # Text generation + gates
    "gen_edits":       "s1_phase1",
    "gate_text_align": "sq1_qc_A",
    "gate_quality":    "sq3_qc_E",
    # Deletion branch
    "del_mesh":        "s5b_del_mesh",
    "preview_del":     "s6p_del",
    # Flux branch
    "flux_2d":         "s4_flux_2d",
    "trellis_3d":      "s5_trellis",
    "preview_flux":    "s6p_flux",
    "render_3d":       "s6_render_3d",
    # Inactive
    # "reencode_del":  "s6b_del_reencode",
}


def _step_to_status_key(step: str) -> str:
    """Return the status.json key for a given step identifier."""
    return _STATUS_KEYS.get(step, step)


if __name__ == "__main__":
    main()
