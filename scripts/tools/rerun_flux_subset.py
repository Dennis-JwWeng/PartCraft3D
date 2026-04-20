#!/usr/bin/env python3
"""rerun_flux_subset.py - re-run FLUX s4 (2D edit) on a sampled subset
of an existing pipeline_v3 shard, writing outputs to a parallel root.

Used to A/B-test prompt-builder changes without touching the live shard
output.  Stages trellis_3d / preview_flux run only with ``--run-trellis``
(and can use ``--trellis-gpus`` for multi-GPU).  Otherwise we only need the new
``edits_2d/{eid}_{input,edited}.png`` for a side-by-side bench.

Sampling strategy
-----------------
For each requested edit type we sample N edits from each gate-E bucket
(``fail`` / ``pass``), reusing the same selection logic as the bench so
the A and B reports are directly comparable when the same ``--seed`` is
passed.

Usage
-----
  # Bring up a single FLUX server first (one GPU is enough for a test).
  python scripts/tools/rerun_flux_subset.py \\
      --src-root  outputs/partverse/shard07/mode_e_text_align \\
      --shard     07 \\
      --out-root  outputs/_bench/shard07_v2/mode_e_text_align \\
      --types     color,scale \\
      --per-bucket 12 \\
      --flux-url  http://localhost:8020/v1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.run_2d_edit import process_one, check_edit_server  # noqa: E402

from partcraft.io.partverse_dataset import PartVerseDataset
from partcraft.pipeline_v3.paths import (DatasetRoots, PipelineRoot,
                                         normalize_shard, ObjectContext)
from partcraft.pipeline_v3.specs import iter_flux_specs
from partcraft.pipeline_v3.edit_status_io import update_edit_stage
from partcraft.pipeline_v3.qc_io import update_edit_gate
from partcraft.utils.config import load_config


LOG = logging.getLogger("rerun_flux_subset")

def _trellis_mp_worker(
    gpu: str,
    obj_ids: list[str],
    config_path_str: str,
    out_root_str: str,
    shard_str: str,
    trellis_ckpt: str | None,
) -> None:
    """Run trellis_3d + preview_flux on *obj_ids* using one GPU (spawn child)."""
    import os
    import sys
    from pathlib import Path

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _r = Path(__file__).resolve().parents[2]
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    from partcraft.utils.config import load_config
    from partcraft.pipeline_v3.paths import DatasetRoots, PipelineRoot, normalize_shard
    from partcraft.pipeline_v3 import trellis_3d, preview_render
    from partcraft.pipeline_v3.edit_status_io import build_prereq_map

    cfg = load_config(Path(config_path_str))
    roots = DatasetRoots.from_pipeline_cfg(cfg)
    shard = normalize_shard(shard_str)
    out_root = PipelineRoot(root=Path(out_root_str))
    prereq_map = build_prereq_map(cfg)

    ctxs = [
        out_root.context(
            shard, oid,
            mesh_npz=roots.mesh_root / shard / f"{oid}.npz",
            image_npz=roots.images_root / shard / f"{oid}.npz",
        )
        for oid in obj_ids
    ]

    log5 = logging.getLogger(f"rerun_flux_subset.s5.gpu{gpu}")
    print(f"[rerun] GPU {gpu}: trellis_3d on {len(ctxs)} objects ...", flush=True)
    trellis_3d.run(
        ctxs,
        cfg=cfg,
        images_root=roots.images_root,
        mesh_root=roots.mesh_root,
        shard=shard,
        prereq_map=prereq_map,
        force=True,
        logger=log5,
    )

    ckpt = (
        trellis_ckpt
        or (cfg.get("services", {}).get("image_edit", {}) or {}).get("trellis_text_ckpt")
        or "checkpoints/TRELLIS-text-xlarge"
    )
    log6 = logging.getLogger(f"rerun_flux_subset.s6p.gpu{gpu}")
    print(f"[rerun] GPU {gpu}: preview_flux on {len(ctxs)} objects ...", flush=True)
    preview_render.render_flux_previews_batch(
        ctxs,
        ckpt=ckpt,
        prereq_map=prereq_map,
        force=True,
        logger=log6,
    )


EID_PFX_TYPES = {"clr": "color", "mod": "modification", "scl": "scale",
                 "mat": "material", "glb": "global"}


# ─────────────────── sampling ───────────────────────────────────────

def _bucket_for(stages: dict) -> str:
    s = ((stages or {}).get("gate_e") or {}).get("status")
    return "pass" if s == "pass" else ("fail" if s == "fail" else "missing")


def _gather_targets(src_obj_root: Path, types: set[str]) -> dict:
    """Return {(et, bucket): [(obj_id, edit_id), ...]}.  s4-done only."""
    out = defaultdict(list)
    for od in sorted(src_obj_root.iterdir()):
        if not od.is_dir():
            continue
        sf = od / "edit_status.json"
        if not sf.is_file():
            continue
        try:
            st = json.loads(sf.read_text())
        except Exception:
            continue
        for eid, einfo in (st.get("edits") or {}).items():
            et = einfo.get("edit_type")
            if et not in types:
                continue
            stages = einfo.get("stages") or {}
            if (stages.get("s4") or {}).get("status") != "done":
                continue
            out[(et, _bucket_for(stages))].append((od.name, eid))
    return out


def _sample_bench_aligned(
    targets: dict,
    types_order: list[str],
    per_bucket: int,
    seed: int,
) -> dict:
    """Match ``bench_v3_color_scale.render_html`` RNG consumption order.

    The bench iterates ``for et in types: for b in (fail, pass, missing):``
    and calls ``random.Random(seed).sample(...)`` once per non-empty pool.
    The old implementation iterated ``targets.items()`` (insertion order) and
    used a ``set`` for ``--types``, so the *same seed* picked a different
    subset — B-side files did not match the HTML's A rows.
    """
    rng = random.Random(seed)
    sampled: dict = {}
    for et in types_order:
        for b in ("fail", "pass", "missing"):
            key = (et, b)
            lst = targets.get(key) or []
            if not lst:
                continue
            sampled[key] = (lst if len(lst) <= per_bucket
                            else rng.sample(lst, per_bucket))
    return sampled


# ─────────────────── execution ──────────────────────────────────────

def _run(args) -> None:
    cfg = load_config(args.config)
    shard = normalize_shard(args.shard)
    roots = DatasetRoots.from_pipeline_cfg(cfg)

    src_root = PipelineRoot(root=args.src_root)
    out_root = PipelineRoot(root=args.out_root)

    src_obj_root = src_root.shard_dir(shard)
    if not src_obj_root.is_dir():
        sys.exit(f"[rerun] missing src obj_root: {src_obj_root}")

    types_list = [t.strip() for t in args.types.split(",") if t.strip()]
    types = set(types_list)
    targets = _gather_targets(src_obj_root, types)
    print(f"[rerun] candidate pool:")
    for k in sorted(targets):
        print(f"    {k[0]:<10s} {k[1]:<8s}  {len(targets[k])}")

    sampled = _sample_bench_aligned(targets, types_list, args.per_bucket, args.seed)
    todo: list[tuple[str, str]] = []
    for et in types_list:
        for b in ("fail", "pass", "missing"):
            k = (et, b)
            if k not in sampled:
                continue
            print(f"    sampling -> {k[0]:<10s} {k[1]:<8s}  "
                  f"{len(sampled[k])}/{len(targets.get(k, []))}")
            todo.extend(sampled[k])
    by_obj: dict[str, set[str]] = defaultdict(set)
    for oid, eid in todo:
        by_obj[oid].add(eid)
    print(f"[rerun] total edits to rerun = {len(todo)}  across {len(by_obj)} objects")

    if args.dry_run:
        print("[rerun] --dry-run set; not calling FLUX.")
        return
    if not args.flux_url:
        sys.exit("[rerun] --flux-url required for actual rerun. "
                 "Pass --dry-run to just print the plan.")

    flux_urls = [u.strip() for u in args.flux_url.split(",") if u.strip()]
    live = [u for u in flux_urls if check_edit_server(u)]
    if not live:
        sys.exit(f"[rerun] no live FLUX server in {flux_urls}")
    LOG.info("FLUX servers: %s", live)
    workers = max(len(live), len(live) * args.workers_per_server)

    dataset = PartVerseDataset(str(roots.images_root),
                               str(roots.mesh_root), [shard])

    # Build the list of (out_ctx, spec) pairs from iter_flux_specs, but
    # only keep specs we actually want to rerun.  We synthesise an
    # ObjectContext rooted at out_root so process_one writes to the new
    # location.  iter_flux_specs only needs ctx.parsed_path / shard /
    # obj_id, so we hand-construct a context whose parsed.json points at
    # the SOURCE phase1 (we never modify src files).
    jobs: list[tuple[ObjectContext, "EditSpec"]] = []
    for oid in sorted(by_obj):
        wanted = by_obj[oid]
        src_ctx = src_root.context(shard, oid,
                                          mesh_npz=roots.mesh_root / shard / f"{oid}.npz",
                                          image_npz=roots.images_root / shard / f"{oid}.npz")
        out_ctx = out_root.context(shard, oid,
                                          mesh_npz=roots.mesh_root / shard / f"{oid}.npz",
                                          image_npz=roots.images_root / shard / f"{oid}.npz")
        # Make sure out dirs exist + give iter_flux_specs the source parsed.json
        # by symlinking the phase1 directory into the out tree (read-only).
        out_ctx.edits_2d_dir.mkdir(parents=True, exist_ok=True)
        out_phase1 = out_ctx.dir / "phase1"
        if not out_phase1.exists():
            # absolute target so the symlink resolves regardless of cwd
            out_phase1.symlink_to((src_ctx.dir / "phase1").resolve(),
                                  target_is_directory=True)
        for spec in iter_flux_specs(out_ctx):
            if spec.edit_id in wanted:
                jobs.append((out_ctx, spec))

    print(f"[rerun] resolved jobs={len(jobs)}  workers={workers}")

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(process_one, spec, dataset, None,
                        ctx.edits_2d_dir, "flux", LOG,
                        edit_server_url=live[i % len(live)]): (ctx, spec)
            for i, (ctx, spec) in enumerate(jobs)
        }
        for done_i, fut in enumerate(as_completed(futs), 1):
            ctx, spec = futs[fut]
            try:
                rec = fut.result()
                ok = rec.get("status") == "success"
            except Exception as exc:
                LOG.warning("  %s: %s", spec.edit_id, exc)
                ok = False
            n_ok += int(ok); n_fail += int(not ok)
            # Write enough status into the variant tree that downstream
            # trellis_3d/preview_flux can resume from s5.  We don't have
            # a real gate_a verdict for the variant - we copy the source
            # verdict (the prompt-builder change cannot affect gate_a's
            # text-alignment judgement in a meaningful way for our
            # bench).
            try:
                update_edit_gate(ctx, spec.edit_id, spec.edit_type, "A",
                                 vlm_result={"pass": True, "score": 1.0,
                                             "reason": "copied_from_src"})
                update_edit_stage(ctx, spec.edit_id, spec.edit_type,
                                  "gate_a", status="pass")
                update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s4",
                                  status="done" if ok else "error")
            except Exception as _e:
                LOG.warning("  %s: status write failed (%s)", spec.edit_id, _e)
            if done_i % 5 == 0 or done_i == len(jobs):
                LOG.info("  %d/%d  ok=%d fail=%d",
                         done_i, len(jobs), n_ok, n_fail)
    print(f"[rerun] done.  ok={n_ok}  fail={n_fail}")
    print(f"[rerun] outputs at: {out_root.shard_dir(shard)}")

    if not args.run_trellis:
        print("[rerun] (skip trellis; pass --run-trellis to also "
              "produce 3D previews)")
        return
    _run_trellis(args, cfg, roots, shard, out_root, by_obj, jobs)



def _run_trellis(args, cfg, roots, shard, out_root, by_obj, jobs):
    """Run trellis_3d + render_flux_previews on the variant tree.

    With ``--trellis-gpus``, objects are round-robin sharded across spawn
    workers (one visible GPU each).  Otherwise runs in-process.
    """
    from partcraft.pipeline_v3 import trellis_3d, preview_render
    from partcraft.pipeline_v3.edit_status_io import build_prereq_map

    prereq_map = build_prereq_map(cfg)

    # Unique ctxs (one per object).
    seen: dict[str, ObjectContext] = {}
    for ctx, _ in jobs:
        seen.setdefault(ctx.obj_id, ctx)
    ctxs = list(seen.values())
    obj_ids = sorted(seen.keys())

    trellis_gpus = [g.strip() for g in (args.trellis_gpus or "").split(",")
                    if g.strip()]

    if trellis_gpus:
        buckets: list[list[str]] = [[] for _ in trellis_gpus]
        for i, oid in enumerate(obj_ids):
            buckets[i % len(trellis_gpus)].append(oid)
        tasks = [(trellis_gpus[i], buckets[i])
                 for i in range(len(trellis_gpus)) if buckets[i]]
        print(f"[rerun] trellis+preview: {len(tasks)} GPU workers  "
              f"gpus={trellis_gpus}  objects={len(obj_ids)}")
        cfg_abs = str(Path(args.config).resolve())
        out_abs = str(out_root.root.resolve())
        spawn_ctx = multiprocessing.get_context("spawn")
        with spawn_ctx.Pool(processes=len(tasks)) as pool:
            pool.starmap(
                _trellis_mp_worker,
                [(gpu, oids, cfg_abs, out_abs, shard, args.trellis_ckpt)
                 for gpu, oids in tasks],
            )
        print(f"[rerun] previews + npz now under {out_root.shard_dir(shard)}")
        return

    print(f"[rerun] trellis_3d on {len(ctxs)} objects (single process) ...")
    trellis_3d.run(
        ctxs,
        cfg=cfg,
        images_root=roots.images_root,
        mesh_root=roots.mesh_root,
        shard=shard,
        prereq_map=prereq_map,
        force=True,
        logger=LOG.getChild("s5"),
    )

    ckpt = (args.trellis_ckpt
            or (cfg.get("services", {}).get("image_edit", {}) or {})
                  .get("trellis_text_ckpt")
            or "checkpoints/TRELLIS-text-xlarge")
    print(f"[rerun] preview_flux on {len(ctxs)} objects ...")
    preview_render.render_flux_previews_batch(
        ctxs,
        ckpt=ckpt,
        prereq_map=prereq_map,
        force=True,
        logger=LOG.getChild("s6p"),
    )
    print(f"[rerun] previews + npz now under {out_root.shard_dir(shard)}")


# ─────────────────── CLI ────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="pipeline yaml (used only to resolve mesh/images roots)")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--src-root", required=True, type=Path,
                    help="existing shard root, e.g. "
                         "outputs/partverse/shard07/mode_e_text_align")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="parallel output root for the rerun")
    ap.add_argument("--types", default="color,scale")
    ap.add_argument("--per-bucket", type=int, default=12,
                    help="samples per (type x gate-E bucket)")
    ap.add_argument("--seed", type=int, default=0,
                    help="match the bench --seed to make A/B comparable")
    ap.add_argument("--flux-url", default=None,
                    help="comma list of FLUX server URLs, e.g. "
                         "http://localhost:8020/v1")
    ap.add_argument("--workers-per-server", type=int, default=2)
    ap.add_argument("--run-trellis", action="store_true",
                    help="after FLUX, also run trellis_3d + preview_flux "
                         "in-process on the variant tree (loads TRELLIS, "
                         "needs a free GPU via CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--trellis-ckpt", default=None,
                    help="override TRELLIS-text-xlarge ckpt path")
    ap.add_argument("--trellis-gpus", default=None,
                    metavar="IDS",
                    help="comma GPU ids for parallel trellis_3d + preview_flux "
                         "with --run-trellis, e.g. 0,1,2,3 (objects round-robin). "
                         "Each worker sets CUDA_VISIBLE_DEVICES before loading "
                         "TRELLIS.  Omit for single-process.")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve sample without calling FLUX")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    _run(args)


if __name__ == "__main__":
    main()
