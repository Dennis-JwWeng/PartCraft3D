"""Step s4 — FLUX 2D image editing (object-centric, multi-server).

For every flux-needing edit (mod/scl/mat/glb) in each ``ObjectContext``
we call the existing ``scripts.run_2d_edit.process_one`` worker against
a pool of FLUX servers, writing into ``ctx.edits_2d_dir``::

    ctx.edits_2d_dir/
        {edit_id}_input.png
        {edit_id}_edited.png

Multi-server scheduling matches the legacy
``scripts.standalone.run_phase1_v2_2d_edit``: round-robin URL
assignment, ``workers = max(N, N * workers_per_server)``,
``ThreadPoolExecutor``.

Resume: an edit whose ``edits_2d/{edit_id}_edited.png`` already exists
is skipped unless ``force=True``. The per-object ``s4_flux_2d`` status
entry counts ``ok`` / ``fail`` over the just-processed batch.
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.run_2d_edit import process_one, check_edit_server  # noqa: E402

from .paths import ObjectContext
from .specs import EditSpec, iter_flux_specs
from .status import update_step, STATUS_OK, STATUS_FAIL, step_done


@dataclass
class Flux2DResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0




def _live_servers(urls: list[str]) -> list[str]:
    return [u for u in urls if check_edit_server(u)]


def run(
    ctxs: Iterable[ObjectContext],
    *,
    edit_urls: list[str],
    workers_per_server: int = 2,
    images_root: Path,
    mesh_root: Path,
    shard: str = "01",
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[Flux2DResult]:
    """Edit every flux spec across the given objects on the FLUX pool."""
    log = logger or logging.getLogger("pipeline_v2.s4")

    live = _live_servers(edit_urls)
    if not live:
        raise SystemExit(f"no live FLUX servers in {edit_urls}")
    workers = max(len(live), len(live) * workers_per_server)
    log.info(f"FLUX servers ({len(live)}): {live}  workers={workers}")

    # Dataset shim required by process_one (for view selection / image fetch).
    from partcraft.io.hy3d_loader import HY3DPartDataset
    dataset = HY3DPartDataset(str(images_root), str(mesh_root), [shard])

    # Flatten work: (ctx, spec)
    jobs: list[tuple[ObjectContext, EditSpec]] = []
    per_obj_results: dict[str, Flux2DResult] = {}
    for ctx in list(ctxs):
        per_obj_results[ctx.obj_id] = Flux2DResult(ctx.obj_id)
        ctx.edits_2d_dir.mkdir(parents=True, exist_ok=True)
        for spec in iter_flux_specs(ctx):
            out = ctx.edit_2d_output(spec.edit_id)
            if out.is_file() and not force:
                per_obj_results[ctx.obj_id].n_skip += 1
                continue
            jobs.append((ctx, spec))

    log.info(f"pending={len(jobs)} "
             f"skip={sum(r.n_skip for r in per_obj_results.values())}")
    if not jobs:
        for ctx in list(ctxs):
            r = per_obj_results[ctx.obj_id]
            update_step(ctx, "s4_flux_2d", status=STATUS_OK,
                        n=r.n_skip, n_fail=0, skipped=True)
        return list(per_obj_results.values())

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, (ctx, spec) in enumerate(jobs):
            url = live[i % len(live)]
            fut = pool.submit(
                process_one, spec, dataset, None,
                ctx.edits_2d_dir, "flux", log, edit_server_url=url,
            )
            futures[fut] = (ctx, spec)
        for k, fut in enumerate(as_completed(futures)):
            ctx, spec = futures[fut]
            try:
                rec = fut.result()
                ok = rec.get("status") == "success"
            except Exception as e:
                log.warning(f"  {spec.edit_id}: {e}")
                ok = False
            r = per_obj_results[ctx.obj_id]
            if ok:
                r.n_ok += 1
            else:
                r.n_fail += 1
            if (k + 1) % 5 == 0:
                done = sum(r.n_ok + r.n_fail for r in per_obj_results.values())
                log.info(f"  {done}/{len(jobs)}")

    dt = time.time() - t0
    log.info(f"done in {dt:.1f}s")

    # Per-object status
    for ctx in list(ctxs):
        r = per_obj_results[ctx.obj_id]
        # If we did nothing for this object (already done) and it had any
        # flux specs, we still want to mark s4 done if all outputs exist.
        all_present = all(
            ctx.edit_2d_output(s.edit_id).is_file()
            for s in iter_flux_specs(ctx)
        )
        update_step(
            ctx, "s4_flux_2d",
            status=STATUS_OK if (r.n_fail == 0 and all_present) else STATUS_FAIL,
            n_ok=r.n_ok, n_fail=r.n_fail, n_skip=r.n_skip,
        )
    return list(per_obj_results.values())


__all__ = ["Flux2DResult", "run"]
