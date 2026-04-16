"""Step s4 — FLUX 2D image editing (edit-centric, multi-server).

Overview
--------
For every *flux-type* edit (modification / scale / material / global / color)
that has passed gate_a, this step calls the FLUX image-edit service to
produce a before→after image pair.  The edited PNG is later consumed by
``s5_trellis_3d`` as the 2D conditioning reference.

Edit-based control
------------------
Each edit is independently skip-gated via :func:`edit_status_io.edit_needs_step`,
which reads ``edit_status.json`` and checks:

  * gate_a ``"pass"``  — required prerequisite (unless no gate is active)
  * s4 stage absent or ``"error"``  — triggers (re-)processing
  * s4 stage ``"done"`` or ``"pass"`` — skip (already done)

Use ``force=True`` to override the own-stage check (gate_a is always enforced).

Input
-----
From the pipeline root / config:

``images_root/<shard>/<obj_id>.npz``
    Source NPZ for object views.  Used by ``HY3DPartDataset`` to load the
    best-view image for each edit.  The view is either ``spec.npz_view``
    (pre-selected by gate_a) or computed on-the-fly from part pixel counts.

``mesh_root/<shard>/<obj_id>.npz``
    Source NPZ for part masks.  Used alongside ``images_root`` by
    ``HY3DPartDataset`` to identify part pixels for view selection.

``phase1/parsed.json`` (via ``iter_flux_specs``)
    Parsed VLM edit list.  ``iter_flux_specs`` reads this and yields
    one :class:`~specs.EditSpec` per flux-type edit with the fields needed
    by ``process_one``:

    ==================  =================================================
    Field               Used for
    ==================  =================================================
    ``edit_id``         Output filename stem, state key
    ``edit_type``       Determines part selection strategy
    ``selected_part_ids`` Parts to highlight / use for view selection
    ``npz_view``        Pre-selected best view (>=0) or -1 to auto-select
    ``prompt``          Natural-language edit instruction -> FLUX prompt
    ``target_part_desc`` Before-state part description
    ``new_parts_desc``  After-state description
    ``part_labels``     Human-readable part name(s)
    ==================  =================================================

Output
------
Written to ``<output_dir>/objects/<shard>/<obj_id>/edits_2d/``:

``{edit_id}_input.png``   518 x 518 RGB -- the best-view source image
``{edit_id}_edited.png``  518 x 518 RGB -- FLUX-edited result

edit_status.json  (via :func:`edit_status_io.update_edit_stage`)::

    edits:
      <edit_id>:
        stages:
          s4: {status: "done"|"error", ts: "...", [reason: "..."]}

Object-level summary (via ``status.update_step``):
  ``steps.flux_2d: {status: "ok"|"fail", n_ok, n_fail, n_skip}``
  This is a per-run counter written after processing completes.
  The authoritative per-edit state is in ``edits.<edit_id>.stages.s4``.

Prerequisite chain
------------------
``gate_a (pass) -> s4 (done) -> s5 (trellis_3d)``

``deletion`` edits do NOT go through s4; they go directly to ``s5b_deletion``.

Resume behaviour
----------------
``edit_needs_step`` handles all skip logic.  Calling ``run()`` again on the
same objects is always safe: already-done edits are skipped, errored edits
are retried.

Server pool
-----------
``edit_urls`` is a list of FLUX server base URLs (e.g.
``["http://localhost:8020/v1", ...]``).  Only live servers (responding to
``check_edit_server``) are used.  Jobs are round-robin-distributed.
Worker count = ``len(live_servers) x workers_per_server``.
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
from .status import update_step, STATUS_OK, STATUS_FAIL
from .edit_status_io import edit_needs_step, update_edit_stage


@dataclass
class Flux2DResult:
    """Per-object tally returned by :func:`run`."""
    obj_id: str
    n_ok: int = 0     # edits successfully edited by FLUX
    n_fail: int = 0   # edits that errored (FLUX call failed)
    n_skip: int = 0   # edits skipped (done or gate_a failed)


def _live_servers(urls: list[str]) -> list[str]:
    """Return only the FLUX server URLs that are currently reachable."""
    return [u for u in urls if check_edit_server(u)]


def run(
    ctxs: Iterable[ObjectContext],
    *,
    edit_urls: list[str],
    workers_per_server: int = 2,
    images_root: Path,
    mesh_root: Path,
    shard: str = "01",
    prereq_map: dict[str, str | None],
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[Flux2DResult]:
    """Run FLUX 2D editing for all flux-type edits across the given objects.

    Parameters
    ----------
    ctxs:
        Object contexts to process.  Materialised internally to a list
        so the iterable can be traversed twice safely.
    edit_urls:
        FLUX server base URLs.  At least one must be reachable.
    workers_per_server:
        Number of concurrent threads per live FLUX server.
    images_root:
        Root directory of image NPZs (``<images_root>/<shard>/<obj_id>.npz``).
    mesh_root:
        Root directory of mesh NPZs (``<mesh_root>/<shard>/<obj_id>.npz``).
    shard:
        Two-digit shard string, e.g. ``"08"``.
    prereq_map:
        Gate prerequisite map from :func:`edit_status_io.build_prereq_map`.
        Each edit is checked with ``edit_needs_step(ctx, edit_id, "s4",
        prereq_map, force=force)`` before being queued.  Typically
        ``{"s4": "gate_a"}`` when gate_text_align is active.
    force:
        If True, re-process already-done edits (gate_a is still enforced).
    logger:
        Optional logger; defaults to ``logging.getLogger("pipeline_v3.s4")``.

    Returns
    -------
    list[Flux2DResult]
        One entry per input object context with ok/fail/skip counts.

    Files written per edit
    ----------------------
    ``edits_2d/{edit_id}_input.png``   -- 518x518 best-view source image
    ``edits_2d/{edit_id}_edited.png``  -- 518x518 FLUX-edited result

    State written per edit
    ----------------------
    ``edit_status.json`` -> ``edits.<edit_id>.stages.s4``
      status: ``"done"``  on success
      status: ``"error"`` on failure (with ``reason`` field)
    """
    log = logger or logging.getLogger("pipeline_v3.s4")

    # Materialise once — used in job build AND final status loop.
    ctxs = list(ctxs)

    live = _live_servers(edit_urls)
    if not live:
        raise SystemExit(f"no live FLUX servers in {edit_urls}")
    workers = max(len(live), len(live) * workers_per_server)
    log.info("FLUX servers (%d): %s  workers=%d", len(live), live, workers)

    # Dataset backed directly by the pipeline input NPZs.
    # PartVerseDataset reads images_root/<shard>/<obj_id>.npz (views) and
    # mesh_root/<shard>/<obj_id>.npz (parts).  No pyrender dependency.
    from partcraft.io.partverse_dataset import PartVerseDataset
    dataset = PartVerseDataset(str(images_root), str(mesh_root), [shard])

    # Flatten work: (ctx, spec) — only edits that pass edit_needs_step.
    # edit_needs_step checks gate_a prerequisite AND own-stage skip logic.
    jobs: list[tuple[ObjectContext, EditSpec]] = []
    per_obj_results: dict[str, Flux2DResult] = {
        ctx.obj_id: Flux2DResult(ctx.obj_id) for ctx in ctxs
    }
    for ctx in ctxs:
        ctx.edits_2d_dir.mkdir(parents=True, exist_ok=True)
        for spec in iter_flux_specs(ctx):
            if edit_needs_step(ctx, spec.edit_id, "s4", prereq_map, force=force):
                jobs.append((ctx, spec))
            else:
                per_obj_results[ctx.obj_id].n_skip += 1

    log.info("pending=%d  skip=%d",
             len(jobs), sum(r.n_skip for r in per_obj_results.values()))

    if not jobs:
        for ctx in ctxs:
            update_step(ctx, "s4_flux_2d", status=STATUS_OK,
                        n=per_obj_results[ctx.obj_id].n_skip, n_fail=0, skipped=True)
        return list(per_obj_results.values())

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_one, spec, dataset, None,
                ctx.edits_2d_dir, "flux", log, edit_server_url=live[i % len(live)],
            ): (ctx, spec)
            for i, (ctx, spec) in enumerate(jobs)
        }
        n_done = 0
        for fut in as_completed(futures):
            ctx, spec = futures[fut]
            err_reason: str | None = None
            try:
                rec = fut.result()
                ok = rec.get("status") == "success"
                if not ok:
                    err_reason = rec.get("error", "flux_failed")
            except Exception as exc:
                log.warning("  %s: %s", spec.edit_id, exc)
                ok = False
                err_reason = str(exc)[:200]

            r = per_obj_results[ctx.obj_id]
            if ok:
                r.n_ok += 1
                update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s4", status="done")
            else:
                r.n_fail += 1
                update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s4",
                                  status="error", reason=err_reason)
            n_done += 1
            if n_done % 5 == 0 or n_done == len(jobs):
                log.info("  %d/%d", n_done, len(jobs))

    log.info("done in %.1fs", time.time() - t0)

    # Per-object object-level summary written after all edits are processed.
    # The authoritative per-edit state lives in edit_status.json
    # edits.<edit_id>.stages.s4 (written above per edit as done/error).
    for ctx in ctxs:
        r = per_obj_results[ctx.obj_id]
        update_step(
            ctx, "s4_flux_2d",
            status=STATUS_OK if r.n_fail == 0 else STATUS_FAIL,
            n_ok=r.n_ok, n_fail=r.n_fail, n_skip=r.n_skip,
        )
    return list(per_obj_results.values())


__all__ = ["Flux2DResult", "run"]
