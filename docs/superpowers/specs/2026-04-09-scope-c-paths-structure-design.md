# Scope C — pipeline path structure (design, first pass)

**Date:** 2026-04-09  
**Status:** Implemented (first pass)

## Problem

- Default dataset roots (`data/partverse/images`, `…/mesh`) and the
  `{mesh,images}_root/<shard>/<obj_id>.npz` pattern were duplicated in
  `pipeline_v2/run.py` (`resolve_ctxs` and `run_step`).

## Resolution

- **`DatasetRoots`** in `partcraft/pipeline_v2/paths.py`: `from_pipeline_cfg(cfg)`,
  `input_npz_paths(shard, obj_id)` → `(mesh_npz, image_npz)`.
- **`DEFAULT_IMAGES_ROOT` / `DEFAULT_MESH_ROOT`** module constants — single literals.
- **`run.py`** uses `DatasetRoots` in both `resolve_ctxs` and `run_step`.

## Later (optional)

- Thread `DatasetRoots` into s4/s5/s5b signatures instead of separate Path args.
- Document `DatasetRoots` in `dataset-path-contract.md` if external callers need it.
