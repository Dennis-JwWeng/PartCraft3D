# Scope B — dataset path semantics (design)

**Date:** 2026-04-09  
**Status:** Implemented (first pass)

## Problem

- `pipeline_v2` reads `data.images_root` / `data.mesh_root`.
- `load_config` / `derive_dataset_subpaths` historically emphasized `image_npz_dir` / `mesh_npz_dir`.
- Without an explicit contract, the two naming schemes drift; logs omitted `images_root` / `mesh_root`.

## Resolution

1. **Documentation:** `docs/dataset-path-contract.md` — single table + synonym rules.
2. **Code:** `partcraft.utils.config._sync_pipeline_v2_data_paths` runs after `_apply_data_roots_and_layout` when `not for_prerender`, merges synonyms, conflicts fail with `[CONFIG_ERROR]`.
3. **Logging:** `[CONFIG_PATH]` includes `data.images_root` and `data.mesh_root`.

## Out of scope (later)

- Deduplicate `paths.dataset_root` vs `data.data_dir` for pipeline-only flows.
- Broader refactor (Scope C) for centralizing path helpers.
