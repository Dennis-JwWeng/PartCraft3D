# Phase 2 — config hygiene (Blender resolution)

**Date:** 2026-04-10  
**Status:** Implemented (wave 1)

## Problem

`pipeline_v2/run.py` used a **machine-specific default** path for Blender when YAML omitted it, which breaks other hosts and contradicts “config-driven” rules.

## Resolution

- `resolve_blender_executable(cfg)` in `partcraft/pipeline_v2/paths.py` with explicit precedence (see `docs/smoke-pipeline.md`).
- Default fallback is `"blender"` on `PATH`, not a repo-embedded absolute path.

## Next waves (not done here)

- Optional: `load_config` normalizes `tools.*` for pipeline mode the same way as prerender.
- Unify `phase0.vlm_model` vs `VLM_CKPT` (launcher vs Python) — needs broader design.
