# Scope D — smoke & repeatable checks (design)

**Date:** 2026-04-09  
**Status:** Implemented (first pass)

## Problem

- `LIMIT` was documented in `ARCH.md` but not enforced by `pipeline_v2.run`.
- No single place describing smoke commands (config load, `--dry-run`, LIMIT).

## Resolution

1. **`LIMIT`**: `run.py` applies `_apply_obj_limit` after GPU shard slicing; logs when trimming.
2. **Docs**: `docs/smoke-pipeline.md` — ordered checks from host → config → dry-run → LIMIT → full run.
3. **Tests**: `tests/test_pipeline_limit.py` covers `_apply_obj_limit`.

## Out of scope

- CI that runs all phases on GPU (machine-dependent).
- Replacing `--obj-ids` with LIMIT-only UX (CLI unchanged).
