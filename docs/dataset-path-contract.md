# Dataset I/O path contract (pipeline v2)

**Scope:** Explains how **input/output directory keys** in YAML relate to each other and to `partcraft.utils.config.load_config`. Complements `docs/ARCH.md` (architecture); this file focuses on **naming** so configs stay consistent across machines.

## Pipeline v2 runtime (`python -m partcraft.pipeline_v2.run`)

The orchestrator reads **these** keys under `data:` (see `partcraft/pipeline_v2/run.py`):

| Key | Role |
|-----|------|
| `images_root` | Root containing `images/<shard>/{obj_id}.npz` |
| `mesh_root` | Root containing `mesh/<shard>/{obj_id}.npz` |
| `slat_dir` | SLAT features (`slat/<shard>/…`) |
| `output_dir` | Run outputs, phase caches, logs (relative `cache_dir` / `log_dir` resolve here) |

## Synonyms merged at load time

Historical configs and `derive_dataset_subpaths` use **`image_npz_dir`** / **`mesh_npz_dir`** (same on-disk trees as `images_root` / `mesh_root`). `load_config` **syncs** them when only one side is set:

- If only `images_root` is set → `image_npz_dir` is set to the same resolved path.
- If only `image_npz_dir` is set → `images_root` is set to the same resolved path.
- Same for `mesh_root` ↔ `mesh_npz_dir`.

If **both** names are set and resolve to **different** paths, loading fails with `[CONFIG_ERROR] data.images_root vs data.image_npz_dir …`.

## Env overrides (still apply first)

- `PARTCRAFT_DATA_ROOT` → overrides `data.data_dir` (used with `derive_dataset_subpaths`).
- `PARTCRAFT_OUTPUT_ROOT` → overrides `data.output_dir`.

After env overrides, `derive_dataset_subpaths: true` can fill `image_npz_dir` / `mesh_npz_dir` / `slat_dir` under `data_dir`; the sync step then fills `images_root` / `mesh_root` when missing.

## Preprocessing / prerender (`for_prerender=True`)

Uses `paths.dataset_root` and `paths.images_npz_dir` / `paths.mesh_npz_dir` — see `ARCH.md` “数据预处理入口”. The merge rules above apply to **pipeline** configs; prerender follows `_apply_prerender_paths`.

## Logging

`[CONFIG_PATH]` lines include `data.images_root`, `data.mesh_root`, `data.image_npz_dir`, `data.mesh_npz_dir` when present so audits show one coherent set of roots.
## Code: `DatasetRoots`

`partcraft.pipeline_v2.paths.DatasetRoots` holds `images_root` / `mesh_root`, parses from `cfg["data"]`, and builds per-object `*.npz` paths via `input_npz_paths(shard, obj_id)`. Defaults: module constants `DEFAULT_IMAGES_ROOT` / `DEFAULT_MESH_ROOT`.

