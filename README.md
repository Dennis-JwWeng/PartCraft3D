# PartCraft3D

Part-segmented 3D assets to **(before, after, edit instruction)** training pairs. Active orchestration: **`partcraft.pipeline_v3`**.

---

## Entry

**Full shard run (recommended):** starts VLM / FLUX per stage, drives `pipeline.stages` from YAML, logs under `logs/v3_<tag>/`.

```bash
bash scripts/tools/run_pipeline_v3_shard.sh <tag> <config.yaml>
# Example: tag `shard08` → Python sees `--shard 08`
# Scope objects: OBJ_IDS_FILE=path/to/ids.txt bash scripts/tools/run_pipeline_v3_shard.sh ...
```

**Direct Python:**

```bash
python -m partcraft.pipeline_v3.run \
  --config <config.yaml> --shard <NN> --all \
  --steps gen_edits,gate_text_align,del_mesh,preview_del,flux_2d,trellis_3d,preview_flux,render_3d,gate_quality
# Or: --stage <name>  (must match pipeline.stages[].name in YAML)
# Scope: --obj-ids ID1 ID2 ... | --obj-ids-file ids.txt | --all
```

**Bench + tmux + obj list:** `scripts/tools/run_pipeline_v3_bench.sh` (see `--help`).

> **Single entrypoint:** all new orchestration capability goes into `partcraft/pipeline_v3/`. Do **not** add `scripts/run_*.py` or sibling pipelines — `pipeline_v2` was deleted on purpose.

---

## Data

Set paths in YAML under `data:` (no hardcoded roots in code).

| Key | Layout |
|-----|--------|
| `mesh_root` | `<mesh_root>/<shard>/<obj_id>.npz` |
| `images_root` | `<images_root>/<shard>/<obj_id>.npz` |
| `slat_dir` | `<slat_dir>/<shard>/<obj_id>_coords.pt` and `_feats.pt` (when Trellis runs) |
| `output_dir` | Run output root: `objects/<shard>/<obj_id>/` |

**Scope objects:** `OBJ_IDS_FILE=path/to/ids.txt` with `run_pipeline_v3_shard.sh`, or `--obj-ids` / `--obj-ids-file` on the Python CLI.

Naming/synonym rules (`images_root` ↔ `image_npz_dir`, env overrides, `derive_dataset_subpaths`): see **[`docs/dataset-path-contract.md`](docs/dataset-path-contract.md)**.

---

## Config

1. Copy **`configs/templates/pipeline_v3_bench.template.yaml`** → your file; fill `blender`, `ckpt_root`, `data.*`, `services.*`.

2. **`pipeline:`** `gpus`, `vlm_port_base` / `vlm_port_stride`, `flux_port_base` / `flux_port_stride`, workers (`prerender_workers`, `s6p_del_workers`, `trellis_workers_per_gpu`), concurrency knobs (`gate_a_concurrency`, `gate_a_per_obj_concurrency`), and **`stages`** (`name`, `steps`, `servers`: `vlm` | `flux` | `none`, `use_gpus`, `parallel_group`).

3. **`services:`** VLM model + URLs; image-edit / Trellis checkpoints.

4. **`qc:`** gate thresholds by edit type.

**Machine env:** `configs/machine/$(hostname).env` (conda envs, `VLM_CKPT`, `EDIT_CKPT`, `TRELLIS_CKPT_ROOT`, `DATA_DIR`, `OUTPUT_ROOT`, `ATTN_BACKEND`, `CUDA_HOME`, optional `VLM_MEM_FRAC` / `VLM_MAX_RUNNING` / `VLM_AUTO_RESTART` / `VLM_MIN_HEALTHY_S`). Override per-run with `MACHINE_ENV=path/to/file.env`. Existing host files (`node03.env`, `H200.env`, `wm1A800.env`, …) are good copy starting points; `local_symlink.env.example` shows how to alias one host to another.

---

## Input validation (before long runs)

Cheap checks that catch most misconfig before any GPU is touched:

```bash
# 1. Machine env + conda layout + required paths exist
bash scripts/tools/check_machine_env_for_pipeline.sh
bash scripts/tools/setup_pipeline_env.sh --check

# 2. Config loads cleanly (path sync + key validation)
python -c "from partcraft.utils.config import load_config; load_config('<config.yaml>'); print('[OK]')"

# 3. Dry-run object resolution (no steps executed)
python -m partcraft.pipeline_v3.run --config <config.yaml> --shard <NN> --all --dry-run

# 4. Pending count for a stage (e.g. how many objects still need Gate E)
python -m partcraft.pipeline_v3.run --config <config.yaml> --shard <NN> --all \
    --stage E --count-pending

# 5. Single-object debug subset (env LIMIT trims after --gpu-shard slicing)
LIMIT=1 python -m partcraft.pipeline_v3.run --config <config.yaml> --shard <NN> --all --stage A
```

If `[CONFIG_ERROR] data.images_root vs data.image_npz_dir …` fires, two roots resolve to different paths — pick one form (see `dataset-path-contract.md`).

---

## Pipeline → H3D_v1 dataset

Pipeline runs land per-shard under `outputs/.../objects/<NN>/<obj_id>/`. To promote one shard at a time into the **H3D_v1** dataset (`data/H3D_v1/`), run the three `pull_*` CLIs in order:

```bash
# 1) Deletions — filter by gate_a, encode any missing after.npz on GPU,
#    then hardlink the dataset bundle. The only GPU CLI of the three.
python -m scripts.cleaning.h3d_v1.pull_deletion \
    --pipeline-cfg configs/pipeline_v3_shard08.yaml --shard 08 \
    --dataset-root data/H3D_v1 \
    --device cuda:0 --blender /usr/local/bin/blender --workers 8

# 2) Flux edits (modification | scale | material | color | global) —
#    filter by gate_a AND gate_e, then hardlink. Pure IO.
python -m scripts.cleaning.h3d_v1.pull_flux \
    --pipeline-cfg configs/pipeline_v3_shard08.yaml --shard 08 \
    --dataset-root data/H3D_v1 --workers 8

# 3) Additions — backfill from each paired deletion already in dataset.
#    Run **after** pull_deletion. Pure IO.
python -m scripts.cleaning.h3d_v1.pull_addition \
    --pipeline-cfg configs/pipeline_v3_shard08.yaml --shard 08 \
    --dataset-root data/H3D_v1 --workers 8

# 4) Build the whole-dataset index (manifests/all.jsonl) + validate.
python -m scripts.cleaning.h3d_v1.build_h3d_v1_index \
    --dataset-root data/H3D_v1 --validate

# 5) (optional) Pack one shard into a single tarball for transfer/upload.
python -m scripts.cleaning.h3d_v1.pack_shard \
    --dataset-root data/H3D_v1 --shard 08 \
    --out releases/H3D_v1__shard08.tar
```

Add `--dry-run` to any `pull_*` CLI for a count-only preview, or `--limit N` / `--obj-id <uuid>` to scope work for testing. Multi-GPU = run multiple `pull_deletion` invocations with disjoint `--obj-id` allowlists and different `--device cuda:N`.

Layout, gate rules, asset-pool semantics, and concurrency model: **`docs/superpowers/specs/2026-04-19-h3d-v1-design.md`**. End-to-end runbook (incl. multi-machine sharding): **`docs/runbooks/h3d-v1-promote.md`**.

---

## More

- Architecture & data contracts: **[`docs/ARCH.md`](docs/ARCH.md)**
- Dataset path naming/synonyms: **[`docs/dataset-path-contract.md`](docs/dataset-path-contract.md)**
- Smoke checks (legacy v2 examples, concepts still apply): [`docs/smoke-pipeline.md`](docs/smoke-pipeline.md)
- Cross-machine restart runbook: [`docs/runbooks/restart-shard-K2.md`](docs/runbooks/restart-shard-K2.md)
- New machine onboarding: [`docs/new-machine-onboarding.md`](docs/new-machine-onboarding.md)
