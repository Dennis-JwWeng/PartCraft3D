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

## Pipeline → unified edit dataset

Pipeline runs land per-shard under `outputs/.../objects/<NN>/<obj_id>/`. To consolidate cleaned passes from one or more runs (mix of v2 + v3) into a versioned dataset (`data/partverse_edit_v1/`), use `partcraft.cleaning.v1` + the `scripts/cleaning/` CLIs.

```bash
# (v2-source only) backfill Gate E / Gate A on a v2 run using the v3 judges
python -m scripts.cleaning.run_gate_quality_on_v2 \
    --v2-run outputs/partverse/pipeline_v2_shard05 \
    --v3-config <v3_config.yaml> --shards 05 --concurrency 4
python -m scripts.cleaning.run_gate_text_align_on_v2 \
    --v2-run outputs/partverse/pipeline_v2_shard05 \
    --v3-config <v3_config.yaml> --shards 05 --force

# Promote rule-passing edits from runs into v1 (hardlink/symlink/copy)
python -m scripts.cleaning.promote_to_v1 \
    --source-runs outputs/partverse/pipeline_v2_shard05 \
                  outputs/partverse/shard08/mode_e_text_align \
    --rules configs/cleaning/promote_v1.yaml

# Encode pending deletion latents (after_new.glb → after.npz with SLAT + DINOv2)
python -m scripts.cleaning.encode_del_latent --rules configs/cleaning/promote_v1.yaml --num-gpus 4

# Rebuild data/partverse_edit_v1/index/{objects,edits}.jsonl
python -m scripts.cleaning.rebuild_v1_index --rules configs/cleaning/promote_v1.yaml
```

Promotion rules (required passes, allowed edit types, `link_mode`, `before_assets.*`, `source_blocklist`) live in `configs/cleaning/promote_v1.yaml`. End-to-end smoke walkthrough: **`docs/superpowers/runbooks/2026-04-19-edit-data-v1-smoke.md`**. Design / spec: `docs/superpowers/specs/2026-04-19-edit-data-v1-design.md`.

---

## More

- Architecture & data contracts: **[`docs/ARCH.md`](docs/ARCH.md)**
- Dataset path naming/synonyms: **[`docs/dataset-path-contract.md`](docs/dataset-path-contract.md)**
- Smoke checks (legacy v2 examples, concepts still apply): [`docs/smoke-pipeline.md`](docs/smoke-pipeline.md)
- Cross-machine restart runbook: [`docs/runbooks/restart-shard-K2.md`](docs/runbooks/restart-shard-K2.md)
- New machine onboarding: [`docs/new-machine-onboarding.md`](docs/new-machine-onboarding.md)
