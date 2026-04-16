# PartCraft3D

Part-segmented 3D assets to **(before, after, edit instruction)** training pairs. Active orchestration: **`partcraft.pipeline_v3`**.

---

## Entry

**Full shard run (recommended):** starts VLM / FLUX per stage, uses `parallel_group` in YAML, logs under `logs/v3_<tag>/`.

```bash
bash scripts/tools/run_pipeline_v3_shard.sh <tag> <config.yaml>
# Example: tag `shard08` → Python sees `--shard 08`
```

**Direct Python:**

```bash
python -m partcraft.pipeline_v3.run \
  --config <config.yaml> --shard <NN> --all \
  --steps gen_edits,gate_text_align,del_mesh,preview_del,gate_quality
# Or: --stage <name>  (must match pipeline.stages[].name in YAML)
```

**Bench + tmux + obj list:** `scripts/tools/run_pipeline_v3_bench.sh` (see `--help`).

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

---

## Config

1. Copy **`configs/templates/pipeline_v3_bench.template.yaml`** → your file; fill `blender`, `ckpt_root`, `data.*`, `services.*`.

2. **`pipeline`:** `gpus`, `vlm_port_base` / `vlm_port_stride`, `flux_port_base` / `flux_port_stride`, workers (`prerender_workers`, `s6p_del_workers`), **`stages`** (`name`, `steps`, `servers`: `vlm` | `flux` | `none`, `use_gpus`, `parallel_group`).

3. **`services`:** VLM model; image edit / Trellis checkpoints.

4. **`qc`:** gate thresholds by edit type.

**Machine env:** `configs/machine/$(hostname).env` (conda, `VLM_CKPT`, `EDIT_CKPT`), or override with `MACHINE_ENV`.

---

Further architecture and contracts: **`docs/ARCH.md`** (if present in your tree).
