# Pipeline v3 架构与运行约定

本文档只描述 **`partcraft.pipeline_v3`**（Mode E / text-driven 编辑）的入口、配置与产物布局。  
**Pipeline v2、预渲染、训练数据、清洗、H3D_v1 promote 等**不在此展开；见代码与历史提交。

---

## 主入口

- **Shell（推荐）**：`bash scripts/tools/run_pipeline_v3_shard.sh <tag> <config_yaml>`  
  - 按 YAML 中的 `pipeline.stages` 解析 batch / chain；按需起停 **VLM**、**FLUX** 服务。  
  - 常用环境变量：`STAGES`（逗号分隔 stage **名称**）、`SHARD=NN`（若 `TAG` 不是 `shardNN` 形式必须显式设）、`FORCE=1`、`LIMIT=N`、`OBJ_IDS_FILE=...`、`MACHINE_ENV=configs/machine/<host>.env`。  
  - 代理：脚本会 `unset` `http_proxy` / `all_proxy` 等，避免本机 VLM 请求误走 Squid。
- **Python**：`python -m partcraft.pipeline_v3.run --config <yaml> --shard <NN> --all --stage <stage_name>`  
  - 单 stage 调试时使用；多 GPU 的 `trellis_3d` / `preview_flux` / `render_3d` 由 `run.py` 内 `dispatch_gpus()` 派生子进程。

**模块目录**：`partcraft/pipeline_v3/`（`run.py`、`scheduler.py`、`paths.py`、`specs.py`、`status`、`validators`、`flux_2d.py`、`trellis_3d.py`、`preview_render.py`、`vlm_core.py` 等）。

---

## 配置结构（YAML）

典型键：

| 区域 | 作用 |
|------|------|
| 顶层 `ckpt_root` | 权重根；可被环境变量 **`PARTCRAFT_CKPT_ROOT`** 覆盖（见下）。 |
| `data.output_dir` | 管线产出根（如 `.../pipeline_v3_shard09`）。 |
| `data.mesh_root` / `images_root` / `slat_dir` | 物体网格、多视角 NPZ、SLAT 张量根。 |
| `pipeline.gpus` | 物理 GPU 索引列表，派生 VLM/FLUX 端口与 Trellis worker 数。 |
| `pipeline.stages` | 具名阶段：`name`、`steps`（step 名列表）、`servers`（`vlm` / `flux` / `none`）、`parallel_group`、`chain_id` / `chain_order`（链式阶段顺序）。 |
| `services.vlm` | 如 `model`（本地路径或 id）。 |
| `services.image_edit` | `trellis_text_ckpt`、`image_edit_backend`、`trellis_workers_per_gpu`、`export_ply_for_deletion` 等。 |
| `qc.thresholds_by_type` | Gate E（`gate_quality`）按编辑类型的阈值。 |

**调度**：`partcraft.pipeline_v3.scheduler` 将 stages 编成 batch；同 batch 内不同 `parallel_group` 可并行；`chain_id` + `chain_order` 把 `flux_2d` → `trellis_preview` 串成一条链（先关 FLUX 再占 GPU 跑 Trellis）。

---

## Post-stage hooks（spec 2026-04-21）

在 `pipeline.hooks` 中声明的 **post-stage hook** 会被调度器接在其 `after_stage` 所在链的末尾，以 `<name>@hook` 形式出现在 chain dump 里。语义：

| YAML 键 | 作用 |
|---------|------|
| `name` | 唯一标识（不得与任何 `pipeline.stages[*].name` 冲突）。 |
| `after_stage` | hook 要追加到哪个 stage 末尾；若该 stage 不在本次 `STAGES=` 选择内则整条 hook 静默丢弃（§4.2）。 |
| `uses` | v1 只支持 `cpu` / `none`；`gpu` 保留给后续 spec。CPU-only hook 会自然和同 `parallel_group` 的 GPU 链并行执行。 |
| `command` | 进程 argv 列表，支持 `{placeholder}` 替换。 |
| `env_passthrough` | shell driver 额外透传到 hook 进程的环境变量名列表。 |

**支持的 placeholder**（闭集合，`partcraft.pipeline_v3.scheduler.resolve_hook_command` 里登记）：`{py_pipe}`、`{cfg}`、`{shard}`、`{blender}`、`{h3d_dataset_root}`、`{h3d_encode_work_dir}`。未知 placeholder 在解析时直接 `ValueError`。Dataset / encode 路径默认来自环境变量 `H3D_DATASET_ROOT` / `H3D_ENCODE_WORK_DIR`，未设时分别回退到 `data/H3D_v1` / `outputs/h3d_v1_encode/<SHARD>`。

**skip / log 约定**：

- 设置环境变量 **`SKIP_HOOKS=1`** 可全局跳过所有 hook 执行（chain dump 中仍列出 `<name>@hook`，但驱动会直接 `return 0`）。  
- hook 日志落在 `logs/v3_<tag>/hook_<name>.log`；若 Python resolver 失败，traceback 落在 `logs/v3_<tag>/hook_<name>.resolve.err`。  
- hook 失败等同 stage 失败：驱动会调用 `show_stage_errors` 并中止后续 batch。

**v1 限制**：不含 retry / timeout / artifact 声明；hook 一律串联在链尾（chain tail），不支持在链中间插入；同一 `after_stage` 若声明多 hook，按声明顺序串联（并发出 WARNING）。

示例见 `configs/pipeline_v3_shard06.yaml` 的 `pipeline.hooks.pull_deletion_render`（CPU-only，自动和 `flux_2d > trellis_preview` 并行跑）。

---

## 权重路径（`load_config`）

`partcraft.utils.config.load_config` 中 **`PARTCRAFT_CKPT_ROOT`** 优先于 YAML 的 `ckpt_root`。  
Machine env 常写 **`TRELLIS_CKPT_ROOT`**，但不会自动等同前者；建议在 `configs/machine/*.env` 中：

```bash
export PARTCRAFT_CKPT_ROOT="${PARTCRAFT_CKPT_ROOT:-${TRELLIS_CKPT_ROOT}}"
```

**`services.image_edit.trellis_text_ckpt`** 建议写**相对名**（如 `TRELLIS-text-xlarge`），在 `ckpt_root` 下解析为绝对路径，换挂载点只改 env。

---

## Trellis 多 worker（`trellis_workers_per_gpu`）

- 仅 **`trellis_3d`** 使用：`dispatch_gpus` 在每张物理 GPU 上 fork **K** 个子进程（`K = trellis_workers_per_gpu`，环境变量 **`TRELLIS_WORKERS_PER_GPU`** 可临时覆盖）。  
- **`preview_flux` / `render_3d`** 每 GPU 固定单进程（避免 OOM）。  
- NFS 冷读权重大时可将 **K=1** 降低并行读盘。

---

## Active steps（`partcraft.pipeline_v3.run`）

含：`gen_edits`、`gate_text_align`、`del_mesh`、`preview_del`（可选）、`flux_2d`、`trellis_3d`、`preview_flux`、`render_3d`（可选）、`gate_quality` 等。  
GPU 类 step 集合见 `GPU_STEPS`（`trellis_3d`、`preview_flux`、`render_3d`）。

---

## 产物布局（每个 object）

根路径：`<data.output_dir>/objects/<shard>/<obj_id>/`

| 路径 | 含义 |
|------|------|
| `phase1/parsed.json`、`overview.png` | Phase 1 编辑指令与拼图。 |
| `edit_status.json` | 每编辑的 gate / stage 状态（含 Gate A / Gate E）。 |
| `edits_2d/<edit_id>_edited.png` | FLUX 2D 输出。 |
| `edits_3d/<edit_id>/` | 3D：`before.npz` / `after.npz`、`preview_0..4.png`（FLUX 类）、`after_new.glb`（deletion）等。 |

**Gate E**：`gate_quality` 用 **image_npz 五视角** 与 **`preview_0..4.png`** 拼 2×5 collage；缺 preview 会记 `missing_previews`。

---

## Machine env（与 v3 相关）

运行 `run_pipeline_v3_shard.sh` 前需 `source configs/machine/$(hostname).env`（或由 `MACHINE_ENV` 指定）。  
至少包含：`CONDA_INIT`、`CONDA_ENV_SERVER`、`CONDA_ENV_PIPELINE`、`VLM_CKPT`、`EDIT_CKPT`、`TRELLIS_CKPT_ROOT`（并建议导出 **`PARTCRAFT_CKPT_ROOT`**）、`DATA_DIR` / `OUTPUT_ROOT`（若脚本或配置引用）。  
离线 DINOv2 / CLIP 时常见：`PARTCRAFT_DINOV2_WEIGHTS`、`PARTCRAFT_CLIP_MODEL_DIR`、`TRANSFORMERS_OFFLINE`、`HF_HUB_OFFLINE`（见各 machine env 注释）。

---

## 与其他文档的衔接

- **`docs/new-machine-onboarding.md`** — 新主机环境与 **pipeline_v2** 跑通步骤（若仍使用 v2 编排）。  
- **`docs/smoke-pipeline.md`** — v2 smoke、`LIMIT` 等调试约定。  
- **`docs/dataset-path-contract.md`** — 数据集路径键名与 `load_config` 合并规则。  
- 更细的 H3D_v1、showcase、历史决策：**见 `docs/superpowers/`、`docs/runbooks/` 与 git 历史**。
