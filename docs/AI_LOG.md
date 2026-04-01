# AI_LOG

按时间倒序记录与 AI/管线相关的决策与变更；**实现细节以仓库内代码与 `docs/ARCH.md` 为准**，此处不重复已迁入架构文档的协议说明。

---

## 2026-03-31 — `_align_masks_to_slat` 体素重叠导致 Empty mask 修复

**问题**：shard09 批量跑 Step4 时大量编辑失败，日志报 `Final mask: 0 voxels`。GPU0 共 251 次空 mask，其中 208 次是有编辑体素但最终被清零。

**根因**：64³ 体素化分辨率下，编辑部件与保留部件的网格在相邻区域共享体素格子，导致 `_align_masks_to_slat` 中同一 SLAT 体素同时被标记为 `edit` 和 `preserved`。后续 `_compute_editing_region` 执行 `mask & ~preserved_parts` 时把这些重叠的编辑体素也剔除，小编辑区域的 mask 因此变为全零。

**修复**（`partcraft/phase2_assembly/trellis_refine.py` — `_align_masks_to_slat`）：在计算 `unassigned` 之前检测 `overlap = slat_is_edit & slat_is_preserved`，将重叠体素的 `preserved` 标记清除（优先保留 `edit` 身份），并增加日志记录重叠数量。

**影响**：仅 Modification 和 Scale 编辑类型受此 bug 影响（经过 `_compute_editing_region` 的 `mask & ~preserved` 路径）。Deletion 由 `direct_delete_mesh` 直接操作 GT mesh，Material 直接使用 `edit_parts.clone()`，Addition 虽经过同一路径但其编辑区域在空白空间、与 preserved 重叠极少，均不受实质影响。

---

## 2026-03-31 — Object-Centric 训练数据重组 + DataLoader

**问题**：平铺 `mesh_pairs/{edit_id}/` 下同一物体的 `before` 大量重复；训练侧需要可缓存的 Dataset。

**代码**（当前仓库）：

| 路径 | 作用 |
|------|------|
| `scripts/tools/repack_to_object_dirs.py` | 从 `mesh_pairs/` + `edit_specs.jsonl` 转为 `partverse_pairs/shard_XX/{obj_id}/`：`original.npz`、按编辑的 `after` NPZ、`metadata.json`、`manifest.jsonl`；addition/identity 仅元数据 |
| `partcraft/io/edit_pair_dataset.py` | `EditPairDataset`：`manifest.jsonl` 索引、`lru_cache` 加载 `original.npz`、解析 addition/identity 引用、`collate_fn` 双路 SparseTensor + SS |
| `partcraft/io/edit_pair_sampler.py` | `ObjectGroupedSampler`：按物体聚类索引，提高 original 缓存命中 |

**用法示例**：见 `docs/ARCH.md`「训练数据契约（Object-Centric Edit Pairs）」一节。

---

## 2026-03-31 — `migrate_slat_to_npz.py` 对象级去重

**问题**：同一 `obj_id` 的多条编辑重复计算 SS（`z_s_before`）并重复写入相同内容的 `before.npz`。

**实现要点**（`scripts/tools/migrate_slat_to_npz.py`）：

- `_obj_id_from_edit_id()`、`_link_or_copy()`（`os.link`，跨设备则 `copy2`）
- Phase 1：按物体分组，首条编辑写 `before.npz`，同物体后续硬链
- Phase 2 deletion：每物体一次 `encode_ss` + canonical `before.npz`，其余硬链
- Phase 3 addition：同物体内对重复的 `add.after.npz` 等硬链
- Phase 4 identity：`before.npz` / `after.npz` 均硬链到该物体已有非 identity 编辑的 canonical `before.npz`

统计类数字为当时全量迁移估算，**以实际跑出来的日志为准**。

---

## 2026-03-31 — 历史 shard 产物格式与 NPZ 迁移

**背景**：早期 shard 存在 PLY、`*_slat/`、`before.npz`/`after.npz` 混用；需统一到 NPZ（keys：`slat_feats`, `slat_coords`, `ss`）。

**迁移工具**：`scripts/tools/migrate_slat_to_npz.py`（多 phase，幂等：已有 `*.npz` 默认跳过）。

**配置**：仓库内仅有 `configs/partverse_H200_shard00.yaml` 等少数模板；**跑其它 shard 时请复制 YAML 并将 `data.shards`（及 `mesh_pairs` / `edit_specs` 路径）改为目标 shard**。Phase 2 依赖 `dataset.load_object(shard, obj_id)`，config 中的 shards 必须包含目标 shard；Phase 1 不依赖 dataset。

**示例（shard 00）**：

```bash
python scripts/tools/migrate_slat_to_npz.py \
    --config configs/partverse_H200_shard00.yaml \
    --mesh-pairs <OUTPUT>/shard_00/mesh_pairs_shard00 \
    --specs-jsonl <OUTPUT>/shard_00/cache/phase1/edit_specs_shard00.jsonl
```

仅预览：`--dry-run`。仅跑部分 phase：`--phase 2,3,4`（例如 GPU 编辑已是 NPZ 时）。

---

## 2026-03-31 — `prerender.py` 多 GPU encode + 多进程 pack

**代码**：

- `scripts/datasets/partverse/prerender.py`：`--pack-workers`，`ProcessPoolExecutor` 并行 pack；`_pack_worker` 顶层函数以便 pickle
- `scripts/datasets/partverse/pack_npz.py`：`--workers`，可独立调用

---

## 2026-03-30 — 环境配置脚本（合并条目）

**代码**：

- `scripts/tools/setup_env_common.sh`：`detect_cuda_suffix`、`check_flash_attn` / `check_xformers`、`resolve_attn_backend`、统一参数与 machine env 加载
- `scripts/tools/setup_pipeline_env.sh`：管线依赖（含 spconv 与 CUDA 后缀匹配、attention 后端选择）
- `scripts/tools/setup_deploy_env.sh`：VLM + 图像编辑服务依赖（`diffusers` 等版本约束）

`requirements.txt` 中有与上述脚本交叉引用的说明。

---

## 2026-03-30 — 单机 PartVerse 路径与 `partverse_wm1A800` 模板

**代码**：`configs/machine/wm1A800.env`、`configs/partverse_wm1A800_shard00.yaml`、`configs/prerender_partverse_wm1A800.yaml` — 将数据与输出根目录绑到本机 cfs 布局；`scripts/tools/download_local_missing_weights.sh` 用于拉取缺失权重。

具体挂载点与容量为**当时机器状态**，以当前 `*.env` 为准。

---

## 2026-03-30 — Step4 保存 NPZ 时 `detach`、多 GPU 启动

**问题**：Sampling 输出 tensor `requires_grad=True`，直接 `.numpy()` 报错。

**代码**：`partcraft/phase2_assembly/trellis_refine.py` 的 `_save_npz()` 对 feats/coords/ss 使用 `.detach()` 再转 numpy。多 GPU Step4 需为 `run_pipeline.py` 传入 `--gpus ...`，否则会走单卡分支。

---

## 2026-03-30 — Step4 spconv / attention 与 worker 日志

**背景**：CUDA 工具链与 spconv 构建需匹配；部分环境需 `flash_attn` 或 `xformers`。

**代码**：`scripts/run_pipeline.py` 启用 `faulthandler`；GPU worker 日志路径见脚本与 orchestrator。`third_party/trellis/modules/**/full_attn.py` 等对 xformers API 变更有兼容分支（以文件内实现为准）。

---

## 2026-03-30 — Step4 产出 `before.npz` / `after.npz`

**代码要点**：

- `third_party/interweave_Trellis.py`：`interweave_Trellis_TI` 返回含 `slat`、`z_s_before`、`z_s_after`
- `partcraft/phase2_assembly/trellis_refine.py`：`encode_ss`、`export_pair` / `export_deletion_pair` 等写 NPZ；`load_pair_npz()` 兼容旧 `*_slat/` 目录
- `scripts/pipeline_step_3d.py`、`partcraft/streaming_lookahead.py`：各编辑类型导出路径
- 下游：`scripts/merge_streaming_workers.py`、`partcraft/phase3_filter/vlm_filter.py`、`scripts/vis/render_gs_pairs.py` 等对 `*.npz` 或旧目录的检测

**说明**：当前 GPU 编辑路径以 **NPZ** 为主；`export_ply` 等开关仍影响是否写 PLY（如 `direct_delete_mesh` 的 GT mesh）。旧日志中「始终写 `before_slat`/`after_slat` 目录」的表述已过时，以 `trellis_refine.py` 现实现为准。

---

## 2026-03-30 — node39 配置、`img_enc_dir` 与 `derive_dataset_subpaths`

**代码**：

- `configs/partverse_node39_shard01.yaml`、`configs/prerender_partverse_node39.yaml`、`configs/machine/node39.env`
- `partcraft/utils/config.py`：`derive_dataset_subpaths` 只自动补 `images`/`mesh`/`slat` 子路径，**不**自动填 `img_enc_dir`
- `scripts/pipeline_common.py`：`data.slat_dir` 必填，`img_enc_dir` 可选（缺省时 TrellisRefiner 可走 mesh NPZ fallback）

`scripts/tools/run_shard_batch_pipeline.sh` 中 `ATTN_BACKEND` 等可由 machine env 覆盖。

---

## 2026-03-29 — Config 显式失败与预渲染路径

**代码**：`partcraft/utils/config.py` 的路径来源与 `[CONFIG_ERROR]`；`scripts/datasets/prerender_common.py` 要求显式 `dataset_root`；管线侧取消不安全的 URL/路径隐式回退（详见 `config.py` 与相关脚本）。

**校验**：缺 `data.slat_dir` 等关键项应失败；`img_enc_dir` 对编辑管线**不一定**为硬错误（见上节）。

---

## 2026-03-29 — 一键环境脚本初版

`setup_deploy_env.sh` / `setup_pipeline_env.sh` / `setup_env_common.sh` 初版与 `--machine-env`、`--check`、`--reinstall` 等；后续在 2026-03-30 条目中合并增强（见上）。

---

## 2026-03-29 — Batch 管线配置化与 `node39` machine env

**代码**：`scripts/tools/run_shard_batch_pipeline.sh` 由 `configs/machine/<hostname>.env` 驱动；`configs/machine/node39.env`；`scripts/tools/launch_local_vlm.sh` 等与 `VLM_MEM_FRAC` 等可配置项。

---

## 2026-03-29 — Step1 多 Worker 先合并再失败

**代码**：`scripts/pipeline_dispatch.py` — `wait_for_workers(..., fail_fast=False)`、`discover_step1_worker_results`、`reconcile_worker_results`（`reconcile_step4_results` 为兼容包装）；`scripts/run_pipeline.py` 中 Step1/Step4 在 worker 异常时先合并再抛错。

---

## 2026-03-29 — Phase1 Prompt 多样性（enricher / planner）

**代码**：`partcraft/phase1_planning/enricher.py` 中 `scale_edits`、丰富 material/global 指令等；`partcraft/phase1_planning/planner.py` 接入 VLM 与模板兜底。责任边界与流程见 `docs/ARCH.md`。
