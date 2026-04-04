# AI_LOG

按时间倒序记录与 AI/管线相关的决策与变更；**实现细节以仓库内代码与 `docs/ARCH.md` 为准**，此处不重复已迁入架构文档的协议说明。

---

## 2026-04-04 — Deletion SLAT 改为 PLY 渲染重编码 + 全类型 `dino_voxel_mean`

**问题 1**：Deletion 的 SLAT 之前通过从原始 SLAT 按 mask 过滤 `feats[keep]` 产出。由于 SLAT 特征在完整物体上下文中编码（DINOv2 多视角聚合 + SLAT encoder），直接子集化后剩余体素的特征不再自洽，TRELLIS Gaussian decode 时出现严重毛边/模糊。PLY 直接渲染验证确认 mesh 本身是干净的，问题出在 SLAT latent 层。

**问题 2**：训练侧需要 `dino_voxel_mean`（SLAT encoder 的输入，多视角 DINOv2 patch 特征投影到体素后取均值，`[N, 1024]` float16），但之前编码流程未持久化该中间产物。

**决策**：

- Deletion after：走与原始物体完全相同的编码路径 — `after.ply → Blender Cycles 渲染 40 views → Open3D 体素化 64³ → DINOv2 → SLAT encoder → SS encoder`，产出完整 `after.npz`（slat_feats + slat_coords + ss + dino_voxel_mean）
- 其他类型 after：保留 TRELLIS 生成的 SLAT+SS，通过 `after.ply → Blender 渲染 → DINOv2` 提取 `dino_voxel_mean` 注入已有 NPZ
- Before 侧：从 `slat_dir` 加载预存的 `{obj_id}_dino_voxel_mean.pt`（需在有 `img_Enc` 的机器上重跑 `encode_into_SLAT`），或 fallback 渲染 `before.ply`

**代码变更**：

| 文件 | 改动 |
|------|------|
| `third_party/encode_asset/encode_into_SLAT.py` | 抽出 `extract_dino_voxel_mean(render_dir, num_views)` 可复用函数；`encode_into_SLAT()` 新增 `save_dino_voxel_mean=True` 参数 |
| `scripts/tools/migrate_slat_to_npz.py` | `_save_npz()` 增加 `dino_voxel_mean` 参数；新增 `_render_and_extract_dino()`、`_inject_dino_into_npz()`；新增 **Phase 5**（PLY 渲染 + DINOv2 提取，deletion 含 SLAT+SS 完整重编码） |
| `partcraft/phase2_assembly/trellis_refine.py` | `_save_npz()` 增加 `dino_voxel_mean` 参数 |
| `partcraft/io/edit_pair_dataset.py` | `_load_npz()` 加载 `dino_voxel_mean`；`__getitem__` 返回 `before_dino`/`after_dino`；`collate_fn` 对 dino 做 `torch.cat`（与 SLAT 共享稀疏布局）|

**NPZ 格式扩展**（向后兼容）：

```python
{"slat_feats": [N, 8], "slat_coords": [N, 4], "ss": [C, R, R, R],
 "dino_voxel_mean": [N, 1024]}  # float16, 可选
```

**存储开销**：dino_voxel_mean 每个 NPZ 增加约 26 MB（13K voxels × 1024 × 2B），总量级 TB。

---

## 2026-04-03 — Material/Global 体素匹配策略定稿

**问题**：TRELLIS S2-only 编辑（material/global）理论上不改变几何，但实际产出会有 1-2 个体素的微小差异。之前的 `coords_match`（`np.array_equal`）和 `voxel_count_match`（精确相等）导致 material 类型全部被误杀（0% 通过率）。

**决策**：统一移除严格体素匹配检查，不论是否有 SS，material/global 均使用宽松的 `voxel_count_close`（1% 容差）。

**代码**：`partcraft/cleaning/pair_checks.py` 的 `check_material()`
- 移除 `require_coords_match` / `coords_match`（`np.array_equal`）
- 移除 `voxel_count_match`（精确 `n_before == n_after`）
- 统一使用 `voxel_count_close`：`abs(ratio - 1.0) <= 0.01`，weight=2.0
- `feat_change` 检查不再依赖体素数完全一致（`feat_change_ratio` 已兼容不同长度）
- SS match 检查保留（`ss_match_tol: 1e-3`，有 SS 时生效）

**验证**：shard01 全量 cleaning，修复前 material 0%，修复后预期 ~95%+。

---

## 2026-04-03 — 统一后处理管线 `run_postprocess.py`

**问题**：shard01 的旧格式（`feats.pt+coords.pt`，无 SS）和 PLY-only deletion 无法直接用现有 cleaning 管线处理；所有编辑类型缺少 VLM 语义评判。

**新增文件**：

| 路径 | 作用 |
|------|------|
| `scripts/tools/run_postprocess.py` | 四阶段统一编排入口（Phase A-D） |
| `partcraft/cleaning/ply_checks.py` | PLY mesh 几何检查（水密性、连通性、退化、体积比） |

**修改文件**：

| 路径 | 改动 |
|------|------|
| `partcraft/cleaning/npz_checks.py` | `require_ss=False` 模式；`load_slat_dir_arrays()` 支持旧 `*_slat/` 目录 |
| `partcraft/cleaning/pair_checks.py` | 所有 checker 新增 `require_ss` 参数，无 SS 时跳过 SS 检查 |
| `partcraft/cleaning/cleaner.py` | 支持从旧 `mesh_pairs/` 平铺布局加载；`require_ss` 和 `mesh_pairs_dir` 透传 |
| `partcraft/phase3_filter/vlm_filter.py` | `render_ply_views()`、`evaluate_edit_from_ply()`、`evaluate_edit_from_slat_dir()` |
| `scripts/tools/migrate_slat_to_npz.py` | `--include-list` 过滤，仅处理通过预筛的子集 |

**设计要点**：

- Phase A（几何预筛）无需 GPU，按数据格式自动分流：SLAT 编辑用 feats+coords 检查，PLY 编辑用 trimesh 检查
- Phase B（VLM 语义）所有编辑类型都走 VLM 评判 prompt 对齐度；PLY 用 Blender 渲染，SLAT 用 TRELLIS Gaussian
- Phase C 仅对 A+B 通过子集 encode SS，避免浪费 GPU
- addition/identity 跟随配对 deletion 的结果，不独立评判

---

## 2026-04-03 — 数据清洗关键 Bug 修复

**问题**：审核 cleaning 管线代码时发现三处影响正确性的 bug。

**修复**：

| 文件 | 问题 | 修复 |
|------|------|------|
| `scripts/tools/run_cleaning.py` | `ss_match_tol: 1e-4` 过于严格，浮点运算误差导致 material/global 编辑被大量误杀 | 放宽到 `1e-3` |
| `scripts/tools/run_cleaning.py` | `--edit-types` 不带参数时 `args.edit_types` 为 `[]`，`set([])` 过滤掉所有编辑，静默产出空结果 | 改为 `if args.edit_types is not None` |
| `partcraft/cleaning/cleaner.py` | identity L1 检查通过 `check_npz_sanity.__wrapped__` 探测，依赖装饰器实现细节，一旦上游改动会静默走错分支 | 直接调用 `_run_l1_on_arrays()` |

---

## 2026-03-31 — 编辑对可视化工具 `render_edit_gallery.py`

**问题**：现有 `render_gs_pairs.py` 仅支持旧平铺 `mesh_pairs/` 格式且输出 MP4 视频，不适合快速浏览大量编辑的合理性。

**代码**：`scripts/vis/render_edit_gallery.py`

- 同时支持 `--input-dir`（新 object-centric `shard_XX/{obj_id}/`）和 `--pairs-dir`（旧平铺 `mesh_pairs/`）
- SLAT → TRELLIS Gaussian decode → N 视角正面环绕渲染
- 输出单张 PNG：header（编辑类型、prompt、quality tier/score）+ before N 视角 + after N 视角
- 支持 `--min-tier` 按 `quality.json` 过滤、`--sample-per-type` 每类采样、`--edit-types`/`--shards` 筛选
- 复用 `render_gs_pairs.py` 的 `load_slat` 兼容逻辑（NPZ + 旧 feats.pt/coords.pt）

**用法**：见 `docs/ARCH.md`「可视化工具」一节。

---

## 2026-03-31 — NPZ 数据清洗管线 + Step 7

**问题**：现有 Step 5 (VLM filter) 基于 PLY mesh 做几何检查，无法直接对 NPZ（SLAT/SS）格式的训练数据做质量过滤；且未按编辑类型特化检查逻辑。

**代码**：

| 路径 | 作用 |
|------|------|
| `partcraft/cleaning/npz_checks.py` | Layer 1：NPZ 健全性（体素数、特征值域、SS 值域、坐标合法性/唯一性） |
| `partcraft/cleaning/pair_checks.py` | Layer 2：7 类编辑各自的对比检查（deletion 子集关系、modification 局部性、material/global S2-only 坐标不变约束等） |
| `partcraft/cleaning/cleaner.py` | 主入口：遍历 object-centric 目录，执行 L1+L2，输出 `quality.json` + `manifest_clean.jsonl` |
| `scripts/tools/run_cleaning.py` | CLI 入口，含完整默认阈值配置 |

**修改现有文件**：

- `partcraft/phase3_filter/filter.py`：提取 `weighted_score()` 为公共函数
- `partcraft/phase3_filter/vlm_filter.py`：公开 `build_judge_prompt`、添加 `__all__` 导出
- `partcraft/io/edit_pair_dataset.py`：新增 `quality_dir` / `min_tier` 参数，加载时自动过滤
- `scripts/run_pipeline.py`：新增 `run_step_cleaning()` + `--cleaning-input-dir` / `--cleaning-workers`
- `scripts/pipeline_orchestrator.py`：Step 7 调度（opt-in，不在默认 steps 中）

**设计要点**：

- 纯 numpy + scipy，无 GPU 依赖（Layer 1/2）
- material/global 类型强制 `slat_coords` 和 `ss` 完全一致（S2-only 不改几何的约束）
- scale 比 modification 有更严格的 SS 余弦相似度和 bbox 轴向比阈值
- 所有阈值可通过 YAML `cleaning:` 段或 CLI 默认配置覆盖

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
