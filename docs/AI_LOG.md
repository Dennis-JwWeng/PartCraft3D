# AI_LOG

按时间倒序记录与 AI/管线相关的决策与变更；**实现细节以仓库内代码与 `docs/ARCH.md` 为准**，此处不重复已迁入架构文档的协议说明。

---

## 2026-04-05 — 代码库精简：消除重复、删除死代码

**问题**：清洗/过滤管线有 3 个独立入口 + 4 套重复指标体系，NPZ 保存和 SS 编码函数在 2 处重复实现，4 个 vis 工具共享 ~300 行重复代码。

**删除**（~1200 行）：
- `partcraft/phase4_filter/filter.py` — 死代码，指标与 phase3 完全重复
- `scripts/tools/run_postprocess.py` — Phase A-D 被独立工具替代
- `partcraft/cleaning/ply_checks.py` — 仅被 run_postprocess 引用，5 个指标重复实现
- `migrate_slat_to_npz.py` Phase 2 函数 + `_create_refiner` — 已被 Phase 5 替代

**重构**：
- `phase3_filter/filter.py` → `_mesh_metrics.py`（内部模块，仅 vlm_filter 的 mesh prefilter 使用）
- 提取 `partcraft/io/npz_utils.py`：`save_npz()` / `encode_ss()` / `load_ss_encoder()`，`migrate_slat_to_npz.py` 和 `trellis_refine.py` 改为 import
- 提取 `scripts/vis/_vis_common.py`：`load_slat()` / `render_gaussian_views()` / `make_text_bar()` / `make_label_bar()`，4 个 vis 脚本改为 import
- `hy3d_loader.py` 加 DeprecationWarning

**精简后管线**：
```
Step 1-6 → migrate (Phase 1,3,4,5) → repack → run_vlm_cleaning → 训练
```

---

## 2026-04-04 — VLM 独立清洗 `run_vlm_cleaning.py`

**问题**：计算型 Layer 2 pair checks（`pair_checks.py`）使用间接几何代理指标（connected_components ≤ 3、edit_locality、center_drift 等），与人眼视觉判断相关性弱。deletion 仅 39.1% 通过率，大量视觉合理的编辑被误杀。

**决策**：新增 `scripts/tools/run_vlm_cleaning.py`，用本地 Qwen3.5-27B VLM 替代 Layer 2 计算检查。渲染 before/after 4 视角对比图 → VLM 结构化评分 → tier 分类。

**关键设计**：
- deletion：有真实 PLY → Blender 渲染（无需 TRELLIS GPU）
- mod/scl/mat/glb：PLY 是点云（0 faces，TRELLIS 导出）→ 从 NPZ 加载 SLAT → TRELLIS decode → Gaussian 渲染
- addition：不评，继承 deletion 分数（before/after 互换，质量等价）
- identity：自动 high
- 复用 `partcraft/phase3_filter/vlm_filter.py` 的 `call_vlm_judge` + `build_judge_prompt` + 5 维评分

**多 GPU 调度**：`scripts/tools/run_vlm_cleaning_multi_gpu.sh`，GPU 0 跑 Qwen SGLang，其余 GPU 跑 TRELLIS 渲染 worker。Resume 支持：增量 JSONL + 渲染缓存。

**有效 VLM 调用量**：~48K（跳过 identity 2.4K + addition 4.2K 继承）

---

## 2026-04-04 — Phase 5：Deletion PLY 渲染 → SLAT+SS 重编码

**问题**：Deletion 的 SLAT 之前通过 mask 过滤 `feats[keep]` 产出（旧 Phase 2），特征在缺失零件的上下文中不自洽，Gaussian decode 时有严重毛边/模糊。

**决策**：在 `migrate_slat_to_npz.py` 中新增 Phase 5，废弃旧 Phase 2。仅对 deletion 的 `after.ply` 做完整重编码（`after.ply` → Blender Cycles 40 views → voxelize → DINOv2 → SLAT encoder → SS encoder → 重写 `after.npz`）。其他类型已有 TRELLIS 产出的有效 SLAT+SS，不处理。

**`migrate_slat_to_npz.py` 完整 Phase 列表**：

| Phase | 功能 | 状态 |
|-------|------|------|
| 1 | 旧 `*_slat/` → NPZ（编码 SS） | 有效 |
| 2 | Deletion mask 过滤 | **废弃**（被 Phase 5 替代） |
| 3 | Addition backfill（硬链接互换） | 有效 |
| 4 | Identity backfill（硬链接） | 有效 |
| 5 | Deletion PLY → SLAT+SS 重编码 | **新增** |

推荐执行：`--phase 1,3,4,5`

**兼容性修复**：
- `blender_script/render.py`：Blender 4.x PLY import 用 `bpy.ops.wm.ply_import`
- `dinov2_hub.py`：新版 hub Weights enum 不接受文件路径，fallback 到 `load_state_dict`

**多 GPU 调度**：`scripts/tools/run_phase5_multi_gpu.sh`，按 edit_id 分片到各 GPU

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

## 2026-04-03 — 后处理预筛 `run_postprocess.py` + 清洗扩展

**问题**：shard01 的旧格式（`feats.pt+coords.pt`，无 SS）和 PLY-only deletion 无法直接用现有 cleaning 管线处理；所有编辑类型缺少 VLM 语义评判。

**新增文件**：`scripts/tools/run_postprocess.py`（预筛编排）、`partcraft/cleaning/ply_checks.py`（PLY 几何检查）

**修改文件**：`npz_checks.py`（`require_ss=False`）、`pair_checks.py`（`require_ss` 透传）、`cleaner.py`（旧布局支持）、`vlm_filter.py`（PLY 渲染）、`migrate_slat_to_npz.py`（`--include-list`）

**注意**：`run_postprocess.py` 的 Phase A/B 是 `migrate_slat_to_npz.py` Phase 1-5 的**上游可选预筛**，不是替代关系。预筛结果通过 `--include-list` 传递给 migrate，减少无效编码。详见 `ARCH.md`。

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
- material/global 使用宽松 `voxel_count_close`（1% 容差），不再要求严格一致（见 2026-04-03 体素匹配策略定稿）
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

## 2026-03-31 — `migrate_slat_to_npz.py` 对象级去重 + NPZ 迁移

**问题**：同一 `obj_id` 的多条编辑重复计算 SS / 重复写入 `before.npz`；早期 shard 存在 PLY、`*_slat/`、NPZ 混用。

**迁移工具**：`scripts/tools/migrate_slat_to_npz.py`（多 phase，幂等）。Phase 1-5 的完整说明见 `ARCH.md`。

**注意**：旧 Phase 2（deletion mask 过滤）已在 2026-04-04 被 Phase 5（PLY 渲染重编码）替代，推荐 `--phase 1,3,4,5`。

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
