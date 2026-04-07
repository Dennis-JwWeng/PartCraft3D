# AI_LOG

按时间倒序记录与 AI/管线相关的决策与变更；**实现细节以仓库内代码与 `docs/ARCH.md` 为准**，此处不重复已迁入架构文档的协议说明。

---

## 2026-04-07 — Phase 1 v2：prompt-driven part selection（重构）

**问题**：旧 phase 1 是 group-driven —— phase0 LLM 先给每个 part 出语义描述、再 group、再为每个 group 生成 prompt。链路长、prompt 风格塌缩、对 cluster_size 小的 part 过滤不一致；多次清洗发现 deletion 13% 空 prompt 主要源于这条管线的中间环节。

**决策**：参照 Vinedresser，**一次 VLM 调用同时输出 object 元信息 + N 条编辑指令**，跳过 phase0 group。喂给 VLM 的是一张 5×2 拼图（top 5 张原始照片 + bottom 5 张同视角 part-colored Workbench 渲染），加一份 part menu（id / 颜色名 / cluster_size）。VLM 直接根据图选 part，不依赖任何上游 caption。

**5 视角集**（来自 dataset 已存的 16 视角）：
- `89/90/91/100`：4 个 cam 在 +z 上方、~30° 俯视，覆盖 4 个 yaw 象限
- `8`：cam 在 -z 下方、~52° 仰视，前方，看物体底面
- 颜色用 16 名色调色板（red/orange/.../gray），part_id `% 16` 取色

**输出 schema**：
```json
{ "object": { "full_desc", "full_desc_stage1", "full_desc_stage2", "parts":[{part_id,color,name}] },
  "edits":  [{ edit_type, prompt, view_index, selected_part_ids,
               target_part_desc, after_desc_full, after_desc_stage{1,2},
               new_parts_desc{,_stage1,_stage2}, edit_params,
               rationale, confidence }, ...] }
```
- 双轨 stage1/stage2 描述供 TRELLIS 两阶段条件
- `view_index` 0..4 由 VLM 自选（哪张视角最能看清 target），下游可视化直接还原 frame
- deletion 的 `after_desc_*` 设为 null（object.full_desc 减去删除 part 已自洽）

**Per-N 配额**（N = part 数，N>16 跳过；3 shard 共保留 3431/3609 = 95.1%）：

| N | del | mod | sc | mat | glob | 总 |
|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 1 | 1 | 5 |
| 3 | 3 | 3 | 1 | 1 | 1 | 9 |
| 4 | 4 | 4 | 2 | 2 | 1 | 13 |
| 5 | 5 | 5 | 2 | 2 | 1 | 15 |
| 6 | 6 | 6 | 2 | 2 | 1 | 17 |
| 7-8 | 8 | 8 | 3 | 3 | 1 | 23 |
| 9-10 | 10 | 10 | 3 | 4 | 1 | 28 |
| 11-12 | 12 | 12 | 4 | 4 | 2 | 34 |
| 13-14 | 14 | 14 | 4 | 5 | 2 | 39 |
| 15-16 | 16 | 16 | 5 | 5 | 2 | 44 |

del 与 mod 同步增长（mod 不破坏拓扑、del 是最便宜的有效编辑）。

**硬规则 R1-R9**（违规丢弃单条；非 deletion 缺 after_desc 算违规；view_index 必须 ∈ [0,4]）。代码侧实际校验 R1/R3 prompt 部分的 R5/R7/R8/R9，R2/R4 是纯语义提示。

**多 GPU 部署**：`--vlm-url` 接受逗号分隔多 server，runner 用 round-robin 把 jobs 分到每个 server，**每 server 单 semaphore 串行**避免 SGLang KV cache thrashing（之前 concurrency=8 实测比 sequential 慢 3×）。每 server 一张 GPU + 独立 27B 权重。

**实测**（shard01 头 20 obj，N≤16 保留 18 个，6 GPU）：
- Phase A 渲染 ~1.5s/obj（Workbench flat，无光照）
- Phase B VLM ~36s/obj wall（≈ 215s/obj sequential ÷ 6）
- 编辑通过率 196/197（99.5%）
- 单 GPU 全 shard01 估算 ~68h，6 GPU ~11.5h

**新增/修改文件**：
- `scripts/standalone/run_phase1_v2.py`：runner，含 `quota_for(N)`、`extract_json_object`、`validate`、`run_async` 多 server 调度
- `scripts/blender_render_parts.py`：Workbench flat 渲染器，按 `transform_matrix` 摆相机
- `scripts/tools/render_part_overview.py`：5×2 grid 拼图（`VIEW_INDICES = [89,90,91,100,8]`）
- `scripts/blender_render.py` / `scripts/vis/render_ply_pairs.py`：`--ref_object` 让 before/after 共用归一化
- `scripts/tools/run_vlm_cleaning.py`：渲染 deletion after 时传 before 作 ref

提交：`b07fa8c` on `feature/prompt-driven-part-selection`。

---

## 2026-04-05 — 统一 VLM 评分 prompt + 3-view 渲染方案

**问题**：
1. VLM 清洗只评编辑质量（5 维），不评 prompt 质量。shard01/03 的 deletion 有 ~13% 空 prompt + ~0.5% 零部件删除，浪费 VLM 调用且无法修正低质量 prompt。
2. 4 视角对比图拥挤，VLM 难以看清细节；4 个等高 pitch 视角无法看到物体顶部。

**决策**：

- **统一 prompt**（`vlm_filter.py:build_judge_prompt`）：单次 VLM 调用同时完成质量评分 + prompt 评价 + prompt 改写，零额外成本。分 Part 1（编辑质量 5 维）和 Part 2（prompt 质量 3 维）。空 prompt 时提示 VLM 从图像推断。
- **3-view 最优覆盖**：(0°, 26°) + (120°, 26°) + (240°, 63°)。前两个视角覆盖正面和右后侧面，第三个高俯角补顶部。Blender 和 TRELLIS Gaussian 使用完全一致的角度。

**新增字段**（`VLMScore`）：
- `prompt_quality` (1-5)：edit_prompt 与实际视觉变化的匹配度
- `improved_prompt`：VLM 改写的 edit_prompt（始终填写）
- `improved_after_desc`：VLM 描述的 AFTER 物体（始终填写）

**代码变更**：
- `partcraft/phase3_filter/vlm_filter.py`：`VLMScore` 加 3 字段，`build_judge_prompt` 改为双部分统一 prompt，`render_views` / `render_ply_views` 默认 3 views + 最优角度，`call_vlm_judge` 通过 `extra_body` 禁用 thinking（含 `TypeError` fallback 兼容非 SGLang 后端），`_VLM_YAWS` / `_VLM_PITCHES` 作为统一角度定义导出
- `scripts/tools/run_vlm_cleaning.py`：`_render_ply_pair` 改用 `render_3views`，`_render_slat_views` 从 `vlm_filter` 导入角度，`_score_one` 捕获新字段
- `scripts/vis/render_ply_pairs.py`：新增 `_THREE_VIEWS` + `render_3views()` + 通用 `_render_views()`
- `scripts/tools/run_vlm_cleaning_multi_gpu.sh`：新增 `VLM_URLS` 支持每 GPU 独立 VLM 实例（6x 吞吐），默认 3 views + 1024 max_tokens

**先清洗后编码流程**（详见 `ARCH.md`）：
```
Step4 → repack → Phase 1 → VLM 清洗 → Phase 5（仅通过的 deletion）→ Phase 3,4
```

**实测结果**（shard01 deletion，6× A800 并行）：
- 1442/2039 已评，73.8% high / 22.9% negative / 3.1% low / 0% rejected
- 速率 5.7 条/min（6 GPU），~63s/条/GPU（Blender 渲染 ~20s + VLM ~40s）
- `improved_prompt` 全部正确填写，`prompt_quality=1` 的 203 条与空 prompt 统计一致

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
