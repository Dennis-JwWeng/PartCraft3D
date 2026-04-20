# AI_LOG

按时间倒序记录与 AI/管线相关的决策与变更；**实现细节以仓库内代码与 `docs/ARCH.md` 为准**，此处不重复已迁入架构文档的协议说明。

---

## 2026-04-20 — H3D_v1 per-edit 视图降为单张 `before.png` / `after.png`

**背景**：上一版只把 `views.best_view_index` 写进 `meta.json`，但 `<edit_dir>/before_views/` 和 `after_views/` 仍然各 hardlink 5 张 PNG，下游要先读 json 再索引 viewK，目录自解释性差。

**变更**：

- **Layout 扁平化** (`partcraft/cleaning/h3d_v1/layout.py`)：
  - 新增 `before_image()` / `after_image()` → `<edit_dir>/{before,after}.png`。
  - 废弃并删除 `before_views_dir()` / `after_views_dir()` / `before_view(k)` / `after_view(k)`。
- **Promoter** (`partcraft/cleaning/h3d_v1/promoter.py`)：
  `promote_deletion` / `promote_flux` / `promote_addition` 各只 hardlink 两张图，K 取自已写好的 `record["views"]["best_view_index"]`，确保 `meta.json` 里的 K 与磁盘上的图是同一视角。addition 的 `before.png` 直接 hardlink 配对 deletion 的 `after.png`（K 已镜像），自然共享 inode。
- **Index validator** (`scripts/cleaning/h3d_v1/build_h3d_v1_index.py`)：`--validate` 只检查每条 edit 的 `before.png` + `after.png` 是否在位，不再按 N_VIEWS 展开。
- **Decode inspect** (`scripts/datasets/h3d_v1/decode_inspect.py`)：GT strip 由"5 张视图拼条"改为单张图（保持 `_gt_strip` 入口名以兼容调用点）。
- **Docs**：`docs/superpowers/specs/2026-04-19-h3d-v1-design.md` §3/§4/§6/§8/§13 全部同步到扁平 layout。

**迁移策略**：不回填老 shard 上的 `before_views/after_views/` 五张旧图——直接把 shard08 已 promote 的目录清掉、跑三个 pull_* CLI 重新 promote 一遍（用户明确同意，见 2026-04-20 会话）。

**测试**：`tests/cleaning/h3d_v1/` 74 pass（含 test_smoke_e2e 从 `pull_deletion` 一路跑到 `pack_shard`，已对单张 `before.png`/`after.png` 的 inode 做了断言）。

---

## 2026-04-20 — H3D_v1 `meta.json` 瘦身 + `views.best_view_index`

**背景**：H3D_v1 per-edit 目录里一直有 5 张 `before/after` 五视角图，但 `meta.json` 里没有记录"哪一张是最值得看的那张"。同时之前的 schema 冗余字段（`part_labels` / `n_parts_selected` / `stats.*` / `lineage.{pipeline_config,pipeline_git_sha,promoted_at,paired_edit_id}`）与"释放给下游消费"的定位不匹配。

**变更**：

- **Schema v3 final** (`partcraft/cleaning/h3d_v1/promoter.py`)：
  - 新增 `meta.views.best_view_index`（0..4，对应 `view{k}.png`）。    选择规则：flux/deletion 读 `edit_status.json.gates.A.vlm.best_view`    （pipeline_v3 对 selected part 掩码在 5 视角 overview 上取 argmax）；    addition 镜像配对 deletion；`global` 类及所有兜底走     `DEFAULT_FRONT_VIEW_INDEX = 4`（`view4` = front upward，yaw +22° pitch +52°，    见 `partcraft/render/overview.py::VIEW_INDICES`）。
  - `quality.gate_A_score` → `alignment_score`（语义：文本-图像对齐）；    `quality.gate_E_score` → `quality_score`（3D 渲染质量）。    addition 的 `alignment_score` 镜像配对 deletion 的 gate A 分数。
  - `lineage` 只保留 `source_dataset` + `pipeline_version`。    `paired_edit_id` 下游靠 `del_<obj>_NNN` ↔ `add_<obj>_NNN` 编号约定重建。
  - `instruction` 删掉 `part_labels`、`n_parts_selected`（见     `partcraft/cleaning/h3d_v1/instruction.py`）。
  - `stats.{before,after}_n_voxels` / `delta_voxels` 整块移除。

- **Repro 审计** (`manifests/_internal/promote_log.jsonl`)：  每次 `promote_*` 成功后 append 一行，带 `pipeline_config` /   `pipeline_git_sha` / `promoted_at`。`pack_shard` 不打包   `manifests/_internal/`，仅本机留底。

- **回填** (`scripts/tools/h3d_v1_backfill_meta.py`)：  `--dataset-root data/H3D_v1 [--shard 08] [--pipeline-cfg <yaml>]`  幂等重写已有 `meta.json` 到 v3 final；带 `--pipeline-cfg` 时从   `edit_status.json` 填 `views.best_view_index`，否则走 default front。  shard08 已回填（15 条 meta：8 × pipeline argmax + 5 × paired-deletion   镜像 + 2 × global default）。

**Design choice**：GLB object 级"整体 overview 图"从本版本砍掉（用户决定不出）。下游要看对象概览就直接从 `_assets/<NN>/<obj>/orig_views/view{0..4}.png` 中 任取一张；同一张 `DEFAULT_FRONT_VIEW_INDEX` 会出现在所有 `global` 编辑的 `best_view_index` 里，语义统一。

**测试**：`tests/cleaning/h3d_v1/` 74 pass（含 5 个新 views/promote_log 断言）。顺带把 `test_asset_pool.py` / `test_smoke_e2e.py` 的 fixture 接上合法 `images.npz`（`ensure_object_views` 在早先的 fix 之后已强依赖该输入）。

---

## 2026-04-19 — Showcase / gallery 编辑对人工挑选脚本三件套

**背景**：H3D Dataset Overview hero 报告（28 picks，shard08）+ gallery 候选（50 picks 目标，shard07+08）需要稳定的"扫一遍 → 挑出来 → 出最终报告"流程。把临时脚本沉淀成可复用的三件套，并写 runbook。

**新增**：

- `scripts/tools/build_showcase_candidates.py` — 单 shard 候选 HTML（沿用 2026-04-19 已有版本，含 `_flatten_to_white`、addition 反向 BEFORE/AFTER）
- `scripts/tools/build_gallery_candidates.py` — 多 shard 候选 HTML，card id 形如 `<shard>/<obj>/<eid>`，新增 `glb ready / needs decode` 徽章（区分 del/add vs FLUX 五类），toolbar 加 shard / glb 过滤
- `scripts/tools/build_showcase_hero.py` — picks JSON → 高质量 hero HTML（已有；这次只整理文档）

**关键约定**（详见 `docs/runbooks/showcase-pick-workflow.md`）：

- BEFORE/AFTER 数据来源：deletion / FLUX 取 NPZ 5 帧（view `[89,90,91,100,8]`）+ `preview_{0..4}.png`；**addition 是 deletion 的反向**——BEFORE 走 `source_del_id` 的 preview，AFTER 走原 NPZ 帧（不能用 add 自己的 preview，会带 TRELLIS 重建伪影）。
- 所有图先 `_flatten_to_white`（RGBA → RGB on white），与 gate_a / gate_e 看到的版本对齐。
- `eid` 必须严格复刻 `partcraft/pipeline_v3/specs.iter_all_specs` 的 seq 规则（deletion 独立 `del_seq`、FLUX 五类共享 `flux_seq`、addition 不在 parsed.json 里需读 `meta.json`），否则 prompt 会贴错卡。

**GLB 现状（gallery 阶段会用到）**：

- 仅 `del_*/after_new.glb` 在磁盘上（s5b `del_mesh` 写出，shard07/08 各 ~3500 个）。
- `addition` 不需要新建文件，引用 `source_del_id` 的 `after_new.glb` + 输入 mesh NPZ 内 `full.glb` 即可。
- `modification / material / color / global / scale` 在 `<eid>/after.npz` 里只有 TRELLIS slat tokens，**需要解码**才能给 Blender 用。当前没有现成 picks → GLB CLI；最朴素做法是仿 `partcraft/pipeline_v3/preview_render.py` 的 `after_new.glb` / `after.ply` 路径写一个解码 CLI，或者对这些 picks 重跑 trellis_preview。

**用法摘要**：

```bash
# 1) 多 shard 候选页（seed 决定抽样次序）
python scripts/tools/build_gallery_candidates.py \
  --shards 07 08 \
  --root outputs/partverse/shard07/mode_e_text_align \
  --root outputs/partverse/shard08/mode_e_text_align \
  --images-root data/partverse/inputs/images \
  --seed 2026 --target 50 \
  --out reports/h3d_gallery_candidates.html

# 2) 浏览器里挑星标 → 右下角 copy JSON → 落盘 reports/<tag>_picks.json

# 3) 出最终 hero（当前是单 shard 入口）
python scripts/tools/build_showcase_hero.py \
  --root outputs/partverse/shard08/mode_e_text_align --shard 08 \
  --images-root data/partverse/inputs/images \
  --picks reports/shard08_picks.json \
  --out reports/shard08_showcase_hero.html
```

**遗留**：

- `build_showcase_hero.py` 仍只接受单 shard `--root`，多 shard picks 现需按 shard 拆开跑两次再拼或扩展脚本。
- FLUX 五类 picks → GLB 的解码 CLI 还没写。

---

## 2026-04-08 — pipeline_v2：object-centric 全管线入口 + validator 修复

**问题**：phase1 v2 之后,后续 step（FLUX 2D / TRELLIS 3D / rerender / addition backfill）原本仍走旧 batch 入口（`scripts/run_pipeline.py`），与 object-centric 输出布局（`outputs/<root>/objects/<shard>/<obj_id>/{phase1,edits_2d,edits_3d,...}`）不对齐；并且没有按 phase 自动起停 VLM/FLUX 服务的统一调度。

**决策**：把 phase1 v2 的全部下游搬进新模块 `partcraft/pipeline_v2/`,以 object 为单位串起 s1→s2→s4→s5→s5b→s6→s6b→s7,并新增 shell 调度器 `scripts/tools/run_pipeline_v2_shard.sh` 按 config 的 `pipeline.phases` 拉起/拆卸服务。

**新入口**：

- Python：`python -m partcraft.pipeline_v2.run --config <yaml> --shard <NN> --all --phase <A|C|D|D2|E|F>`
- Shell：`PHASES="A,C,D,D2,E,F" bash scripts/tools/run_pipeline_v2_shard.sh shard01 configs/pipeline_v2_shard01.yaml`
  - 由 `partcraft/pipeline_v2/scheduler.py:dump_shell_env()` 给 bash 吐 `GPUS / VLM_PORTS / FLUX_PORTS / DEFAULT_PHASES`,bash 只负责 `start_vlm`/`start_flux`/`stop_*` 和 phase 循环
  - 任意 N-GPU 自动派生端口(VLM `base+i*stride`,FLUX `base+i*stride`)
  - 单 phase 失败立刻清理服务并退出

**模块布局**(`partcraft/pipeline_v2/`):

| 文件 | 作用 |
|---|---|
| `run.py` | CLI；按 `--phase` 或 `--steps` 调度多卡;每个 step 跑完调用 validators 翻 status |
| `scheduler.py` | 纯控制面: `gpus_for / vlm_urls_for / flux_urls_for / phases_for / dump_shell_env` |
| `specs.py` | `iter_all_specs / iter_flux_specs / iter_deletion_specs`,从 parsed.json 派生 edit 任务 |
| `paths.py` | `ObjectContext`,统一所有产物路径(parsed/overview/highlights/edits_2d/edits_3d) |
| `status.py` | `status.json` 读写 + 状态常量(OK/FAIL/SKIP/PENDING) + `step_done` |
| `validators.py` | 每 step 的纯文件检查,翻 status 用 |
| `s1_phase1_vlm.py` | 异步 producer-consumer:1 producer + N consumer,每 VLM server `Semaphore(1)`;`ProcessPoolExecutor` 跑 blender prerender;parsed.json 存在即跳过 |
| `s2_highlights.py` | 渲染 highlights/e{idx:02d}.png |
| `s4_flux_2d.py` | 调 FLUX 服务做 2D 编辑,per-edit 看 `_edited.png` resume |
| `s5_trellis_3d.py` | TRELLIS 编辑产 before/after.npz |
| `s5b_deletion.py` | deletion 专用 mesh 路径 |
| `s6_render_3d.py` | 3D rerender(`s6` Gaussian 渲染 + `s6b` deletion 重编码) |
| `s7_addition_backfill.py` | addition 互换硬链接 |

**Resume 协议(per-edit 粒度)**:

1. 每 step 跑完 → validators 扫磁盘 → status flip(OK/FAIL/SKIP)
2. 下次启动 → s5/s6 等先看 `step_done(ctx, step)` 跳过物体级 OK
3. 物体级 FAIL/PENDING → 进入 step,内部 per-edit 检查 `before.npz/after.npz/_edited.png` 等,只补缺失文件
4. s1 SKIP(`too_many_parts`) 不会被 validator 翻成 FAIL

**Validator bug 修复**(本次配套):

| 问题 | 修复 |
|---|---|
| `_check_files` 在 expected=0 时返回 ok=True,导致没 parsed.json 的物体被 s4/s5/s5b/s6/s6b/s7 标成"通过" | 新增 `_require_phase1(step, ctx)` 网关:s1 SKIP → ok=True expected=0;parsed.json 缺失 → ok=False missing=['parsed.json'];否则继续 |
| `check_s1` 把 too_many_parts 的 SKIP 物体翻成 FAIL(parsed.json 不存在) | 先看 `_phase1_skipped`,SKIP 短路返回 ok=True |
| `dispatch_gpus` 子进程 exit=1 时 orchestrator 不抛错,直接报"ALL PHASES DONE" + 旧 status 数字 | (待修)`run.py` 的 `dispatch_gpus` 返回 rc 应让 phase 失败传播 |

**配置示例**:`configs/pipeline_v2_shard01.yaml`

```yaml
pipeline:
  gpus: [0, 4, 5, 6, 7]              # N-GPU agnostic, 单一事实源
  prerender_workers: 8
  vlm_port_base:    8002
  vlm_port_stride:  10               # → 8002,8012,8022,8032,8042
  flux_port_base:   8004
  flux_port_stride: 1                # → 8004..8008
  phases:
    - { name: A,  desc: "phase1 VLM",        servers: vlm,  steps: [s1],       use_gpus: false }
    - { name: C,  desc: "FLUX 2D",           servers: flux, steps: [s4],       use_gpus: false }
    - { name: D,  desc: "TRELLIS 3D edit",   servers: none, steps: [s5],       use_gpus: true  }
    - { name: D2, desc: "deletion mesh",     servers: none, steps: [s5b],      use_gpus: false }
    - { name: E,  desc: "3D rerender",       servers: none, steps: [s6, s6b],  use_gpus: true  }
    - { name: F,  desc: "addition backfill", servers: none, steps: [s7],       use_gpus: false }
```

**实测**(shard01,5 GPU):
- Phase A:5 sglang 并行 → ~40s/obj wall,982 个剩余 obj 约 2.5h
- Phase D:TRELLIS 单 GPU ~4 min/edit,5 GPU 并行 ≈ 1.25 edit/min 整体吞吐
- 与其他用户共享卡时会被显著拖慢(GPU 4-7 上每张卡有 13-23 GB 别人的 python 进程,实测 D 约 1 edit/min)

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
- `partcraft/pipeline_v2/s1_phase1_vlm.py`（旧 `scripts/standalone/run_phase1_v2.py` 已迁入包）：含 `quota_for(N)`、`extract_json_object`、`validate`、`run_async` 多 server 调度
- `scripts/blender_render_parts.py`：Workbench flat 渲染器，按 `transform_matrix` 摆相机
- `partcraft/render/overview.py`（library）+ `scripts/tools/render_part_overview.py`（CLI shim）：5×2 grid 拼图（`VIEW_INDICES = [89,90,91,100,8]`）
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
- `partcraft/cleaning/vlm_filter.py`：`VLMScore` 加 3 字段，`build_judge_prompt` 改为双部分统一 prompt，`render_views` / `render_ply_views` 默认 3 views + 最优角度，`call_vlm_judge` 通过 `extra_body` 禁用 thinking（含 `TypeError` fallback 兼容非 SGLang 后端），`_VLM_YAWS` / `_VLM_PITCHES` 作为统一角度定义导出
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

## 2026-04-04 — VLM 独立清洗 `run_vlm_cleaning.py`

**问题**：计算型 Layer 2 pair checks（`pair_checks.py`）使用间接几何代理指标（connected_components ≤ 3、edit_locality、center_drift 等），与人眼视觉判断相关性弱。deletion 仅 39.1% 通过率，大量视觉合理的编辑被误杀。

**决策**：新增 `scripts/tools/run_vlm_cleaning.py`，用本地 Qwen3.5-27B VLM 替代 Layer 2 计算检查。渲染 before/after 4 视角对比图 → VLM 结构化评分 → tier 分类。

**关键设计**：
- deletion：有真实 PLY → Blender 渲染（无需 TRELLIS GPU）
- mod/scl/mat/glb：PLY 是点云（0 faces，TRELLIS 导出）→ 从 NPZ 加载 SLAT → TRELLIS decode → Gaussian 渲染
- addition：不评，继承 deletion 分数（before/after 互换，质量等价）
- identity：自动 high
- 复用 `partcraft/cleaning/vlm_filter.py` 的 `call_vlm_judge` + `build_judge_prompt` + 5 维评分

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

- `partcraft/cleaning/vlm_filter.py`：提取 `weighted_score()` 为公共函数
- `partcraft/cleaning/vlm_filter.py`：公开 `build_judge_prompt`、添加 `__all__` 导出
- `partcraft/io/edit_pair_dataset.py`：新增 `quality_dir` / `min_tier` 参数，加载时自动过滤
- `partcraft/pipeline_v2/run.py`：Phase F 对应 `s7_addition_backfill`（cleaning 作为独立工具调用，见 `scripts/tools/run_vlm_cleaning.py`）

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

**代码**：`configs/machine/<hostname>.env`、对应 `configs/partverse_*.yaml` — 机器相关路径集中在 `*.env`，`scripts/tools/download_local_missing_weights.sh` 用于拉取缺失权重。以当前 `*.env` 为准。

---

## 2026-03-29 — Config 显式失败与预渲染路径

**代码**：`partcraft/utils/config.py` 的路径来源与 `[CONFIG_ERROR]`；`scripts/datasets/prerender_common.py` 要求显式 `dataset_root`；关键路径缺失时启动期 fail-fast，不再静默 fallback（详见 `config.py`）。

---

## 2026-03-29 — 一键环境脚本初版

`setup_deploy_env.sh` / `setup_pipeline_env.sh` / `setup_env_common.sh` 初版与 `--machine-env`、`--check`、`--reinstall` 等；后续在 2026-03-30 条目中合并增强（见上）。

---

## 2026-04-11 — Gate A 内联可见性检查，移除 Stage B 和 C_qc

### 背景

Stage B（s2_highlights）的职责是对每个 edit 用 Blender 渲染高亮图，用于：
1. `zero_visible_pixels` 检查（判断选中 part 在该视角是否有像素）
2. 作为 C_qc（sq2_qc_c）的 VLM 输入图像

发现的两个问题：
- Stage B 每个 edit 单独 spawn 一次 Blender 进程（35 个 edits = 35 次启动），代价极高
- Stage C_qc 依赖 Stage B 的高亮图，但 Stage B 与 Stage C 并行执行，可能导致依赖不满足；且 C_qc 整体价值有限

### 解决方案

**关键洞察**：Phase A 在 `phase1/overview.png` 中保存了一张 5×2 网格图，底排就是同一次 Blender 渲染的 per-part 染色视图（每个 part 用 `_PALETTE[pid % 16]` 固定颜色）。这张图已经落盘，无需再次调用 Blender。

**改动**：
1. **`qc_rules.py`**：新增 `count_part_pixels_in_overview(overview_img, view_index, selected_part_ids)` 函数，从 overview.png 底排指定视角格子中数出选中 part 的调色板颜色像素数。
2. **`sq1_qc_a.py`**：在 Gate A 规则检查中，解码 overview.png 一次，对每个 edit 调用上述函数；若像素数为 0，则写入 `zero_visible_pixels` 规则失败，不进入 VLM 队列也不进入后续 2D/3D 编辑。
3. **所有 YAML configs**：从 `stages` 列表中移除 Stage B（`s2_highlights`）和 Stage C_qc（`sq2_qc_c`）。
4. **`run_pipeline_v2_shard.sh`**：移除 Stage B 后台并行执行逻辑（`run_stage_bg` 函数及其调用），主循环回归简单串行。

**效果**：
- 不再需要 Blender 做 zero_visible_pixels 检查，毫秒级完成（纯图像像素统计）
- 高亮图（highlights/）不再生成，节省磁盘和 Blender 时间
- Gate C（2D region QC）整体移除，只保留 Gate A 和最终 Gate E
- 管线阶段由 A→B→C→C_qc→D→D2→E→E_qc→F 简化为 **A→C→D→D2→E→E_qc→F**

### 注意事项
- `count_part_pixels_in_overview` 使用颜色容限 30（L2 距离），抗锯齿
- 当 part_id ≥ 16 时调色板循环（`pid % 16`），极少数情况下可能误判；实际对象绝大多数 < 16 个 parts
- `s2_highlights.py` 和 `sq2_qc_c.py` 文件保留（不删除），仅从 YAML 和调度器中移除


---

## 2026-04-14 — mesh NPZ 从 PLY 改为 GLB（延迟 VD 变换）

### 背景与问题

旧 `pack_npz.py` 从 `source/anno_infos/<uuid>/<uuid>_segmented.glb`（粗网格）加载几何体，导出为仅含顶点色的 `.ply` 文件存入 NPZ。这导致：
- UV 纹理完全丢失（PLY 格式不支持 UV + PBR 材质）
- `s5b_deletion` 只能通过 KD-tree 近似匹配做部件去除，代码复杂且精度低
- `partcraft_loader` 需要在加载时将顶点色 bake 为图像纹理，再多一次转换

### 解决方案：GLB 直接拷贝 + 延迟 VD 变换

**核心思路**：pack 阶段不做任何几何变换，直接将原始 Y-up GLB 字节拷贝进 NPZ，把坐标系变换参数（`vd_scale`, `vd_offset`，来自 `transforms.json`）也存入 NPZ，由读取端在运行时懒惰地应用。

**新 mesh NPZ 格式**（`full.glb` 存在时为 GLB 格式）：

| key | 内容 |
|---|---|
| `full.glb` | 整体纹理 GLB 原始字节（Y-up，源坐标系） |
| `part_N.glb` | 第 N 个部件 GLB 原始字节（Y-up） |
| `vd_scale` | float64 标量，VD 空间均匀缩放因子 |
| `vd_offset` | float64[3] 向量，VD 空间平移（Y-up 坐标系） |

旧格式（`full.ply`）继续向后兼容。

### 改动模块

| 文件 | 改动 |
|---|---|
| `scripts/datasets/partverse/pack_npz.py` | 新增 `_pack_mesh_glb()`：直接拷贝 GLB 字节 + 存 `vd_scale`/`vd_offset`；新增 `--textured-part-glbs-dir`、`--normalized-glb-dir`、`--mesh-out-dir` CLI 参数；修复 multiprocessing pickling（`_pack_worker` 提升为 module-level）|
| `scripts/datasets/partverse/prerender.py` | 透传新 GLB 路径参数 |
| `partcraft/io/partcraft_loader.py` | `_mesh_fmt()` 检测格式；`_load_mesh_bytes()` 加载 GLB 并懒惰应用 VD 变换 |
| `partcraft/render/overview.py` | `extract_parts()` 对新格式 GLB 应用 VD 变换后再写临时文件 |
| `scripts/blender_render_parts.py` | 支持 `.glb` import（`bpy.ops.import_scene.gltf`），before/after 对象快照模式 |
| `partcraft/trellis/refiner.py` | `build_part_mask()` 检测 `full.glb`/`full.ply`，对新格式 GLB 懒惰应用 VD 变换 |
| `partcraft/pipeline_v2/s1_vlm_core.py` | `build_part_menu()` 同时解析 `part_N.glb` 和 `part_N.ply` key |
| `partcraft/pipeline_v2/s5b_deletion.py` | 新增 `_build_deletion_from_npz()`：直接从 NPZ 拼接非选中部件 GLB，无需 KD-tree；旧路径作为 PLY 格式 fallback |
| `configs/prerender_partverse_*.yaml` | 新增 `textured_part_glbs_dir`、`normalized_glb_dir` 配置项 |
| `tests/test_mesh_npz_glb.py` | 7 个 TDD 测试覆盖 pack/load/deletion/refiner/overview 完整路径 |

### 性能收益

pack 阶段消除 `trimesh.load` + `export` 开销，变为纯字节拷贝（I/O bound）。每个对象从 ~2-5s 降至 <0.1s，1203 个对象的单 shard 重 pack 从数小时降至分钟级。

### Repack 指令（各 shard）

```bash
DATA=/mnt/zsn/data/partverse
PY=/mnt/zsn/3dobject/envs/trellis2/bin/python
SCRIPT=/mnt/zsn/zsn_workspace/PartCraft3D/scripts/datasets/partverse/pack_npz.py
COMMON="--data-root $DATA --num-shards 10 \
  --textured-part-glbs-dir $DATA/textured_part_glbs \
  --normalized-glb-dir $DATA/normalized_glbs \
  --force --workers 8"

$PY $SCRIPT $COMMON --shard 00 --mesh-out-dir $DATA/inputs/mesh/00
$PY $SCRIPT $COMMON --shard 02 --mesh-out-dir $DATA/inputs/mesh/02
$PY $SCRIPT $COMMON --shard 06 --mesh-out-dir $DATA/inputs/mesh/06
$PY $SCRIPT $COMMON --shard 07 --mesh-out-dir $DATA/inputs/mesh/07
$PY $SCRIPT $COMMON --shard 08 --mesh-out-dir $DATA/inputs/mesh/08
```

Shard 05 以旧 VD-space GLB 格式重 pack（无 `vd_scale`），代码已向后兼容。

### 注意事项
- `vd_scale`/`vd_offset` 缺失时（旧 GLB 格式或 PLY 格式），各读取端直接使用原始顶点坐标（旧行为不变）
- `_pack_mesh_glb` 中 `_is_int_stem` 过滤非整数文件名，防止 `int(p.stem)` 崩溃
- `s5b_deletion._build_deletion_from_npz` 失败时自动 fallback 到 KD-tree 路径

## 2026-04-19 — H3D_v1 dataset + retire partverse_edit_v1

Replaced the entire `partverse_edit_v1` promotion stack with a
streamlined per-edit-type workflow under `partcraft/cleaning/h3d_v1/`
(library) and `scripts/cleaning/h3d_v1/` (CLIs), targeting
`data/H3D_v1/`.

### Why
- v1 promotion required one big YAML rules file + a half-dozen
  scripts, and the GPU encode step (`encode_del_latent.py`) was a
  separate orchestrator that the pipeline didn't drive automatically.
- New design: one CLI per edit-type group (`pull_deletion`,
  `pull_flux`, `pull_addition`), each consuming `pipeline_v3` output
  directly via `--pipeline-cfg <yaml>`. `pull_deletion` inlines the
  s6b GPU encode (Blender → DINOv2 → SLAT + SS) for any edit whose
  `after.npz` is missing, then hardlinks the dataset bundle in one
  pass. Flux + addition are pure IO.

### What's in (commits `2e2b41d` … `cfe3e10` on
`feature/prompt-driven-part-selection`)
- `partcraft/cleaning/h3d_v1/`: layout, filter (gate-status rules),
  pipeline_io (config-driven walk), manifest (atomic JSONL via
  `fcntl.flock`), asset_pool (idempotent `_assets/` materialise),
  promoter (per-type hardlink + `meta.json`).
- `scripts/cleaning/h3d_v1/`: `pull_{deletion,flux,addition}`,
  `build_h3d_v1_index`, `pack_shard`, `_common.py` (shared CLI bits).
- `tests/cleaning/h3d_v1/`: 54 unit + smoke tests (incl. a synthetic
  E2E that drives all 5 CLIs via subprocess).
- `docs/superpowers/specs/2026-04-19-h3d-v1-design.md`,
  `docs/superpowers/plans/2026-04-19-h3d-v1-plan.md` (with E2/E3
  findings), `docs/runbooks/h3d-v1-promote.md`.

### What's out
- `partcraft/cleaning/v1/` (whole package), `tests/cleaning/v1/`,
  `scripts/cleaning/{encode_del_latent,link_add_npz_from_del,promote_to_v1,rebuild_v1_index,v1_status,run_gate_quality_on_v2,run_gate_text_align_on_v2}.py`,
  `configs/cleaning/promote_v1*.yaml`, related runbooks/specs.
  35 files, 6800 LOC removed.

### Known gotcha
- `partcraft.io.npz_utils.load_ss_encoder` matches only the bare
  string `"cuda"`; passing `cuda:0` silently keeps the model on CPU
  and a torch type mismatch follows on the first conv. `pull_deletion`
  works around this with `_normalize_device_env()` (pins
  `CUDA_VISIBLE_DEVICES=N`, passes `"cuda"` downstream). If the
  underlying loader is ever fixed, the wrapper can simplify.
- Dataset `data/H3D_v1/` must live on the same filesystem as
  `outputs/.../objects/` to keep the hardlink dedup. Cross-FS
  promotes degrade to copies (warnings printed once per source dir).
