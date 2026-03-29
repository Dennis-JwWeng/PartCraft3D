# AI_LOG

## 2026-03-28 — Batch 管线配置化 & node39 适配

### 改动
- `run_shard_batch_pipeline.sh` 改为 `configs/machine/<hostname>.env` 驱动，消除所有硬编码机器路径
- 支持双 conda 环境（服务 vs 管线）、TP 多卡 VLM、`LIMIT` 调试模式
- `launch_local_vlm.sh` 的 `mem_fraction_static` 改为可配置（`VLM_MEM_FRAC`）
- 新增 `configs/machine/node39.env`、`configs/partverse_node39_shard01.yaml`

### 修复的 Bug
- VLM 服务 kill 后子进程（sglang::scheduler/detokenizer）残留导致 GPU 显存不释放，Step3 OOM
- `conda activate` 在非交互 shell 函数中不可用（缺少 `source conda.sh`）
- `set -u` 与 conda activate 脚本中未定义变量冲突（`ADDR2LINE: unbound variable`）
- SGLang CuDNN 版本检查误报阻止启动（`SGLANG_DISABLE_CUDNN_CHECK=1`）
- `mem_fraction_static=0.5` 对 Qwen3.5-27B 单卡 80GB 不够分配 KV cache（改为 0.85）
- 图像编辑服务未显式传 `--model` 路径，fallback 到不存在的 `/mnt/zsn/ckpts`

## 2026-03-29 — Step1 多 Worker 中断恢复 & 容错

### 背景
shard_01 运行 Step1 时，worker 4 (GPU 7) 在处理 39/240 个对象后 crash (exit code 1)。
`wait_for_workers` 直接 raise，merge 从未执行，主文件 `semantic_labels_shard01.jsonl` 未更新。
w0-w3 已完成的 960 条结果散落在 worker 文件中，重启时会被 unlink 导致全部重跑。

### 改动

**`scripts/pipeline_dispatch.py`**
- `wait_for_workers` 新增 `fail_fast` 参数（默认 True，向后兼容）。`fail_fast=False` 时返回失败列表而非直接 raise，让调用方先合并数据再决定是否中断
- 新增 `discover_step1_worker_results()`：发现历史 `semantic_labels{tag}_w*.jsonl` worker 文件
- `reconcile_step4_results` 泛化为 `reconcile_worker_results`（新增 `id_key`、`stage` 参数），Step1/Step4 共用。原函数保留为向后兼容包装

**`scripts/run_pipeline.py`**
- `run_step_semantic_multi_gpu` (Step1)：
  - 分发前调用 `discover_step1_worker_results` + `reconcile_worker_results` 合并历史 worker 文件到主文件，再算 pending（不重复跑已完成的工作）
  - worker crash 后先 `merge_jsonl_by_key` + `write_records` 保存已有结果，再 raise
  - worker 子进程 stdout/stderr 重定向到 `cache/phase0/worker_w{i}.log`，方便排查 crash 原因
- `run_step_3d_edit_multi_gpu` (Step4)：同样改为 crash 时先合并再报错

### 效果
- 重启 pipeline 时自动发现并合并历史 worker 产出，只补跑缺失部分
- worker crash 不再导致已完成数据丢失
- worker crash 原因可通过日志文件定位

## 2026-03-29 — 编辑 Prompt 多样性改进

### 问题
- material 和 scale 编辑由 8 种硬编码模板生成，句式完全固定（如 "Make the {part} taller"、"Change the {part} to wooden material"），缺乏创意
- global 编辑虽由 VLM 生成，但 few-shot 示例仅有 "Make the entire object wooden" 和 "Transform into sci-fi style" 两个，导致 VLM 输出严重趋同（102 次 "Make the entire object wooden"、66 次 "Transform into sci-fi style"）
- identity 编辑仅 8 条固定 prompt，1197 个 object 中唯一率 0.7%
- material 编辑中 VLM 已生成 902 条高质量 prompt（句式多样），但 planner 额外追加了 ~7400 条模板 prompt 稀释了多样性

### 改动

**`partcraft/phase1_planning/enricher.py`**
- `_build_orthogonal_prompt()`：
  - `materials` 从 "0-1" 改为 "2-3" 条，新增 CREATIVE/DIVERSE 指令 + 丰富材质示例列表（volcanic basalt, woven rattan, iridescent beetle shell 等）
  - 新增 `scale_edits` 字段：要求 VLM 为每个 non-core group 生成 1-2 条语义化缩放描述（非模板化的 "bigger/smaller"）
  - `global_edits` 从 2 个 few-shot 示例扩展为 3 个风格各异的示例，并加入强 diversity 指令，禁用 generic 词（"futuristic", "sci-fi", "wooden"），引导 VLM 生成特定艺术流派 / 历史时期 / 自然现象的风格
- `_build_prompt_action()`：同步更新，新增 per-part `materials` 和 `scale_edits` 字段说明
- `_result_groups_to_record()`：提取 VLM 返回的 `scale_edits` 到 group_edits（type="scale"）
- `_result_to_phase0_record()`（legacy path）：同样提取 per-part `materials` 和 `scale_edits`

**`partcraft/phase1_planning/planner.py`**
- group_edits 循环新增 `type=="scale"` 处理分支
- per-part edits 循环新增 VLM 生成的 material 和 scale 提取
- 跟踪 `vlm_material_pids` / `vlm_scale_pids` 集合
- 模板 material/scale 仅对没有 VLM 生成编辑的 part 生效（fallback）

### 效果
- VLM 发挥创造力生成 material/scale/global prompt，不再受限于固定模板
- 模板仅作为 VLM 未覆盖 part 的兜底
- 需重跑 Phase 0（Step 1）使新 prompt 格式生效

### 架构对齐说明
- `docs/ARCH.md` 已同步补充 Step1/Step4 的中断恢复协议（`wait_for_workers` / `reconcile_worker_results` / 先 merge 再 raise）
- `docs/ARCH.md` 已同步补充 Step1 Prompt 责任边界：`enricher.py` 负责 VLM 生成，`planner.py` 负责接入与模板兜底
