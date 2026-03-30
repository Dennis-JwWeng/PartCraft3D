# AI_LOG

## 2026-03-30 — 环境配置脚本完善（跨机器一键部署）

### 背景
在当前机器调试过程中积累了大量环境级修复（spconv CUDA 版本、flash_attn、diffusers、warp-lang），换机器时需要重新手动操作。将这些修复固化到 `setup_pipeline_env.sh` 和 `setup_deploy_env.sh` 中，实现跨机器一键配置。

### 改动

**`scripts/tools/setup_env_common.sh`**
- 新增 `detect_cuda_suffix()`：从 nvcc / nvidia-smi / torch.version.cuda 自动检测 CUDA 版本后缀（如 `cu124`）
- 新增 `check_flash_attn()` / `check_xformers()`：实际执行 GPU 算子验证可用性（不仅仅检查 import）
- 新增 `resolve_attn_backend()`：优先 flash_attn，fallback 到 xformers

**`scripts/tools/setup_pipeline_env.sh`**（5 阶段流程）
1. 核心依赖（requirements.txt + editable install）
2. spconv + cumm 自动匹配 CUDA 版本（`spconv-cu{CUDA_SUFFIX}`），检测已安装版本是否匹配当前 CUDA
3. warp-lang 安装
4. Attention backend 智能选择：
   - 优先 `pip install flash-attn --no-build-isolation`
   - 失败则扫描同机器其他 conda 环境拷贝已编译的 flash_attn
   - 均失败则 fallback 到 xformers
   - 自动更新 `machine.env` 的 `ATTN_BACKEND` 和 YAML 配置的 `attn_backend`
5. 最终验证（所有模块 import + GPU 可用性）

**`scripts/tools/setup_deploy_env.sh`**
- `diffusers` 版本约束从无约束改为 `>=0.37.1`（FLUX.2-klein-9B 的 `Flux2KleinPipeline` 需要）
- 新增 `Flux2KleinPipeline` import 验证
- `--check` 模式增加版本和 API 可用性报告

**`requirements.txt`**
- 新增 `warp-lang>=1.0`
- spconv 注释更新为 CUDA 版本自适应说明
- 新增 `setup_pipeline_env.sh` 引导说明

### 跨机器部署流程
```bash
# 1. 创建 machine env（从模板复制，修改路径和 GPU 配置）
cp configs/machine/wm1A800.env configs/machine/$(hostname).env
vi configs/machine/$(hostname).env

# 2. 一键配置 pipeline 环境（自动处理 spconv/flash_attn/xformers）
bash scripts/tools/setup_pipeline_env.sh

# 3. 一键配置 deploy 环境（VLM + image edit 服务）
bash scripts/tools/setup_deploy_env.sh
```

### attention backend 决策逻辑
| 优先级 | 条件 | 选择 |
|--------|------|------|
| 1 | flash_attn import 成功且 GPU 算子可执行 | `flash_attn` |
| 2 | xformers import 成功且 memory_efficient_attention 可执行 | `xformers` |
| 3 | 均不可用 | 报错退出，给出手动安装指引 |

选择结果自动写入 `configs/machine/*.env` 和 `configs/partverse_*.yaml`。

## 2026-03-30 — 本机管线配置修复（存储迁移 + diffusers 升级 + attn_backend）

### 背景
shard09 运行到 Step3 时，4 个图像编辑服务器全部启动失败，报错 `AttributeError: module diffusers has no attribute Flux2KleinPipeline`。同时发现 vffey4 存储占用 99%，以及 YAML 配置中存在多处与本机不符的参数。

### 问题诊断

| 问题 | 根因 |
|------|------|
| Step3 edit server 启动失败 | `diffusers==0.36.0` 缺少 `Flux2KleinPipeline`，模型 `model_index.json` 要求 `>=0.37.0.dev0` |
| Step4 TRELLIS 会 crash | `pipeline.attn_backend: flash_attn`，但本机 `vinedresser3d` 环境未安装 `flash_attn` |
| 存储空间不足 | `/mnt/cfs/vffey4`（20T）已用 99%，仅剩 392G；`/mnt/cfs/8o00i7`（256T）空闲 |
| 无本机 hostname machine env | `run_shard_batch_pipeline.sh` 无法按 `$(hostname).env` 自动加载 |
| image_edit_base_urls 多一条 | 原配置 5 条（端口 8004-8008），本机只有 4 GPU，第 5 条永远不健康 |

### 改动

**`qwen_test` conda 环境**
- `diffusers 0.36.0` → `0.37.1`（从 PyPI main 安装）
- 验证：`from diffusers import Flux2KleinPipeline` 可正常导入

**新增 `configs/machine/aibox-rd3996bf91f9-7fb898c998-6kmcq.env`**
- 本机 hostname 对应的 machine env，`run_shard_batch_pipeline.sh` 启动时自动加载
- GPU 分配：4× A800-SXM4-80GB，`VLM_TP=1`（单卡 80GB 可容纳 Qwen3.5-27B ~54GB BF16）
- `VLM_GPUS=0,1,2,3` → 4 个 VLM 实例并行（端口 8003-8006），Step1/2 吞吐 ×4
- `EDIT_GPUS=0,1,2,3` → 4 个 FLUX edit server（端口 8004-8007）
- `ATTN_BACKEND=xformers`
- `OUTPUT_ROOT=/mnt/cfs/8o00i7/3dedit/outputs/partverse`（vffey4 已满，改写 8o00i7）
- `DATA_DIR`/`TRELLIS_CKPT_ROOT`/`VLM_CKPT`/`EDIT_CKPT` 保留在 vffey4（只读，已有数据）

**`configs/partverse_wm1A800_shard00.yaml`**
- `pipeline.attn_backend: flash_attn` → `xformers`
- `image_edit_base_urls`: 5 条 → 4 条（端口 8004-8007，匹配 4 GPU）
- `image_edit_workers: 10` → `8`（4 servers × 2 workers）
- `data.output_dir` → `/mnt/cfs/8o00i7/3dedit/outputs/partverse`

### 存储格局（修复后）

| 挂载点 | 容量 | 用途 |
|--------|------|------|
| `/mnt/cfs/vffey4` | 20T，99% 占用 | ckpts + data（只读） |
| `/mnt/cfs/8o00i7` | 256T，0% 占用 | **新输出根目录** |
| `/mnt/pfs/ca41bi` | 201T，35% 占用 | 备用 |

### shard09 续跑说明
shard09 的 phase0/phase1 cache 已迁移到 pfs（`/mnt/pfs/ca41bi/omni3d/partverse/shard_09/`），直接使用新配置续跑即可：
```bash
SHARD=09 STEPS=3,4,5,6 bash scripts/tools/run_shard_batch_pipeline.sh
```

### 存储格局（最终）

| 挂载点 | 容量 | 用途 |
|--------|------|------|
| `/mnt/cfs/vffey4` | 20T，99% 占用 | ckpts 只读（Qwen/FLUX/TRELLIS） |
| `/mnt/pfs/ca41bi/omni3d/partverse` | 201T，33% 占用 | **主数据目录 + 输出根目录** |
| `/mnt/cfs/8o00i7` | 256T，空闲 | 备用 |

`DATA_DIR` 与 `OUTPUT_ROOT` 均指向 pfs，shard_XX 输出产物直接写入数据目录下。

## 2026-03-30 — Step4 Sampling 后保存崩溃修复（detach + 多 GPU 启动参数）

### 问题
Step4 的 4 个 GPU worker Sampling 全部正常完成（`Sampling: 100%` × 25 steps），但每条编辑在保存 NPZ 时全部失败：
```
partcraft.pipeline ERROR: Failed xxx: Can't call numpy() on Tensor that requires grad.
Use tensor.detach().numpy() instead.
```
所有 per-GPU 结果 `edit_results_shard09_gpu{N}.jsonl` 中 ok=0、fail=全部。

### 根因
`trellis_refine.py` 的 `_save_npz()` 方法在将 Sampling 产出的 SLAT tensor 和 SS latent 保存为 `.npz` 时，直接调用 `.cpu().float().numpy()`。Sampling 返回的 `slat.feats` 和 `z_s` 带有 `requires_grad=True`（因 diffusion 反向采样中的梯度追踪），PyTorch 禁止对需要梯度的 tensor 调用 `.numpy()`。

此外首次重启时启动命令缺少 `--gpus 0 1 2 3`，导致 `pipeline_orchestrator.py` 走了单 GPU 分支（`run_step_3d_edit` 而非 `run_step_3d_edit_multi_gpu`），只用 1 卡串行处理 29732 条编辑。

### 改动

| 文件 | 改动 |
|------|------|
| `partcraft/phase2_assembly/trellis_refine.py` | `_save_npz()` 中 `slat.feats` / `slat.coords` / `z_s` 在 `.cpu()` 前加 `.detach()` |

启动命令修正：
```bash
python scripts/run_pipeline.py --config ... --steps 4 --shard 09 --gpus 0 1 2 3
```

### 验证
修复后 4 卡 GPU worker 全部正常产出成功结果（gpu0: 2 ok, gpu1: 2 ok, gpu2: 1 ok, gpu3: 1 ok），显存 16-27 GB/卡，GPU 利用率 100%。

## 2026-03-30 — Step4 GPU worker SIGFPE 修复（spconv 升级 + flash_attn 安装）

### 问题
Step4 的 4 个 GPU worker（处理 modification/scale/material 类型）在 TRELLIS Sampling 第一步全部被 SIGFPE (signal 8) 杀死，非 GPU worker（deletion/addition/identity）正常完成。

### 诊断过程

1. 初始怀疑 `xformers.ops.fmha.BlockDiagonalMask` 在 0.0.28 中被移到了 `attn_bias` 子模块 → 修复了 import 但 SIGFPE 依旧
2. 安装 `flash_attn` 切换 attention backend → SIGFPE 依旧，排除 attention 层
3. 在 `run_pipeline.py` 添加 `faulthandler.enable()` + GPU worker stdout 重定向到独立日志 → 捕获到 C 级崩溃堆栈
4. 堆栈定位：`interweave_Trellis_TI → RF_sample_once → inference_model → structured_latent_flow.forward → SparseConv3d → spconv.implicit_gemm → cumm.tensorview.from_numpy → SIGFPE`

### 根因
`spconv-cu120` 2.3.6 + `cumm-cu120` 0.4.11 在 CUDA 12.4 运行时的 `implicit_gemm` 内核触发浮点异常。cu120 编译的算子与 cu124 驱动不完全兼容。

### 改动

| 操作 | 说明 |
|------|------|
| `spconv-cu120` 2.3.6 → `spconv-cu124` 2.3.8 | 匹配 CUDA 12.4，修复 `implicit_gemm` SIGFPE |
| `cumm-cu120` 0.4.11 → `cumm-cu124` 0.7.11 | spconv-cu124 的依赖 |
| 安装 `flash_attn` 2.7.3 | 从 `trellis2` 环境拷贝（torch 2.6+cu124 编译，兼容 torch 2.5） |
| `ATTN_BACKEND` → `flash_attn` | `wm1A800.env` + `partverse_wm1A800_shard00.yaml` |
| `run_pipeline.py` 添加 `faulthandler` | GPU worker 崩溃时输出 C 级堆栈 |
| GPU worker stdout 重定向 | 写入 `cache/phase2_5/worker_gpu{N}.log`，不再混入主日志 |
| `xformers` BlockDiagonalMask 兼容 | `full_attn.py` / `windowed_attn.py` / `serialized_attn.py` fallback 到 `attn_bias` 子模块 |

### 验证
Step4 4 卡 GPU worker 全部完成首次 Sampling (25 steps)，显存占用 18-26 GB/卡，正常写入 `edit_results_shard09_gpu{N}.jsonl`。

## 2026-03-30 — Save SS + SLAT as NPZ for all edit types

### 改动

**`third_party/interweave_Trellis.py`**
- `interweave_Trellis_TI` 返回值从 `slat_new` 改为 `{"slat": slat_new, "z_s_before": z_s, "z_s_after": z_s_new}`
- TextureOnly 分支新增 SS encoder 编码（`z_s = z_s_new`，结构不变）
- Deletion/HybridDeletion 分支新增 SS encoder 编码 before/after

**`partcraft/phase2_assembly/trellis_refine.py`**
- 新增 `encode_ss(coords)` 方法：从坐标构建占据网格并编码为 SS VAE latent
- `edit()` 返回值改为 `list[dict]`（每个 dict 包含 `slat`, `z_s_before`, `z_s_after`）
- `export_pair()` / `export_pair_shared_before()` 重写为保存 `before.npz` / `after.npz`（keys: `slat_feats`, `slat_coords`, `ss`）
- 新增 `export_deletion_pair()` 方法：对 deletion 编辑类型进行 SLAT 过滤 + SS 编码 + npz 保存
- 新增 `load_pair_npz()` 静态方法：兼容读取新 npz 格式和旧 `*_slat/` 目录格式

**`scripts/pipeline_step_3d.py`**
- GPU 编辑路径：解包 `edit()` 返回的 dict，传递 `z_s_before`/`z_s_after` 到 `export_pair`
- Deletion 路径：新增 `ensure_refiner()` + `build_part_mask` + `export_deletion_pair` 以产出 SLAT+SS npz
- Addition 路径：复制 npz 文件（before/after 互换）
- Identity 路径：复制 `before.npz` 到 before/after
- 复用 deletion 已加载的 `ori_slat`，避免 GPU 编辑路径重复加载

**`partcraft/streaming_lookahead.py`**
- Deletion/GPU/Addition/Identity 路径同步更新（与 pipeline_step_3d.py 对齐）

**下游消费者兼容更新**
- `scripts/merge_streaming_workers.py`：完成度检测增加 `after.npz`
- `partcraft/phase3_filter/vlm_filter.py`：存在性检测增加 `before.npz`/`after.npz`
- `scripts/vis/render_gs_pairs.py`：`load_slat()` 优先从 npz 加载，fallback 到旧格式
- `scripts/datasets/partobjaverse/build_dataset.py`：数据集构建同时记录 npz 和旧格式路径

### 产物格式

```
mesh_pairs/{edit_id}/
  before.npz   # keys: slat_feats [N,8], slat_coords [N,4], ss [C,R,R,R]
  after.npz    # keys: slat_feats [N,8], slat_coords [N,4], ss [C,R,R,R]
```

### 编辑类型覆盖

| 编辑类型 | SS 来源 | SLAT 来源 |
|----------|---------|-----------|
| modification, scale | interweave S1 repaint z_s/z_s_new | interweave S2 repaint |
| material, global | SS encoder 从 coords 编码（before==after） | interweave S2 repaint |
| deletion | SS encoder 从原始/过滤后 coords 编码 | 原始 SLAT 过滤 |
| addition | 从 deletion 复制并互换 before/after | 从 deletion 复制并互换 |
| identity | 从首个编辑对复制 before.npz | 从首个编辑对复制 |

## 2026-03-30 — node39 配置整理 & 管线启动修复

### 改动

**配置文件**
- 重写 `configs/partverse_node39_shard01.yaml`：
  - 所有路径与 `configs/machine/node39.env` 对齐（ckpt_root、blender_path 等）
  - 补充 `export_ply: false` / `export_ply_for_deletion: true`（默认不写 PLY）
  - `phase2_5` 多服务端口与实际运行一致（5 GPU × 5 端口 8004-8008）
  - `pipeline.attn_backend` 改为 `xformers`（node39 无 flash_attn）
  - 各 section 加中文注释分隔 + 字段对齐，提升可读性
- 新建 `configs/prerender_partverse_node39.yaml`：
  - 所有路径绑定 node39 绝对路径
- `configs/machine/node39.env`：新增 `ATTN_BACKEND=xformers`
- `scripts/tools/run_shard_batch_pipeline.sh`：`ATTN_BACKEND` 从硬编码 `flash_attn` 改为 `${ATTN_BACKEND:-xformers}`

**管线数据路径修复**（`img_enc_dir` 不应被管线要求）
- 问题：`derive_dataset_subpaths` 会自动派生 `img_enc_dir = data_dir/img_Enc`，但编辑管线只需要打包后的 `images/*.npz` + `slat/`，不需要预渲染原始输出 `img_Enc/`。之前跑通是因为旧代码没有对 `img_enc_dir` 做存在性校验，"Config 驱动与显式失败"改动加了强校验后导致启动报错。
- `partcraft/utils/config.py`：`derive_dataset_subpaths` 不再自动派生 `img_enc_dir`（只派生 `image_npz_dir`、`mesh_npz_dir`、`slat_dir`）。预渲染链路仍通过 `paths.img_enc_dir` 显式配置。
- `scripts/pipeline_common.py`：`resolve_data_dirs()` 的 `img_enc_dir` 从必需改为可选（返回 `None` 时由 TrellisRefiner 走 mesh NPZ fallback）
- `partcraft/phase2_assembly/trellis_refine.py`：`img_enc_dir` 初始化校验改为 warning + fallback（`None` 时从 `mesh.npz` 读 `full.ply`）

### 数据路径职责边界
- **预渲染**需要 `img_enc_dir`（`paths.img_enc_dir`）：产出原始渲染图和 `mesh.ply`
- **编辑管线**只需要打包后的产物：`images/*.npz`（`image_npz_dir`）、`mesh/*.npz`（`mesh_npz_dir`）、`slat/`（`slat_dir`）
- `img_enc_dir` 在管线中唯一的消费点是 `trellis_refine.py` 加载 VD mesh.ply 的 fallback 路径，而 `mesh.npz` 中的 `full.ply` 已经是等价替代

## 2026-03-30 — Step4 默认不落盘 PLY（保留可选导出）

### 改动
- `phase2_5` 新增导出开关：
  - `export_ply: false`（默认关闭 GPU 编辑的 `before.ply/after.ply` 落盘）
  - `export_ply_for_deletion: true`（默认保留 deletion 的 GT mesh PLY，保证兼容链路）
- `partcraft/phase2_assembly/trellis_refine.py`：
  - `export_pair()` / `export_pair_shared_before()` 改为始终写 `before_slat/after_slat`，仅在 `export_ply=true` 时写 PLY
  - `direct_delete_mesh()` 增加 `export_ply` 参数（支持按配置关闭 deletion PLY）
- `scripts/pipeline_step_3d.py` 与 `partcraft/streaming_lookahead.py`：
  - 透传 `export_ply` / `export_ply_for_deletion`
  - identity 逻辑从“必须有 before.ply”调整为“before_slat 或 before.ply 均可”
- `scripts/merge_streaming_workers.py`：
  - mesh pair 完成统计从仅检测 `after.ply` 扩展为 `after.ply` 或 `after_slat`

### 目的
- 将 `mesh_pairs` 的磁盘占用从“PLY 主导”切换为“SLAT 主导”，默认显著降低单 shard 存储压力；
- 同时保留显式开关，便于在需要 mesh 预筛/可视化时恢复 PLY 导出。

## 2026-03-29 — Config 驱动与显式失败（预渲染 + 管线）

### 改动
- `partcraft/utils/config.py`：
  - 新增关键路径来源记录：`[CONFIG_PATH] key=value source=...`
  - 关键配置错误统一为 `[CONFIG_ERROR] ...`
  - 移除 `ckpt_root` 的隐式机器 fallback（必须来自 config 或 env 覆盖）
- 预渲染侧：
  - `scripts/datasets/prerender_common.py` 禁止通过 `img_enc_dir.parent` 推导数据根，必须显式传 `dataset_root`
  - 无 GPU 时渲染/编码直接失败（不再降级继续）
  - `scripts/datasets/partverse/prerender.py`：`captions_json` 缺失、空 shard、空对象集均改为硬失败
- 管线侧：
  - Step3/Step4 统一 `2d_edits_{run_token}` 契约，并要求显式 `edit_dir`
  - 移除 `image_edit_base_url` 的隐式回退（不再回退到 `vlm_base_url` 或 `localhost`）
  - Step5/Step6 前置产物缺失改为硬失败；manifest/worker 结果缺失不再静默跳过
  - `TrellisRefiner` 禁止 `slat_dir/img_enc_dir` 默认回退到 `partobjaverse_tiny`

### 验证用例
- 负向（应立即报错）：
  - 缺 `paths.source_glb_dir` / `paths.captions_json`
  - 缺 `phase2_5.image_edit_base_url`
  - 缺 `data.slat_dir` 或 `data.img_enc_dir`
- 正向（应正常启动）：
  - 完整 config 下执行 `prerender.py --encode-only`
  - 完整 config 下执行 `run_pipeline.py --steps 3 4`

## 2026-03-29 — 一键环境初始化脚本（分开部署/管线）

### 改动
- 新增 `scripts/tools/setup_deploy_env.sh`：初始化 `CONDA_ENV_SERVER`（VLM + image edit 服务依赖）
- 新增 `scripts/tools/setup_pipeline_env.sh`：初始化 `CONDA_ENV_PIPELINE`（pipeline 依赖）
- 新增 `scripts/tools/setup_env_common.sh`：统一参数解析与 machine env 加载逻辑，支持：
  - `--machine-env <path>`
  - `--check`（只校验不安装）
  - `--reinstall`（强制重装 pip 依赖）

### 使用方式
```bash
bash scripts/tools/setup_deploy_env.sh
bash scripts/tools/setup_pipeline_env.sh
```

### 边界
- 当前版本只做 conda/python 依赖安装与校验，不包含 checkpoint/权重下载。

## 2026-03-29 — PartVerse 本机预渲染与管线适配（/mnt/cfs）

### 改动
- 新增 `configs/machine/wm1A800.env`，将 batch 运行器所需路径统一切到本机：
  - `VLM_CKPT=/mnt/cfs/vffey4/3dedit/ckpts/Qwen3.5-27B`
  - `EDIT_CKPT=/mnt/cfs/vffey4/3dedit/ckpts/FLUX.2-klein-9B`
  - `TRELLIS_CKPT_ROOT=/mnt/cfs/vffey4/3dedit/ckpts`
  - `DATA_DIR=/mnt/cfs/vffey4/3dedit/data/partverse`
  - `OUTPUT_ROOT=/mnt/cfs/vffey4/3dedit/outputs/partverse`
- 新增 `configs/partverse_wm1A800_shard00.yaml`，作为本机 shard00 的 batch 配置模板（ckpt/data/output 全部指向 `/mnt/cfs/vffey4/3dedit`）。
- 新增 `configs/prerender_partverse_wm1A800.yaml`，用于本机 PartVerse 预渲染，显式绑定：
  - `paths.source_glb_dir=/mnt/cfs/vffey4/3dedit/data/partverse/source/normalized_glbs`
  - `paths.captions_json=/mnt/cfs/vffey4/3dedit/data/partverse/source/text_captions.json`
- 新增 `scripts/tools/download_local_missing_weights.sh`：
  - 默认下载根目录为 `/mnt/cfs/vffey4/3dedit/ckpts`（可由 `PARTCRAFT_CKPT_ROOT` 覆盖）
  - 支持 `MODE=all|vlm|edit`
  - 目录已存在时自动 skip
  - 支持 `VLM_REPO_ID` / `EDIT_REPO_ID` 覆盖 repo_id，便于替换镜像或仓库

### 目的
- 将 node39 历史路径（`/Node11_nvme/...`）从本机运行路径中剥离，保证当前机器开箱可跑。
- 为“缺 VLM + 图像编辑权重”的场景提供统一下载入口，减少手工操作和路径不一致问题。

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
