# Shard04 Config Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清理旧配置、新建本机 machine env、迁移 shard04 产出目录、创建 pipeline_v2_shard04.yaml，并验证可 dry-run。

**Architecture:** 纯配置操作，无代码改动。machine env 给 run 脚本提供路径锚点；YAML 提供 shard 级运行参数；已有产出 mv 到新输出根以统一管理。

**Tech Stack:** bash, YAML, `check_machine_env_for_pipeline.sh`, `python -m partcraft.pipeline_v2.run --dry-run`

---

### Task 1: 删除 11 个旧配置文件

**Files:**
- Delete: `configs/_phase1v2_mirror5.yaml`
- Delete: `configs/hybrid_streaming.yaml`
- Delete: `configs/partverse_H200_shard00.yaml`
- Delete: `configs/partverse_local.yaml`
- Delete: `configs/partverse_local_parallel_shard00.yaml`
- Delete: `configs/partverse_node39_shard01.yaml`
- Delete: `configs/partverse_node39_shard03.yaml`
- Delete: `configs/partverse_wm1A800_shard00.yaml`
- Delete: `configs/pipeline_v2_gpu01.yaml`
- Delete: `configs/pipeline_v2_local_symlink.yaml`
- Delete: `configs/pipeline_v2_mirror5.yaml`

- [ ] **Step 1: 删除全部旧配置**

```bash
cd /mnt/cfs/vffey4/omni3d/workspace/PartCraft3D
rm configs/_phase1v2_mirror5.yaml \
   configs/hybrid_streaming.yaml \
   configs/partverse_H200_shard00.yaml \
   configs/partverse_local.yaml \
   configs/partverse_local_parallel_shard00.yaml \
   configs/partverse_node39_shard01.yaml \
   configs/partverse_node39_shard03.yaml \
   configs/partverse_wm1A800_shard00.yaml \
   configs/pipeline_v2_gpu01.yaml \
   configs/pipeline_v2_local_symlink.yaml \
   configs/pipeline_v2_mirror5.yaml
```

- [ ] **Step 2: 确认剩余 configs/ 只有预期文件**

```bash
ls configs/ configs/machine/
```

Expected configs/: `pipeline_v2_shard00–03.yaml`, `pipeline_v2_test_shard00.yaml`, `prerender_*.yaml`, `machine/`

- [ ] **Step 3: Commit**

```bash
git add -A configs/
git commit -m "chore: remove stale pre-pipeline_v2 and legacy configs"
```

---

### Task 2: 新建 machine env

**Files:**
- Create: `configs/machine/aibox-rd3996bf91f9-68f4cd496c-nsm56.env`

- [ ] **Step 1: 创建 machine env 文件**

写入以下内容（`CONDA_ENV_*` 使用全路径，因为这两个 env 没有注册短名）：

```bash
# Machine config: aibox (4× NVIDIA A800-SXM4-80GB)
# CFS mount: /mnt/cfs/vffey4/omni3d/
# To onboard: see docs/new-machine-onboarding.md

# ── Conda ──────────────────────────────────────────────────────────
CONDA_INIT=/root/miniforge/etc/profile.d/conda.sh
CONDA_ENV_SERVER=/mnt/cfs/vffey4/envs/qwen_test      # sglang 0.5.9 (VLM server)
CONDA_ENV_PIPELINE=/mnt/cfs/vffey4/envs/vinedresser3d # partcraft.pipeline_v2

# ── Checkpoints ────────────────────────────────────────────────────
VLM_CKPT=/mnt/cfs/vffey4/omni3d/ckpt/Qwen3.5-27B
EDIT_CKPT=/mnt/cfs/vffey4/omni3d/ckpt/FLUX.2-klein-9B
TRELLIS_CKPT_ROOT=/mnt/cfs/vffey4/omni3d/ckpt

# ── Data I/O ───────────────────────────────────────────────────────
DATA_DIR=/mnt/cfs/vffey4/omni3d/partverse
OUTPUT_ROOT=/mnt/cfs/vffey4/omni3d/partverse/outputs/partverse

# ── Optional tools ─────────────────────────────────────────────────
BLENDER_PATH=/usr/local/bin/blender

# ── DINOv2 offline weights ──────────────────────────────────────────
PARTCRAFT_DINOV2_WEIGHTS=/mnt/cfs/vffey4/omni3d/ckpt/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

- [ ] **Step 2: 创建 OUTPUT_ROOT 目录（check 脚本要求路径存在）**

```bash
mkdir -p /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse
```

- [ ] **Step 3: 运行 check 脚本验证 machine env**

```bash
bash scripts/tools/check_machine_env_for_pipeline.sh \
    --machine-env configs/machine/aibox-rd3996bf91f9-68f4cd496c-nsm56.env
```

Expected: `[CHECK] Machine env OK (conda envs + paths): ...` 以及 `[OK] Machine profile is ready for pipeline_v2 ...`

- [ ] **Step 4: Commit**

```bash
git add configs/machine/aibox-rd3996bf91f9-68f4cd496c-nsm56.env
git commit -m "feat(config): add machine env for aibox A800 node"
```

---

### Task 3: 迁移 shard04 产出目录

**Files:** 纯 filesystem 操作，不涉及 git。

- [ ] **Step 1: 移动 shard04 产出到新路径**

```bash
mv /mnt/cfs/vffey4/omni3d/partverse/phase1_v2_shard04 \
   /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard04
```

- [ ] **Step 2: 确认目录结构完整**

```bash
ls /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard04/
# Expected: _global/  objects/
ls /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard04/objects/04/ | wc -l
# Expected: 1203
```

---

### Task 4: 创建 pipeline_v2_shard04.yaml

**Files:**
- Create: `configs/pipeline_v2_shard04.yaml`
- Reference: `configs/pipeline_v2_shard00.yaml`（模板）

- [ ] **Step 1: 创建 YAML 文件**

```yaml
# Pipeline v2 — shard04 on aibox (4× A800-SXM4-80GB)
# Machine env: configs/machine/aibox-rd3996bf91f9-68f4cd496c-nsm56.env
#
# Inputs:
#   /mnt/cfs/vffey4/omni3d/partverse/mesh/04/<obj_id>.npz
#   /mnt/cfs/vffey4/omni3d/partverse/images/04/<obj_id>.npz
#   /mnt/cfs/vffey4/omni3d/partverse/slat/04/<obj_id>_{feats,coords}.pt
#
# Output root:
#   /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard04/
#
# Resume status (2026-04-11):
#   s1 (VLM) : 915/1203 ok
#   s4 (FLUX): 1144/1203 ok
#   s5 (3D)  :  229/1203 ok,  915 fail → resume from stage D
#
# Quick start (resume from D):
#   STAGES=D,D2,E,F bash scripts/tools/run_pipeline_v2_shard.sh shard04 \
#       configs/pipeline_v2_shard04.yaml
#
# Full run (all stages):
#   bash scripts/tools/run_pipeline_v2_shard.sh shard04 \
#       configs/pipeline_v2_shard04.yaml

ckpt_root: /mnt/cfs/vffey4/omni3d/ckpt
blender: /usr/local/bin/blender
data:
  output_dir: /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard04
  mesh_root: /mnt/cfs/vffey4/omni3d/partverse/mesh
  images_root: /mnt/cfs/vffey4/omni3d/partverse/images
  slat_dir: /mnt/cfs/vffey4/omni3d/partverse/slat
pipeline:
  gpus:
  - 0
  - 1
  - 2
  - 3
  prerender_workers: 8
  n_vlm_servers: 4
  vlm_port_base: 8002
  vlm_port_stride: 10
  flux_port_base: 8004
  flux_port_stride: 1
  stages:
  - {name: A,    desc: "phase1 VLM + QC-A",  servers: vlm,  steps: [s1, sq1]}
  - {name: B,    desc: "highlights",          servers: none, steps: [s2]}
  - {name: C,    desc: "FLUX 2D",             servers: flux, steps: [s4]}
  - {name: C_qc, desc: "QC-C 2D region",     servers: vlm,  steps: [sq2]}
  - {name: D,    desc: "TRELLIS 3D edit",     servers: none, steps: [s5],      use_gpus: true}
  - {name: D2,   desc: "deletion mesh",       servers: none, steps: [s5b]}
  - {name: E,    desc: "3D rerender",         servers: none, steps: [s6, s6b], use_gpus: true}
  - {name: E_qc, desc: "QC-E final quality",  servers: vlm,  steps: [sq3]}
  - {name: F,    desc: "addition backfill",   servers: none, steps: [s7]}
services:
  vlm:
    model: /mnt/cfs/vffey4/omni3d/ckpt/Qwen3.5-27B
  image_edit:
    enabled: true
    trellis_text_ckpt: /mnt/cfs/vffey4/omni3d/ckpt/TRELLIS-text-xlarge
    image_edit_backend: local_diffusers
    workers_per_server: 2
    export_ply: false
    export_ply_for_deletion: true
    large_part_threshold: 0.35
    repaint_mode: image
step_params:
  s5:
    num_views: 40
qc:
  vlm_score_threshold: 0.7
  thresholds_by_type:
    deletion:     {min_visual_quality: 3}
    modification: {min_visual_quality: 3, require_preserve_other: true}
    scale:        {min_visual_quality: 3, require_preserve_other: true}
    material:     {min_visual_quality: 3}
    global:       {min_visual_quality: 3}
    addition:     {min_visual_quality: 3}
```

- [ ] **Step 2: dry-run 验证配置路径正确**

```bash
cd /mnt/cfs/vffey4/omni3d/workspace/PartCraft3D
/mnt/cfs/vffey4/envs/vinedresser3d/bin/python -m partcraft.pipeline_v2.run \
    --config configs/pipeline_v2_shard04.yaml \
    --shard 04 --all --dry-run 2>&1 | head -40
```

Expected: 打印 `[CONFIG_PATH]` 路径审计行，显示 `images_root`、`mesh_root`、`slat_dir`、`output_dir` 均为 CFS 真实路径；无 `[CONFIG_ERROR]`。

- [ ] **Step 3: Commit**

```bash
git add configs/pipeline_v2_shard04.yaml
git commit -m "feat(config): add pipeline_v2_shard04 for aibox A800 node"
```

---

### Task 5: 提交 spec 文档

**Files:**
- Already created: `docs/superpowers/specs/2026-04-11-shard04-shard09-config-design.md`
- Already created: `docs/superpowers/plans/2026-04-11-shard04-config-setup.md`

- [ ] **Step 1: Commit spec 和 plan**

```bash
git add docs/superpowers/specs/2026-04-11-shard04-shard09-config-design.md \
        docs/superpowers/plans/2026-04-11-shard04-config-setup.md
git commit -m "docs: add spec and plan for shard04/09 config setup on aibox"
```
