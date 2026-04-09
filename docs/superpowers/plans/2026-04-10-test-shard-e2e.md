# Test Shard E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 `pipeline_v2_test_shard00.yaml` 配置，用 LIMIT=20 在全新输出目录跑 20 个对象的完整管线以验证每个 step 的真实运行情况。

**Architecture:** 复用 shard00 的所有输入数据（mesh/images/slat），只改 output_dir 和 n_vlm_servers=8。通过 LIMIT=20 环境变量截取前 20 个对象，output_dir 全新保证从头执行。

**Tech Stack:** bash, Python, YAML, partcraft.pipeline_v2

---

### Task 1: 创建 pipeline_v2_test_shard00.yaml

**Files:**
- Create: `configs/pipeline_v2_test_shard00.yaml`

- [ ] **Step 1: 新建配置文件**

```yaml
# Pipeline v2 — test_shard00：20 对象全流程 E2E 验证
# 机器：dedicated-developjob-saining-r0iyw (8× L20X 48GB)
# 运行：LIMIT=20 bash scripts/tools/run_pipeline_v2_shard.sh test_shard00 configs/pipeline_v2_test_shard00.yaml
# ⚠️ 端口冲突：不能与 shard00/shard02 正式管线同时运行

ckpt_root: /root/workspace/PartCraft3D/checkpoints
blender: /usr/local/bin/blender
data:
  output_dir: /mnt/zsn/data/partverse/outputs/partverse/pipeline_v2_test_shard00
  mesh_root: /mnt/zsn/data/partverse/inputs/mesh
  images_root: /mnt/zsn/data/partverse/inputs/images
  slat_dir: /mnt/zsn/data/partverse/inputs/slat
pipeline:
  gpus: [0,1,2,3,4,5,6,7]
  prerender_workers: 8
  n_vlm_servers: 8
  vlm_port_base: 8002
  vlm_port_stride: 1
  flux_port_base: 8010
  flux_port_stride: 1
  stages:
  - {name: A,    desc: "phase1 VLM + QC-A",  servers: vlm,  steps: [s1, sq1]}
  - {name: B,    desc: "highlights",          servers: none, steps: [s2], optional: true}
  - {name: C,    desc: "FLUX 2D",             servers: flux, steps: [s4]}
  - {name: C_qc, desc: "QC-C 2D region",     servers: vlm,  steps: [sq2]}
  - {name: D,    desc: "TRELLIS 3D edit",     servers: none, steps: [s5],      use_gpus: true}
  - {name: D2,   desc: "deletion mesh",       servers: none, steps: [s5b]}
  - {name: E,    desc: "3D rerender",         servers: none, steps: [s6, s6b], use_gpus: true}
  - {name: E_qc, desc: "QC-E final quality",  servers: vlm,  steps: [sq3]}
  - {name: F,    desc: "addition backfill",   servers: none, steps: [s7]}
services:
  vlm:
    model: /mnt/zsn/ckpts/Qwen3.5-27B
  image_edit:
    enabled: true
    trellis_text_ckpt: /root/workspace/PartCraft3D/checkpoints/TRELLIS-text-xlarge
    image_edit_backend: local_diffusers
    workers_per_server: 2
    export_ply: false
    export_ply_for_deletion: true
    large_part_threshold: 0.35
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

- [ ] **Step 2: 验证 config 可加载**

```bash
python -c "from partcraft.utils.config import load_config; load_config('configs/pipeline_v2_test_shard00.yaml'); print('[OK] load_config')"
```
Expected: `[OK] load_config`

- [ ] **Step 3: dry-run 确认 20 个对象被正确解析**

```bash
LIMIT=20 python -m partcraft.pipeline_v2.run \
  --config configs/pipeline_v2_test_shard00.yaml \
  --shard 00 --all --dry-run 2>&1 | tail -30
```
Expected: 列出 20 个 obj_id，无报错

- [ ] **Step 4: commit**

```bash
git add configs/pipeline_v2_test_shard00.yaml
git commit -m "config: add pipeline_v2_test_shard00 for 20-object E2E validation"
```

---

### Task 2: 启动管线并监控

- [ ] **Step 1: 确认没有冲突的 VLM/FLUX 进程在跑**

```bash
pgrep -a python | grep -E "vllm|image_edit_server|flux" || echo "no conflict processes"
```

- [ ] **Step 2: 在 tmux 中启动管线**

```bash
tmux new-session -d -s test_shard00 \
  "cd /mnt/zsn/zsn_workspace/PartCraft3D && \
   LIMIT=20 bash scripts/tools/run_pipeline_v2_shard.sh test_shard00 \
   configs/pipeline_v2_test_shard00.yaml 2>&1 | tee /tmp/test_shard00.log"
```

- [ ] **Step 3: 确认 tmux session 已启动**

```bash
tmux list-sessions | grep test_shard00
```

- [ ] **Step 4: 监控状态**

每 60 秒打印一次各 step 完成情况：
```bash
watch -n 60 'python3 -c "
import json
from pathlib import Path
from collections import defaultdict
base = Path(\"/mnt/zsn/data/partverse/outputs/partverse/pipeline_v2_test_shard00/objects/00\")
if not base.exists():
    print(\"output dir not yet created\")
    exit()
step_counts = defaultdict(lambda: defaultdict(int))
total = 0
for obj in base.iterdir():
    sf = obj / \"status.json\"
    if sf.exists():
        total += 1
        for step, info in json.loads(sf.read_text()).get(\"steps\", {}).items():
            step_counts[step][info.get(\"status\",\"?\")] += 1
print(f\"Objects with status: {total}\")
for step in sorted(step_counts):
    print(step, dict(step_counts[step]))
"'
```
