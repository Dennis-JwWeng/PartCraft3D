# Test Shard E2E Design: 20-Object Full Pipeline Validation

**Date**: 2026-04-10  
**Status**: Approved  
**Machine**: dedicated-developjob-saining-r0iyw (8× L20X 48GB)

## Background

当前正式管线（shard00/shard02）实际运行中存在问题：shard00 有 855 个对象处于不同状态，
其中 793 个完成了 s1（VLM Phase1），但只有 12 个通过了 sq1（QC-A），后续 C/D/E 各阶段
有 s4/s5/s5b/s6/s6b 等失败记录。需要用一个隔离的小规模 test shard 从头验证每个 step 的
真实运行情况。

## Goal

- 从 shard00 输入数据中取 **20 个全新对象**（从未被正式管线处理的）
- 创建独立的 `test_shard00` 配置，输出目录与正式 shard 完全隔离
- 运行所有 Stage（A → C → C_qc → D → D2 → E → E_qc → F）
- 使用 **全部 8 个 GPU**（VLM 阶段：每 GPU 一个 server，TP=1）

## Design

### 配置文件

**新增**：`configs/pipeline_v2_test_shard00.yaml`

与正式 `pipeline_v2_shard00.yaml` 的差异：

| 字段 | 正式 shard00 | test_shard00 |
|------|-------------|--------------|
| `data.output_dir` | `.../pipeline_v2_shard00` | `.../pipeline_v2_test_shard00` |
| `pipeline.n_vlm_servers` | 4 | **8**（每 GPU 一个，TP=1） |
| `pipeline.vlm_port_stride` | 10 | 1 |
| 输入数据路径 | — | 完全复用（mesh/images/slat 同 shard00） |
| stages | A,B,C,C_qc,D,D2,E,E_qc,F | 相同 |

`n_vlm_servers: 8` 配合 `vlm_port_base: 8002, vlm_port_stride: 1` 产生端口
8002~8009，GPU 0~7 各一个 Qwen-27B 实例（TP=1 适合 48GB 卡）。

### 运行命令

唯一入口（与正式管线相同脚本）：

```bash
LIMIT=20 bash scripts/tools/run_pipeline_v2_shard.sh test_shard00 \
    configs/pipeline_v2_test_shard00.yaml
```

- `LIMIT=20`：pipeline 从 `inputs/mesh/00/` sorted 列表取前 20 个对象，
  由于 output_dir 全新（无任何 status.json），所有 step 从头执行
- `test_shard00`：tmux 会话名

### GPU 资源分配

| Stage | Step(s) | GPU 使用方式 |
|-------|---------|-------------|
| A | s1 (VLM Phase1), sq1 (QC-A) | 8× VLM server，GPU 0-7 各一个 |
| C | s4 (FLUX 2D edit) | 8× FLUX server，GPU 0-7 各一个 |
| C_qc | sq2 (QC-C) | 8× VLM server |
| D | s5 (TRELLIS 3D) | 全 8 GPU（use_gpus: true） |
| D2 | s5b (deletion mesh) | CPU 为主 |
| E | s6+s6b (3D rerender) | 全 8 GPU（use_gpus: true） |
| E_qc | sq3 (QC-E) | 8× VLM server |
| F | s7 (addition backfill) | CPU 为主 |

### 验证方法

运行中观察：
```bash
tmux attach -t test_shard00
```

Stage 完成后汇总进度：
```bash
python3 -c "
import json
from pathlib import Path
from collections import defaultdict
base = Path('/mnt/zsn/data/partverse/outputs/partverse/pipeline_v2_test_shard00/objects/00')
step_counts = defaultdict(lambda: defaultdict(int))
for obj in base.iterdir():
    sf = obj / 'status.json'
    if sf.exists():
        for step, info in json.loads(sf.read_text()).get('steps', {}).items():
            step_counts[step][info.get('status','?')] += 1
for step in sorted(step_counts):
    print(step, dict(step_counts[step]))
"
```

失败分析：每个对象的 `status.json` 记录 fail step 及原因；stage 日志在 tmux 终端。

> ⚠️ **端口冲突约束**：test_shard00 使用 `vlm_port_stride: 1`（端口 8002~8009），
> 而正式 shard00/shard02 用 `vlm_port_stride: 10`（端口 8002,8012,...）。
> 两个管线**不能同时运行 VLM/FLUX server**，运行 test_shard 前需确认正式管线已停止。

## Implementation Plan

1. 新建 `configs/pipeline_v2_test_shard00.yaml`（基于 shard00，改 output_dir 和 n_vlm_servers）
2. 启动 test shard：`LIMIT=20 bash scripts/tools/run_pipeline_v2_shard.sh test_shard00 configs/pipeline_v2_test_shard00.yaml`

## Success Criteria

- 20 个对象全部完成 Stage A（s1+sq1），至少 15 个通过 QC-A（pass_rate ≥ 75%）
- 后续 C/D/E 阶段无系统性崩溃（所有 step 能走完流程，失败率可观察）
- 每个 step 的 status.json 有完整记录，便于调试
