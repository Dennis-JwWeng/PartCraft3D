# 新机器接入：跑通 pipeline_v2 全阶段

本文是 **新主机** 上从「已有打包数据」到 **跑完 `pipeline_v2` 各阶段** 的唯一详细步骤说明；`docs/ARCH.md` 仅作索引与契约摘要，不重复本页流程。

数据目录键名（`images_root` / `image_npz_dir` 等）与 `load_config` 对齐规则见 [`dataset-path-contract.md`](dataset-path-contract.md)。

## 目标与前提

- **目标**：在新机器上配置 conda / 路径一次，然后用仓库约定的 shell 入口跑 **全部** `pipeline_v2` 阶段（`A`、`C`、`D`、`D2`、`E`、`F`；可选阶段 `B` 是否启用以所选 YAML 的 `pipeline.stages` 为准）。
- **前提**：**打包 NPZ 等数据已在该机器可用**（例如 `images` / `mesh` 等目录与 config 中 `data.*` 一致），本指南不负责从原始数据重新预处理。

## 1. 复制并编辑 machine 环境文件

机器相关路径（conda、权重、数据根目录）集中在 **`configs/machine/<hostname>.env`**，运行时默认按 `$(hostname)` 加载，也可用 `MACHINE_ENV` 指向任意文件。

任选其一作为模板复制后改名并编辑路径：

- `configs/machine/node39.env`（参考机器完整示例）
- `configs/machine/local_symlink.env.example`（本机 `data/`、`outputs/` 为仓库下软链时的模板）

将其中 **`CONDA_INIT`、`CONDA_ENV_SERVER`、`CONDA_ENV_PIPELINE`、`VLM_CKPT`、`EDIT_CKPT`、`TRELLIS_CKPT_ROOT`、`DATA_DIR`、`OUTPUT_ROOT`** 等改为本机真实路径；若模板使用 `OUTPUT_DIR` 与 `OUTPUT_ROOT=${OUTPUT_DIR}`，保持与校验脚本要求一致即可。

## 2. 校验 machine 环境（推荐最先执行）

在仓库根目录执行（默认读取 `configs/machine/$(hostname).env`）：

```bash
bash scripts/tools/check_machine_env_for_pipeline.sh
```

若主机名与文件名不一致，或需临时指定 profile：

```bash
bash scripts/tools/check_machine_env_for_pipeline.sh --machine-env configs/machine/<your-profile>.env
```

通过则表示必填变量已设置且路径存在，适合作为后续 `--check` 与正式跑批的前置条件。

## 3. 部署环境与管线环境初始化

首次在该机器上建议依次执行（安装/校验 conda 环境与依赖）：

```bash
bash scripts/tools/setup_deploy_env.sh
bash scripts/tools/setup_pipeline_env.sh
```

**仅检查** machine env、conda 可激活等、**不安装包**时：

```bash
bash scripts/tools/setup_deploy_env.sh --check
bash scripts/tools/setup_pipeline_env.sh --check
```

（`--check` 模式下同样支持 `--machine-env <path>`，与其他 setup 脚本一致。）

## 3.5. 确认数据可用性

pipeline 运行前须确认以下三类数据在新机器上可访问（路径与 YAML 中 `data.*` 对应）：

```
<mesh_root>/<shard>/<obj_id>.npz          # 网格 NPZ
<images_root>/<shard>/<obj_id>.npz        # 多视角渲染图 NPZ
<slat_dir>/<shard>/<obj_id>_feats.pt      # SLAT 特征
<slat_dir>/<shard>/<obj_id>_coords.pt
```

- **NFS / 共享存储**（如 `/mnt/zsn`）：挂载后直接可用，无需额外操作。
- **需要拷贝**：用 `rsync -avz` 同步对应 shard 的 `inputs/` 和 `slat/` 目录。
- 快速验证（以 shard02 为例）：

  ```bash
  ls <mesh_root>/02/ | wc -l      # 应与预期对象数一致
  ls <slat_dir>/02/ | wc -l
  ```

## 3.6. 适配新机器的 pipeline YAML

若新机器的数据路径与现有 YAML 中的 `data.*` 不同，需创建机器专用配置，
**不要直接修改共享的 `configs/pipeline_v2_shardXX.yaml`**：

```bash
cp configs/pipeline_v2_shard02.yaml configs/pipeline_v2_shard02_<newmachine>.yaml
```

编辑以下字段（其他保持不变）：

| 字段 | 说明 |
|------|------|
| `data.output_dir` | 输出根目录（新机器上的绝对路径） |
| `data.mesh_root` | 网格 NPZ 根目录 |
| `data.images_root` | 渲染图 NPZ 根目录 |
| `data.slat_dir` | SLAT 特征根目录 |
| `ckpt_root` | checkpoints 根目录 |
| `blender` | Blender 可执行文件路径 |
| `services.vlm.model` | VLM 模型路径（须与 machine env 的 `VLM_CKPT` 保持一致） |

## 3.7. 冒烟验证：先跑 10 个对象全流程

**在提交大规模批跑前，必须先用 `LIMIT=10` 跑通全流程**，验证数据、模型、服务、输出路径均正常：

```bash
# 在 tmux 会话内（仓库根目录）
LIMIT=10 bash scripts/tools/run_pipeline_v2_shard.sh shard02_test10 \
    configs/pipeline_v2_shard02_<newmachine>.yaml
```

预期行为：脚本依次启动 VLM → 跑 Stage A → 停 VLM → 启动 FLUX → 跑 Stage C → … → 全部 stages done。
检查产物：

```bash
# 10 个对象应各有 phase1/parsed.json、edits_2d/ 等
ls <output_dir>/objects/<shard>/ | head -15
```

也可只验证某单个阶段：

```bash
LIMIT=10 STAGES=A bash scripts/tools/run_pipeline_v2_shard.sh shard02_smoke \
    configs/pipeline_v2_shard02_<newmachine>.yaml
```

冒烟通过后再去掉 `LIMIT=10` 提交全量批跑。

## 3.8. 清除错误的运行结果（重跑前重置）

若发现已跑批次的结果有误（如配置错误、VLM 输出不对），需在重跑前清除 phase1 及下游产物：

```python
# 清除指定 shard 下所有对象的 phase1 及依赖步骤
import json, os, shutil, glob

SHARD_OBJ_DIR = "<output_dir>/objects/<shard>/"
DOWNSTREAM_KEYS = {"s1_phase1", "s4_flux_2d", "s5_trellis", "s5b_del_mesh"}

for status_path in glob.glob(SHARD_OBJ_DIR + "*/status.json"):
    obj_dir = os.path.dirname(status_path)
    for sub in ["phase1", "edits_2d"]:
        d = os.path.join(obj_dir, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
    with open(status_path) as f:
        s = json.load(f)
    for k in list(s.get("steps", {}).keys()):
        if k in DOWNSTREAM_KEYS:
            del s["steps"][k]
    with open(status_path, "w") as f:
        json.dump(s, f, indent=2)
```

清除后再按步骤 3.7 先做 `LIMIT=10` 冒烟，确认正常再全量重跑。

## 4. 运行 shard 管线（默认阶段与 `STAGES` 覆盖）

在仓库根目录：

```bash
bash scripts/tools/run_pipeline_v2_shard.sh <tag> configs/pipeline_v2_shardXX.yaml
```

- **`<tag>`**：日志与输出子目录标识（例如 `shard00`），与 `SHARD` 等约定见 `docs/ARCH.md`。
- **`configs/pipeline_v2_shardXX.yaml`**：换成你实际使用的 shard 配置。

**默认阶段**：未设置环境变量 `PHASES` 时，脚本使用配置里 `pipeline.stages` 的默认列表（与 `WITH_OPTIONAL` 等逻辑见 `run_pipeline_v2_shard.sh` 注释）。

**覆盖阶段**（逗号分隔、无空格）：

```bash
STAGES=A,C,D,D2,E,F bash scripts/tools/run_pipeline_v2_shard.sh <tag> configs/pipeline_v2_shardXX.yaml
```

需要单次指定非默认 machine 文件时：

```bash
MACHINE_ENV=configs/machine/<your-profile>.env bash scripts/tools/run_pipeline_v2_shard.sh <tag> configs/pipeline_v2_shardXX.yaml
```

## machine env 与 pipeline YAML 的分工

| 层级 | 典型文件 | 回答的问题 |
|------|-----------|------------|
| **Machine profile** | `configs/machine/<hostname>.env` | 本机 **在哪**：conda、权重根路径、数据/输出根目录、可选 GPU/工具路径 |
| **Pipeline 运行** | `configs/pipeline_v2_*.yaml` | 本次任务 **跑什么**：shard、数据子路径、`pipeline.stages`、端口/步长、各 phase 选项 |

原则：**同一语义的路径不应在两处各写一份绝对路径**；当前仓库里个别键仍存在历史重复，迁移中请以「单一真源」为准逐步收敛。

### VLM 路径：`VLM_CKPT` 与 YAML 中的 `services.vlm.model`

- **SGLang / VLM 启动脚本**使用 machine env 中的 **`VLM_CKPT`** 作为权重目录。
- YAML 里若存在 **`services.vlm.model`**（或等价字段），其值应与实际加载的模型路径/命名 **人工保持一致**，直到代码侧完成 env/YAML 去重；否则易出现「校验通过但运行时指向不一致」的隐蔽问题。

## 边缘情况：`DATA_DIR` / `OUTPUT_ROOT` 与校验

`check_machine_env_for_pipeline.sh` 与 setup 脚本的 **`--check`** 会要求 **`DATA_DIR`、`OUTPUT_ROOT` 所指路径在磁盘上存在**。

若你的数据布局 **并不使用这两个目录字面路径**（例如数据全部在别处、仅通过软链或 YAML 指过去），可任选其一满足校验：

- **创建空目录**再挂载或绑定到真实数据；或  
- **创建符号链接**，使 `DATA_DIR` / `OUTPUT_ROOT` 解析到实际使用的根路径。

这样路径校验可通过，且与「machine profile = 本机锚点」的约定一致。若长期希望弱化该约束，需另行变更校验列表或文档契约。
