# 新机器接入：跑通 pipeline_v2 全阶段

本文是 **新主机** 上从「已有打包数据」到 **跑完 `pipeline_v2` 各阶段** 的唯一详细步骤说明；`docs/ARCH.md` 仅作索引与契约摘要，不重复本页流程。

## 目标与前提

- **目标**：在新机器上配置 conda / 路径一次，然后用仓库约定的 shell 入口跑 **全部** `pipeline_v2` 阶段（`A`、`C`、`D`、`D2`、`E`、`F`；可选阶段 `B` 是否启用以所选 YAML 的 `pipeline.phases` 为准）。
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

## 4. 运行 shard 管线（默认阶段与 `PHASES` 覆盖）

在仓库根目录：

```bash
bash scripts/tools/run_pipeline_v2_shard.sh <tag> configs/pipeline_v2_shardXX.yaml
```

- **`<tag>`**：日志与输出子目录标识（例如 `shard00`），与 `SHARD` 等约定见 `docs/ARCH.md`。
- **`configs/pipeline_v2_shardXX.yaml`**：换成你实际使用的 shard 配置。

**默认阶段**：未设置环境变量 `PHASES` 时，脚本使用配置里 `pipeline.phases` 的默认列表（与 `WITH_OPTIONAL` 等逻辑见 `run_pipeline_v2_shard.sh` 注释）。

**覆盖阶段**（逗号分隔、无空格）：

```bash
PHASES=A,C,D,D2,E,F bash scripts/tools/run_pipeline_v2_shard.sh <tag> configs/pipeline_v2_shardXX.yaml
```

需要单次指定非默认 machine 文件时：

```bash
MACHINE_ENV=configs/machine/<your-profile>.env bash scripts/tools/run_pipeline_v2_shard.sh <tag> configs/pipeline_v2_shardXX.yaml
```

## machine env 与 pipeline YAML 的分工

| 层级 | 典型文件 | 回答的问题 |
|------|-----------|------------|
| **Machine profile** | `configs/machine/<hostname>.env` | 本机 **在哪**：conda、权重根路径、数据/输出根目录、可选 GPU/工具路径 |
| **Pipeline 运行** | `configs/pipeline_v2_*.yaml` | 本次任务 **跑什么**：shard、数据子路径、`pipeline.phases`、端口/步长、各 phase 选项 |

原则：**同一语义的路径不应在两处各写一份绝对路径**；当前仓库里个别键仍存在历史重复，迁移中请以「单一真源」为准逐步收敛。

### VLM 路径：`VLM_CKPT` 与 YAML 中的 `phase0.vlm_model`

- **SGLang / VLM 启动脚本**使用 machine env 中的 **`VLM_CKPT`** 作为权重目录。
- YAML 里若存在 **`phase0.vlm_model`**（或等价字段），其值应与实际加载的模型路径/命名 **人工保持一致**，直到代码侧完成 env/YAML 去重；否则易出现「校验通过但运行时指向不一致」的隐蔽问题。

## 边缘情况：`DATA_DIR` / `OUTPUT_ROOT` 与校验

`check_machine_env_for_pipeline.sh` 与 setup 脚本的 **`--check`** 会要求 **`DATA_DIR`、`OUTPUT_ROOT` 所指路径在磁盘上存在**。

若你的数据布局 **并不使用这两个目录字面路径**（例如数据全部在别处、仅通过软链或 YAML 指过去），可任选其一满足校验：

- **创建空目录**再挂载或绑定到真实数据；或  
- **创建符号链接**，使 `DATA_DIR` / `OUTPUT_ROOT` 解析到实际使用的根路径。

这样路径校验可通过，且与「machine profile = 本机锚点」的约定一致。若长期希望弱化该约束，需另行变更校验列表或文档契约。
