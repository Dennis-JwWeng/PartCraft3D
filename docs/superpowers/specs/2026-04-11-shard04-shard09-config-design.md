# Config 整理：shard04 / shard09 on aibox (4× A800-SXM4-80GB)

**日期:** 2026-04-11  
**机器:** `aibox-rd3996bf91f9-68f4cd496c-nsm56`  
**目标:** 清理过时配置，为本机新建 machine env 和 shard04/shard09 pipeline YAML，迁移已有产出目录。

---

## 1. 清理（删除 11 个旧配置）

以下文件已不对应任何在线机器或已被 pipeline_v2 体系取代，统一删除：

**旧格式 YAML（非 pipeline_v2 体系）：**
- `configs/_phase1v2_mirror5.yaml`
- `configs/hybrid_streaming.yaml`
- `configs/partverse_H200_shard00.yaml`
- `configs/partverse_local.yaml`
- `configs/partverse_local_parallel_shard00.yaml`
- `configs/partverse_node39_shard01.yaml`
- `configs/partverse_node39_shard03.yaml`
- `configs/partverse_wm1A800_shard00.yaml`

**过时 pipeline_v2 YAML（机器/模式已弃用）：**
- `configs/pipeline_v2_gpu01.yaml`
- `configs/pipeline_v2_local_symlink.yaml`
- `configs/pipeline_v2_mirror5.yaml`

保留：`pipeline_v2_shard00–03`、`pipeline_v2_test_shard00`、`configs/machine/*.env`（现有）。

---

## 2. 新建 machine env

**文件：** `configs/machine/aibox-rd3996bf91f9-68f4cd496c-nsm56.env`

```bash
CONDA_INIT=/root/miniforge/etc/profile.d/conda.sh
CONDA_ENV_SERVER=/mnt/cfs/vffey4/envs/qwen_test      # sglang 0.5.9
CONDA_ENV_PIPELINE=/mnt/cfs/vffey4/envs/vinedresser3d # partcraft.pipeline_v2

VLM_CKPT=/mnt/cfs/vffey4/omni3d/ckpt/Qwen3.5-27B
EDIT_CKPT=/mnt/cfs/vffey4/omni3d/ckpt/FLUX.2-klein-9B
TRELLIS_CKPT_ROOT=/mnt/cfs/vffey4/omni3d/ckpt

DATA_DIR=/mnt/cfs/vffey4/omni3d/partverse
OUTPUT_ROOT=/mnt/cfs/vffey4/omni3d/partverse/outputs/partverse

BLENDER_PATH=/usr/local/bin/blender
PARTCRAFT_DINOV2_WEIGHTS=/mnt/cfs/vffey4/omni3d/ckpt/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

**注意：** `check_machine_env_for_pipeline.sh` 要求 `DATA_DIR` 和 `OUTPUT_ROOT` 路径在磁盘上存在，需提前 `mkdir -p`。

### CONDA_ENV_* 全路径说明

`qwen_test` 和 `vinedresser3d` 在 `conda env list` 中以全路径出现（无短名），`check_machine_env_for_pipeline.sh` 的 awk 检查以 `$1 == env` 匹配——全路径字符串可通过。`run_pipeline_v2_shard.sh` 用 `conda run -n <val>` 解析，全路径同样被 conda 接受。

---

## 3. 产出目录迁移

### 现状

| 目录 | 内容 | 动作 |
|------|------|------|
| `partverse/phase1_v2_shard04/` | 1203 obj，s1✓ 915，s4✓ 1144，s5✓ 229，s5✗ 915 | **mv → 新路径** |
| `partverse/phase1_v2_shard09/` | 388 obj，s1✓ 274，s4✓ 363，s5✓ 89，s5✗ 274 | 待 shard04 验证后再迁 |
| `partverse/shard_09/` | 老式 mesh_pairs 27650 条 | 不动 |

### 迁移命令

```bash
# 创建输出根目录
mkdir -p /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse

# shard04（先行）
mv /mnt/cfs/vffey4/omni3d/partverse/phase1_v2_shard04 \
   /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard04

# shard09（待 shard04 验证后执行）
# mv /mnt/cfs/vffey4/omni3d/partverse/phase1_v2_shard09 \
#    /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard09
```

---

## 4. 新建 pipeline YAML

### `configs/pipeline_v2_shard04.yaml`

模板：`pipeline_v2_shard00.yaml`（含 QC 阶段 sq1/sq2/sq3）

关键字段：
- `ckpt_root: /mnt/cfs/vffey4/omni3d/ckpt`
- `blender: /usr/local/bin/blender`
- `data.output_dir: /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v2_shard04`
- `data.mesh_root: /mnt/cfs/vffey4/omni3d/partverse/mesh`
- `data.images_root: /mnt/cfs/vffey4/omni3d/partverse/images`
- `data.slat_dir: /mnt/cfs/vffey4/omni3d/partverse/slat`
- `pipeline.gpus: [0, 1, 2, 3]`
- `pipeline.n_vlm_servers: 4`
- `services.vlm.model: /mnt/cfs/vffey4/omni3d/ckpt/Qwen3.5-27B`
- `services.image_edit.trellis_text_ckpt: /mnt/cfs/vffey4/omni3d/ckpt/TRELLIS-text-xlarge`
- `services.image_edit.repaint_mode: image`
- `step_params.s5.num_views: 40`

shard09 同上，仅 `output_dir` 后缀改为 `pipeline_v2_shard09`（迁移后添加）。

---

## 5. 验证步骤

```bash
# 1. 检查 machine env
bash scripts/tools/check_machine_env_for_pipeline.sh \
    --machine-env configs/machine/aibox-rd3996bf91f9-68f4cd496c-nsm56.env

# 2. dry-run shard04（确认配置读取正确，不实际执行）
python -m partcraft.pipeline_v2.run \
    --config configs/pipeline_v2_shard04.yaml \
    --shard 04 --all --dry-run

# 3. 正式 resume（仅跑 s5 失败对象，pipeline 自动 skip 已完成步骤）
STAGES=D,D2,E,F bash scripts/tools/run_pipeline_v2_shard.sh shard04 \
    configs/pipeline_v2_shard04.yaml
```

---

## 实施顺序

1. 删除 11 个旧配置文件
2. 创建 machine env 文件
3. `mkdir -p` 输出根目录
4. `mv` shard04 产出目录
5. 创建 `pipeline_v2_shard04.yaml`
6. 运行 `check_machine_env_for_pipeline.sh` 验证
7. `dry-run` 确认配置路径正确
8. （shard09 迁移和 YAML 在 shard04 验证通过后执行）
