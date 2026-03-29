# AI_LOGS

## 2026-03-29 — 本机 machine config 与 strict-config 收口

### 新增
- `configs/machine/dedicated-developjob-saining-r0iyw.env`
  - `CONDA_INIT=/root/miniconda3/etc/profile.d/conda.sh`
  - `CONDA_ENV_SERVER=qwen_test`
  - `CONDA_ENV_PIPELINE=partcraft3d`
  - 本机默认 `DATA_DIR/OUTPUT_ROOT` 与 checkpoint 根路径已指向当前工作区

### 说明
- 本机可直接用以下命令启动 shard runner：
  - `SHARD=06 bash scripts/tools/run_shard_batch_pipeline.sh`
- 若环境名或 checkpoint 路径与机器实际不一致，优先修改 `configs/machine/dedicated-developjob-saining-r0iyw.env`。
