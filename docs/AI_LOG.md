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
