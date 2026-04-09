# Pipeline smoke checks (Scope D)

Lightweight ways to verify config + wiring before long GPU runs.

## 1. Machine env + conda (host)

```bash
bash scripts/tools/check_machine_env_for_pipeline.sh
bash scripts/tools/setup_pipeline_env.sh --check
```

## 2. Config loads (Python)

```bash
python -c "from partcraft.utils.config import load_config; load_config('configs/pipeline_v2_shard00.yaml'); print('[OK] load_config')"
```

Use your real pipeline YAML path.

## 3. Resolve objects without running steps (`--dry-run`)

Requires valid `data.*` paths and either existing `objects/<shard>/…` output dirs or input NPZ under `mesh_root/<shard>/` when using `--all`.

```bash
python -m partcraft.pipeline_v2.run \
  --config configs/pipeline_v2_shard00.yaml \
  --shard 00 \
  --all \
  --dry-run
```

## 4. Debug subset: `LIMIT`

`docs/ARCH.md` documents `LIMIT`. **`partcraft.pipeline_v2.run` reads it** after `--gpu-shard` slicing: only the first *N* objects are processed.

```bash
LIMIT=1 python -m partcraft.pipeline_v2.run \
  --config configs/pipeline_v2_shard00.yaml \
  --shard 00 \
  --all \
  --phase A
```

(Phase A still needs VLM + data; this only caps object count.)

## 5. Full phased run

See `docs/new-machine-onboarding.md` and `bash scripts/tools/run_pipeline_v2_shard.sh`.

## Blender 可执行文件

`partcraft.pipeline_v2.paths.resolve_blender_executable(cfg)` 决定 s1/s2/s6b 使用的 Blender，优先级：

1. `cfg["tools"]["blender_path"]`
2. 顶层 `cfg["blender"]`（多数 pipeline YAML）
3. 环境变量 `BLENDER_PATH`（仍支持，建议迁到 YAML）
4. 字符串 `"blender"`（依赖 `PATH`，**无**仓库内写死的机器路径）

