#!/usr/bin/env bash
# One-shot backfill for shard05:
#  1) preview_del --best-view-only  (Blender, GPUs 4-7)
#  2) pull_deletion (promote-only, --skip-encode since after.npz already done)
#  3) pull_addition (IO only)
#
# Referenced by docs/runbooks/h3d-v1-promote.md §2b.
set -euo pipefail

REPO=/mnt/zsn/zsn_workspace/PartCraft3D
cd "$REPO"

CFG=configs/pipeline_v3_shard05.yaml
SHARD=05
DATASET=data/H3D_v1
GPUS=0,1,2,3,4,5,6,7

LOG_DIR=logs/s05_backfill_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vinedresser3d

echo "=== [1/3] preview_del --best-view-only  gpus=${GPUS}  $(date -Iseconds) ==="
python -m partcraft.pipeline_v3.run \
  --config "$CFG" --shard "$SHARD" --all \
  --steps preview_del --best-view-only --skip-input-check \
  --gpus "$GPUS" --force 2>&1 | tee "$LOG_DIR/01_preview_del.log"

echo "=== [2/3] pull_deletion --skip-encode  $(date -Iseconds) ==="
PARTCRAFT_CKPT_ROOT="$PWD/checkpoints" \
python -m scripts.cleaning.h3d_v1.pull_deletion \
  --pipeline-cfg "$CFG" --shard "$SHARD" --dataset-root "$DATASET" \
  --skip-encode --workers 8 2>&1 | tee "$LOG_DIR/02_pull_deletion.log"

echo "=== [3/3] pull_addition  $(date -Iseconds) ==="
python -m scripts.cleaning.h3d_v1.pull_addition \
  --pipeline-cfg "$CFG" --shard "$SHARD" --dataset-root "$DATASET" \
  --workers 8 2>&1 | tee "$LOG_DIR/03_pull_addition.log"

echo "=== all done  $(date -Iseconds)  log_dir=$LOG_DIR ==="
