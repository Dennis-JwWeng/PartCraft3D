#!/bin/bash
# Multi-GPU Phase 5: PLY render → DINOv2 + SLAT re-encode
# Splits edit pairs across GPUs, one process per GPU.
#
# Usage:
#   GPUS=0,3,4,5,6,7 SHARD=01 bash scripts/tools/run_phase5_multi_gpu.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

SHARD="${SHARD:-01}"
GPUS="${GPUS:-0,3,4,5,6,7}"
DINO_VIEWS="${DINO_VIEWS:-40}"
BLENDER_PATH="${BLENDER_PATH:-/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender}"
CONFIG="${CONFIG:-configs/partverse_node39_shard${SHARD}.yaml}"
MESH_PAIRS="${MESH_PAIRS:-outputs/partverse/shard_${SHARD}/mesh_pairs_shard${SHARD}}"
SPECS="${SPECS:-outputs/partverse/shard_${SHARD}/cache/phase1/edit_specs_shard${SHARD}.jsonl}"
WORK_DIR="${WORK_DIR:-outputs/partverse/shard_${SHARD}/_dino_render_tmp}"

# ── conda env ──
CONDA_ENV="${CONDA_ENV:-vinedresser3d}"
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "$CONDA_ENV" 2>/dev/null || true
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "[INFO] Phase 5 multi-GPU launcher"
echo "  Shard:       $SHARD"
echo "  GPUs:        ${GPU_ARRAY[*]} ($NUM_GPUS)"
echo "  Config:      $CONFIG"
echo "  Mesh pairs:  $MESH_PAIRS"
echo "  Specs:       $SPECS"
echo "  Blender:     $BLENDER_PATH"
echo "  Views:       $DINO_VIEWS"
echo "  Work dir:    $WORK_DIR"
echo ""

# ── Collect all edit_ids that have after.ply ──
TMPDIR_SPLIT=$(mktemp -d)
trap "rm -rf $TMPDIR_SPLIT" EXIT

python3 -c "
import os, sys
mesh_pairs = '$MESH_PAIRS'
out_dir = '$TMPDIR_SPLIT'
n = $NUM_GPUS
ids = []
for d in sorted(os.listdir(mesh_pairs)):
    p = os.path.join(mesh_pairs, d)
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'after.ply')):
        ids.append(d)
print(f'Found {len(ids)} edit pairs with after.ply')
# Split into N chunks
for i in range(n):
    chunk = ids[i::n]
    with open(os.path.join(out_dir, f'chunk_{i}.txt'), 'w') as f:
        f.write('\n'.join(chunk) + '\n')
    print(f'  GPU {i}: {len(chunk)} pairs')
"

# ── Launch one process per GPU ──
LOG_DIR="outputs/partverse/shard_${SHARD}/logs"
mkdir -p "$LOG_DIR"

PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    CHUNK_FILE="$TMPDIR_SPLIT/chunk_${i}.txt"
    LOG_FILE="$LOG_DIR/phase5_gpu${GPU_ID}.log"

    if [ ! -s "$CHUNK_FILE" ]; then
        echo "[GPU $GPU_ID] No work, skipping"
        continue
    fi

    echo "[GPU $GPU_ID] Starting (log: $LOG_FILE)"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    BLENDER_PATH="$BLENDER_PATH" \
    PARTCRAFT_DINOV2_WEIGHTS="${PARTCRAFT_DINOV2_WEIGHTS:-/home/artgen/.cache/torch/hub/partcraft_ckpts/dinov2/dinov2_vitl14_reg4_pretrain.pth}" \
    PARTCRAFT_SLAT_ENC_CKPT="${PARTCRAFT_SLAT_ENC_CKPT:-${PROJECT_ROOT}/checkpoints/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16}" \
    python scripts/tools/migrate_slat_to_npz.py \
        --config "$CONFIG" \
        --mesh-pairs "$MESH_PAIRS" \
        --specs-jsonl "$SPECS" \
        --phase 5 \
        --include-list "$CHUNK_FILE" \
        --blender-path "$BLENDER_PATH" \
        --dino-views "$DINO_VIEWS" \
        --dino-work-dir "${WORK_DIR}/gpu${GPU_ID}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "[INFO] ${#PIDS[@]} workers launched, waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    GPU_ID="${GPU_ARRAY[$i]}"
    if wait "$PID"; then
        echo "[GPU $GPU_ID] Done (PID $PID)"
    else
        echo "[GPU $GPU_ID] FAILED (PID $PID, exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "[ERROR] $FAILED workers failed. Check logs in $LOG_DIR"
    exit 1
fi

echo "[INFO] All Phase 5 workers completed successfully."
