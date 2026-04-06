#!/bin/bash
# Multi-GPU VLM cleaning launcher.
#
# GPU 0: Qwen VLM server (start separately before running this)
# Remaining GPUs: TRELLIS rendering workers (one per GPU)
#
# Each worker processes a subset of objects (round-robin split).
# Deletion edits use Blender rendering (CPU, no TRELLIS GPU needed).
# mod/scl/mat/glb edits use TRELLIS decode on the assigned GPU.
#
# Usage:
#   # Start VLM server first:
#   conda activate qwen_test
#   CUDA_VISIBLE_DEVICES=0 VLM_MODEL=/Node11_nvme/zsn/checkpoints/Qwen3.5-27B \
#       bash scripts/tools/launch_local_vlm.sh
#
#   # Then launch multi-GPU cleaning:
#   GPUS=3,4,5,6,7 SHARD=01 bash scripts/tools/run_vlm_cleaning_multi_gpu.sh
#
#   # Deletion-only (no TRELLIS, fast):
#   ONLY_TYPES=deletion GPUS=3 SHARD=01 bash scripts/tools/run_vlm_cleaning_multi_gpu.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

SHARD="${SHARD:-01}"
GPUS="${GPUS:-3,4,5,6,7}"
VLM_URL="${VLM_URL:-http://localhost:8002/v1}"
# Per-GPU VLM URLs: comma-separated, one per GPU. If set, each worker uses its own VLM.
# E.g. VLM_URLS="http://localhost:8002/v1,http://localhost:8003/v1,..."
VLM_URLS="${VLM_URLS:-}"
VLM_MODEL="${VLM_MODEL:-Qwen3.5-27B}"
ROOT="${ROOT:-outputs/partverse/partverse_pairs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/partverse}"
BLENDER_PATH="${BLENDER_PATH:-/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender}"
ONLY_TYPES="${ONLY_TYPES:-}"          # e.g. "deletion" or "modification scale material global"
NUM_VIEWS="${NUM_VIEWS:-3}"
VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-1024}"
TRELLIS_CKPT="${TRELLIS_CKPT:-checkpoints/TRELLIS-image-large}"

# ── conda env ──
CONDA_ENV="${CONDA_ENV:-vinedresser3d}"
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "$CONDA_ENV" 2>/dev/null || true
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Parse per-GPU VLM URLs (if provided)
if [ -n "$VLM_URLS" ]; then
    IFS=',' read -ra VLM_URL_ARRAY <<< "$VLM_URLS"
else
    VLM_URL_ARRAY=()
    for i in "${!GPU_ARRAY[@]}"; do
        VLM_URL_ARRAY+=("$VLM_URL")
    done
fi

echo "[INFO] VLM cleaning multi-GPU launcher"
echo "  Shard:       $SHARD"
echo "  GPUs:        ${GPU_ARRAY[*]} ($NUM_GPUS workers)"
echo "  VLM:         ${VLM_URL_ARRAY[*]}"
echo "  Root:        $ROOT"
echo "  Output root: $OUTPUT_ROOT"
echo "  Only types:  ${ONLY_TYPES:-all (del/mod/scl/mat/glb)}"
echo "  Views:       $NUM_VIEWS"
echo ""

# ── Split objects across GPUs ──
TMPDIR_SPLIT=$(mktemp -d)
trap "rm -rf $TMPDIR_SPLIT" EXIT

python3 -c "
import os
root = '$ROOT'
shard = '$SHARD'
n = $NUM_GPUS
out_dir = '$TMPDIR_SPLIT'

shard_dir = os.path.join(root, f'shard_{shard}')
objs = sorted(
    d for d in os.listdir(shard_dir)
    if os.path.isdir(os.path.join(shard_dir, d))
)
print(f'Found {len(objs)} objects in shard_{shard}')

for i in range(n):
    chunk = objs[i::n]
    with open(os.path.join(out_dir, f'chunk_{i}.txt'), 'w') as f:
        f.write('\n'.join(chunk) + '\n')
    print(f'  Worker {i}: {len(chunk)} objects')
"

# ── Launch one worker per GPU ──
LOG_DIR="outputs/partverse/shard_${SHARD}/logs"
mkdir -p "$LOG_DIR"
RENDER_CACHE="${ROOT}/_vlm_render_cache"
mkdir -p "$RENDER_CACHE"

PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    CHUNK_FILE="$TMPDIR_SPLIT/chunk_${i}.txt"
    LOG_FILE="$LOG_DIR/vlm_clean_gpu${GPU_ID}.log"
    SCORES_FILE="${ROOT}/vlm_scores_shard${SHARD}_gpu${GPU_ID}.jsonl"

    if [ ! -s "$CHUNK_FILE" ]; then
        echo "[GPU $GPU_ID] No work, skipping"
        continue
    fi

    # Build --only-types argument
    ONLY_TYPES_ARG=""
    if [ -n "$ONLY_TYPES" ]; then
        ONLY_TYPES_ARG="--only-types $ONLY_TYPES"
    fi

    WORKER_VLM_URL="${VLM_URL_ARRAY[$i]:-$VLM_URL}"
    echo "[GPU $GPU_ID] Starting (VLM: $WORKER_VLM_URL, log: $LOG_FILE)"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python scripts/tools/run_vlm_cleaning.py \
        --root "$ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --vlm-url "$WORKER_VLM_URL" \
        --vlm-model "$VLM_MODEL" \
        --vlm-max-tokens "$VLM_MAX_TOKENS" \
        --shards "$SHARD" \
        --trellis-ckpt "$TRELLIS_CKPT" \
        --blender-path "$BLENDER_PATH" \
        --num-views "$NUM_VIEWS" \
        --include-objects "$CHUNK_FILE" \
        --scores-file "$SCORES_FILE" \
        --render-cache "$RENDER_CACHE" \
        $ONLY_TYPES_ARG \
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

# ── Merge per-GPU scores ──
echo ""
echo "[INFO] Merging per-GPU scores..."
python3 -c "
import json
from pathlib import Path

root = Path('$ROOT')
shard = '$SHARD'
merged = {}

# Load any previous merged scores first
main_scores = root / 'vlm_scores.jsonl'
if main_scores.exists():
    with open(main_scores) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    merged[rec['edit_id']] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f'Loaded {len(merged)} existing scores from {main_scores}')

# Merge GPU-specific scores
for f in sorted(root.glob(f'vlm_scores_shard{shard}_gpu*.jsonl')):
    n = 0
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                merged[rec['edit_id']] = rec
                n += 1
            except (json.JSONDecodeError, KeyError):
                pass
    print(f'  {f.name}: {n} scores')

with open(main_scores, 'w') as fh:
    for s in merged.values():
        fh.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f'Merged total: {len(merged)} scores -> {main_scores}')
"

# ── Generate quality.json (finalization run, no new scoring) ──
echo "[INFO] Generating quality.json per object..."
python scripts/tools/run_vlm_cleaning.py \
    --root "$ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --vlm-url "$VLM_URL" \
    --vlm-model "$VLM_MODEL" \
    --shards "$SHARD"

echo ""
echo "[INFO] VLM cleaning completed for shard_${SHARD}."
