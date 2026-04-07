#!/usr/bin/env bash
# pipeline_v2 phased scheduler.
#
# Drives a full shard through the new object-centric pipeline_v2 by
# starting/stopping the right servers around each phase. Each phase
# logs to logs/v2_<shard>/<phase>.log and the script aborts on the
# first non-zero exit.
#
# Usage:
#   bash scripts/tools/run_pipeline_v2_shard.sh shard01 \
#        configs/pipeline_v2_shard01.yaml \
#        4,5,6,7
#
# Phases (only the requested ones run; default = all):
#   PHASES="A,B,C,D,E,F,G" bash scripts/tools/run_pipeline_v2_shard.sh ...
#
#   A : VLM up on GPU0 of pool → s1 → VLM down
#   B : s2 (CPU only)
#   C : FLUX up on all 4 GPUs (one per GPU) → s4 → FLUX down
#   D : s5 + s5b on GPU pool (multi-GPU dispatch)
#   E : s6 + s6b on GPU pool
#   F : s7 (CPU only)
#
# Resume: each pipeline_v2 step is idempotent + has product validators,
# so re-running this script picks up where it left off.

set -euo pipefail

# ─── args ─────────────────────────────────────────────────────────────
TAG="${1:-shard01}"
CFG="${2:-configs/pipeline_v2_shard01.yaml}"
GPUS="${3:-4,5,6,7}"
PHASES="${PHASES:-A,B,C,D,E,F}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# ─── env paths from node39.env ───────────────────────────────────────
ENV_FILE="configs/machine/node39.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

CONDA_INIT="${CONDA_INIT:-/home/artgen/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:-qwen_test}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:-vinedresser3d}"
VLM_CKPT="${VLM_CKPT:-/Node11_nvme/zsn/checkpoints/Qwen3.5-27B}"
EDIT_CKPT="${EDIT_CKPT:-/Node11_nvme/wjw/checkpoints/FLUX.2-klein-9B}"

PY_PIPE="/Node11_nvme/artgen/.miniconda3/envs/${CONDA_ENV_PIPELINE}/bin/python"
PY_SRV="/Node11_nvme/artgen/.miniconda3/envs/${CONDA_ENV_SERVER}/bin/python"

LOG_DIR="logs/v2_${TAG}"
mkdir -p "$LOG_DIR"

# ─── derived ─────────────────────────────────────────────────────────
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}
VLM_GPU="${GPU_ARR[0]}"
VLM_PORT=8002
VLM_URL="http://localhost:${VLM_PORT}/v1"
FLUX_BASE_PORT=8004
FLUX_PORTS=()
for i in $(seq 0 $((N_GPUS-1))); do
    FLUX_PORTS+=( $((FLUX_BASE_PORT+i)) )
done

echo "============================================================"
echo "  pipeline_v2 phased run"
echo "============================================================"
echo "  tag         : $TAG"
echo "  config      : $CFG"
echo "  gpus        : $GPUS  (N=$N_GPUS)"
echo "  phases      : $PHASES"
echo "  log dir     : $LOG_DIR"
echo "  python(pipe): $PY_PIPE"
echo "  python(srv) : $PY_SRV"
echo "============================================================"

run_phase() { [[ ",${PHASES}," == *",${1},"* ]]; }

# ─── server lifecycle ─────────────────────────────────────────────────

start_vlm() {
    echo "[VLM] starting on GPU $VLM_GPU port $VLM_PORT"
    local log="$LOG_DIR/vlm_server.log"
    CUDA_VISIBLE_DEVICES="$VLM_GPU" \
    SGLANG_DISABLE_CUDNN_CHECK=1 \
        "$PY_SRV" -m sglang.launch_server \
            --model-path "$VLM_CKPT" \
            --host 0.0.0.0 --port "$VLM_PORT" \
            --tp-size 1 --mem-fraction-static 0.85 \
            --context-length 32768 \
            > "$log" 2>&1 &
    VLM_PID=$!
    echo "[VLM] pid=$VLM_PID  log=$log"
    echo $VLM_PID > "$LOG_DIR/vlm.pid"
    # wait until /v1/models responds (timeout 600s)
    local deadline=$(( $(date +%s) + 600 ))
    while :; do
        if curl -s -m 2 "${VLM_URL}/models" >/dev/null 2>&1; then
            echo "[VLM] ready"; return 0
        fi
        if ! kill -0 "$VLM_PID" 2>/dev/null; then
            echo "[VLM] DIED — see $log"; tail -30 "$log"; return 1
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "[VLM] TIMEOUT"; kill "$VLM_PID" 2>/dev/null || true; return 1
        fi
        sleep 5
    done
}

stop_vlm() {
    if [ -f "$LOG_DIR/vlm.pid" ]; then
        local pid; pid=$(cat "$LOG_DIR/vlm.pid")
        echo "[VLM] killing pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
        pkill -9 -f "sglang.launch_server.*${VLM_CKPT}" 2>/dev/null || true
        rm -f "$LOG_DIR/vlm.pid"
        sleep 2
    fi
}

start_flux() {
    echo "[FLUX] starting ${N_GPUS} servers on GPUs $GPUS"
    : > "$LOG_DIR/flux.pids"
    for i in $(seq 0 $((N_GPUS-1))); do
        local gpu="${GPU_ARR[i]}"
        local port="${FLUX_PORTS[i]}"
        local log="$LOG_DIR/flux_${port}.log"
        echo "  GPU $gpu -> port $port"
        CUDA_VISIBLE_DEVICES="$gpu" \
            "$PY_SRV" scripts/tools/image_edit_server.py \
                --model "$EDIT_CKPT" --port "$port" \
                > "$log" 2>&1 &
        echo $! >> "$LOG_DIR/flux.pids"
    done
    # wait for all
    local deadline=$(( $(date +%s) + 600 ))
    for port in "${FLUX_PORTS[@]}"; do
        while :; do
            if curl -s -m 2 -o /dev/null -w "%{http_code}" \
                    "http://localhost:${port}/health" 2>/dev/null \
                    | grep -q "200"; then
                echo "[FLUX] :$port ready"; break
            fi
            if [ "$(date +%s)" -gt "$deadline" ]; then
                echo "[FLUX] :$port TIMEOUT"
                tail -20 "$LOG_DIR/flux_${port}.log" || true
                stop_flux; return 1
            fi
            sleep 5
        done
    done
    echo "[FLUX] all ${N_GPUS} servers ready"
}

stop_flux() {
    if [ -f "$LOG_DIR/flux.pids" ]; then
        echo "[FLUX] killing all servers"
        while read -r pid; do
            kill -9 "$pid" 2>/dev/null || true
        done < "$LOG_DIR/flux.pids"
        pkill -9 -f "image_edit_server.py" 2>/dev/null || true
        rm -f "$LOG_DIR/flux.pids"
        sleep 2
    fi
}

# global trap so any abort tears down servers
cleanup_all() { stop_vlm; stop_flux; }
trap cleanup_all EXIT

# ─── pipeline_v2 invocations ──────────────────────────────────────────

pv2() {
    local steps="$1"; shift
    local log="$LOG_DIR/$1.log"; shift
    echo "  -> $steps  (log: $log)"
    ATTN_BACKEND=xformers \
    "$PY_PIPE" -m partcraft.pipeline_v2.run \
        --config "$CFG" \
        --shard "${TAG#shard}" \
        --all \
        --steps "$steps" \
        "$@" \
        2>&1 | tee "$log"
}

# ═══ PHASES ═══════════════════════════════════════════════════════════

if run_phase A; then
    echo; echo "▶ Phase A — s1 (VLM)"
    start_vlm
    pv2 s1 phaseA
    stop_vlm
fi

if run_phase B; then
    echo; echo "▶ Phase B — s2 (highlights, CPU)"
    pv2 s2 phaseB
fi

if run_phase C; then
    echo; echo "▶ Phase C — s4 (FLUX 2D)"
    start_flux
    pv2 s4 phaseC
    stop_flux
fi

if run_phase D; then
    echo; echo "▶ Phase D — s5 + s5b (TRELLIS 3D + deletion mesh)"
    pv2 s5,s5b phaseD --gpus "$GPUS"
fi

if run_phase E; then
    echo; echo "▶ Phase E — s6 + s6b (3D rerender + deletion reencode)"
    pv2 s6,s6b phaseE --gpus "$GPUS"
fi

if run_phase F; then
    echo; echo "▶ Phase F — s7 (addition backfill, CPU)"
    pv2 s7 phaseF
fi

echo; echo "=== ALL PHASES DONE ==="
