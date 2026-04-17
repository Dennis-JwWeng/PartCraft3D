#!/bin/bash
# Auto-spawn N Qwen3.5-VL SGLang servers, one per GPU.
#
# Reads:
#   GPUS=2,3,4,5,6,7        comma-separated GPU ids (required)
#   BASE_PORT=8002          first port; subsequent servers use BASE_PORT+i
#   VLM_MODEL=...           default /Node11_nvme/zsn/checkpoints/Qwen3.5-27B
#   VLM_MEM_FRAC=0.85       per-server static mem fraction
#   VLM_MAX_LEN=32768
#   LOG_DIR=/tmp/vlm_logs
#
# On success prints a single line:
#   VLM_URLS=http://localhost:8002/v1,http://localhost:8003/v1,...
# Suitable to copy-paste into the runner --vlm-url arg.
#
# Usage:
#   GPUS=2,3,4,5,6,7 bash scripts/tools/launch_multi_vlm.sh
#   # then point any VLM consumer (e.g. partcraft.pipeline_v3.run) at the
#   # printed VLM_URLS line.
#
# Stop:
#   pkill -f sglang.launch_server
set -e

GPUS="${GPUS:?GPUS is required, e.g. GPUS=2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8002}"
_CKPT_ROOT="${PARTCRAFT_CKPT_ROOT:-/Node11_nvme/zsn/checkpoints}"
VLM_MODEL="${VLM_MODEL:-${_CKPT_ROOT}/Qwen3.5-27B}"
VLM_MEM_FRAC="${VLM_MEM_FRAC:-0.85}"
VLM_MAX_LEN="${VLM_MAX_LEN:-16384}"   # match node39 default; SGLang KV mem
export SGLANG_DISABLE_CUDNN_CHECK=1   # required: PyTorch 2.9.1 + CuDNN 9.10 startup check
LOG_DIR="${LOG_DIR:-/tmp/vlm_logs}"
mkdir -p "$LOG_DIR"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
N=${#GPU_ARRAY[@]}

# auto-detect CUDA_HOME for FlashInfer JIT
if [ -z "$CUDA_HOME" ]; then
    nvcc_path=$(which nvcc 2>/dev/null) || true
    if [ -n "$nvcc_path" ]; then
        export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$nvcc_path")")")"
    fi
fi
rm -rf "$HOME/.cache/flashinfer" 2>/dev/null || true

echo "============================================================" >&2
echo "  Launching $N Qwen3.5-VL SGLang servers" >&2
echo "  Model:    $VLM_MODEL" >&2
echo "  GPUs:     $GPUS  (one server per GPU, TP=1)" >&2
echo "  Ports:    $BASE_PORT..$((BASE_PORT + N - 1))" >&2
echo "  Logs:     $LOG_DIR" >&2
echo "============================================================" >&2

PIDS=()
URLS=()
for i in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$i]}"
    port=$((BASE_PORT + i))
    log="$LOG_DIR/vlm_gpu${gpu}_p${port}.log"
    echo "  [$i] GPU $gpu  →  port $port  log=$log" >&2
    CUDA_VISIBLE_DEVICES="$gpu" \
    nohup python -m sglang.launch_server \
        --model-path "$VLM_MODEL" \
        --port "$port" \
        --tp 1 \
        --max-total-tokens "$VLM_MAX_LEN" \
        --mem-fraction-static "$VLM_MEM_FRAC" \
        --attention-backend triton \
        > "$log" 2>&1 &
    PIDS+=("$!")
    URLS+=("http://localhost:${port}/v1")
done

echo >&2
echo "PIDs: ${PIDS[*]}" >&2
echo "Waiting for /health on all servers (timeout 600s)…" >&2

deadline=$(( $(date +%s) + 600 ))
ready=()
for url in "${URLS[@]}"; do
    base="${url%/v1}"
    while :; do
        if curl -s -m 1 "${base}/health" >/dev/null 2>&1 \
           || curl -s -m 1 "${base}/v1/models" >/dev/null 2>&1; then
            echo "  [OK] $url" >&2
            ready+=("$url")
            break
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "  [TIMEOUT] $url" >&2
            break
        fi
        sleep 3
    done
done

echo >&2
echo "============================================================" >&2
echo "  Ready: ${#ready[@]}/$N servers" >&2
echo "============================================================" >&2

# Single-line copy-paste output (stdout)
( IFS=,; echo "VLM_URLS=${ready[*]}" )

# Hand control back; don't wait — caller can monitor logs / pkill to stop
