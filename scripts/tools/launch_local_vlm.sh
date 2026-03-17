#!/bin/bash
# Launch a local vLLM server for VLM to save API token costs.
#
# Prerequisites:
#   pip install vllm>=0.6.0
#   # or for SGLang:
#   pip install sglang[all]
#
# Usage:
#   bash scripts/launch_local_vlm.sh                    # default: local Qwen3-VL-2B-Instruct
#   bash scripts/launch_local_vlm.sh /path/to/model 1   # custom model path
#   BACKEND=sglang bash scripts/launch_local_vlm.sh     # use SGLang instead
#
# Then in configs/local_vlm.yaml, set:
#   phase0:
#     vlm_backend: "local"
#     local_base_url: "http://localhost:8000/v1"

set -e

MODEL="${1:-/Node11_nvme/wjw/checkpoints/Qwen3-VL-2B-Instruct}"
TP="${2:-1}"
GPU_IDS="${3:-0}"
PORT="${4:-8000}"
BACKEND="${BACKEND:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "============================================"
echo "  Local VLM Server"
echo "  Model:  $MODEL"
echo "  TP:     $TP GPU(s)"
echo "  GPUs:   $GPU_IDS"
echo "  Port:   $PORT"
echo "  Backend: $BACKEND"
echo "============================================"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

if [ "$BACKEND" = "sglang" ]; then
    echo "Starting SGLang server..."
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port "$PORT" \
        --tp "$TP" \
        --max-total-tokens "$MAX_MODEL_LEN" \
        --chat-template chatml
else
    echo "Starting vLLM server..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --dtype auto
fi
