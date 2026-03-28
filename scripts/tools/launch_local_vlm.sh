#!/bin/bash
# Launch local model servers via SGLang/vLLM for PartCraft3D pipeline.
#
# Prerequisites:
#   conda activate qwen_test   # SGLang 0.5.6 pre-installed
#   # or: pip install sglang[all]
#   # or: pip install vllm>=0.6.0
#
# Usage:
#   # Single VLM server (default — sufficient for local_sglang.yaml)
#   bash scripts/tools/launch_local_vlm.sh
#
#   # Custom model paths
#   VLM_MODEL=/path/to/model bash scripts/tools/launch_local_vlm.sh
#
#   # Use vLLM instead of SGLang
#   BACKEND=vllm bash scripts/tools/launch_local_vlm.sh
#
# Note: Image editing (Qwen-Image-Edit-2511) is now loaded directly via
# diffusers in the pipeline — no separate server needed.
#
# Then use: --config configs/local_sglang.yaml

set -e

# Save outer CUDA_VISIBLE_DEVICES if set (so launch functions respect it)
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
    _OUTER_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

# ---- Mode: vlm | image-edit | both ----
MODE="${MODE:-vlm}"
BACKEND="${BACKEND:-sglang}"

# ---- VLM server settings (Qwen3.5-VL-27B) ----
_CKPT_ROOT="${PARTCRAFT_CKPT_ROOT:-/mnt/zsn/ckpts}"
VLM_MODEL="${VLM_MODEL:-${_CKPT_ROOT}/Qwen3.5-27B}"
VLM_PORT="${VLM_PORT:-8002}"
VLM_TP="${VLM_TP:-1}"
VLM_GPUS="${VLM_GPUS:-0}"
VLM_MAX_LEN="${VLM_MAX_LEN:-32768}"
VLM_MEM_FRAC="${VLM_MEM_FRAC:-0.5}"

# ---- Image edit server settings (qwen-image-2511) ----
IMG_MODEL="${IMG_MODEL:-${_CKPT_ROOT}/Qwen-Image-Edit-2511}"
IMG_PORT="${IMG_PORT:-8001}"
IMG_TP="${IMG_TP:-1}"
IMG_GPUS="${IMG_GPUS:-2}"
IMG_MAX_LEN="${IMG_MAX_LEN:-8192}"

# ---- Legacy single-model overrides (backward compat) ----
# If positional args given, use them for single-server mode
if [ $# -ge 1 ]; then VLM_MODEL="$1"; fi
if [ $# -ge 2 ]; then VLM_TP="$2"; fi
if [ $# -ge 3 ]; then VLM_GPUS="$3"; fi
if [ $# -ge 4 ]; then VLM_PORT="$4"; fi

launch_sglang() {
    local model="$1" port="$2" tp="$3" gpus="$4" max_len="$5"

    # If CUDA_VISIBLE_DEVICES is already set externally, respect it
    # (don't override with VLM_GPUS/IMG_GPUS)
    if [ -n "${_OUTER_CUDA_VISIBLE_DEVICES+x}" ]; then
        gpus="$_OUTER_CUDA_VISIBLE_DEVICES"
    fi
    echo "Starting SGLang: model=$model port=$port tp=$tp gpus=$gpus"

    # Auto-detect CUDA_HOME from nvcc location (fixes FlashInfer JIT)
    if [ -z "$CUDA_HOME" ]; then
        local nvcc_path
        nvcc_path=$(which nvcc 2>/dev/null) || true
        if [ -n "$nvcc_path" ]; then
            export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$nvcc_path")")")"
            echo "  Auto-detected CUDA_HOME=$CUDA_HOME"
        fi
    fi

    # Clear stale FlashInfer JIT cache (may have wrong nvcc path baked in)
    if [ -d "$HOME/.cache/flashinfer" ]; then
        echo "  Clearing FlashInfer JIT cache..."
        rm -rf "$HOME/.cache/flashinfer"
    fi

    CUDA_VISIBLE_DEVICES="$gpus" \
    CUDA_HOME="$CUDA_HOME" \
    python -m sglang.launch_server \
        --model-path "$model" \
        --port "$port" \
        --tp "$tp" \
        --max-total-tokens "$max_len" \
        --mem-fraction-static "$VLM_MEM_FRAC" \
        --attention-backend triton
}

launch_vllm() {
    local model="$1" port="$2" tp="$3" gpus="$4" max_len="$5"
    echo "Starting vLLM: model=$model port=$port tp=$tp gpus=$gpus"
    CUDA_VISIBLE_DEVICES="$gpus" python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --tensor-parallel-size "$tp" \
        --max-model-len "$max_len" \
        --trust-remote-code \
        --dtype auto
}

launch_server() {
    if [ "$BACKEND" = "sglang" ]; then
        launch_sglang "$@"
    else
        launch_vllm "$@"
    fi
}

echo "============================================"
echo "  PartCraft3D Local Model Server"
echo "  Mode:    $MODE"
echo "  Backend: $BACKEND"
echo "============================================"

case "$MODE" in
    vlm)
        echo ""
        echo "  VLM Model:  $VLM_MODEL"
        echo "  VLM Port:   $VLM_PORT"
        echo "  VLM GPUs:   $VLM_GPUS (TP=$VLM_TP)"
        echo "============================================"
        launch_server "$VLM_MODEL" "$VLM_PORT" "$VLM_TP" "$VLM_GPUS" "$VLM_MAX_LEN"
        ;;
    image-edit)
        echo ""
        echo "  IMG Model:  $IMG_MODEL"
        echo "  IMG Port:   $IMG_PORT"
        echo "  IMG GPUs:   $IMG_GPUS (TP=$IMG_TP)"
        echo "============================================"
        launch_server "$IMG_MODEL" "$IMG_PORT" "$IMG_TP" "$IMG_GPUS" "$IMG_MAX_LEN"
        ;;
    both)
        echo ""
        echo "  VLM Model:  $VLM_MODEL"
        echo "  VLM Port:   $VLM_PORT"
        echo "  VLM GPUs:   $VLM_GPUS (TP=$VLM_TP)"
        echo ""
        echo "  IMG Model:  $IMG_MODEL"
        echo "  IMG Port:   $IMG_PORT"
        echo "  IMG GPUs:   $IMG_GPUS (TP=$IMG_TP)"
        echo "============================================"
        launch_server "$VLM_MODEL" "$VLM_PORT" "$VLM_TP" "$VLM_GPUS" "$VLM_MAX_LEN" &
        PID_VLM=$!
        launch_server "$IMG_MODEL" "$IMG_PORT" "$IMG_TP" "$IMG_GPUS" "$IMG_MAX_LEN" &
        PID_IMG=$!
        echo "VLM server PID: $PID_VLM"
        echo "IMG server PID: $PID_IMG"
        trap "kill $PID_VLM $PID_IMG 2>/dev/null" EXIT
        wait
        ;;
    *)
        echo "ERROR: Unknown MODE=$MODE (use: vlm, image-edit, both)"
        exit 1
        ;;
esac
