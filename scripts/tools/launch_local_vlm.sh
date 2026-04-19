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

    # Validate / auto-detect CUDA_HOME for FlashInfer JIT (sampling kernels).
    # FlashInfer reads $CUDA_HOME/bin/nvcc — if that path is missing/stale
    # (e.g. user shell exports cuda-12.1 but the dir was removed), JIT fails
    # with "/usr/local/cuda-12.1/bin/nvcc: not found" and the server crashes
    # the moment the first sampling op runs. Re-detect whenever the configured
    # nvcc isn't actually executable.
    if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
        local _bad_home="$CUDA_HOME"
        unset CUDA_HOME
        local nvcc_path
        nvcc_path=$(command -v nvcc 2>/dev/null) || true
        if [ -n "$nvcc_path" ] && [ -x "$nvcc_path" ]; then
            export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$nvcc_path")")")"
        else
            for _try in /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda; do
                if [ -x "$_try/bin/nvcc" ]; then
                    export CUDA_HOME="$_try"; break
                fi
            done
        fi
        if [ -n "$_bad_home" ] && [ "$_bad_home" != "$CUDA_HOME" ]; then
            echo "  CUDA_HOME corrected: $_bad_home -> $CUDA_HOME"
        else
            echo "  CUDA_HOME auto-detected: $CUDA_HOME"
        fi
        if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
            echo "  ERROR: no usable nvcc found; FlashInfer JIT will fail." >&2
            exit 1
        fi
        export PATH="$CUDA_HOME/bin:$PATH"
    fi

    # Clear stale FlashInfer JIT cache (may have wrong nvcc path baked in)
    if [ -d "$HOME/.cache/flashinfer" ]; then
        echo "  Clearing FlashInfer JIT cache..."
        rm -rf "$HOME/.cache/flashinfer"
    fi

    # VLM_MAX_RUNNING caps in-flight requests per SGLang server.  Without
    # this, an aggressive client (e.g. gate_text_align fan-out across many
    # objects) can submit far more multimodal requests than the KV cache
    # can hold; SGLang then queues them but the OpenAI client times out
    # before they ever get scheduled.  8 is safe for Qwen3.5-27B on a
    # single A800 with mem-fraction-static=0.85.
    local _maxrun_args=()
    if [ -n "${VLM_MAX_RUNNING:-}" ]; then
        _maxrun_args=(--max-running-requests "$VLM_MAX_RUNNING")
    fi

    # Watchdog: SGLang has a known multimodal shm race
    # (mm_utils.__setstate__ → SharedMemory(name='/psm_xxx') →
    # FileNotFoundError → scheduler crash → server dies) that can fire
    # under multi-hour load when the OpenAI client cancels in-flight
    # requests.  When VLM_AUTO_RESTART > 0, we restart the server up to
    # that many times after non-fatal exits (runtime ≥ MIN_HEALTHY_S).
    # Default 0 = preserve legacy single-shot behaviour.
    local _max_restart="${VLM_AUTO_RESTART:-0}"
    local _min_healthy="${VLM_MIN_HEALTHY_S:-120}"
    local _attempt=0 _start_ts _end_ts _runtime _exit
    # Forward kill signals from the parent (run_pipeline_v3_shard.sh
    # stop_vlm) to the child python process so cleanup is prompt.
    local _child_pid=
    _fwd() { [ -n "$_child_pid" ] && kill -TERM "$_child_pid" 2>/dev/null; exit 143; }
    trap _fwd TERM INT HUP

    while true; do
        _start_ts=$(date +%s)
        set +e
        CUDA_VISIBLE_DEVICES="$gpus" \
        CUDA_HOME="$CUDA_HOME" \
        python -m sglang.launch_server \
            --model-path "$model" \
            --port "$port" \
            --tp "$tp" \
            --max-total-tokens "$max_len" \
            --mem-fraction-static "$VLM_MEM_FRAC" \
            --attention-backend triton \
            "${_maxrun_args[@]}" &
        _child_pid=$!
        wait "$_child_pid"
        _exit=$?
        set -e
        _child_pid=
        _end_ts=$(date +%s)
        _runtime=$((_end_ts - _start_ts))

        if [ "$_max_restart" = "0" ]; then
            return $_exit
        fi
        if [ "$_runtime" -lt "$_min_healthy" ]; then
            echo "[watchdog] sglang :$port exited too fast (${_runtime}s, code=$_exit) — fatal, giving up"
            return $_exit
        fi
        _attempt=$((_attempt + 1))
        if [ "$_attempt" -gt "$_max_restart" ]; then
            echo "[watchdog] sglang :$port: $_attempt restarts reached, giving up"
            return $_exit
        fi
        echo "[watchdog] sglang :$port crashed after ${_runtime}s (exit=$_exit); restarting (attempt $_attempt/$_max_restart) in 10s..."
        sleep 10
    done
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
