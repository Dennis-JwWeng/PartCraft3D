#!/usr/bin/env bash
# pipeline_v2 fully-parallel multi-GPU runner.
#
# Each GPU gets ONE independent worker:
#   - its own VLM server   (port vlm_port_base + i * vlm_port_stride)
#   - its own FLUX server  (port flux_port_base + i)
#   - its own pipeline process (CUDA_VISIBLE_DEVICES=gpu_i, --gpu-shard i/N)
#
# Workers run ALL phases (A → F) in parallel, each serving a static 1/N
# object slice.  Server lifecycle is managed per-phase inside each worker
# (VLM starts before Phase A and stops after; FLUX starts before Phase C
# and stops after), so heavy models never co-reside with TRELLIS on the
# same GPU.
#
# Usage:
#   bash scripts/tools/run_pipeline_v2_parallel.sh shard01 configs/pipeline_v2.yaml
#
#   # Run subset of phases on all workers:
#   STAGES="A,C,D"  bash ... shard01 cfg
#
#   # Override number of workers (uses first N GPUs from config):
#   N_WORKERS=2     bash ... shard01 cfg
#
# Compared with run_pipeline_v2_shard.sh (sequential phases, shared servers):
#   + Each GPU is fully self-contained — one worker failure ≠ global abort
#   + No cross-GPU coordination inside any step
#   + ~N× wall-clock speedup on steps A (VLM) and C (FLUX)
#   - Requires N × VLM_VRAM (27B ≈ 24 GB/GPU at TP=1)
#   - Static load-balancing: long-tail objects may make one worker slower
#
# Logs:  logs/v2_<tag>/worker_<i>/{vlm,flux,phase_A,...}.log

set -euo pipefail

TAG="${1:?usage: $0 <tag> <config>}"
CFG="${2:?usage: $0 <tag> <config>}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[ -f "$CFG" ] || { echo "[ERROR] missing config: $CFG"; exit 1; }

# ─── machine env ────────────────────────────────────────────────────
ENV_FILE="${MACHINE_ENV:-configs/machine/$(hostname).env}"
[ -f "$ENV_FILE" ] || {
    echo "[ERROR] machine env not found: ${ENV_FILE}"
    echo "  copy from: configs/machine/node39.env"
    exit 1
}
# shellcheck disable=SC1090
source "$ENV_FILE"

CONDA_INIT="${CONDA_INIT:?CONDA_INIT not set in ${ENV_FILE}}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:?CONDA_ENV_SERVER not set}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:?CONDA_ENV_PIPELINE not set}"
VLM_CKPT="${VLM_CKPT:?VLM_CKPT not set}"
EDIT_CKPT="${EDIT_CKPT:?EDIT_CKPT not set}"

# shellcheck disable=SC1090
set +u; source "${CONDA_INIT}"; set -u
PY_PIPE="$(conda run -n "${CONDA_ENV_PIPELINE}" which python 2>/dev/null)" \
    || PY_PIPE="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_PIPELINE}/bin/python"
PY_SRV="$(conda run -n "${CONDA_ENV_SERVER}" which python 2>/dev/null)" \
    || PY_SRV="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_SERVER}/bin/python"
[ -x "$PY_PIPE" ] || { echo "[ERROR] pipeline python not found: $PY_PIPE"; exit 1; }
[ -x "$PY_SRV"  ] || { echo "[ERROR] server python not found: $PY_SRV"; exit 1; }

# ─── read GPU/port plan from Python ─────────────────────────────────
eval "$(
    "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg))
"
)"

N="${N_WORKERS:-${#GPUS[@]}}"    # may be fewer than total GPUs for testing
SHARD="${TAG#shard}"             # "shard01" → "01"; "01" → "01"

# Stage selection
if [ -n "${STAGES:-}" ]; then
    IFS=',' read -r -a SELECTED_STAGES <<< "$STAGES"
else
    SELECTED_STAGES=("${DEFAULT_STAGES[@]}")
fi

LOG_DIR="logs/v2_${TAG}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  pipeline_v2 PARALLEL multi-GPU run"
echo "============================================================"
echo "  tag      : $TAG   shard=$SHARD"
echo "  config   : $CFG"
echo "  workers  : $N  (GPUs: ${GPUS[*]:0:$N})"
echo "  stages   : ${SELECTED_STAGES[*]}"
echo "  log dir  : $LOG_DIR"
echo "============================================================"

# ─── helper: query phase metadata ───────────────────────────────────
# Sets STAGE_SERVERS, STAGE_STEPS, STAGE_USE_GPUS for a given phase name.
stage_meta() {
    local stage="$1"
    eval "$(
        "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg, stage_name='$stage'))
"
    )"
}

# ─── per-worker server functions ────────────────────────────────────
# These run inside the worker subshell; they write PID files to wlog/.

w_start_vlm() {
    local gpu="$1" port="$2" wlog="$3"
    echo "[vlm] GPU $gpu port $port starting..."
    set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
    CUDA_VISIBLE_DEVICES="$gpu" \
    VLM_MODEL="$VLM_CKPT" \
    VLM_PORT="$port" \
    VLM_TP=1 \
    SGLANG_DISABLE_CUDNN_CHECK=1 \
        bash scripts/tools/launch_local_vlm.sh \
            > "$wlog/vlm.log" 2>&1 &
    echo $! > "$wlog/vlm.pid"
    # wait for readiness
    local deadline=$(( $(date +%s) + 900 ))
    while :; do
        if curl -s -m 2 "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo "[vlm] :${port} ready"; return 0
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "[vlm] :${port} TIMEOUT (see $wlog/vlm.log)"
            tail -20 "$wlog/vlm.log" || true
            return 1
        fi
        sleep 5
    done
}

w_stop_vlm() {
    local port="$1" wlog="$2"
    local pid
    pid=$(cat "$wlog/vlm.pid" 2>/dev/null || true)
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    pkill -9 -f "sglang.launch_server.*${VLM_CKPT##*/}" 2>/dev/null || true
    rm -f "$wlog/vlm.pid"
    echo "[vlm] :${port} stopped"
    sleep 1
}

w_start_flux() {
    local gpu="$1" port="$2" wlog="$3"
    echo "[flux] GPU $gpu port $port starting..."
    set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
    CUDA_VISIBLE_DEVICES="$gpu" \
        "$PY_SRV" scripts/tools/image_edit_server.py \
            --model "$EDIT_CKPT" --port "$port" \
            > "$wlog/flux.log" 2>&1 &
    echo $! > "$wlog/flux.pid"
    local deadline=$(( $(date +%s) + 900 ))
    while :; do
        if curl -s -m 2 -o /dev/null -w "%{http_code}" \
                "http://localhost:${port}/health" 2>/dev/null \
                | grep -q "200"; then
            echo "[flux] :${port} ready"; return 0
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "[flux] :${port} TIMEOUT (see $wlog/flux.log)"
            tail -20 "$wlog/flux.log" || true
            return 1
        fi
        sleep 5
    done
}

w_stop_flux() {
    local port="$1" wlog="$2"
    local pid
    pid=$(cat "$wlog/flux.pid" 2>/dev/null || true)
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    pkill -9 -f "image_edit_server.py.*${port}" 2>/dev/null || true
    rm -f "$wlog/flux.pid"
    echo "[flux] :${port} stopped"
    sleep 1
}

# ─── per-worker main loop ────────────────────────────────────────────
run_worker() {
    local i="$1"
    local gpu="${GPUS[$i]}"
    local vlm_port="${VLM_PORTS[$i]}"
    local flux_port="${FLUX_PORTS[$i]}"
    local wlog="$LOG_DIR/worker_${i}"
    mkdir -p "$wlog"

    echo "[worker $i] start  GPU=$gpu  vlm=:$vlm_port  flux=:$flux_port"

    # Ensure servers are killed if this subshell exits for any reason
    trap '
        w_stop_vlm  "$vlm_port"  "$wlog" 2>/dev/null || true
        w_stop_flux "$flux_port" "$wlog" 2>/dev/null || true
    ' EXIT

    local rc=0
    for stage in "${SELECTED_STAGES[@]}"; do

        stage_meta "$stage"   # sets STAGE_SERVERS, STAGE_DESC

        echo "[worker $i] ▶ stage $stage — $STAGE_DESC"

        # ── start service if this phase needs one ───────────────────
        case "$STAGE_SERVERS" in
            vlm)  w_start_vlm  "$gpu" "$vlm_port"  "$wlog" || { rc=1; break; } ;;
            flux) w_start_flux "$gpu" "$flux_port" "$wlog" || { rc=1; break; } ;;
        esac

        # ── run the phase ───────────────────────────────────────────
        CUDA_VISIBLE_DEVICES="$gpu" \
        ATTN_BACKEND=flash_attn \
            "$PY_PIPE" -m partcraft.pipeline_v2.run \
                --config "$CFG" \
                --shard  "$SHARD" \
                --all \
                --stage  "$stage" \
                --single-gpu \
                --gpu-shard "${i}/${N}" \
                --vlm-url  "http://localhost:${vlm_port}/v1" \
                --flux-url "http://localhost:${flux_port}" \
                2>&1 | tee "$wlog/stage_${stage}.log"
        rc=${PIPESTATUS[0]}

        # ── stop service ────────────────────────────────────────────
        case "$STAGE_SERVERS" in
            vlm)  w_stop_vlm  "$vlm_port"  "$wlog" ;;
            flux) w_stop_flux "$flux_port" "$wlog" ;;
        esac

        if [ "$rc" -ne 0 ]; then
            echo "[worker $i] stage $stage FAILED (rc=$rc) — aborting worker"
            break
        fi
        echo "[worker $i] ✓ stage $stage done"
    done

    echo "[worker $i] exit rc=$rc"
    return "$rc"
}

# ─── launch all workers in parallel ─────────────────────────────────
declare -a WORKER_PIDS=()
for i in $(seq 0 $(( N - 1 ))); do
    # Each worker runs in a subshell so its trap/EXIT doesn't affect others
    ( run_worker "$i" ) &
    WORKER_PIDS+=($!)
    echo "[main] worker $i spawned (pid=${WORKER_PIDS[-1]})"
done

# ─── collect results ─────────────────────────────────────────────────
final_rc=0
for i in $(seq 0 $(( N - 1 ))); do
    if wait "${WORKER_PIDS[$i]}"; then
        echo "[main] worker $i OK"
    else
        rc=$?
        echo "[main] worker $i FAILED (rc=$rc)"
        final_rc=$rc
    fi
done

echo
if [ "$final_rc" -eq 0 ]; then
    echo "=== ALL ${N} WORKERS DONE ==="
else
    echo "=== ${N} WORKERS: SOME FAILED (rc=$final_rc) ==="
fi
exit "$final_rc"
