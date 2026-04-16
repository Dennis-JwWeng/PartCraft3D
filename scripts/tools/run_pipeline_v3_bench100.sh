#!/usr/bin/env bash
# run_pipeline_v3_bench100.sh — pipeline_v3 Mode-E bench100
#
# GPU utilization per stage (all stages use ALL GPUs):
#   text_gen_gate_a : N VLM servers  (one per GPU, ports from vlm_port_base)
#   deletion_cpu    : CPU only
#   flux_2d         : N FLUX servers (one per GPU, ports from flux_port_base)
#   trellis_preview : N Trellis workers (dispatch_gpus = pipeline.gpus)
#   gate_quality    : N VLM servers  (same ports)
#
# Usage:
#   bash scripts/tools/run_pipeline_v3_bench100.sh [stage1,stage2,...]

set -euo pipefail

CFG="configs/pipeline_v3_shard08_bench100.yaml"
OBJ_IDS_FILE="configs/shard08_bench100_obj_ids.txt"
SHARD="08"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[ -f "$CFG" ]          || { echo "[ERROR] missing config: $CFG"; exit 1; }
[ -f "$OBJ_IDS_FILE" ] || { echo "[ERROR] missing obj list: $OBJ_IDS_FILE"; exit 1; }

# ── machine env ─────────────────────────────────────────────────────
ENV_FILE="${MACHINE_ENV:-configs/machine/$(hostname).env}"
[ -f "$ENV_FILE" ] || { echo "[ERROR] machine env not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"
CONDA_INIT="${CONDA_INIT:?}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:?}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:?}"
VLM_CKPT="${VLM_CKPT:?}"
EDIT_CKPT="${EDIT_CKPT:?}"

set +u; source "${CONDA_INIT}"; set -u
PY_PIPE="$(conda run -n "${CONDA_ENV_PIPELINE}" which python 2>/dev/null)" \
    || PY_PIPE="/root/miniconda3/envs/${CONDA_ENV_PIPELINE}/bin/python"
PY_SRV="$(conda run -n "${CONDA_ENV_SERVER}" which python 2>/dev/null)" \
    || PY_SRV="/root/miniconda3/envs/${CONDA_ENV_SERVER}/bin/python"

LOG_DIR="logs/bench100"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── read config: GPUS, VLM_PORTS, FLUX_PORTS, DEFAULT_STAGES ────────
eval "$(
    "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg))
"
)"

# Stage selection
if [ -n "${1:-}" ]; then
    IFS=',' read -r -a SELECTED_STAGES <<< "$1"
else
    SELECTED_STAGES=("${DEFAULT_STAGES[@]}")
fi

echo "============================================================"
echo "  pipeline_v3 bench100"
echo "  config   : $CFG"
echo "  obj_ids  : $OBJ_IDS_FILE"
echo "  stages   : ${SELECTED_STAGES[*]}"
echo "  GPUs     : ${GPUS[*]}  (${#GPUS[@]} total)"
echo "  VLM ports: ${VLM_PORTS[*]}"
echo "  FLUX ports: ${FLUX_PORTS[*]}"
echo "  log dir  : $LOG_DIR"
echo "============================================================"

# ── helpers: start/stop all VLM servers (one per GPU) ───────────────
start_vlm() {
    echo "[VLM] starting ${#GPUS[@]} servers on GPUs: ${GPUS[*]}"
    local pids=()
    for i in "${!GPUS[@]}"; do
        local gpu="${GPUS[$i]}"
        local port="${VLM_PORTS[$i]}"
        local log="$LOG_DIR/vlm_${port}.log"
        (
            set +u; source "${CONDA_INIT}"; conda activate "${CONDA_ENV_SERVER}"; set -u
            CUDA_VISIBLE_DEVICES="$gpu" \
            VLM_MODEL="$VLM_CKPT" \
            VLM_PORT="$port" \
            VLM_TP=1 \
            VLM_MEM_FRAC="${VLM_MEM_FRAC:-0.57}" \
            SGLANG_DISABLE_CUDNN_CHECK=1 \
                bash scripts/tools/launch_local_vlm.sh
        ) > "$log" 2>&1 &
        pids+=($!)
        echo "[VLM]   GPU=$gpu port=$port pid=${pids[-1]}"
    done
    printf '%s\n' "${pids[@]}" > "$LOG_DIR/vlm.pids"

    local deadline=$(( $(date +%s) + 900 ))
    local ready=0
    while [ "$ready" -lt "${#VLM_PORTS[@]}" ]; do
        [ "$(date +%s)" -gt "$deadline" ] && {
            echo "[VLM] TIMEOUT — check $LOG_DIR/vlm_*.log"; return 1
        }
        ready=0
        for port in "${VLM_PORTS[@]}"; do
            curl -s -m 2 "http://localhost:${port}/v1/models" >/dev/null 2>&1 && (( ready++ )) || true
        done
        [ "$ready" -lt "${#VLM_PORTS[@]}" ] && sleep 5
    done
    echo "[VLM] all ${#VLM_PORTS[@]} servers ready"
}

stop_vlm() {
    [ -f "$LOG_DIR/vlm.pids" ] && {
        while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done < "$LOG_DIR/vlm.pids"
        rm -f "$LOG_DIR/vlm.pids"
    }
    for port in "${VLM_PORTS[@]}"; do pkill -9 -f "sglang.*${port}" 2>/dev/null || true; done
    echo "[VLM] all servers stopped"; sleep 2
}

# ── helpers: start/stop all FLUX servers (one per GPU) ──────────────
start_flux() {
    echo "[FLUX] starting ${#GPUS[@]} servers on GPUs: ${GPUS[*]}"
    local pids=()
    for i in "${!GPUS[@]}"; do
        local gpu="${GPUS[$i]}"
        local port="${FLUX_PORTS[$i]}"
        local log="$LOG_DIR/flux_${port}.log"
        (
            set +u; source "${CONDA_INIT}"; conda activate "${CONDA_ENV_SERVER}"; set -u
            CUDA_VISIBLE_DEVICES="$gpu" \
                "$PY_SRV" scripts/tools/image_edit_server.py \
                    --model "$EDIT_CKPT" --port "$port"
        ) > "$log" 2>&1 &
        pids+=($!)
        echo "[FLUX]  GPU=$gpu port=$port pid=${pids[-1]}"
    done
    printf '%s\n' "${pids[@]}" > "$LOG_DIR/flux.pids"

    local deadline=$(( $(date +%s) + 600 ))
    local ready=0
    while [ "$ready" -lt "${#FLUX_PORTS[@]}" ]; do
        [ "$(date +%s)" -gt "$deadline" ] && {
            echo "[FLUX] TIMEOUT — check $LOG_DIR/flux_*.log"; return 1
        }
        ready=0
        for port in "${FLUX_PORTS[@]}"; do
            curl -s -m 2 -o /dev/null -w "%{http_code}" \
                "http://localhost:${port}/health" 2>/dev/null | grep -q "200" && (( ready++ )) || true
        done
        [ "$ready" -lt "${#FLUX_PORTS[@]}" ] && sleep 5
    done
    echo "[FLUX] all ${#FLUX_PORTS[@]} servers ready"
}

stop_flux() {
    [ -f "$LOG_DIR/flux.pids" ] && {
        while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done < "$LOG_DIR/flux.pids"
        rm -f "$LOG_DIR/flux.pids"
    }
    for port in "${FLUX_PORTS[@]}"; do pkill -9 -f "image_edit_server.*${port}" 2>/dev/null || true; done
    echo "[FLUX] all servers stopped"; sleep 2
}

cleanup() {
    echo "[main] cleanup..."
    stop_vlm  2>/dev/null || true
    stop_flux 2>/dev/null || true
}
trap cleanup EXIT

# ── helpers: single-stage runner ─────────────────────────────

_run_stage_python() {
    # Run one pipeline stage; output goes to log and stdout via tee.
    local stage="$1" log="$2"
    ATTN_BACKEND=xformers \
        "$PY_PIPE" -m partcraft.pipeline_v3.run \
            --config       "$CFG" \
            --shard        "$SHARD" \
            --obj-ids-file "$OBJ_IDS_FILE" \
            --stage        "$stage" \
            2>&1 | tee "$log"
    return "${PIPESTATUS[0]}"
}

_stage_env() {
    # Load STAGE_* shell vars for a given stage into the calling shell.
    local stage="$1"
    eval "$(
        "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg, stage_name='$stage'))
"
    )"
}

# ── stage loop (batch-aware: parallel_group stages run concurrently) ──

# Build ordered batches: each batch is a space-separated list of stage names
# that share the same parallel_group and can run concurrently.
readarray -t BATCHES < <(
    "$PY_PIPE" -c "
import yaml, sys
sys.path.insert(0, '.')
from partcraft.pipeline_v3 import scheduler as sched
cfg = yaml.safe_load(open('$CFG'))
all_names = [s.name for s in sched.stages_for(cfg)]
selected  = [n for n in all_names if n in set('''${SELECTED_STAGES[*]}'''.split())]
for batch in sched.dump_stage_batches(cfg, selected):
    print(' '.join(batch))
"
)

for batch_line in "${BATCHES[@]}"; do
    read -r -a BATCH <<< "$batch_line"

    # ── single stage (solo or no parallel_group) ────────────────────────────────────
    if [ "${#BATCH[@]}" -eq 1 ]; then
        stage="${BATCH[0]}"
        _stage_env "$stage"

        echo ""
        echo "[stage] ▶ $stage — $STAGE_DESC  (servers=$STAGE_SERVERS)"

        case "$STAGE_SERVERS" in
            vlm)  start_vlm  || exit 1 ;;
            flux) start_flux || exit 1 ;;
        esac

        STAGE_LOG="$LOG_DIR/stage_${stage}_${TIMESTAMP}.log"
        _run_stage_python "$stage" "$STAGE_LOG"
        rc=$?

        case "$STAGE_SERVERS" in
            vlm)  stop_vlm  ;;
            flux) stop_flux ;;
        esac

        echo "[stage] ✓ $stage done (rc=$rc)"
        [ $rc -ne 0 ] && { echo "[ERROR] stage $stage failed"; exit $rc; }

    # ── parallel batch: start server once, run all stages concurrently ────────────
    else
        echo ""
        echo "[batch] ▶ parallel: ${BATCH[*]}"

        # Find which server type this batch needs (at most one per group).
        BATCH_SERVERS="none"
        for stage in "${BATCH[@]}"; do
            _stage_env "$stage"
            [ "$STAGE_SERVERS" != "none" ] && BATCH_SERVERS="$STAGE_SERVERS"
        done

        case "$BATCH_SERVERS" in
            vlm)  start_vlm  || exit 1 ;;
            flux) start_flux || exit 1 ;;
        esac

        PIDS=()
        for stage in "${BATCH[@]}"; do
            STAGE_LOG="$LOG_DIR/stage_${stage}_${TIMESTAMP}.log"
            echo "[batch]   launching $stage → $STAGE_LOG"
            _run_stage_python "$stage" "$STAGE_LOG" &
            PIDS+=($!)
        done

        # Wait for all and collect worst exit code.
        BATCH_RC=0
        for pid in "${PIDS[@]}"; do
            wait "$pid" || BATCH_RC=$?
        done

        case "$BATCH_SERVERS" in
            vlm)  stop_vlm  ;;
            flux) stop_flux ;;
        esac

        echo "[batch] ✓ ${BATCH[*]} done (rc=$BATCH_RC)"
        [ $BATCH_RC -ne 0 ] && { echo "[ERROR] parallel batch failed"; exit $BATCH_RC; }
    fi
done

echo ""
echo "=== ALL STAGES DONE ==="
