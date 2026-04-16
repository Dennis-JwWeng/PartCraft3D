#!/usr/bin/env bash
# run_gate_quality.sh — run gate_quality step for any obj_ids_file
#
# Usage:
#   bash scripts/tools/run_gate_quality.sh \
#       --cfg   configs/pipeline_v3_shard08_bench100.yaml \
#       --ids   configs/shard08_test20_obj_ids.txt \
#       --shard 08 \
#       [--force]

set -euo pipefail

CFG="configs/pipeline_v3_shard08_bench100.yaml"
OBJ_IDS_FILE=""
SHARD="08"
FORCE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cfg)   CFG="$2";          shift 2 ;;
        --ids)   OBJ_IDS_FILE="$2"; shift 2 ;;
        --shard) SHARD="$2";        shift 2 ;;
        --force) FORCE="--force";   shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[ -f "$CFG" ]          || { echo "[ERROR] config not found: $CFG"; exit 1; }
[ -f "$OBJ_IDS_FILE" ] || { echo "[ERROR] obj_ids file not found: $OBJ_IDS_FILE"; exit 1; }

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ENV_FILE="${MACHINE_ENV:-configs/machine/$(hostname).env}"
[ -f "$ENV_FILE" ] || { echo "[ERROR] machine env not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"
CONDA_INIT="${CONDA_INIT:?}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:?}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:?}"
VLM_CKPT="${VLM_CKPT:?}"

set +u; source "${CONDA_INIT}"; set -u
PY_PIPE="$(conda run -n "${CONDA_ENV_PIPELINE}" which python 2>/dev/null)" \
    || PY_PIPE="/root/miniconda3/envs/${CONDA_ENV_PIPELINE}/bin/python"

LOG_DIR="logs/gate_quality_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

eval "$(
    "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg))
"
)"

echo "============================================================"
echo "  gate_quality re-run"
echo "  config   : $CFG"
echo "  obj_ids  : $OBJ_IDS_FILE  ($(grep -cv '^#' "$OBJ_IDS_FILE") objects)"
echo "  shard    : $SHARD"
echo "  force    : ${FORCE:---no force}"
echo "  GPUs     : ${GPUS[*]}  (${#GPUS[@]} total)"
echo "  VLM ports: ${VLM_PORTS[*]}"
echo "  log dir  : $LOG_DIR"
echo "============================================================"

start_vlm() {
    echo "[VLM] starting ${#GPUS[@]} servers..."
    local pids=()
    for i in "${!GPUS[@]}"; do
        local gpu="${GPUS[$i]}" port="${VLM_PORTS[$i]}"
        (
            set +u; source "${CONDA_INIT}"; conda activate "${CONDA_ENV_SERVER}"; set -u
            CUDA_VISIBLE_DEVICES="$gpu" VLM_MODEL="$VLM_CKPT" VLM_PORT="$port" \
            VLM_TP=1 VLM_MEM_FRAC="${VLM_MEM_FRAC:-0.57}" \
            SGLANG_DISABLE_CUDNN_CHECK=1 \
                bash scripts/tools/launch_local_vlm.sh
        ) > "$LOG_DIR/vlm_${port}.log" 2>&1 &
        pids+=($!)
        echo "[VLM]   GPU=$gpu port=$port"
    done
    printf '%s\n' "${pids[@]}" > "$LOG_DIR/vlm.pids"
    local deadline=$(( $(date +%s) + 900 )) ready=0
    echo "[VLM] waiting for servers..."
    while [ "$ready" -lt "${#VLM_PORTS[@]}" ]; do
        [ "$(date +%s)" -gt "$deadline" ] && { echo "[VLM] TIMEOUT"; return 1; }
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
    echo "[VLM] stopped"; sleep 2
}

cleanup() { stop_vlm 2>/dev/null || true; }
trap cleanup EXIT

start_vlm || exit 1

STAGE_LOG="$LOG_DIR/gate_quality.log"
"$PY_PIPE" -m partcraft.pipeline_v3.run \
    --config "$CFG" --shard "$SHARD" \
    --obj-ids-file "$OBJ_IDS_FILE" \
    --steps gate_quality $FORCE \
    2>&1 | tee "$STAGE_LOG"
RC="${PIPESTATUS[0]}"

stop_vlm
[ "$RC" -eq 0 ] && echo "=== DONE ===" || echo "=== FAILED (rc=$RC) ==="
exit "$RC"
