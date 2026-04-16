#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ "$#" -lt 4 ]; then
  echo "Usage: bash scripts/tools/run_pipeline_v3_bench.sh <config.yaml> <obj_ids.txt> <shard> <tmux_session>"
  echo "Example:"
  echo "  bash scripts/tools/run_pipeline_v3_bench.sh \\\n    configs/pipeline_v3_shard08_test20.yaml \\\n    configs/shard08_test20_obj_ids.txt \\\n    08 v3_test20_fullrerun"
  exit 2
fi

CFG="$1"
OBJ_IDS_FILE="$2"
SHARD="$3"
SESSION="$4"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

[ -f "$CFG" ] || { echo "[ERROR] config not found: $CFG" >&2; exit 1; }
[ -f "$OBJ_IDS_FILE" ] || { echo "[ERROR] obj_ids file not found: $OBJ_IDS_FILE" >&2; exit 1; }

ENV_FILE="${MACHINE_ENV:-configs/machine/$(hostname).env}"
[ -f "$ENV_FILE" ] || { echo "[ERROR] machine env not found: $ENV_FILE" >&2; exit 1; }
source "$ENV_FILE"

: "${CONDA_INIT:?missing CONDA_INIT}"
: "${CONDA_ENV_SERVER:?missing CONDA_ENV_SERVER}"
: "${CONDA_ENV_PIPELINE:?missing CONDA_ENV_PIPELINE}"
: "${VLM_CKPT:?missing VLM_CKPT}"
: "${EDIT_CKPT:?missing EDIT_CKPT}"

set +u
source "$CONDA_INIT"
set -u

PY_PIPE="$(conda run -n "$CONDA_ENV_PIPELINE" which python 2>/dev/null)" || PY_PIPE="/root/miniconda3/envs/${CONDA_ENV_PIPELINE}/bin/python"
PY_SRV="$(conda run -n "$CONDA_ENV_SERVER" which python 2>/dev/null)" || PY_SRV="/root/miniconda3/envs/${CONDA_ENV_SERVER}/bin/python"

OUT_DIR="$($PY_PIPE - <<'PY' "$CFG"
import yaml,sys
cfg=yaml.safe_load(open(sys.argv[1]))
print((cfg.get('data') or {}).get('output_dir',''))
PY
)"

if [ -z "$OUT_DIR" ]; then
  echo "[ERROR] data.output_dir missing in config: $CFG" >&2
  exit 1
fi

echo "[1/4] validating inputs"
"$PY_PIPE" scripts/tools/validate_v3_inputs.py \
  --config "$CFG" \
  --shard "$SHARD" \
  --obj-ids-file "$OBJ_IDS_FILE" \
  -v

echo "[2/4] cleaning output dir: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/v3_bench_runs/${SESSION}_${TS}"
mkdir -p "$LOG_DIR"

RUNNER="$LOG_DIR/run_inside_tmux.sh"
cat > "$RUNNER" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?}"
CFG="${CFG:?}"
OBJ_IDS_FILE="${OBJ_IDS_FILE:?}"
SHARD="${SHARD:?}"
LOG_DIR="${LOG_DIR:?}"
CONDA_INIT="${CONDA_INIT:?}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:?}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:?}"
VLM_CKPT="${VLM_CKPT:?}"
EDIT_CKPT="${EDIT_CKPT:?}"
PY_PIPE="${PY_PIPE:?}"
PY_SRV="${PY_SRV:?}"

cd "$ROOT"

eval "$($PY_PIPE - <<'PY' "$CFG"
import yaml,sys
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg=yaml.safe_load(open(sys.argv[1]))
print(dump_shell_env(cfg))
PY
)"

start_vlm() {
  echo "[VLM] starting ${#GPUS[@]} servers on GPUs: ${GPUS[*]}"
  local pids=()
  for i in "${!GPUS[@]}"; do
    local gpu="${GPUS[$i]}"
    local port="${VLM_PORTS[$i]}"
    local log="$LOG_DIR/vlm_${port}.log"
    (
      set +u; source "$CONDA_INIT"; conda activate "$CONDA_ENV_SERVER"; set -u
      CUDA_VISIBLE_DEVICES="$gpu" \
      VLM_MODEL="$VLM_CKPT" \
      VLM_PORT="$port" \
      VLM_TP=1 \
      VLM_MEM_FRAC="${VLM_MEM_FRAC:-0.57}" \
      SGLANG_DISABLE_CUDNN_CHECK=1 \
      bash scripts/tools/launch_local_vlm.sh
    ) > "$log" 2>&1 &
    pids+=($!)
  done
  printf '%s\n' "${pids[@]}" > "$LOG_DIR/vlm.pids"

  local deadline=$(( $(date +%s) + 900 ))
  while true; do
    local ready=0
    for port in "${VLM_PORTS[@]}"; do
      curl -s -m 2 "http://localhost:${port}/v1/models" >/dev/null 2>&1 && ((ready++)) || true
    done
    [ "$ready" -eq "${#VLM_PORTS[@]}" ] && break
    [ "$(date +%s)" -gt "$deadline" ] && { echo "[VLM] TIMEOUT"; return 1; }
    sleep 5
  done
  echo "[VLM] ready"
}

stop_vlm() {
  [ -f "$LOG_DIR/vlm.pids" ] && {
    while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done < "$LOG_DIR/vlm.pids"
    rm -f "$LOG_DIR/vlm.pids"
  }
  for port in "${VLM_PORTS[@]}"; do pkill -9 -f "sglang.*${port}" 2>/dev/null || true; done
}

start_flux() {
  echo "[FLUX] starting ${#GPUS[@]} servers on GPUs: ${GPUS[*]}"
  local pids=()
  for i in "${!GPUS[@]}"; do
    local gpu="${GPUS[$i]}"
    local port="${FLUX_PORTS[$i]}"
    local log="$LOG_DIR/flux_${port}.log"
    (
      set +u; source "$CONDA_INIT"; conda activate "$CONDA_ENV_SERVER"; set -u
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PY_SRV" scripts/tools/image_edit_server.py \
        --model "$EDIT_CKPT" --port "$port"
    ) > "$log" 2>&1 &
    pids+=($!)
  done
  printf '%s\n' "${pids[@]}" > "$LOG_DIR/flux.pids"

  local deadline=$(( $(date +%s) + 600 ))
  while true; do
    local ready=0
    for port in "${FLUX_PORTS[@]}"; do
      curl -s -m 2 -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null | grep -q "200" && ((ready++)) || true
    done
    [ "$ready" -eq "${#FLUX_PORTS[@]}" ] && break
    [ "$(date +%s)" -gt "$deadline" ] && { echo "[FLUX] TIMEOUT"; return 1; }
    sleep 5
  done
  echo "[FLUX] ready"
}

stop_flux() {
  [ -f "$LOG_DIR/flux.pids" ] && {
    while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done < "$LOG_DIR/flux.pids"
    rm -f "$LOG_DIR/flux.pids"
  }
  for port in "${FLUX_PORTS[@]}"; do pkill -9 -f "image_edit_server.*${port}" 2>/dev/null || true; done
}

cleanup() {
  stop_vlm || true
  stop_flux || true
}
trap cleanup EXIT

run_stage() {
  local stage="$1"
  local log="$LOG_DIR/stage_${stage}.log"
  echo "[stage] >>> $stage"
  ATTN_BACKEND=xformers "$PY_PIPE" -m partcraft.pipeline_v3.run \
    --config "$CFG" \
    --shard "$SHARD" \
    --obj-ids-file "$OBJ_IDS_FILE" \
    --stage "$stage" \
    2>&1 | tee "$log"
}

readarray -t BATCHES < <(
  "$PY_PIPE" - <<'PY' "$CFG"
import yaml,sys
from partcraft.pipeline_v3 import scheduler as sched
cfg=yaml.safe_load(open(sys.argv[1]))
all_names=[s.name for s in sched.stages_for(cfg)]
for batch in sched.dump_stage_batches(cfg, all_names):
    print(' '.join(batch))
PY
)

for line in "${BATCHES[@]}"; do
  read -r -a batch <<< "$line"

  if [ "${#batch[@]}" -eq 1 ]; then
    stage="${batch[0]}"
    eval "$($PY_PIPE - <<'PY' "$CFG" "$stage"
import yaml,sys
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg=yaml.safe_load(open(sys.argv[1]))
print(dump_shell_env(cfg, stage_name=sys.argv[2]))
PY
)"

    case "$STAGE_SERVERS" in
      vlm)  start_vlm ;;
      flux) start_flux ;;
    esac

    run_stage "$stage"

    case "$STAGE_SERVERS" in
      vlm)  stop_vlm ;;
      flux) stop_flux ;;
    esac

  else
    batch_servers="none"
    for stage in "${batch[@]}"; do
      eval "$($PY_PIPE - <<'PY' "$CFG" "$stage"
import yaml,sys
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg=yaml.safe_load(open(sys.argv[1]))
print(dump_shell_env(cfg, stage_name=sys.argv[2]))
PY
)"
      [ "$STAGE_SERVERS" != "none" ] && batch_servers="$STAGE_SERVERS"
    done

    case "$batch_servers" in
      vlm)  start_vlm ;;
      flux) start_flux ;;
    esac

    pids=()
    for stage in "${batch[@]}"; do
      (
        run_stage "$stage"
      ) > "$LOG_DIR/stage_${stage}.log" 2>&1 &
      pids+=($!)
      echo "[batch] launched $stage (pid=${pids[-1]})"
    done

    batch_rc=0
    for pid in "${pids[@]}"; do
      wait "$pid" || batch_rc=1
    done

    case "$batch_servers" in
      vlm)  stop_vlm ;;
      flux) stop_flux ;;
    esac

    [ "$batch_rc" -eq 0 ] || { echo "[ERROR] parallel batch failed"; exit 1; }
  fi
done

echo "[DONE] full pipeline finished"
EOS

chmod +x "$RUNNER"

echo "[3/4] creating tmux session: $SESSION"
tmux kill-session -t "$SESSION" 2>/dev/null || true
TMUX_CMD="cd '$ROOT' && ROOT='$ROOT' CFG='$CFG' OBJ_IDS_FILE='$OBJ_IDS_FILE' SHARD='$SHARD' LOG_DIR='$LOG_DIR' CONDA_INIT='$CONDA_INIT' CONDA_ENV_SERVER='$CONDA_ENV_SERVER' CONDA_ENV_PIPELINE='$CONDA_ENV_PIPELINE' VLM_CKPT='$VLM_CKPT' EDIT_CKPT='$EDIT_CKPT' PY_PIPE='$PY_PIPE' PY_SRV='$PY_SRV' bash '$RUNNER'"
tmux new-session -d -s "$SESSION" "$TMUX_CMD"

echo "[4/4] started"
echo "  session : $SESSION"
echo "  logs    : $LOG_DIR"
echo "  attach  : tmux attach -t $SESSION"
echo "  monitor : tmux capture-pane -pt $SESSION:0 | tail -n 80"
