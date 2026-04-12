#!/usr/bin/env bash
# watch_and_report.sh
# 监控 pipeline_v2 管线完成，自动 copy 已完成对象到 test 目录并生成 HTML 报告
#
# Usage:
#   bash scripts/tools/watch_and_report.sh [tmux-session] [run-dir] [test-dir] [interval-sec]
#
# Defaults:
#   tmux-session = pc3d_shard02
#   run-dir      = /mnt/zsn/data/partverse/outputs/partverse/pipeline_v2_shard02
#   test-dir     = /mnt/zsn/data/partverse/outputs/partverse/pipeline_v2_shard02_test
#   interval     = 60

set -euo pipefail
TMUX_SESSION="${1:-pc3d_shard02}"
RUN_DIR="${2:-/mnt/zsn/data/partverse/outputs/partverse/pipeline_v2_shard02}"
TEST_DIR="${3:-/mnt/zsn/data/partverse/outputs/partverse/pipeline_v2_shard02_test}"
INTERVAL="${4:-60}"
WORKSPACE="${WORKSPACE:-/mnt/zsn/zsn_workspace/PartCraft3D}"
REPORT_SCRIPT="$WORKSPACE/scripts/vis/generate_qc_report.py"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

is_pipeline_done() {
    # Return 0 (true) if tmux session shows shell prompt (no active pipeline command)
    local out
    out=$(tmux capture-pane -t "$TMUX_SESSION" -p -S -5 2>/dev/null || true)
    if echo "$out" | grep -q "=== ALL STAGES DONE ==="; then
        return 0
    fi
    # Also check: no python pipeline process alive in that session
    local pids
    pids=$(tmux list-panes -t "$TMUX_SESSION" -F "#{pane_pid}" 2>/dev/null || true)
    for pid in $pids; do
        local children
        children=$(pgrep -P "$pid" 2>/dev/null | head -3 || true)
        for cpid in $children; do
            local cmd
            cmd=$(ps -p "$cpid" -o comm= 2>/dev/null || true)
            if [[ "$cmd" == "python"* ]]; then
                return 1   # still running
            fi
        done
    done
    return 0   # no python children = done
}

copy_and_report() {
    log "Pipeline done. Starting copy + report..."

    mkdir -p "$TEST_DIR/objects"

    # detect shard subdir
    local shard_src shard_dst
    shard_src=$(ls -d "$RUN_DIR/objects/"*/ 2>/dev/null | head -1 || true)
    if [[ -z "$shard_src" ]]; then
        log "ERROR: no shard dirs found under $RUN_DIR/objects/"
        exit 1
    fi
    local shard_name
    shard_name=$(basename "$shard_src")
    shard_dst="$TEST_DIR/objects/$shard_name"
    mkdir -p "$shard_dst"

    # copy objects that have s6p_preview ok
    local count=0
    for obj_dir in "$shard_src"/*/; do
        local status_file="$obj_dir/status.json"
        if [[ ! -f "$status_file" ]]; then continue; fi

        local s6p_status
        s6p_status=$(python3 -c "
import json,sys
d=json.load(open('$status_file'))
print(d.get('steps',{}).get('s6p_preview',{}).get('status','none'))
" 2>/dev/null || echo "none")

        if [[ "$s6p_status" == "ok" ]]; then
            local obj_id
            obj_id=$(basename "$obj_dir")
            local dst="$shard_dst/$obj_id"
            if [[ ! -d "$dst" ]]; then
                cp -r "$obj_dir" "$dst"
                log "  Copied $obj_id"
            else
                log "  Skip (exists) $obj_id"
            fi
            (( count++ ))
        fi
    done
    log "Copied $count objects to $TEST_DIR"

    # generate report for test dir
    local ts
    ts=$(date '+%Y%m%d_%H%M%S')
    local report_path="$TEST_DIR/qc_report_${ts}.html"
    python3 "$REPORT_SCRIPT" \
        --run-dir "$TEST_DIR" \
        --out "$report_path" \
        --min-stage s6p 2>&1 | while IFS= read -r line; do log "  $line"; done

    # also overwrite a fixed-name "latest"
    cp "$report_path" "$TEST_DIR/qc_report_latest.html"
    log "Report: $report_path"
    log "Latest: $TEST_DIR/qc_report_latest.html"
}

# ─── Main poll loop ───────────────────────────────────────────────────────────
log "Watching tmux session '$TMUX_SESSION' every ${INTERVAL}s"
log "Run dir : $RUN_DIR"
log "Test dir: $TEST_DIR"

while true; do
    if is_pipeline_done; then
        log "Pipeline appears done."
        copy_and_report
        log "Done. Exiting watcher."
        exit 0
    fi

    # live progress
    s6p_ok=$(python3 - <<'PYEOF' 2>/dev/null || echo "?"
import json
from pathlib import Path
base = Path("$shard_src")
n = sum(1 for d in base.iterdir()
        if (d/"status.json").is_file()
        and json.loads((d/"status.json").read_text())
               .get("steps",{}).get("s6p_preview",{}).get("status") == "ok")
print(n)
PYEOF
)
    log "s6p_preview ok: $s6p_ok  — sleeping ${INTERVAL}s..."
    sleep "$INTERVAL"
done
