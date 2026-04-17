# Runbook: Restart a running v3 shard with `trellis_workers_per_gpu=2`

**Purpose**: kill a long-running `pipeline_v3` shard mid-`trellis_preview` and
relaunch it so the trellis_3d step uses K=2 workers per GPU (1.3-1.8x faster on
144 GB cards).

**Use this when**:
- Shard is currently running with K=1 (default) on `pipeline_v3`.
- Cards are L20X / H800 (>=80 GB) — verify in step 1.
- You don't want to wait for the natural shard completion.

**Don't use this when**:
- Shard is in `text_gen_gate_a`, `flux_2d`, or `deletion_cpu` stage — K=2 only
  helps `trellis_3d`. Wait until trellis_preview is the active stage.
- Cards have <80 GB free memory.

**Telling Cursor to do it for you**:

> "In this repo, git pull, then follow `docs/runbooks/restart-shard-K2.md` to
> restart shard `<NN>` (config `configs/pipeline_v3_shard<NN>.yaml`)."

Cursor reads this file as a checklist and walks the steps. All decisions and
verification gates are inlined here so no extra context is needed.

---

## 0. Replace these placeholders

| Placeholder | Where to find it |
|---|---|
| `<NN>` | shard number, e.g. `08`, `05`, `04`. Matches `configs/pipeline_v3_shard<NN>.yaml`. |
| `<TAG>` | shard tag, usually `shard<NN>`. |
| `<GPUS>` | space-separated list from `pipeline.gpus` in the YAML, e.g. `2 3 4 5 6 7`. |
| `<N_GPUS>` | length of `<GPUS>`, e.g. `6`. |
| `<TOTAL_WORKERS>` | `<N_GPUS> * 2`, e.g. `12`. |

Quick way to find them:

```bash
python - <<'PY'
import yaml, sys
shard = sys.argv[1]  # NN
cfg = yaml.safe_load(open(f"configs/pipeline_v3_shard{shard}.yaml").read())
gpus = cfg["pipeline"]["gpus"]
print(f"NN={shard}  GPUS=({' '.join(map(str,gpus))})  N={len(gpus)}  TOTAL={2*len(gpus)}")
PY <NN>
```

---

## 1. Pre-flight (must all pass before killing)

### 1a. Pull latest code (the K=2 fan-out lives here)

```bash
cd <repo-root>
git pull --ff-only origin feature/prompt-driven-part-selection  # or main, whichever has the K=2 commits
```

Required commits in `git log` (any order):

| SHA | Subject |
|---|---|
| `4eb1dbb` (or descendant) | `feat(pipeline_v3): add trellis_workers_per_gpu config accessor` |
| `7ba1811` | `feat(pipeline_v3): K workers per GPU for trellis_3d dispatch` |
| `c70b8de` | `config: enable trellis_workers_per_gpu=2 on shard08` (template-only on others) |

Verify:

```bash
git log --oneline | grep -E "trellis_workers_per_gpu|K workers per GPU" | head
```

### 1b. Confirm config has K=2 (or set it)

```bash
python - <<'PY'
import yaml
from partcraft.pipeline_v3.services_cfg import trellis_workers_per_gpu
cfg = yaml.safe_load(open("configs/pipeline_v3_shard<NN>.yaml").read())
k = trellis_workers_per_gpu(cfg)
gpus = cfg["pipeline"]["gpus"]
print(f"K={k}  GPUs={gpus}  total_workers={k*len(gpus)}")
assert k == 2, f"K is {k}, expected 2"
print("OK")
PY
```

If `K=1`, edit `configs/pipeline_v3_shard<NN>.yaml`, in `services.image_edit:`
add (4-space indent, sibling of `workers_per_server`):

```yaml
    trellis_workers_per_gpu: 2
```

Then commit + push (the running process won't pick it up, but the restart will).

### 1c. Confirm cards have memory budget

```bash
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | awk -F, '{printf "GPU%s: total=%6dMB free=%6dMB\n", $1, $2, $3}'
```

For each GPU in `<GPUS>`:
- `total >= 80000` MB (80 GB) — minimum for K=2.
- `free + ~25000 >= 50000` MB after subtracting the current trellis worker
  (~17-30 GB). K=2 needs ~34-60 GB resident; 144 GB cards have enormous
  headroom, 80 GB cards are tight but workable.

If any target GPU is hosting unrelated heavy jobs (e.g. another shard's FLUX
servers using 100+ GB), **stop** and resolve that first.

### 1d. Confirm the shard is in `trellis_preview` stage

```bash
ps -ef | grep -E "pipeline_v3.*shard <NN>.*--stage" | grep -v grep
```

Expect ONE line containing `--stage trellis_preview`. If you see `--stage flux_2d`
or `--stage text_gen_gate_a`, **don't restart now** — K=2 only helps trellis_3d.

---

## 2. Snapshot baseline (so you can measure speedup later)

```bash
cd <repo-root>
{
  echo "=== Baseline at $(date -Is) ==="
  for typ in scl glb mod clr mat; do
    done_=$(ls outputs/partverse/<TAG>/mode_e_text_align/objects/<NN>/*/edits_3d/${typ}_*/after.npz 2>/dev/null | wc -l)
    total=$(ls outputs/partverse/<TAG>/mode_e_text_align/objects/<NN>/*/edits_2d/${typ}_*_edited.png 2>/dev/null | wc -l)
    printf "  %s : %5d / %5d\n" "$typ" "$done_" "$total"
  done
  TOTAL=$(ls outputs/partverse/<TAG>/mode_e_text_align/objects/<NN>/*/edits_3d/*/after.npz 2>/dev/null | wc -l)
  echo "TOTAL_DONE=$TOTAL"
} | tee /tmp/<TAG>_baseline.txt
```

Save the timestamp `T0` of this snapshot.

---

## 3. Find the live process tree

```bash
echo "--- bash orchestrator ---"
ps -ef | grep "run_pipeline_v3_shard.sh <TAG>" | grep -v grep
echo "--- python stage launcher (with --stage) ---"
ps -ef | grep "pipeline_v3.run.*shard <NN>.*--stage" | grep -v grep
echo "--- N worker children (one per GPU, --gpu-shard k/N) ---"
ps -ef | grep -E "pipeline_v3.run.*shard <NN>.*--single-gpu --gpu-shard.*/<N_GPUS>" | grep -v grep
```

Note three sets of PIDs:
- **`ORCH_PID`** — bash `run_pipeline_v3_shard.sh`
- **`STAGE_PID`** — python `pipeline_v3.run --stage trellis_preview`
- **`WORKER_PIDS`** — N python `--gpu-shard k/<N_GPUS>` children (their parent
  is `STAGE_PID`)

---

## 4. Graceful kill

> **Critical know-how 1**: SIGTERM on the bash orchestrator does **not**
> propagate to the worker children — `dispatch_gpus` uses `subprocess.Popen`
> without `start_new_session=True` / process group. Workers become orphans
> (PPID=1) and keep running. You must SIGTERM the workers directly.

> **Critical know-how 2**: `partcraft/io/npz_utils.save_npz` writes
> `after.npz` directly without a tempfile-then-rename atomic dance
> (`np.savez(path, **data)`). A worker SIGKILLed mid-write would leave a
> truncated file that the resume logic
> (`partcraft/pipeline_v3/trellis_3d.py:122-126`) treats as "done", causing
> permanent corruption for that edit. Step 4 + Step 5 prevent this.

### 4a. Mark the kill window (used by the cache scan in Step 5)

```bash
KILL_MARK=/tmp/kill_mark_$(date +%s)
touch "$KILL_MARK"
echo "$KILL_MARK" > /tmp/last_kill_mark.txt
echo "kill mark: $KILL_MARK ($(date -Is))"
```

### 4b. SIGTERM orchestrator + stage launcher + workers, in that order

```bash
kill -TERM <ORCH_PID> <STAGE_PID> <WORKER_PIDS>
```

### 4c. Wait up to 60s for workers to finish their current edit

Each edit cycle is ~10-30s. Workers handle SIGTERM by exiting after the current
`refiner.edit()` returns (which always finishes by writing `after.npz`).
Empirically they exit within 5-15s of SIGTERM at edit boundaries.

```bash
for i in 6 5 4 3 2 1; do
  sleep 10
  alive=$(ps -p <WORKER_PIDS> --no-headers 2>/dev/null | wc -l)
  echo "  $((i*10))s left, $alive workers still alive"
  [ "$alive" = "0" ] && break
done
```

### 4d. SIGKILL stragglers

```bash
kill -9 <WORKER_PIDS> 2>/dev/null || echo "(none left)"
sleep 3
echo "=== survivors (should be empty) ==="
ps -ef | grep -E "pipeline_v3.*shard <NN>" | grep -v grep || echo "all clear"
```

If anything still has `shard <NN>`, manually kill it before continuing.

---

## 5. Cache safety scan (mandatory)

Find any `after.npz` modified at or after `KILL_MARK` — these are the only files
that *could* be partial writes. Validate them; delete corrupt ones.

```bash
KILL_MARK=$(cat /tmp/last_kill_mark.txt)
SUSPECT=$(find outputs/partverse/<TAG>/mode_e_text_align/objects/<NN> \
  -name "after.npz" -newer "$KILL_MARK" 2>/dev/null)
echo "suspect count: $(echo "$SUSPECT" | grep -c .)"

echo "$SUSPECT" | while read -r f; do
  [ -z "$f" ] && continue
  size=$(stat -c %s "$f")
  if [ "$size" -lt 4096 ]; then
    echo "  TINY ($size B) — DELETING: $f"
    rm -f "$f"
  elif ! python -c "import numpy as np; np.load('$f')" 2>/dev/null; then
    echo "  CORRUPT ($size B) — DELETING: $f"
    rm -f "$f"
  else
    echo "  OK ($size B): $f"
  fi
done
```

**Expected on a healthy kill**: 0 suspects. Workers usually exit between edits.
If you see TINY/CORRUPT files, the deletion is necessary so the resume logic
re-runs those edits.

---

## 6. Restart with K=2

The `STAGES` env var skips already-completed stages, jumping straight back to
`trellis_preview`.

### 6a. If a tmux session already exists for this shard

```bash
tmux send-keys -t <TAG> 'STAGES=trellis_preview bash scripts/tools/run_pipeline_v3_shard.sh <TAG> configs/pipeline_v3_shard<NN>.yaml' Enter
```

### 6b. Otherwise create one

```bash
tmux new -d -s <TAG> 'STAGES=trellis_preview bash scripts/tools/run_pipeline_v3_shard.sh <TAG> configs/pipeline_v3_shard<NN>.yaml'
```

### 6c. Confirm launch

```bash
sleep 5
tmux capture-pane -t <TAG> -p | tail -10
```

Expect to see "loading machine env", "starting stage trellis_preview", etc.

---

## 7. Verify K=2 actually engaged (must check ALL four)

### 7a. Dispatch fan-out log line

```bash
sleep 60  # wait for dispatch to log
grep -E "dispatching: gpus=.*workers_per_gpu=2 total_workers=<TOTAL_WORKERS>" \
  logs/v3_<TAG>/stage_trellis_preview.log
grep -E "waiting on <TOTAL_WORKERS> children" \
  logs/v3_<TAG>/stage_trellis_preview.log
```

Both must produce a hit. If `workers_per_gpu=1`, the K=2 setting didn't reach
`dispatch_gpus` — re-check Step 1b.

### 7b. <TOTAL_WORKERS> worker children spawned

```bash
ps -ef | grep -E "pipeline_v3.run.*shard <NN>.*--gpu-shard.*/<TOTAL_WORKERS>" \
  | grep -v grep | wc -l
```

Must equal `<TOTAL_WORKERS>`.

### 7c. 2 trellis processes per target GPU

```bash
for g in <GPUS>; do
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $g \
    | wc -l)
  echo "GPU $g: $n compute processes (expect >=2)"
done
```

Each must be >=2 (may be more if there are unrelated CUDA jobs like Blender on
the same GPU).

### 7d. First edit completes within 10-15 minutes of restart

Bootstrap is slow because <TOTAL_WORKERS> processes simultaneously read
TRELLIS-image-large + TRELLIS-text-xlarge + DINOv2 + CLIP from NFS. Expect
7-10 min until first worker logs "TRELLIS models loaded", another 1-2 min for
first `[s5] ... ok`.

```bash
sleep 600  # 10 min
echo "--- TRELLIS load count (expect >=N_GPUS after 10 min) ---"
grep -c "TRELLIS models loaded" logs/v3_<TAG>/stage_trellis_preview.log
echo "--- first [s5] ok lines (expect >=1) ---"
grep -E "\[s5\] .* ok" logs/v3_<TAG>/stage_trellis_preview.log | head -3
```

### 7e. Per-worker memory in steady state

After 15-20 min, workers should be in steady-state range:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | awk -F, '{print $1, $2}'
```

Per-worker mem: **15-30 GB** typical, **up to 43 GB** on heavy objects. If
any worker is at 60+ GB **and growing**, suspect a leak — kill that worker (its
sibling on the same GPU will keep going).

---

## 8. Measure speedup (after ~30 min of warm steady-state)

```bash
cd <repo-root>
NEW_TOTAL=$(ls outputs/partverse/<TAG>/mode_e_text_align/objects/<NN>/*/edits_3d/*/after.npz 2>/dev/null | wc -l)
BASELINE_TOTAL=$(grep TOTAL_DONE /tmp/<TAG>_baseline.txt | cut -d= -f2)
echo "delta=$((NEW_TOTAL - BASELINE_TOTAL)) edits since baseline"
echo "stage runtime so far:"
ps -o etime= -p $(pgrep -f "pipeline_v3.run.*shard <NN>.*--stage trellis_preview")
```

Compute edits/hour and compare to the K=1 baseline rate (typically 80-90/hr/GPU
on L20X). Expected K=2 throughput: **120-150/hr/GPU** (1.5-1.8x).

---

## 9. Rollback (if K=2 OOMs or destabilizes)

1. Repeat Steps 4-5 to graceful-kill the K=2 run.
2. Edit `configs/pipeline_v3_shard<NN>.yaml`: change
   `trellis_workers_per_gpu: 2` to `trellis_workers_per_gpu: 1` (or delete the
   line entirely — default is 1).
3. Repeat Step 6 to restart at K=1.

Alternatively, set `TRELLIS_WORKERS_PER_GPU=1` in the launch env to override
without touching the config:

```bash
tmux send-keys -t <TAG> 'STAGES=trellis_preview TRELLIS_WORKERS_PER_GPU=1 bash scripts/tools/run_pipeline_v3_shard.sh <TAG> configs/pipeline_v3_shard<NN>.yaml' Enter
```

---

## Appendix: anchors in the codebase

| Concern | File:line |
|---|---|
| K resolution (env > YAML > default) | `partcraft/pipeline_v3/services_cfg.py:55-73` |
| Dispatch fan-out (the `K*N` fork) | `partcraft/pipeline_v3/run.py:429-487` |
| K-only-for-`trellis_3d` gate | `partcraft/pipeline_v3/run.py:451-454` |
| Resume skip (after.npz exists) | `partcraft/pipeline_v3/trellis_3d.py:122-126` |
| `after.npz` write (NOT atomic) | `partcraft/io/npz_utils.py:14-35` |
| YAML default | `configs/templates/pipeline_v3_bench.template.yaml` (`services.image_edit.trellis_workers_per_gpu: 1`) |

For background on why K=2 helps, see `docs/ARCH.md` -> "Trellis 单卡多 worker
（`trellis_workers_per_gpu`，pipeline_v3）".
