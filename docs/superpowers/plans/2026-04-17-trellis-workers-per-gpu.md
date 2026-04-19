# Trellis Workers-per-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `trellis_3d` to run K independent worker processes on every physical GPU, so DINOv2 conditioning / voxel-mask building / NFS I/O on one worker can overlap with Trellis flow-model sampling on another worker on the same GPU. K=1 must remain identical to today's behavior.

**Architecture:** Replace `dispatch_gpus` 1-process-per-GPU spawning with K-processes-per-GPU. Each child still inherits a single `CUDA_VISIBLE_DEVICES` value but advertises a wider `--gpu-shard` (`i / (K*N)`) so `slice_for_gpu`'s mod-N round-robin partitions edits across all K*N workers without overlap. Knob lives in `services.image_edit.trellis_workers_per_gpu` (default 1) with `TRELLIS_WORKERS_PER_GPU` env override.

**Tech Stack:** Python 3.10+, `subprocess.Popen`, existing `partcraft.pipeline_v3` orchestration, no new deps.

---

## Context (read first)

### Current call path

1. `scripts/tools/run_pipeline_v3_shard.sh` → `python -m partcraft.pipeline_v3.run --steps trellis_3d --gpus 0,1,2,3,4,5`.
2. `partcraft/pipeline_v3/run.py:main` (~line 645) calls `dispatch_gpus("trellis_3d", ...)`.
3. `dispatch_gpus` (`run.py:429-472`) spawns **N children**, one per GPU id, each child gets `CUDA_VISIBLE_DEVICES=<gpu>` and `--gpu-shard i/N`.
4. Each child re-enters `main` with `--single-gpu`, calls `slice_for_gpu(ctxs, i, N)` (`run.py:157-158`: round-robin mod-N), then `run_step("trellis_3d", ...)` → `partcraft.pipeline_v3.trellis_3d.run` → `_ensure_refiner` (loads ~20 GB Trellis text+image flow models, S1/S2 codecs, DINOv2) → loops `run_for_object` per object → `refiner.edit` per spec.

### Why K=2-3 is safe

- Per-process GPU memory of one Trellis worker observed at 22-43 GB (mixed flow + S1/S2 + DINOv2 + intermediate SLAT tensors).
- Target hardware (H100/H800): ~80-144 GB total; 35-100 GB headroom per GPU.
- K=2 ⇒ ~40-86 GB peak per GPU (safe). K=3 ⇒ ~66-129 GB peak (risky on 80 GB cards, OK on 144 GB H800).
- CPU bottleneck is per-process (DINOv2 conditioning, voxel mask via `scipy.ndimage`, NFS reads of `mesh.npz`); duplicating processes naturally parallelizes them.

### Files in scope

- **Modify**: `partcraft/pipeline_v3/run.py` (dispatch loop), `partcraft/pipeline_v3/services_cfg.py` (one accessor), `configs/_templates/pipeline_v3_bench.template.yaml` + `configs/pipeline_v3_shard08.yaml` (knob), `docs/ARCH.md` (one paragraph).
- **No change**: `partcraft/pipeline_v3/trellis_3d.py`, `partcraft/trellis/refiner.py`, `third_party/interweave_Trellis.py`, `scripts/tools/run_pipeline_v3_shard.sh`, scheduler. Shell orchestrator passes `--gpus` as today; only the Python side fans out wider.
- **No change to status / lock paths**: `status.json` and `update_edit_stage` already use file locks per object/edit; K extra writers don't change correctness.

---

## File Structure

```
partcraft/pipeline_v3/run.py                          # MODIFIED: dispatch_gpus widens to K*N children
partcraft/pipeline_v3/services_cfg.py                 # MODIFIED: + trellis_workers_per_gpu(cfg) helper
configs/_templates/pipeline_v3_bench.template.yaml    # MODIFIED: documented default = 1
configs/pipeline_v3_shard08.yaml                      # MODIFIED: opt-in to 2 (after smoke test)
docs/ARCH.md                                          # MODIFIED: one paragraph
```

No new files. No public API changes outside `services_cfg`.

---

## Task 1: Add `trellis_workers_per_gpu` accessor

**Files:**
- Modify: `partcraft/pipeline_v3/services_cfg.py:38-53` (add helper after `trellis_image_edit_flat`)

- [ ] **Step 1: Add accessor with env override + clamp**

```python
def trellis_workers_per_gpu(cfg: dict, *, default: int = 1) -> int:
    """Number of Trellis 3D worker subprocesses per physical GPU.

    Resolution order:
      1. ``TRELLIS_WORKERS_PER_GPU`` env var (ad-hoc override)
      2. ``services.image_edit.trellis_workers_per_gpu`` in YAML
      3. ``default`` (= 1, current behavior)

    Always clamped to >= 1.
    """
    import os
    raw = os.environ.get("TRELLIS_WORKERS_PER_GPU", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    ie = image_edit_service(cfg)
    v = ie.get("trellis_workers_per_gpu", default)
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = default
    return max(1, v)
```

- [ ] **Step 2: Export it**

Append `"trellis_workers_per_gpu"` to `__all__` at the bottom of the file.

- [ ] **Step 3: Smoke-test the accessor**

```bash
python - <<'PY'
import os
from partcraft.pipeline_v3.services_cfg import trellis_workers_per_gpu

cfg_no  = {"services": {"image_edit": {}}}
cfg_yes = {"services": {"image_edit": {"trellis_workers_per_gpu": 2}}}
cfg_bad = {"services": {"image_edit": {"trellis_workers_per_gpu": "x"}}}

assert trellis_workers_per_gpu(cfg_no) == 1
assert trellis_workers_per_gpu(cfg_yes) == 2
assert trellis_workers_per_gpu(cfg_bad) == 1

os.environ["TRELLIS_WORKERS_PER_GPU"] = "3"
assert trellis_workers_per_gpu(cfg_yes) == 3
os.environ["TRELLIS_WORKERS_PER_GPU"] = "0"
assert trellis_workers_per_gpu(cfg_yes) == 1
del os.environ["TRELLIS_WORKERS_PER_GPU"]
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add partcraft/pipeline_v3/services_cfg.py
git commit -m "feat(pipeline_v3): add trellis_workers_per_gpu config accessor"
```

---

## Task 2: Widen `dispatch_gpus` to spawn K processes per GPU

**Files:**
- Modify: `partcraft/pipeline_v3/run.py:429-472` (replace `dispatch_gpus` body; signature unchanged)

- [ ] **Step 1: Replace `dispatch_gpus` with K-aware version**

```python
def dispatch_gpus(
    step: str,
    cfg_path: Path,
    args: argparse.Namespace,
) -> int:
    """Spawn ``K * N`` children where ``N = #GPUs`` and ``K`` is the
    per-GPU worker count for this step.

    K is read from ``services.image_edit.trellis_workers_per_gpu`` (env
    override ``TRELLIS_WORKERS_PER_GPU``). Only ``trellis_3d`` honors
    K > 1 — the other GPU steps (``preview_flux``, ``render_3d``) keep
    K = 1 because they are compute-bound, not I/O-bound.

    Each child receives ``CUDA_VISIBLE_DEVICES=<gpu>`` and a global
    ``--gpu-shard k/(K*N)`` so ``slice_for_gpu`` partitions the edit
    list across all workers without overlap.
    """
    gpus = [g.strip() for g in (args.gpus or "").split(",") if g.strip()]
    n = len(gpus)
    if n == 0:
        return run_single_gpu(step, cfg_path, args)

    k = 1
    if step == "trellis_3d":
        cfg = load_config(cfg_path)
        k = psvc.trellis_workers_per_gpu(cfg)

    if n == 1 and k == 1:
        return run_single_gpu(step, cfg_path, args)

    total = n * k
    LOG.info(
        "[%s] dispatching: gpus=%s workers_per_gpu=%d total_workers=%d",
        step, gpus, k, total,
    )

    procs: list[tuple[str, int, subprocess.Popen]] = []
    for gpu_idx, gpu in enumerate(gpus):
        for w in range(k):
            shard_id = gpu_idx * k + w
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("ATTN_BACKEND", "flash_attn")
            cmd = [
                sys.executable, "-m", "partcraft.pipeline_v3.run",
                "--config", str(cfg_path),
                "--shard", args.shard,
                "--steps", step,
                "--single-gpu",
                "--gpu-shard", f"{shard_id}/{total}",
            ]
            if args.obj_ids:
                cmd += ["--obj-ids", *args.obj_ids]
            elif getattr(args, "obj_ids_file", None) and args.obj_ids_file:
                cmd += ["--obj-ids-file", str(args.obj_ids_file)]
            if args.all:
                cmd += ["--all"]
            if args.force:
                cmd += ["--force"]
            LOG.info(
                "  GPU %s worker %d/%d (shard %d/%d): %s",
                gpu, w + 1, k, shard_id, total, " ".join(cmd[-6:]),
            )
            procs.append((gpu, w, subprocess.Popen(cmd, env=env)))

    rc = 0
    for gpu, w, p in procs:
        r = p.wait()
        LOG.info("[%s] GPU %s worker %d exit=%d", step, gpu, w, r)
        if r != 0:
            rc = r
    return rc
```

Note: `load_config` and `psvc` (alias for `services_cfg`) are already imported at the top of `run.py` — no new imports needed.

- [ ] **Step 2: Verify imports already in scope**

```bash
grep -n "services_cfg\|load_config" partcraft/pipeline_v3/run.py | head -10
```

Expected: shows `from . import services_cfg as psvc` (or equivalent) and `load_config` import. If alias differs, adapt the call.

- [ ] **Step 3: Verify dispatch math without burning GPU time**

```bash
python - <<'PY'
from pathlib import Path
from partcraft.pipeline_v3 import run as r
from partcraft.pipeline_v3 import services_cfg as psvc
import os

os.environ["TRELLIS_WORKERS_PER_GPU"] = "2"
cfg = r.load_config(Path('configs/pipeline_v3_shard08.yaml'))
k = psvc.trellis_workers_per_gpu(cfg)
gpus = ['0', '1']
total = k * len(gpus)
print(f"K={k} N={len(gpus)} total={total}")
for gi, g in enumerate(gpus):
    for w in range(k):
        sid = gi * k + w
        print(f"  shard={sid}/{total}  GPU={g}  worker={w}")
PY
```

Expected:

```
K=2 N=2 total=4
  shard=0/4  GPU=0  worker=0
  shard=1/4  GPU=0  worker=1
  shard=2/4  GPU=1  worker=0
  shard=3/4  GPU=1  worker=1
```

- [ ] **Step 4: Commit**

```bash
git add partcraft/pipeline_v3/run.py
git commit -m "feat(pipeline_v3): K workers per GPU for trellis_3d dispatch"
```

---

## Task 3: Real GPU smoke test on a tiny subset

**Files:** none modified; verification only.

- [ ] **Step 1: Run K=2 on 4 objects with 1 GPU**

Pick a quiet GPU (check `nvidia-smi` first). Use `LIMIT` to cap the test:

```bash
LIMIT=4 TRELLIS_WORKERS_PER_GPU=2 \
  python -m partcraft.pipeline_v3.run \
    --config configs/pipeline_v3_shard08.yaml \
    --shard 08 --all --steps trellis_3d \
    --gpus 0 2>&1 | tee /tmp/trellis_workers_smoke.log
```

- [ ] **Step 2: Confirm two children on the same GPU during the run**

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
           --format=csv | sort
```

Expected: two distinct `python` PIDs sharing one `gpu_uuid`. Total used memory ≈ 2× single-worker footprint.

- [ ] **Step 3: Confirm correctness — no edit_id processed twice**

```bash
ls -1 outputs/partverse/shard08/mode_e_text_align/objects/*/*/edits_3d/*/after.npz \
  | wc -l

grep -h '"s5_trellis"' outputs/partverse/shard08/mode_e_text_align/objects/*/*/status.json \
  | wc -l
```

Expected: `after.npz` count > 0; `s5_trellis` appears at most once per object's status.json.

- [ ] **Step 4: Compare wall time vs K=1 baseline (recommended)**

Re-run the same `LIMIT=4` workload with `TRELLIS_WORKERS_PER_GPU=1` (use `--force` and a fresh output dir or restore from backup), then:

```bash
python - <<'PY'
import json, glob
from pathlib import Path
def total_wall(root):
    t = 0.0
    for p in glob.glob(f"{root}/objects/*/*/status.json"):
        d = json.loads(Path(p).read_text())
        s = (d.get("steps") or {}).get("s5_trellis") or {}
        t += float(s.get("wall_s", 0) or 0)
    return t
print("sum wall_s:", total_wall("outputs/partverse/shard08/mode_e_text_align"))
PY
```

Expected: K=2 sum(wall_s) is meaningfully lower than K=1 (target speedup 1.3-1.7×). If ≤1×, the GPU is already saturated — stop here, don't push K higher.

---

## Task 4: Wire the knob into shard configs (opt-in only)

**Files:**
- Modify: `configs/_templates/pipeline_v3_bench.template.yaml` (commented-out knob, default 1)
- Modify: `configs/pipeline_v3_shard08.yaml` (set to 2 only after Task 3 succeeds)

- [ ] **Step 1: Document knob in template**

In `configs/_templates/pipeline_v3_bench.template.yaml`, find the `services.image_edit:` block and add (preserve 2-space indent):

```yaml
  image_edit:
    # ... existing keys ...
    # Per-GPU concurrent Trellis 3D workers (1 = current behavior).
    # Recommended: 2 on 80+ GB cards, 3 only on 144 GB H800. K > 1 helps
    # because each worker overlaps its CPU/IO (DINOv2, voxel mask, NFS)
    # with another worker's GPU sampling. Only trellis_3d honors this;
    # preview_flux / render_3d ignore it.
    trellis_workers_per_gpu: 1
```

- [ ] **Step 2: Enable on shard08**

In `configs/pipeline_v3_shard08.yaml`, in the same `services.image_edit:` block:

```yaml
    trellis_workers_per_gpu: 2
```

- [ ] **Step 3: Verify YAML still parses**

```bash
python - <<'PY'
from partcraft.utils.config import load_config
from partcraft.pipeline_v3.services_cfg import trellis_workers_per_gpu
for p in ['configs/_templates/pipeline_v3_bench.template.yaml',
          'configs/pipeline_v3_shard08.yaml']:
    print(p, '->', trellis_workers_per_gpu(load_config(p)))
PY
```

Expected: template prints `1`, shard08 prints `2`. (Adapt loader import if `partcraft.utils.config.load_config` is not the canonical entrypoint — see top of `partcraft/pipeline_v3/run.py` for the actual import.)

- [ ] **Step 4: Commit**

```bash
git add configs/_templates/pipeline_v3_bench.template.yaml configs/pipeline_v3_shard08.yaml
git commit -m "config: enable trellis_workers_per_gpu=2 on shard08"
```

---

## Task 5: Document in `docs/ARCH.md`

- [ ] **Step 1: Locate the GPU dispatch section**

```bash
grep -n "dispatch_gpus\|GPU dispatch\|multi-GPU\|trellis_3d" docs/ARCH.md | head
```

If a "GPU dispatch" section exists, append below it. If not, add a short subsection at the end of the "Pipeline v3" section.

- [ ] **Step 2: Append paragraph**

```markdown
### Per-GPU worker fan-out (`trellis_workers_per_gpu`)

`trellis_3d` is mixed CPU/GPU (DINOv2 conditioning, voxel-mask
construction, NFS writes interleaved with Rectified-Flow sampling). To
keep the GPU busy while one worker preprocesses the next edit, the
dispatcher can spawn K worker subprocesses per physical GPU
(`services.image_edit.trellis_workers_per_gpu`, default 1; env
override `TRELLIS_WORKERS_PER_GPU`). Each worker pins the same
`CUDA_VISIBLE_DEVICES` and takes shard `k / (K * N_gpus)`, so
`slice_for_gpu`'s mod-N round-robin partitions edits with no overlap.
Only `trellis_3d` honors K > 1; `preview_flux` and `render_3d` are
compute-bound and stay at K=1. Memory budget per worker is roughly
20-40 GB, so K=2 is safe on 80 GB cards and K=3 only on 144 GB H800.
```

- [ ] **Step 3: Commit**

```bash
git add docs/ARCH.md
git commit -m "docs: explain trellis_workers_per_gpu dispatch fan-out"
```

---

## Self-Review

**1. Spec coverage.**
- "Multiple workers per GPU" → Task 2 widens `dispatch_gpus`.
- "Config knob" → Tasks 1 & 4 (`trellis_workers_per_gpu` + env override).
- "K=1 must reproduce current behavior" → Task 2 step 1 (early-return when `n == 1 and k == 1`); Task 3 baseline comparison.
- "Don't affect preview_flux / render_3d" → Task 2 limits K resolution to `step == "trellis_3d"`.
- "No core Trellis algorithm changes" → confirmed: `trellis_3d.py`, `refiner.py`, `interweave_Trellis.py` untouched.

**2. Placeholder scan.** No TBDs / TODOs. All steps include exact commands or full code blocks.

**3. Type / signature consistency.**
- `dispatch_gpus(step, cfg_path, args) -> int` — signature unchanged.
- `psvc.trellis_workers_per_gpu(cfg, *, default=1) -> int` — used identically in Task 1, Task 2, and Task 4.
- `--gpu-shard "i/n"` format unchanged; `slice_for_gpu(ctxs, i, n)` already does `k % n == i` round-robin (`run.py:158`), so widening `n` works without modifying the slicer.

**4. Risk callouts.**
- OOM if K too high → default 1, doc memory budget, smoke-test in Task 3 step 4 before bumping.
- NFS write contention → status updates lock-protected; K=2-3 adds bounded write rate.
- Server lifecycle: trellis_3d uses `servers: none` in the v3 yaml, so no edit-server race.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-trellis-workers-per-gpu.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task with review between tasks; fast iteration, isolated context per task.
2. **Inline Execution** — execute tasks in this session using executing-plans, with checkpoints after Tasks 1, 2, and 3.

Which approach?
