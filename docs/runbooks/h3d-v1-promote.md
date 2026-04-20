# H3D_v1 promote runbook

End-to-end procedure to promote one or more pipeline_v3 shards into
`data/H3D_v1/`. Spec: `docs/superpowers/specs/2026-04-19-h3d-v1-design.md`.
Plan: `docs/superpowers/plans/2026-04-19-h3d-v1-plan.md`.

## 0. Prerequisites

- Pipeline_v3 shard config exists and the shard has finished phases A
  (gate_a) and E (gate_e). Verify with:
  `python -m scripts.cleaning.h3d_v1.pull_deletion --pipeline-cfg <cfg> --shard <NN> --dataset-root /tmp/probe --dry-run --skip-encode`
- Conda env containing `trellis`, `open3d`, `xformers`, `spconv`, etc. is
  active. Production env: `vinedresser3d`. The `pull_deletion` CLI needs
  this env; `pull_flux` / `pull_addition` / index / pack are pure IO and
  run fine in any Python with `numpy + opencv-python + pyyaml`.
- Blender 3.5 binary path known (`$BLENDER_PATH`, default `/usr/local/bin/blender`).
- TRELLIS checkpoints present at `$PARTCRAFT_CKPT_ROOT` (default
  `<repo>/checkpoints/`). Required subtrees:
  `TRELLIS-text-xlarge/ckpts/ss_enc_conv3d_16l8_fp16.{json,safetensors}`,
  `TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.{json,safetensors}`,
  `dinov2/`.
- `data/H3D_v1/` is on the **same filesystem** as the pipeline output
  dir (otherwise hardlinks degrade to copies — search for "cross-FS
  hardlink failed" warnings in the log).

## 1. Dry-run sanity (no GPU, zero writes)

Get the projected accept/reject counts before committing GPU time.

```bash
SHARD=08
CFG=configs/pipeline_v3_shard${SHARD}.yaml
DATASET=data/H3D_v1

for c in pull_deletion pull_flux pull_addition; do
  extra=""; [ "$c" = pull_deletion ] && extra="--skip-encode"
  python -m scripts.cleaning.h3d_v1.$c \
    --pipeline-cfg $CFG --shard $SHARD --dataset-root $DATASET \
    --dry-run $extra --log-level WARNING
done
```

`pull_flux --dry-run` reports the top reject reasons; if `gate_E='fail'`
dominates, that just means a lot of edits failed gate_e — expected.

## 2. Wet promote — one shard, one machine

```bash
SHARD=08
CFG=configs/pipeline_v3_shard${SHARD}.yaml
DATASET=data/H3D_v1
# Set which physical cards to use (one process fans out to all of them):
GPUS=0,1,2,3,4,5,6,7  # e.g. 0 for a single-GPU box

# 1) Deletions (GPU). Encodes missing `after.npz` (s6b) then hardlinks
#    into H3D_v1. A **single** `pull_deletion` process fans out encode
#    (Blender + DINOv2 + SLAT) across ``--gpu-ids`` (default 0..7) via
#    `multiprocessing` ``spawn`` workers; each worker loads `ss_encoder` once
#    and round-robins its edit partition. For a **single** GPU, use
#    ``--gpu-ids 0`` (or legacy ``--device cuda:0``).
PARTCRAFT_CKPT_ROOT=$PWD/checkpoints \
python -m scripts.cleaning.h3d_v1.pull_deletion \
  --pipeline-cfg $CFG --shard $SHARD --dataset-root $DATASET \
  --gpu-ids $GPUS --blender ${BLENDER_PATH:-/usr/local/bin/blender} \
  --workers 8

# 2) Flux (pure IO; safe to run concurrently with pull_deletion).
python -m scripts.cleaning.h3d_v1.pull_flux \
  --pipeline-cfg $CFG --shard $SHARD --dataset-root $DATASET --workers 8

# 3) Additions — must run AFTER deletions for the same shard.
python -m scripts.cleaning.h3d_v1.pull_addition \
  --pipeline-cfg $CFG --shard $SHARD --dataset-root $DATASET --workers 8
```

`pull_addition` will skip any addition whose paired `del_<obj>_NNN` is
not yet in the dataset (counted under `top reasons`). Rerun once more
deletions land; the operation is idempotent.

Expected per-deletion encode time: 5-15 s on an L20X, dominated by
40 Blender renders. Promote phase is sub-second per edit.

### 2.1 How the encode step (s6b) is parallelised

`pull_deletion` inlines pipeline_v3's s6b encode whenever an accepted
edit lacks `after.npz`. The runner partitions pending edits **round-robin
across `--gpu-ids`** and forks one `multiprocessing` `spawn` worker per
GPU. Each worker runs **two phases** over its bucket:

1. **Phase 1 — Blender render + voxelize** for *every* edit in the
   bucket. No torch / CUDA init yet, so Cycles owns the GPU during this
   phase. Logs: `[gpu3 R 12/47] del_xxx_003 render ok (7.2s)`.
2. **Phase 2 — DINOv2 + SLAT + SS encode** over the staged renders.
   `ss_encoder` loads **once per worker**, then streams the bucket.
   Logs: `[gpu3 E 12/47] del_xxx_003 encode ok (0.9s)`.

Rationale: interleaving Cycles and PyTorch on the same device per edit
caused driver-context thrash in earlier runs. Splitting phases keeps
each backend's state hot for the whole bucket.

Practical knobs:

- `--gpu-ids 0,1,2,3,4,5,6,7` — default, one worker per listed GPU.
- `--gpu-ids 0` — single-GPU box (equivalent to legacy `--device cuda:0`;
  the legacy flag still works but prints a deprecation warning).
- `--encode-work-dir <path>` — staging for per-GPU renders; each worker
  creates `gpu<N>/` under it. Put this on the same FS as the pipeline
  output to keep IO local.
- `--skip-encode` — skip s6b entirely (dry-run / CPU-only accounting).

Workers write `<edit_dir>/after.npz` directly; the parent process
re-enumerates ready edits after the pool finishes, so a partial encode
failure in one worker does not block promotion of the rest of the shard.

## 3. Wet promote — multi-machine, multi-GPU

Each shard is independent; assign each shard to one machine. Within a
machine, **one `pull_deletion` process already fans out over all local
GPUs via `--gpu-ids`** (see §2.1) — do NOT launch one process per GPU.
The asset pool (`_assets/<NN>/<obj>/`) and per-shard manifests are
protected by `fcntl.flock`, so concurrent writers on the same FS are
safe when you do split across machines.

```bash
# Machine A — shard 07, all 8 local GPUs:
python -m scripts.cleaning.h3d_v1.pull_deletion \
  --pipeline-cfg configs/pipeline_v3_shard07.yaml --shard 07 \
  --dataset-root data/H3D_v1 --gpu-ids 0,1,2,3,4,5,6,7

# Machine B — shard 08, all 8 local GPUs (parallel with A):
python -m scripts.cleaning.h3d_v1.pull_deletion \
  --pipeline-cfg configs/pipeline_v3_shard08.yaml --shard 08 \
  --dataset-root data/H3D_v1 --gpu-ids 0,1,2,3,4,5,6,7
```

Only split a single shard across multiple machines when one shard is
GPU-bound for too long. In that case partition obj_ids by hash and
pass `--obj-ids-file work/obj_ids_${SHARD}_partK.txt` to each machine
(still using the full local `--gpu-ids` on each). Once all
`pull_deletion` invocations finish, run `pull_addition` once per shard
from any machine.

## 4. Aggregate index + validate

```bash
python -m scripts.cleaning.h3d_v1.build_h3d_v1_index \
  --dataset-root data/H3D_v1 --validate
```

Exit code `0` means every record's files exist and both NPZ files load
with the expected `slat_feats / slat_coords / ss` keys. Exit code `2`
means at least one record failed validation; fix the per-edit issue
and rerun the relevant `pull_*` CLI for that obj_id.

## 5. Pack for upload

```bash
for S in 00 01 02 03 04 05 06 07 08; do
  python -m scripts.cleaning.h3d_v1.pack_shard \
    --dataset-root data/H3D_v1 --shard $S \
    --out releases/H3D_v1__shard${S}.tar
done
```

The `tar` (uncompressed) preserves hardlinks, so `_assets/<NN>/<obj>/object.npz`
is stored once per obj rather than once per edit. Add `--compression gz`
only if you need on-disk compression; gzip will inflate the on-wire
size where SLAT NPZ already compresses internally.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)…` | Old `pull_deletion` without `_normalize_device_env`; SS encoder ended up on CPU | Pull latest `feature/prompt-driven-part-selection` |
| `cross-FS hardlink failed at … falling back to copy` | Dataset root and pipeline output on different mounts | Move `data/H3D_v1` onto the same FS as `outputs/.../` |
| `paired deletion del_<obj>_NNN not promoted yet` | `pull_addition` ran before `pull_deletion` for that obj | Rerun `pull_deletion` for the obj, then rerun `pull_addition` (idempotent) |
| `_assets object.npz missing — promote a deletion or flux for <obj> first` | `pull_addition` invoked on an obj with no completed flux/deletion | Same as above; ensure the source step ran first |
| `no after_new.glb; cannot encode` | Pipeline_v3 s5b never produced the GLB for this edit | Investigate the pipeline run; missing s5b output is a pipeline failure, not a promote issue |
| Encode workers stall / one `gpu<N>` worker logs zero progress | A single hung Blender subprocess or CUDA context wedged on that card | Kill the worker PID; the parent re-enumerates ready edits, so rerunning `pull_deletion` picks up the remainder |
| `--device is deprecated for pull_deletion; use --gpu-ids` | Passing legacy `--device cuda:N` | Replace with `--gpu-ids N` (single) or `--gpu-ids 0,1,…,7` (multi); the legacy flag still works but is removed in a future pass |
