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

> **Automation note (spec 2026-04-21, from shard06 onward):**  
> Shards whose YAML declares a `pull_deletion_render` hook under
> `pipeline.hooks` will have the Blender `--phase render` step kicked
> off automatically by `run_pipeline_v3_shard.sh` as soon as `del_mesh`
> finishes — it runs in parallel with `flux_2d > trellis_preview`. If
> that's already done for your shard, you can skip the `--phase render`
> command below and go straight to the `--phase encode` (GPU) step
> (see §2.1 for the two-phase split).
> To opt out, export `SKIP_HOOKS=1` before running the shard script.

```bash
SHARD=08
CFG=configs/pipeline_v3_shard${SHARD}.yaml
DATASET=data/H3D_v1
# Set which physical cards to use (one process fans out to all of them):
GPUS=0,1,2,3,4,5,6,7  # e.g. 0 for a single-GPU box

# 1) Deletions (GPU). Encodes missing `after.npz` (s6b) then hardlinks
#    into H3D_v1. Uses **fork** workers (not spawn) to avoid numpy
#    re-import race in child processes (see §2.1). A **single**
#    `pull_deletion` process fans out over ``--gpu-ids`` (default 0..7);
#    for a single GPU use ``--gpu-ids 0`` (or legacy ``--device cuda:0``).
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
edit lacks `after.npz`. The encode step is split globally into a
**render pool** and an **encode pool**, each using `multiprocessing`
**fork** workers (CUDA-safe because the parent never initialises CUDA).

**Why fork instead of spawn:** earlier `spawn` runs on shard08 hit a
`PyCapsule_Import could not import module "datetime"` crash caused by
concurrent numpy re-import in spawn children (the parent had already
loaded numpy via `promoter.py`). `fork` copies the parent address space
after numpy is stable, avoiding the race entirely.

**Two pools, two phases (always global, not per-worker):**

1. **Render pool** — Blender Cycles + voxelize for *all* pending edits.
   No torch / trellis imported.  Each render result lands in a flat
   directory `<encode-work-dir>/<shard>_<obj>_<edit>/` and a
   `render.done` marker is written.
   Logs: `[gpu3 R 12/437] del_xxx_003 render ok (7.2s)`.

2. **Encode pool** — load `ss_encoder` once per worker, then
   DINOv2 → SLAT → SS for all staged renders.  Staging is removed on
   success unless `--keep-staging` is set.
   Logs: `[gpu3 E 12/437] del_xxx_003 encode ok (0.9s)`.

**`--phase` flag** lets you run the two pools independently:

```bash
SHARD=08; CFG=configs/pipeline_v3_shard${SHARD}.yaml; DATASET=data/H3D_v1

# Step A — render only (no torch/trellis; can run while other GPU jobs occupy compute)
python -m scripts.cleaning.h3d_v1.pull_deletion \
  --pipeline-cfg $CFG --shard $SHARD --dataset-root $DATASET \
  --phase render --gpu-ids 0,1,5,6,7 \
  --encode-work-dir outputs/h3d_v1_encode \
  --blender ${BLENDER_PATH:-/usr/local/bin/blender}

# Step B — encode + promote (run after GPU compute is free; loads ss_encoder)
PARTCRAFT_CKPT_ROOT=$PWD/checkpoints \
python -m scripts.cleaning.h3d_v1.pull_deletion \
  --pipeline-cfg $CFG --shard $SHARD --dataset-root $DATASET \
  --phase encode --gpu-ids 0,1,2,3,4,5,6,7 \
  --encode-work-dir outputs/h3d_v1_encode
```

`--phase both` (default) runs render → encode → promote in a single
invocation, equivalent to the old behaviour.

Practical knobs:

- `--gpu-ids 0,1,2,3,4,5,6,7` — default, one worker per listed GPU.
- `--gpu-ids 0` — single-GPU box (equivalent to legacy `--device cuda:0`;
  the legacy flag still works but prints a deprecation warning).
- `--encode-work-dir <path>` — staging root; render artifacts land in
  `<path>/<shard>_<obj>_<edit>/`. Put this on the same FS as the pipeline
  output to keep IO local.
- `--keep-staging` — keep render artifacts after successful encode (default:
  removed to reclaim disk; ~40 MB per edit × 1108 = ~43 GiB for shard08).
- `--skip-encode` — skip s6b entirely (dry-run / CPU-only accounting).
  Mutually exclusive with `--phase`.

Workers write `<edit_dir>/after.npz` directly; the parent re-enumerates
ready edits after the encode pool finishes, so a partial encode failure
in one worker does not block promotion of the rest of the shard.

## 2b. Backfill `preview_{k}.png` on shards that skipped `preview_del`

> Applies to shards whose pipeline run never executed the `preview_del`
> stage (discovered on shard05: every `edits_3d/del_*/` had `after.npz`
> and `after_new.glb` but **no** `preview_*.png`).  `pull_deletion` will
> still promote the SLAT NPZ, but H3D_v1 `deletion/<shard>/<obj>/<eid>/
> after.png` stays missing because the promoter hardlinks from the
> pipeline's `preview_{k}.png`.  That in turn makes every paired
> addition `skip=pair_after_image_missing`, so `pull_addition` finishes
> with `promoted=0`.

**Fast fix — render only the single best_view per edit.**  H3D_v1 only
ever reads one preview (`preview_{best_view_index}.png`), so rendering
all five canonical slots is wasted work.  The pipeline_v3 `preview_del`
stage accepts `--best-view-only`, which picks the slot per edit from
`edit_status.json → gates.A.vlm.best_view` (fallback slot 4,
``DEFAULT_FRONT_VIEW_INDEX``) and produces exactly one file per edit.

```bash
SHARD=05
CFG=configs/pipeline_v3_shard${SHARD}.yaml

# vinedresser3d env — the Blender path is resolved from the shard yaml.
python -m partcraft.pipeline_v3.run   --config $CFG --shard $SHARD --all   --steps preview_del --best-view-only   --gpus 0,1,2,3,4,5,6,7 --force
```

Throughput on an 8×L20X box: ≈ 2.5–5 s/edit per GPU →  ~17 min for a
3000-edit shard.  Addition entries are handled in the same run —
`preview_del` copies the paired deletion's single `preview_{k}.png`
into each `add_*/` dir (both sides of an add/del pair always share
best_view via `_views_block`).

After this completes, rerun the normal §2 sequence:

```bash
PARTCRAFT_CKPT_ROOT=$PWD/checkpoints python -m scripts.cleaning.h3d_v1.pull_deletion   --pipeline-cfg $CFG --shard $SHARD --dataset-root data/H3D_v1   --gpu-ids 0,1,2,3,4,5,6,7 --workers 8

python -m scripts.cleaning.h3d_v1.pull_flux   --pipeline-cfg $CFG --shard $SHARD --dataset-root data/H3D_v1 --workers 8

python -m scripts.cleaning.h3d_v1.pull_addition   --pipeline-cfg $CFG --shard $SHARD --dataset-root data/H3D_v1 --workers 8
```

`pull_deletion` is a no-op for NPZ re-encode if `after.npz` already
exists (idempotent); it will simply hardlink the freshly rendered
`preview_{k}.png` as `after.png` and update the promote log.

Notes:

- `--best-view-only` is **additive / non-destructive**: already-rendered
  5-view previews are left alone on other shards.  The skip check uses
  per-slot existence, so a future full `preview_del` run on a
  best-view-only shard will top up the missing 4 slots without
  re-rendering the best slot.
- `preview_del` step status in `edit_status.json` is marked `done` on
  success; drop `--force` if you later only want to pick up new edits.
- Implementation: `partcraft/pipeline_v3/preview_render.py`
  (`render_del_previews_batch(..., best_view_only=True)`).

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

`pack_shard` deliberately does **not** include `manifests/_internal/`
(local promoter audit log — see spec §8); only `_assets/<NN>/`,
`<edit_type>/<NN>/`, `manifests/<edit_type>/<NN>.jsonl` and
`manifests/all.jsonl` are packed.

## 5b. Backfill existing `meta.json` after schema changes

If you have `meta.json` files from a previous promote that pre-date
schema v3 final (no `views` block, old `gate_*_score` keys, `stats`
block, verbose `lineage`), run the idempotent rewriter:

```bash
# Recommended: pass pipeline cfg so views.best_view_index is pulled
# from the pipeline's pixel-mask argmax (gates.A.vlm.best_view).
# Add --rebuild-manifests so manifests/<type>/<NN>.jsonl and
# manifests/all.jsonl are regenerated from the rewritten meta.json files.
python -m scripts.tools.h3d_v1_backfill_meta \
  --dataset-root data/H3D_v1 --shard 08 \
  --pipeline-cfg configs/pipeline_v3_shard08.yaml \
  --rebuild-manifests

# Without --pipeline-cfg, views.best_view_index falls back to
# DEFAULT_FRONT_VIEW_INDEX (4, "front upward") with a warning count
# in the final summary.
```

Safe to run multiple times.  On a clean v3-final dataset the tool
reports `rewrite=0 unchanged=N` and exits 0.  ``--rebuild-manifests``
may be used on its own (without meta rewrites) to just bring the
manifest JSONLs back in sync with the per-edit ``meta.json`` files.

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
| `--skip-encode and --phase are mutually exclusive` | Both flags passed together | Use `--phase both --skip-encode` (equivalent to old `--skip-encode` alone) or drop one |
| `missing_staged_render` skip reason in summary | `--phase encode` ran but render artifacts absent | Run `--phase render` first with the same `--encode-work-dir`, then re-run `--phase encode` |
| `render.done missing` in encode worker log | Individual edit's render failed or staging deleted | Rerun `--phase render` (idempotent; already-staged edits are skipped) |
| `pull_addition --dry-run` reports `skip=pair_after_image_missing` for most adds | Shard's pipeline run skipped the `preview_del` stage, so `deletion/<shard>/<obj>/<eid>/after.png` never populated | Run `preview_del --best-view-only` for that shard (see §2b), then rerun `pull_deletion` + `pull_addition` |
| `gate_a.vlm.best_view` absent and `--best-view-only` falls back everywhere | VLM gate_a never wrote `best_view` (rare) | Single-view render uses :data:`partcraft.pipeline_v3.preview_render.DEFAULT_FRONT_VIEW_INDEX` (slot 4); check pipeline_v3 VLM log if this is widespread |
