# Edit-Data v1 — End-to-End Smoke Runbook

Runbook for the `partcraft.cleaning.v1` pipeline (spec:
`docs/superpowers/specs/2026-04-19-edit-data-v1-design.md`, plan:
`docs/superpowers/plans/2026-04-19-edit-data-v1-plan.md`).

Goal: take cleaned edits from one or more pipeline runs (v2 + v3) and
materialize them into the unified `data/partverse_edit_v1/` dataset, with
indexes ready for downstream loaders.

---

## Prereqs

| Resource | Required for | Where it lives in this repo |
|---|---|---|
| `data/partverse/img_Enc/<obj_id>/{089,090,091,100,008}.png` | `before/views/` materialization | `data/partverse/img_Enc/` |
| `data/partverse/inputs/slat/<NN>/<obj_id>_{coords,feats}.pt` | `before/slat.npz` packing | `data/partverse/inputs/slat/` |
| `data/partverse/inputs/slat/<NN>/<obj_id>_ss.pt` *(optional)* | `before/ss.npz` packing — falls back to `ss.missing.json` placeholder | `data/partverse/inputs/slat/` |
| `data/partverse/images/<NN>/<obj_id>.npz` | `run_gate_quality_on_v2.py` — loads 5 before-view images | `data/partverse/images/` |
| Trellis SS-encoder checkpoint dir | `encode_del_latent.py` — Phase-5 deletion latent encode | configured via `--ckpt-root` or `--config` |
| Blender executable | `encode_del_latent.py` — render 40 views via Cycles | `--blender-path` or `BLENDER_PATH` env |
| Live VLM endpoint (one per server) | `run_gate_quality_on_v2.py` — visual quality judge | `services.vlm.urls` in v3 config YAML |

> **Default config:** `configs/cleaning/promote_v1.yaml`. The default
> `before_assets.slat_root` is `data/partverse/inputs/slat` (matches this
> machine). Override via `--rules` for other machines.

---

## Step 1 — Pick a small target subset *(no live deps)*

```bash
mkdir -p /tmp/v1_smoke
ls outputs/partverse/pipeline_v2_shard05/objects/05 | head -2 \
    > /tmp/v1_smoke/v2_ids.txt
ls outputs/partverse/shard08/mode_e_text_align/objects/08 | head -2 \
    > /tmp/v1_smoke/v3_ids.txt
cat /tmp/v1_smoke/v2_ids.txt /tmp/v1_smoke/v3_ids.txt
```

**Verified 2026-04-18** on `pipeline_v2_shard05` (1203 objects) and
`shard08/mode_e_text_align` (1203 objects). Sample obj_ids:
`5dc4ca7d607c495bb82eca3d0153cc2c`, `5dc5ffbdedbd4ce8ae8bb34fe4f2dd19`,
`bdd36c94f3f74f22b02b8a069c8d97b7`, `bdde2b5a90c542faa33561b86f3f990c`.

---

## Step 2 — Backfill Gate E on the v2 subset *(live env: VLM)*

v2 runs never executed Gate E, so `gates.E` is `null` everywhere. The
adapter constructs `partcraft.pipeline_v3.paths.PipelineRoot(root=v2_run)`
(v2 and v3 share the on-disk object layout) and invokes
`run_gate_quality(...)` directly. v2 module code is **not** modified.

```bash
python -m scripts.cleaning.run_gate_quality_on_v2 \
    --v2-run outputs/partverse/pipeline_v2_shard05 \
    --v3-config configs/pipeline_v3_shard08_test20_gateQ.yaml \
    --shards 05 \
    --obj-ids /tmp/v1_smoke/v2_ids.txt \
    --concurrency 4
```

**Expected:** for each obj_id in `v2_ids.txt`, the corresponding
`outputs/partverse/pipeline_v2_shard05/objects/05/<obj_id>/edit_status.json`
gets `gates.E` populated for every non-`{deletion,addition}` edit (deletion
and addition inherit Gate A's verdict). Spot check:

```bash
python3 -c "
import json, sys
p = 'outputs/partverse/pipeline_v2_shard05/objects/05/5dc4ca7d607c495bb82eca3d0153cc2c/edit_status.json'
d = json.load(open(p))
for eid, e in d['edits'].items():
    print(eid, e['edit_type'], (e.get('gates') or {}).get('E'))
"
```

> **Tip:** run this against any v2 run dir before promoting, otherwise the
> `gate_quality` required-pass rule will defer those edits in step 3.

---

## Step 3 — Promote both runs into v1 *(live data: img_Enc + inputs/slat)*

```bash
python -m scripts.cleaning.promote_to_v1 \
    --source-runs outputs/partverse/pipeline_v2_shard05 \
                   outputs/partverse/shard08/mode_e_text_align \
    --link-mode hardlink
```

The CLI auto-detects v2 vs v3 layout (`<run>/objects/...` → v2,
`<run>/mode_*/objects/...` → v3). Each promoted edit obeys the rule in
`configs/cleaning/promote_v1.yaml`:

```yaml
required_passes: [gate_text_align, gate_quality]
```

Edits missing a required pass go to `deferred` (re-runnable later);
edits with a failing pass go to `failed` (won't materialize unless re-judged
and re-promoted with `--force`).

**Expected:**
- log line `TOTAL promoted=N skipped=M deferred=K failed=F fallback=...`
- `data/partverse_edit_v1/objects/{05,08}/<obj_id>/{meta.json,before/,edits/}`
  populated
- For each promoted **deletion** edit, an entry appended to
  `data/partverse_edit_v1/_pending/del_latent.txt` and a `_after_pending.json`
  marker dropped in the edit dir (its `after.npz` is produced by step 4).

> **Idempotency:** re-running with the same `--source-runs` is a no-op
> (skipped). Re-running with a *different* `source_run_tag` for the same
> edit_id creates a parallel `<edit_id>__r2/` dir (per spec §10.3 — diversity,
> not dedup).

---

## Step 4 — Encode pending deletion latents *(live env: GPU + Blender + Trellis ckpt)*

Drives `scripts.tools.migrate_slat_to_npz._render_and_full_encode` (renders
40 views via Blender Cycles → DINOv2 → SLAT + SS encode → returns numpy dict
which we `np.savez` into `after.npz`).

Single GPU:

```bash
python -m scripts.cleaning.encode_del_latent \
    --ckpt-root /path/to/trellis/checkpoints \
    --blender-path "$BLENDER_PATH" \
    --num-gpus 1
```

Multi-GPU fan-out (round-robin slice across N children, each pinned via
`CUDA_VISIBLE_DEVICES`):

```bash
python -m scripts.cleaning.encode_del_latent \
    --ckpt-root /path/to/trellis/checkpoints \
    --blender-path "$BLENDER_PATH" \
    --num-gpus 4
```

**Expected:** the pending list shrinks to 0; each `del_*` dir gains an
`after.npz`; `qc.json` gets a `passes.del_latent_encode = pass`; the
`_after_pending.json` marker is removed.

---

## Step 5 — Rebuild index *(no live deps)*

```bash
python -m scripts.cleaning.rebuild_v1_index
cat data/partverse_edit_v1/index/_last_rebuild.json
head -2 data/partverse_edit_v1/index/edits.jsonl
```

**Expected:** `n_objects > 0`, `n_edits > 0`. Each line of `edits.jsonl`
includes `obj_id, shard, edit_id, edit_dir_suffix, edit_type, before_*,
after_*, source_pipeline, source_run_tag` (all paths relative to
`data/partverse_edit_v1/`).

**Verified 2026-04-18** on an empty v1 root:

```text
{"n_objects": 0, "n_edits": 0, "ts": "2026-04-18T19:04:36+00:00"}
```

(empty `objects.jsonl` + `edits.jsonl` written, exit 0).

---

## Verification matrix

| Step | Verified in this commit | Why not |
|---|---|---|
| 1 | yes (real `outputs/partverse/...` listings) | — |
| 2 | no | requires live VLM endpoint |
| 3 | no | requires real `data/partverse/img_Enc/` + `data/partverse/inputs/slat/` writeback (~1.2k objects per run) |
| 4 | no | requires GPU + Blender + Trellis SS-encoder ckpt |
| 5 | yes (empty-root smoke) | — |

CLI `--help` smokes for all four scripts have been verified during their
respective task commits (`promote_to_v1`, `rebuild_v1_index`,
`encode_del_latent`, `run_gate_quality_on_v2`).

---

## Known gotchas

- **`slat_root` default** points to `data/partverse/inputs/slat`. On
  machines that store SLAT pt files elsewhere, override via a copy of
  `configs/cleaning/promote_v1.yaml` and pass `--rules <yourcopy>.yaml`.
- **`promote_to_v1.py` has no `--obj-ids` filter today.** It walks the
  entire `<run>/objects/<NN>/...` tree. For a per-object rerun, the cheapest
  workaround is to delete the target object's v1 dir and re-run; promotion
  is idempotent for matching `source_run_tag`s.
- **Deletion edits before step 4** show `after.npz` missing in the index
  rebuilt at step 5. Re-run step 5 after step 4 for a complete index.
- **`ss.pt` may not exist** under `data/partverse/inputs/slat/<NN>/`. The
  promoter writes a `before/ss.missing.json` placeholder in that case
  rather than aborting; backfill `ss.npz` separately if/when needed.

---

## Reset / cleanup (manual)

```bash
rm -rf data/partverse_edit_v1
```

(All v1 content is rebuildable from the source pipeline runs + the configs in
`configs/cleaning/`.)
