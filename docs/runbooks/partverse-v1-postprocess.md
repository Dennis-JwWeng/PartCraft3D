# Runbook: PartVerse v1 dataset post-processing

**Purpose**: turn a finished `pipeline_v3` shard (or a v2 mode-E run) into a
fully-linked `data/partverse_edit_v1/` slice that downstream training can
consume — including the special handling that **deletion** and **addition**
edits require.

**Use this when**:
- A pipeline shard has finished `gate_text_align` and (for non-add/del edit
  types) `gate_quality`, and you want to materialise it in v1 layout.
- You are unsure which post-processing step to run next on a partially
  populated v1 slice.

**Telling Cursor to do it for you**:

> "In this repo, follow `docs/runbooks/partverse-v1-postprocess.md` for shard
> `<NN>` against pipeline run `<outputs/partverse/.../shard<NN>>`. Run
> `v1_status.py` first; do exactly what it tells you next; re-run
> `v1_status.py` between steps until it says nothing actionable remains."

Cursor reads this as a checklist. The reference command for "where am I and
what's next" is **always**:

```bash
PYTHONPATH=. python -m scripts.cleaning.v1_status \
    --v1-root data/partverse_edit_v1 --shard <NN>
```

Exit code `0` means **done**; exit code `1` means **work left**, and the
script prints the next concrete command to run.

---

## 0. Replace these placeholders

| Placeholder | Meaning | Example |
|---|---|---|
| `<NN>` | Shard id (zero-padded) | `08` |
| `<RUN_DIR>` | The pipeline run dir to promote from | `outputs/partverse/shard08/mode_e_text_align` |
| `<CKPT_ROOT>` | Trellis SS encoder checkpoint root | (same as the one used by trellis_3d step) |
| `<SLAT_ROOT>` | Original-object SLAT/SS .pt directory (`{shard}/{obj}_{feats,coords,ss}.pt`) | `data/partverse/inputs/slat` |

---

## 1. Data contract (what v1 looks like when complete)

```
data/partverse_edit_v1/
├── _pending/
│   └── del_latent.txt              # one line per deletion awaiting encode
├── index/
│   ├── objects.jsonl
│   └── edits.jsonl
└── objects/<shard>/<obj_id>/
    ├── meta.json                   # caption, parts, source pipeline tag
    ├── before/
    │   ├── slat.npz                # keys: feats, coords            (object-shared, legacy schema)
    │   ├── ss.npz                  # SS                              (object-shared)
    │   ├── _combined.npz           # keys: slat_feats, slat_coords, ss  (object-shared, NEW combined schema; written on demand)
    │   └── views/view_0..4.png
    └── edits/<edit_id>[__r<n>]/
        ├── spec.json
        ├── qc.json                 # passes.{gate_text_align, gate_quality, add_npz_link, del_latent_encode}
        ├── after.npz               # keys: slat_feats, slat_coords, ss
        ├── before.npz              # keys: slat_feats, slat_coords, ss   (per-edit, hardlinked)
        ├── views/view_0..4.png
        ├── _after_pending.json     # marker: deletion edit awaiting encode_del_latent
        └── _before_pending.json    # marker: addition edit awaiting source-deletion encode
```

### Per-edit `after.npz` / `before.npz` provenance

| edit type | `before.npz` source | `after.npz` source |
|---|---|---|
| modification, scale, material, color, glb | hardlink from pipeline `<RUN_DIR>/.../edits_3d/<eid>/before.npz` | hardlink from `<RUN_DIR>/.../edits_3d/<eid>/after.npz` |
| deletion | hardlink from object's `before/_combined.npz` (= original object) | written by `encode_del_latent.py` from `after_new.glb` (render → DINOv2 → SLAT → SS) |
| addition | hardlink from source-deletion's `after.npz` (the inverse-deletion state) | hardlink from object's `before/_combined.npz` (= original object) |

**Why deletion + addition are special**: their `after.npz` is never produced
by the in-pipeline trellis_3d step because the s6b stage was not run. They
must be backfilled in v1 layout, in order: encode the deletion first,
then link the addition to it.

### Why deletion + addition skip `gate_quality`

`partcraft/pipeline_v3/vlm_core.py::_GATE_A_ONLY = {"deletion", "addition"}`.
For these two edit types, Gate E (visual quality) is meaningless — the gate
just inherits Gate A. So they can be promoted as soon as `gate_text_align`
finishes; no need to wait for `gate_quality`.

---

## 2. Standard execution flow

There are two parallel tracks. Run **Track A first** (it has no upstream
dependency on `gate_quality`) and **Track B** when `gate_quality` is
complete. Inside Track A, the steps must run in order A1 → A2 → A3.

```
                     gate_text_align            gate_quality
                            │                        │
                            ▼                        ▼
                  ┌─── Track A (del+add) ───┐  ┌── Track B (mod/scl/mat/clr/glb) ──┐
                  │  A1  promote_to_v1      │  │  B1  promote_to_v1                │
                  │      (addel-textalign)  │  │      (default rule)               │
                  │  A2  encode_del_latent  │  │       (also materialises          │
                  │  A3  link_add_npz_…     │  │        per-edit before.npz from   │
                  └─────────────────────────┘  │        pipeline source) [TODO]    │
                                               └───────────────────────────────────┘
```

### Step 0 — Inspect

```bash
PYTHONPATH=. python -m scripts.cleaning.v1_status \
    --v1-root data/partverse_edit_v1 --shard <NN>
```

Read the VERDICT block. It tells you which steps below to run.

### A1 — Promote del+add (CPU, no GPU)

```bash
PYTHONPATH=. python -m scripts.cleaning.promote_to_v1 \
    --rules configs/cleaning/promote_v1_addel_textalign.yaml \
    --source-runs <RUN_DIR>
```

This writes `qc.json` for every passing del/add edit and (for deletions)
appends to `_pending/del_latent.txt`. Re-running is safe and idempotent.

**Verify** the step succeeded: `v1_status.py --shard <NN>` should now show
non-zero promoted counts for `deletion` and `addition`, plus a populated
pending queue.

### A2 — Encode deletion `after.npz` (GPU-heavy)

```bash
PYTHONPATH=. python -m scripts.cleaning.encode_del_latent \
    --v1-root data/partverse_edit_v1 \
    --rules configs/cleaning/promote_v1.yaml \
    --ckpt-root <CKPT_ROOT> \
    --num-gpus 8
```

For each entry in `_pending/del_latent.txt`, this renders the deletion's
`after_new.glb` at 40 views, runs DINOv2 → Trellis SLAT encoder → SS
encoder, and writes `edits/<del_id>/after.npz` with keys
`slat_feats / slat_coords / ss` (matching the flux schema; no `dino_voxel_mean`).
Each successful entry is removed from the pending file.

**Verify**: `_pending/del_latent.txt` should be empty (or nearly so) and
deletion's `after.npz=N/N` in the status report.

### A3 — Link addition `before.npz` / `after.npz` to deletion

```bash
PYTHONPATH=. python -m scripts.cleaning.link_add_npz_from_del \
    --v1-root data/partverse_edit_v1 \
    --slat-root <SLAT_ROOT> \
    --shard <NN>
```

For each addition edit, this:

1. Reads `qc.json.passes.gate_text_align.extra.inherited_from` to find the
   source deletion id (set by `partcraft/cleaning/v1/source_v2.py`).
2. Hardlinks `del.after.npz` → `add.before.npz`.
3. Builds (once per object) `before/_combined.npz` from the SLAT/SS .pt
   files in `<SLAT_ROOT>/<NN>/<obj_id>_{feats,coords,ss}.pt`, then
   hardlinks it → `add.after.npz`.
4. Writes `qc.json.passes.add_npz_link = pass`.

If the source deletion has not yet been encoded, the script writes a
`_before_pending.json` marker and counts the edit as `deferred` (no error).
Re-run after A2 finishes; the marker is removed automatically when the link
succeeds.

**Verify**: `v1_status.py` should show addition's `before.npz=N/N`,
`after.npz=N/N`, `add_npz_link=N/N`.

### B1 — Promote remaining edit types (CPU, requires `gate_quality` upstream)

```bash
PYTHONPATH=. python -m scripts.cleaning.promote_to_v1 \
    --rules configs/cleaning/promote_v1.yaml \
    --source-runs <RUN_DIR>
```

This requires `gate_quality` to have a verdict for each edit. If
`gate_quality` is not done yet, this step will simply not promote those
records (it will not error).

> **Open follow-up (tracked here)**: the promoter currently materialises only
> per-edit `after.npz` for non-del/non-add edits. Per-edit `before.npz` for
> these types is not yet written by the promoter — they should be hardlinked
> from `<RUN_DIR>/.../edits_3d/<eid>/before.npz` (which exists for every
> trellis_3d-produced edit). `v1_status.py` reports this gap as
> "missing before.npz" so the agent can spot it. Tracked as the
> "promoter extension" item in this runbook; consumers that read v1 today
> should fall back to object-level `before/slat.npz` + `before/ss.npz` for
> these edit types.

### Final inspection

```bash
PYTHONPATH=. python -m scripts.cleaning.v1_status \
    --v1-root data/partverse_edit_v1 --shard <NN>
```

Exit code `0` ⇒ done.

---

## 3. Verification table (what "correct" means)

| Check | Tool | Expected |
|---|---|---|
| Every promoted edit has `qc.json` with `passes.gate_text_align.pass = true` | `v1_status.py` | promoted = sum of types |
| Every deletion has `after.npz` with keys `slat_feats / slat_coords / ss` | `v1_status.py` | `after.npz=N/N`; `_pending/del_latent.txt` drained |
| Every addition has `before.npz` (hardlinked from source-del `after.npz`) | `v1_status.py` | `before.npz=N/N`; no `_before_pending.json` markers |
| Every addition has `after.npz` (hardlinked from `before/_combined.npz`) | `v1_status.py` | `after.npz=N/N` |
| Every addition's link is recorded in qc | `v1_status.py` | `add_npz_link=N/N` |
| All hardlinks point at real files (no broken links) | `find data/partverse_edit_v1/objects -xtype l` | empty output |
| Schema sanity on a sample npz | `python -c "import numpy as np; d=np.load('<path>'); print(sorted(d.files))"` | `['slat_coords', 'slat_feats', 'ss']` |

---

## 4. Recovery / re-runs

| Symptom | Action |
|---|---|
| A2 crashed mid-shard | Re-run A2; `_pending/del_latent.txt` only has remaining entries |
| Want to re-promote an edit (qc.json was wrong upstream) | Re-run A1/B1 with `--force`; promoter writes `<eid>__r2/` suffix dirs to avoid clobbering |
| Want to re-link addition npz (e.g. del re-encoded) | `link_add_npz_from_del.py --force` |
| Suspect a hardlink is stale | Delete the per-edit `after.npz` / `before.npz`, re-run A2 / A3 |

---

## 5. Why the design splits Track A from Track B

- **Track A unblocks downstream training fastest**: del + add together
  account for ~40-60% of records in a typical shard, and they only need
  Gate A (cheap, CPU). They can be linked into v1 as soon as A1+A2+A3 run,
  without waiting on the slow VLM-driven `gate_quality` stage.
- **Track B is gated on `gate_quality`** because the remaining edit types
  (modification / scale / material / color / glb) genuinely need a VLM
  visual judge — Gate A only checks text alignment, which is necessary but
  not sufficient.
- The two tracks share no on-disk state, so they are safe to run in
  parallel from different shells / hosts.
