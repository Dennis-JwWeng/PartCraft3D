# Edit Dataset v1 — Cleaned, Pipeline-Version-Agnostic Layout

> **SUPERSEDED 2026-04-19** by [`2026-04-19-h3d-v1-design.md`](./2026-04-19-h3d-v1-design.md).
> The `partverse_edit_v1` layout, the multi-step `promote_to_v1` / `encode_del_latent`
> CLIs, and the `del_latent.txt` pending list are deprecated. New work targets the
> `H3D_v1` dataset with three end-to-end CLIs (`pull_deletion`, `pull_flux`,
> `pull_addition`) sharing a `_assets/` pool and a per-edit-type top-level layout.
> This document is kept for historical context only.

**Status**: Superseded
**Date**: 2026-04-19
**Owners**: PartCraft3D pipeline team
**Related**:
- `docs/ARCH.md` — current pipeline_v2 / pipeline_v3 layouts
- `docs/superpowers/specs/2026-04-15-text-align-gate-design.md` — Gate A
- `docs/superpowers/plans/2026-04-17-trellis-workers-per-gpu.md` — recent v3 work

---

## 1. Problem

Two pipeline versions have produced edit data on disk:

| | `outputs/partverse/pipeline_v2_shard*/` | `outputs/partverse/shard08/mode_e_text_align/` |
|---|---|---|
| Pipeline | v2 (object-centric, all edit types) | v3 Mode E (text-driven, del/add focused) |
| Gate A (text-align) | done (`sq1_qc_A`) | done (`sq1_qc_A` / `gate_text_align`) |
| Gate E (visual quality) | **never run** (`gates.E = null`) | done (`sq3_qc_E`) |
| Quality filter (8-dim cleaning) | **never run** | **never run** |
| Object count | ~1203 / shard | ~1203 / shard |

Goals:

1. The v3 cleaning logic (Gate E + future quality filter) should be re-applied to v2's existing artifacts.
2. Anything that passes both filters — regardless of the pipeline version that produced it — must be promotable into a single, version-agnostic dataset (call it **edit dataset v1**) suitable for downstream training and external upload.
3. Existing pipeline runs must keep working. v1 must not block ongoing v2/v3 runs and must not require modifying past run outputs.
4. The format must be extensible: new cleaning passes, new annotations, new pipeline versions, and new edit types should slot in without redesign.
5. For deletion edits (`del_*`), v2/v3 only produce `after_new.glb` + Blender previews. v1 must additionally hold an SS/SLAT latent for the post-edit object so deletion samples are training-equivalent to flux-branch samples.

Non-goals:

- Migrating or renaming any v2/v3 physical files.
- Inventing a new orchestration entrypoint. v1 is a **post-pipeline cleaning + repackaging** layer; it has its own scripts but does not replace `partcraft.pipeline_v2.run` / `partcraft.pipeline_v3.run`.
- Changing the schema of `edit_status.json` produced by either pipeline.

---

## 2. Key Observations That Drive Simplification

These observations let v1 be smaller and simpler than the source layouts:

1. **`before` assets are object-scoped**, not edit-scoped. The source object's SS, SLAT, and rendered views are identical for every edit on that object → store once per `obj_id`, not per `edit_id`.
2. **The 5 preview views are a fixed canonical view set.** Both v2 and v3 render the same 5 camera angles for `phase1/overview.png` (a 2576×1030 collage of 5 sub-images) and `edits_3d/<edit_id>/preview_{0..4}.png` (5 separate ~518×518 images). There is no per-edit "best view" to track in v1 — view index k ∈ {0..4} is implicit and stable.
3. **`(obj_id, edit_id)` is a natural global primary key.** Edit IDs already encode the type prefix (`del_`, `add_`, `mod_`, `scl_`, `mat_`, `glb_`, `clr_`) and a per-object index, and they collide neither within an object nor in practice across runs (they're produced from the same VLM prompt template).
4. **`edit_id` is unique per object across versions in practice.** If the same object is processed by both v2 and v3 in the future, the resulting edit_id sets may overlap in name but differ in content; v1 records the source `run_tag` per edit and treats the first-promoted edit as canonical (subsequent attempts to promote a colliding id are skipped unless `--upgrade-source` is passed).

---

## 3. v1 Dataset Layout

```
data/partverse_edit_v1/
├── objects/
│   └── <shard>/<obj_id>/
│       ├── meta.json                       # caption, part list, source obj id, dataset origin
│       ├── before/
│       │   ├── ss.npz                      # sparse-structure latent  (link or copy)
│       │   ├── slat.npz                    # sparse latent             (link or copy)
│       │   └── views/
│       │       ├── view_0.png              # 5 fixed canonical views
│       │       ├── view_1.png              #   (sourced from data/partverse/img_Enc/<obj>/)
│       │       ├── view_2.png
│       │       ├── view_3.png
│       │       └── view_4.png
│       └── edits/
│           └── <edit_id>/                  # del_xxx_000 | add_xxx_001 | mod_xxx_002 | ...
│               ├── spec.json               # frozen EditSpec subset (see §4)
│               ├── after.npz               # SS + SLAT post-edit
│               ├── views/
│               │   ├── view_0.png          # = source preview_0.png
│               │   ├── view_1.png
│               │   ├── view_2.png
│               │   ├── view_3.png
│               │   └── view_4.png
│               └── qc.json                 # gate scores + source provenance (see §5)
└── index/
    ├── objects.jsonl                       # one row per object
    ├── edits.jsonl                         # one row per edit (denormalized, ready for dataloader)
    └── _last_rebuild.json                  # timestamp + counts for the last rebuild
```

**Why a separate root, not under `outputs/partverse/`:** v1 is a curated, uploadable dataset — siblings of `data/partverse/img_Enc/`, `data/partverse/slat/`, etc. — not pipeline scratch space. Living under `data/` makes it clear that `outputs/partverse/*` can be deleted or rotated without touching v1.

---

## 4. Per-File Schemas

### 4.1 `objects/<shard>/<obj_id>/meta.json`

```json
{
  "obj_id": "5dc4ca7d607c495bb82eca3d0153cc2c",
  "shard": "05",
  "source_dataset": "partverse",
  "caption": "<phase1 raw caption>",
  "part_list": [
    {"part_id": 0, "name": "handle"},
    {"part_id": 1, "name": "body"}
  ],
  "promoted_at": "2026-04-19T12:00:00Z",
  "promoter_version": "1.0.0"
}
```

### 4.2 `objects/<shard>/<obj_id>/edits/<edit_id>/spec.json`

Frozen subset of `partcraft.pipeline_v2.specs.EditSpec`. Only fields needed for downstream training are kept; pipeline-internal bookkeeping is dropped.

```json
{
  "edit_id": "del_5dc4ca7d607c495bb82eca3d0153cc2c_000",
  "edit_type": "deletion",
  "prompt": "Remove the handle.",
  "selected_part_ids": [0],
  "part_labels": ["handle"],
  "target_part_desc": "the cylindrical handle on the right side",
  "new_parts_desc": null
}
```

### 4.3 `objects/<shard>/<obj_id>/edits/<edit_id>/qc.json`

Records every cleaning pass that touched this edit. `passes` is an open-ended dict so new pass types slot in without schema changes.

```json
{
  "edit_id": "del_5dc4ca7d607c495bb82eca3d0153cc2c_000",
  "source": {
    "pipeline_version": "v2",
    "run_tag": "pipeline_v2_shard05",
    "run_dir": "outputs/partverse/pipeline_v2_shard05/objects/05/5dc4...c2c",
    "promoted_at": "2026-04-19T12:00:00Z"
  },
  "passes": {
    "gate_text_align": {
      "pass": true, "score": 1.0,
      "producer": "v3.gate_text_align@2026-04-18",
      "reason": "...", "ts": "2026-04-18T15:20:05Z"
    },
    "gate_quality": {
      "pass": true, "score": 0.87,
      "producer": "v3.gate_quality@2026-04-18",
      "metrics": {"visual_quality": 0.9, "correct_region": 0.85, "preserve_other": 0.86},
      "ts": "2026-04-18T21:53:38Z"
    },
    "del_latent_encode": {
      "ok": true, "ts": "2026-04-19T13:00:00Z",
      "producer": "encode_del_latent.py@1.0.0"
    }
  }
}
```

A pass entry is treated as "missing" if absent. Promotion requires `gate_text_align.pass` and `gate_quality.pass` by default; the rule is configurable.

### 4.4 `index/edits.jsonl`

One JSON object per line. Denormalized for dataloader convenience.

```json
{
  "obj_id": "5dc4...c2c", "shard": "05", "edit_id": "del_5dc4...c2c_000", "edit_type": "deletion",
  "before_ss":    "objects/05/5dc4...c2c/before/ss.npz",
  "before_slat":  "objects/05/5dc4...c2c/before/slat.npz",
  "before_views": ["objects/05/.../before/views/view_0.png", "...", "...", "...", "..."],
  "after_npz":    "objects/05/.../edits/del_5dc4...c2c_000/after.npz",
  "after_views":  ["objects/05/.../edits/del_5dc4...c2c_000/views/view_0.png", "...", "...", "...", "..."],
  "spec":         "objects/05/.../edits/del_5dc4...c2c_000/spec.json",
  "qc":           "objects/05/.../edits/del_5dc4...c2c_000/qc.json",
  "source_pipeline": "v2",
  "source_run_tag":  "pipeline_v2_shard05"
}
```

All paths are relative to `data/partverse_edit_v1/` so the dataset is self-locating. JSONL (not Parquet) is intentional: 12k lines × ~700 bytes ≈ 8 MB; no extra dependency, easy to grep.

### 4.5 `index/objects.jsonl`

```json
{"obj_id": "5dc4...c2c", "shard": "05", "n_edits": 7,
 "edit_types": {"deletion": 1, "scale": 1, "material": 1, "modification": 1, "glb": 3, "addition": 1}}
```

---

## 5. Promotion Filter Rules

Default rule (configurable via YAML):

```yaml
promote_rules:
  required_passes:
    - gate_text_align
    - gate_quality
  optional_passes:
    - del_latent_encode    # auto-injected for del edits, not a gate
  edit_types_allowed: [deletion, addition, modification, scale, material, color, glb]
```

An edit is promotable iff every entry in `required_passes` evaluates to `pass = true`. Missing pass = not promotable (not failing, just deferred — re-run promotion after the missing pass runs).

For v2 outputs that have no Gate E recorded, the operator runs Gate E first (against v2's `preview_*.png` and `phase1/overview.png` slices) using the same `partcraft.pipeline_v3.gate_quality` code path, writes results back to v2's `edit_status.json` under `gates.E`, then promotes. **No v2 file is rewritten by v1 promotion itself** — only the cleaning step writes to v2's status, and only under the `gates.E` namespace it already reserves.

---

## 6. Tools

Three small scripts under `scripts/cleaning/`. None of them go into `partcraft/pipeline_v2/` — they are post-pipeline cleaning, intentionally separate from the orchestration entrypoint.

### 6.1 `scripts/cleaning/promote_to_v1.py`

```
python -m scripts.cleaning.promote_to_v1 \
  --source-runs outputs/partverse/pipeline_v2_shard05 outputs/partverse/shard08/mode_e_text_align \
  --v1-root data/partverse_edit_v1 \
  --rules configs/cleaning/promote_v1.yaml \
  --link-mode hardlink   # hardlink | symlink | copy
```

Behavior:

1. For each source run, walk `objects/<shard>/<obj_id>/` directories.
2. Read `edit_status.json`, evaluate the promote rule, collect promotable edit IDs.
3. For each promotable `(obj_id, edit_id)`:
   - If `data/partverse_edit_v1/objects/<shard>/<obj_id>/before/` does not exist:
     - Hardlink `data/partverse/img_Enc/<obj_id>/{089,090,091,100,008}.png` → `before/views/view_{0..4}.png` in that order. These absolute frame indices are `VIEW_INDICES = [89, 90, 91, 100, 8]` from `partcraft.pipeline_v3.specs`, the canonical 5-view set used by both `overview.png` (collage of these 5 frames) and `preview_*.png` (rendered after-edit at the same camera positions).
     - Pack `data/partverse/slat/<shard>/<obj_id>_feats.pt` and `_coords.pt` into `before/ss.npz` + `before/slat.npz` (single-file packing avoids `.pt`/`.npz` mixed conventions in v1)
     - Write `meta.json` from `phase1/parsed.json` + caption source.
   - Create `edits/<edit_id>/`:
     - `spec.json` from the source `EditSpec` (drop pipeline-internal fields).
     - `qc.json` with the resolved pass dict.
     - `views/view_{0..4}.png` ← hardlink `edits_3d/<edit_id>/preview_{0..4}.png`.
     - `after.npz`:
       - For flux-branch edits: link `edits_3d/<edit_id>/after.npz` (already exists from `s5_trellis`).
       - For deletion edits: leave a placeholder `_after_pending.json` and add the edit to `data/partverse_edit_v1/_pending/del_latent.txt` for `encode_del_latent.py` to consume.
4. Idempotent within a single source run: re-running the promoter against the same `--source-runs` skips edits whose v1 directory already exists with the same `source_run_tag` in `qc.json`. Re-running against a *different* source run that produces the same `(obj_id, edit_id)` admits a parallel v1 directory with suffix `__r2`/`__r3`/... (see §10.3); `--force` overwrites in place instead of appending a suffix.

### 6.2 `scripts/cleaning/encode_del_latent.py`

Reuses `scripts/datasets/partverse/prerender.py`'s render+encode building blocks. For each `(obj_id, edit_id)` listed in `_pending/del_latent.txt`:

1. Locate `after_new.glb` (recorded in `qc.json.source.run_dir`).
2. Blender render the canonical views to a tmp dir (same camera set as the original prerender).
3. DINOv2 → Trellis SS/SLAT encode → write `edits/<edit_id>/after.npz` packing `feats` and `coords`.
4. Append `del_latent_encode = {"ok": true, ...}` to `qc.json.passes`.
5. Remove `_after_pending.json` and the line from `_pending/del_latent.txt`.

Multi-GPU dispatch follows `scripts/datasets/partverse/prerender.py --num-gpus N` exactly.

### 6.3 `scripts/cleaning/rebuild_v1_index.py`

Single-process scan of `data/partverse_edit_v1/objects/**/edits/**`, writes `index/objects.jsonl`, `index/edits.jsonl`, `index/_last_rebuild.json`. Run after every promote / encode batch. Pure read-then-write, fully idempotent.

---

## 7. Data Flow

```
outputs/partverse/pipeline_v2_shard05/  ──┐
outputs/partverse/shard08/mode_e/         │
outputs/partverse/<future_runs>/          │
                                          │  (1) Re-run Gate E on v2 outputs
                                          │      (writes edit_status.json gates.E only)
                                          ▼
                                    promote_to_v1.py
                                          │
                                          ├──► data/partverse_edit_v1/objects/.../edits/<flux>/   (after.npz ready)
                                          └──► data/partverse_edit_v1/_pending/del_latent.txt
                                                        │
                                                        ▼
                                              encode_del_latent.py    (multi-GPU)
                                                        │
                                                        ▼
                                              rebuild_v1_index.py
                                                        │
                                                        ▼
                                  data/partverse_edit_v1/index/{objects,edits}.jsonl
                                                        │
                                                        ▼
                                              downstream training / tar+upload
```

---

## 8. Extensibility

| Change | What you do |
|---|---|
| New pipeline (v4) writes a different `edit_status.json` | Add a `--source-format=v4` adapter in `promote_to_v1.py`. v1 layout untouched. |
| Stricter cleaning pass v2 | Add a new key to `qc.json.passes` (e.g. `quality_filter_v2`). Re-run `promote_to_v1.py` to surface it; pre-existing v1 edits get the new pass appended in-place. |
| Human label / manual annotation | Drop `edits/<id>/human_label.json`. `rebuild_v1_index.py` picks it up automatically. |
| New edit type | Already supported — `edit_id` prefix encodes type. Add the new prefix to `promote_rules.edit_types_allowed`. |
| Need Parquet later | `rebuild_v1_index.py` can grow a `--format parquet` flag. The JSONL stays as the human-friendly default. |
| New shard / new run | No code change — point `--source-runs` at the new directory and re-run promote+rebuild. |

---

## 9. Disk Footprint (per shard, ~1203 obj × ~10 edits)

| Asset | Per unit | Per shard | Notes |
|---|---|---|---|
| before SS+SLAT | ~10 MB / obj | ~12 GB | hardlinked → 0 cost on same FS |
| before 5 views | ~250 KB × 5 / obj | ~1.5 GB | hardlinked → 0 cost |
| after SS+SLAT (flux) | ~10 MB / edit | ~120 GB | hardlinked from `edits_3d/<id>/after.npz` |
| after SS+SLAT (del, encoded) | ~10 MB / edit | ~10 GB | new bytes; ~1k del per shard |
| after 5 views | ~250 KB × 5 / edit | ~15 GB | hardlinked from `preview_*.png` |
| spec.json + qc.json | ~2 KB / edit | ~25 MB | new bytes |
| index | — | ~10 MB | new bytes |

With `--link-mode hardlink` (default), v1 adds **only the new bytes**: ~10 GB for del latents + ~25 MB metadata per shard. `--link-mode copy` is opt-in for tar+upload (a one-time materialization step before packaging).

---

## 10. Resolved Decisions (2026-04-19)

1. **`before` latent format**: split into two files `before/ss.npz` + `before/slat.npz`. Mirrors `after.npz` semantics on the edit side and keeps the two latents independently readable. Both are produced by packing the existing `{obj_id}_feats.pt` and `{obj_id}_coords.pt` from `data/partverse/slat/<shard>/` into NPZ form (no re-encode; the .pt tensors are written verbatim into the NPZ container).
2. **Gate E on v2 outputs**: one-off script `scripts/cleaning/run_gate_quality_on_v2.py` that imports `partcraft.pipeline_v3.gate_quality` directly and writes results back into v2's `edit_status.json` under the existing `gates.E` namespace (currently `null`). v2 module code is **not** modified. The v3 module is imported as a library, not invoked through `partcraft.pipeline_v3.run`.
3. **Cross-version edits as diversity, not duplicates**: pipeline version is a non-distinguishing facet. If obj X is processed by both v2 and v3, both runs' edits are admitted into v1 with the same `edit_id`. Filesystem disambiguation suffix `__r2`, `__r3`, ... is appended to the v1 directory name when needed; the original `edit_id` and `source_run_tag` are preserved in `spec.json` and `qc.json`. `index/edits.jsonl` carries both fields so downstream can group-by `(obj_id, edit_id)` (treat as duplicate) or by `(obj_id, edit_id, source_run_tag)` (treat as diversity) at its own discretion.

---

## 11. Out of Scope (explicitly)

- Training pipeline changes (this spec only delivers the dataset and its index).
- A web UI / report for v1 (re-use existing `report_vlm.html` style if needed, separate spec).
- Cross-dataset (PartObjaverse) extension — the layout is dataset-agnostic but we only validate against PartVerse here.
