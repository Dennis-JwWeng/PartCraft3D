# H3D_v1 HF publish runbook

Companion to `h3d-v1-promote.md`. That runbook ends at §5 where a
validated `data/H3D_v1/` is on disk and per-shard tars can be produced.
This runbook picks up from there and publishes one shard to
<https://huggingface.co/datasets/ART-3D/H3D_v1> — the canonical Hugging
Face release of the dataset.

The scheme in one sentence: **one uncompressed `tar` per shard**
(hardlinks preserved for NPZ dedup, `orig_views/` and the host-local
`all.jsonl` excluded), tracked under Git LFS in the HF repo, with
per-shard manifest fragments aggregated on the HF side by
`rebuild_manifests.py` into cross-shard indexes / splits / loader
constants.

## 0. Prerequisites (fresh machine)

1. **PartCraft3D checkout** on the current branch — provides
   `scripts/cleaning/h3d_v1/pack_shard.py`,
   `scripts/cleaning/h3d_v1/dedup_manifests.py`, and the
   `partcraft.cleaning.h3d_v1.layout` module they import. Both scripts
   must run in an env where `partcraft` is importable (e.g.
   `vinedresser3d`, or any env where `pip install -e .` was run
   against this repo). `numpy + pyyaml` is enough — no GPU, no trellis.

2. **Validated H3D_v1 root** at `<pipeline>/data/H3D_v1/`. Follow
   `h3d-v1-promote.md` §0–§4 first; expect
   `build_h3d_v1_index --validate` to exit 0.

3. **HF repo clone** at a sibling path **on the same filesystem** as
   `data/H3D_v1/`. Hardlinks inside the tar and `install_shard.py`'s
   same-inode short-circuit only work when source and destination
   share a filesystem.

   ```bash
   cd /mnt/zsn/data                # anywhere on the same FS as data/H3D_v1
   git lfs install
   git clone https://huggingface.co/datasets/ART-3D/H3D_v1 H3D_v1_hf
   cd H3D_v1_hf

   hf auth login                   # paste an ART-3D write token
   hf lfs-enable-largefiles .      # one-time, allows >5 GB LFS objects
   ```

   After clone, expect:

   ```
   H3D_v1_hf/
     scripts/{install_shard.py, publish_shard.sh, rebuild_manifests.py, validate.py}
     src/h3d_v1/                   # loader package
     H3D_v1.py                     # datasets.load_dataset entry
     data/{shards,manifests,splits}/
     README.md CITATION.cff LICENSE
   ```

4. **(optional) symlink into the PartCraft3D workspace** so the HF
   clone is visible from your editor without leaving the pipeline
   tree — purely cosmetic:

   ```bash
   ln -s /mnt/zsn/data/H3D_v1_hf /mnt/zsn/zsn_workspace/PartCraft3D/H3D_v1_hf
   ```

## 1. Dedup per-shard manifests (last-wins)

The pipeline may write `manifests/<edit_type>/<NN>.jsonl` multiple
times for the same `(edit_type, edit_id)` if preview / `best_view`
was re-rendered mid-promote. The on-disk PNG + `meta.json` reflect
the **latest** render, so we keep the last occurrence.

```bash
cd <PartCraft3D>
SHARD=07
python -m scripts.cleaning.h3d_v1.dedup_manifests \
  --dataset-root data/H3D_v1 --shard $SHARD --keep last
```

Flags:

- `--shard 07` (repeatable) — restrict to one shard. Omit to dedup all
  shards present.
- `--edit-type deletion` (repeatable) — restrict to one type. Omit for
  all 7 types.
- `--keep last` (default) — keep the last occurrence of each
  `(edit_type, edit_id)` pair. `first` should **not** be used unless
  you understand the consequence: it makes the manifest's
  `views.best_view_index` disagree with the rendered PNG on disk.
- `--dry-run` — report duplicates, don't touch manifests.
- `--no-backup` — skip `<file>.bak-<ts>`; otherwise a backup is kept.
- `--rebuild-aggregate` — also regenerate `manifests/all.jsonl`. Not
  needed if you only plan to pack, since §2 drops it anyway.

## 2. Pack the shard into an HF-friendly tar

```bash
SHARD=07
python -m scripts.cleaning.h3d_v1.pack_shard \
  --dataset-root data/H3D_v1 \
  --shard $SHARD \
  --out /mnt/zsn/data/H3D_v1_hf/data/shards/H3D_v1__shard${SHARD}.tar \
  --drop-orig-views \
  --drop-agg-manifest
```

Why these flags (all defaults are deliberate):

| flag | rationale |
|---|---|
| *no* `--compression` | uncompressed tar preserves hardlinks, which dedup `_assets/<NN>/<obj>/object.npz` down to one on-disk copy per obj inside the archive (~20–30 % saving). Users who `tar -xf` see the dedup on disk; `datasets.load_dataset` streams bytes and pays zero extra. gzip / xz **defeat** the hardlink dedup. |
| `--drop-orig-views` | `_assets/<NN>/<obj>/orig_views/` (~200 MB / shard) are not consumed by the loader. Users who need them can re-render from the SLAT latent. |
| `--drop-agg-manifest` | `manifests/all.jsonl` indexes whatever shards were on the **packing host**. The HF side regenerates it from per-shard fragments, so packing the host's copy would leak irrelevant state. |
| `--out` lands in the HF clone | puts the tar directly in `<hf-clone>/data/shards/`, so `publish_shard.sh`'s same-inode short-circuit skips the extra copy step. |

Sanity check the tar before publishing:

```bash
TAR=/mnt/zsn/data/H3D_v1_hf/data/shards/H3D_v1__shard${SHARD}.tar

du -h "$TAR"
tar -tf "$TAR" | head

tar -tf "$TAR" | grep -E "^manifests/(deletion|addition|modification|scale|material|color|global)/${SHARD}\.jsonl$"
# expect one fragment per edit type actually present in the shard

! tar -tf "$TAR" | grep -q "^manifests/all\.jsonl$"    # must be absent
! tar -tf "$TAR" | grep -q "/orig_views/"              # must be absent
```

## 3. Publish (HF clone side)

```bash
cd /mnt/zsn/data/H3D_v1_hf
bash scripts/publish_shard.sh data/shards/H3D_v1__shard${SHARD}.tar
```

What it does (source of truth: `scripts/publish_shard.sh`):

1. `git pull --rebase origin main` — pick up other machines' pushes.
2. `install_shard.py <tar>` — extract the per-shard manifest
   fragments into `data/manifests/by_shard/<NN>.jsonl` and hardlink
   the tar into `data/shards/` if it isn't already there.
3. `rebuild_manifests.py` — regenerate from per-shard fragments:
   - `data/manifests/by_type/<edit_type>.jsonl`
   - `data/manifests/all.jsonl`
   - `data/splits/{train,val,test}.jsonl` (`obj_id → split` via
     SHA-256, 95 / 2.5 / 2.5)
   - `data/shards/index.json`
   - `H3D_v1.py`'s `_SHARDS` tuple
   - README "Available shards" table
   Cross-shard `(edit_type, edit_id)` dedup uses **last-wins**, so
   fragments published later for the same key overwrite older ones.
4. `validate.py` — manifest-only consistency pass (required fields,
   split partitioning, shard table, `_SHARDS` tuple regex).
5. `git add` the narrow set (shard tar + `by_shard/<NN>.jsonl` +
   `by_type/` + `all.jsonl` + `splits/` + `index.json` + `H3D_v1.py` +
   `README.md`) and `git commit` with an auto-generated message
   including record + object counts and tar size.
6. `git push origin main`.

Pass `--no-push` to stop after step 5 (useful for review); push later
with `git push origin main`.

## 4. Verify

1. **HF web.** Refresh
   <https://huggingface.co/datasets/ART-3D/H3D_v1/tree/main>. The new
   tar should appear under `data/shards/`, and the README "Available
   shards" table should list the new row with the right record count.

2. **Re-download and count records on any machine:**

   ```bash
   hf download --repo-type=dataset ART-3D/H3D_v1 \
     "data/shards/H3D_v1__shard${SHARD}.tar" \
     --local-dir /tmp/h3d_v1_check
   tar -tf "/tmp/h3d_v1_check/data/shards/H3D_v1__shard${SHARD}.tar" \
     | grep -c "/meta\.json$"   # number of edit records in the shard
   ```

3. **Loader smoke test** — pull one sample end-to-end through the
   published loader:

   ```bash
   python - <<'PY'
   from datasets import load_dataset
   ds = load_dataset(
       "ART-3D/H3D_v1", name="color", split="train",
       streaming=False, trust_remote_code=True,
   )
   r = next(iter(ds))
   print({k: type(v).__name__ for k, v in r.items()})
   PY
   ```

   Expect the 14 canonical fields documented in `src/h3d_v1/dataset.py`
   (no `best_view_index`, no `orig_views`).

4. **Split bookkeeping.** Splits are deterministic (SHA-256 of
   `obj_id` — see `src/h3d_v1/splits.py`); publishing a new shard
   never moves existing objects across splits, it only adds rows.
   If `data/splits/` diffs for more than the new shard's obj_ids,
   something in `rebuild_manifests.py` changed — investigate before
   pushing.

## 5. Concurrency (multi-machine publish)

Each shard is independent, so two machines publishing **different**
shards in parallel is safe: `publish_shard.sh` starts with
`git pull --rebase`, and a push race is resolved by rebase plus
re-run of steps 3–6 (`rebuild_manifests` + `validate` are idempotent).

Do **not** publish the *same* shard from two machines simultaneously —
pack each shard on its designated host. If it happens anyway, the
later push loses, and you just rerun `publish_shard.sh` on the
winning host.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: partcraft` from `pack_shard` / `dedup_manifests` | env missing the repo install | activate `vinedresser3d`, or `pip install -e .` inside PartCraft3D |
| Packed tar size is 2–3× expected | `--compression` set (gzip defeats hardlink dedup) or `--drop-orig-views` missing | repack without `--compression`, with `--drop-orig-views` and `--drop-agg-manifest` |
| `install_shard: dst already exists with different content` | an older tar for the same shard is staged | `rm data/shards/H3D_v1__shard<NN>.tar`, then rerun `publish_shard.sh` |
| `remote: You need to configure your repository to enable upload of files > 5GB` on `git push` | HF largefiles not enabled for this clone | run `hf lfs-enable-largefiles .` in the HF clone once, retry push |
| `validate.py` reports `duplicate edit_id` after rebuild | per-shard dedup was skipped before pack, and cross-shard duplicates remain | rerun §1 with `--keep last` on the affected shard, repack, rerun `publish_shard.sh` |
| `datasets.load_dataset("ART-3D/H3D_v1", name=<type>)` errors on the new shard | `H3D_v1.py`'s `_SHARDS` tuple wasn't updated | confirm `rebuild_manifests.py` ran (it rewrites `_SHARDS`); if hand-editing, match the regex in `scripts/rebuild_manifests.py` — declaration must stay on one line as `_SHARDS: tuple[str, ...] = ( ... )` |
| HF page images broken after editing README | relative image paths | README must use absolute HF `resolve` URLs: `https://huggingface.co/datasets/ART-3D/H3D_v1/resolve/main/assets/<file>` |
| `git push` interrupted mid-LFS upload, local has an extra commit | network blip or Ctrl-C | `git reset --hard origin/main && rm -f data/shards/H3D_v1__shard<NN>.tar`, rerun from §2 |
| `install_shard.py: same inode, short-circuit` but the tar is stale | hardlink into `data/shards/` points at an older `--out` path on the same FS | delete both the `data/shards/` hardlink and the source tar, repack with `--out` pointing into `data/shards/` directly |

## 7. One-command recap (after a fresh pull of H3D_v1 raw data)

```bash
SHARD=07
PARTCRAFT=/mnt/zsn/zsn_workspace/PartCraft3D
HF=/mnt/zsn/data/H3D_v1_hf

# 1. dedup (last-wins) manifest fragments for this shard
cd "$PARTCRAFT"
python -m scripts.cleaning.h3d_v1.dedup_manifests \
  --dataset-root data/H3D_v1 --shard "$SHARD" --keep last

# 2. pack into the HF clone's data/shards/
python -m scripts.cleaning.h3d_v1.pack_shard \
  --dataset-root data/H3D_v1 --shard "$SHARD" \
  --out "$HF/data/shards/H3D_v1__shard${SHARD}.tar" \
  --drop-orig-views --drop-agg-manifest

# 3. publish (handles install + rebuild + validate + commit + push)
cd "$HF"
bash scripts/publish_shard.sh "data/shards/H3D_v1__shard${SHARD}.tar"
```
