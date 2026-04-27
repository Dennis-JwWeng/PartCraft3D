# H3D default val+test bench review packaging

This runbook builds review ZIPs from the default `H3D_v1_hf/data/splits/{val,test}.obj_ids.txt` split. It keeps the benchmark review at object-split granularity for leakage checks, but the actual review list is edit-level.

## Current local package

On this machine the currently reviewable default `val ∪ test` candidates are packaged here:

- Manifest: `/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available/h3d_default_val_test_available_manifest.jsonl`
- Review directory: `/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_zip_review/`
- Portable review bundle: `/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_zip_review.zip`
- Missing shard lists: `/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available/missing/`

The current package contains 3,855 edits from the default `val ∪ test` objects whose local `outputs/partverse` assets are present. It is chunked into 39 `h3d_test_review_XXX_assets.zip` files plus `h3d_review_tool.html`.

## Missing Pipeline Outputs

The missing default `val ∪ test` pipeline outputs are:

| shard | objects | edits |
|---|---:|---:|
| 00 | 57 | 485 |
| 01 | 53 | 544 |
| 03 | 63 | 473 |

Use these files as the source of truth for remote machines:

- `default_val_test_available/missing/shard00_missing_obj_ids.txt`
- `default_val_test_available/missing/shard01_missing_obj_ids.txt`
- `default_val_test_available/missing/shard03_missing_obj_ids.txt`
- `default_val_test_available/missing/shard00_missing_edit_ids.txt`
- `default_val_test_available/missing/shard01_missing_edit_ids.txt`
- `default_val_test_available/missing/shard03_missing_edit_ids.txt`

A shard is considered ready for review only when each required object has the corresponding `outputs/partverse/shardXX/mode_e_text_align/objects/XX/<obj_id>/edits_3d/<edit_id>` directory. For `modification`, `scale`, `material`, `color`, and `global`, the same object directory must also contain:

- `edits_2d/<edit_id>_input.png`
- `edits_2d/<edit_id>_edited.png`

`addition` and `deletion` do not require 2D edit images.

## Rebuild The Review Manifest

After copying or generating shard outputs under the same `outputs/partverse` layout, rebuild the default val+test manifest:

```bash
python bench_review/build_default_val_test_review_manifest.py \
  --repo /mnt/zsn/zsn_workspace/PartCraft3D \
  --out-dir /mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available
```

The command writes:

- `h3d_default_val_test_available_manifest.jsonl`
- `h3d_default_val_test_available_edit_ids.txt`
- `summary.md`
- `missing/*.txt`

If all missing shard outputs have been restored, `summary.md` should no longer report `no_pipeline_shard_dir` or `missing_pipeline_edit_dir` for shards 00/01/03.

## Build Review ZIP Chunks

Reuse the existing ZIP review HTML tool:

```bash
python bench_review/build_zip_review.py \
  --manifest /mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available/h3d_default_val_test_available_manifest.jsonl \
  --out-dir /mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_zip_review \
  --chunk-size 100 \
  --all
```

Then create the portable outer bundle:

```bash
python - <<'PYZIP'
from pathlib import Path
import zipfile
src = Path('/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_zip_review')
out = Path('/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_zip_review.zip')
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as zf:
    for path in sorted(src.rglob('*')):
        if path.is_file():
            zf.write(path, path.relative_to(src.parent))
print(out)
PYZIP
```

## Review And Export Final Benchmark IDs

1. Unzip or open `default_val_test_available_zip_review.zip`.
2. Open `default_val_test_available_zip_review/h3d_review_tool.html` in a browser.
3. Drag one `h3d_test_review_XXX_assets.zip` file into the page.
4. Mark each sample as `要`, `不要`, or `不确定`.
5. Click `导出 review_results_*.json` to preserve full review decisions.
6. Click `导出 selected_edit_ids_*.txt` to export the `要` edit ids for that chunk.
7. Repeat for every chunk.

To merge all selected chunk files into the final benchmark list:

```bash
python - <<'PYIDS'
from pathlib import Path
review_dir = Path('/path/to/downloaded/review/results')
out = review_dir / 'final_selected_edit_ids.txt'
ids = []
seen = set()
for path in sorted(review_dir.glob('selected_edit_ids_*.txt')):
    for line in path.read_text(encoding='utf-8').splitlines():
        edit_id = line.strip()
        if edit_id and edit_id not in seen:
            seen.add(edit_id)
            ids.append(edit_id)
out.write_text('\n'.join(ids) + ('\n' if ids else ''), encoding='utf-8')
print(f'wrote {len(ids)} ids -> {out}')
PYIDS
```

Keep the exported `review_results_*.json` files next to the final id list so rejected and uncertain cases remain auditable.


## Remote Shard Packaging And Upload

Use this section on machines that have the missing pipeline shard outputs. Package each missing shard separately so uploads do not overwrite the already uploaded `assets/h3d_test_review_000_assets.zip` through `assets/h3d_test_review_038_assets.zip`.

Current missing default `val ∪ test` worklist:

| shard | objects | edits | object id list | edit id list |
|---|---:|---:|---|---|
| 00 | 57 | 485 | `bench_review/default_val_test_available/missing/shard00_missing_obj_ids.txt` | `bench_review/default_val_test_available/missing/shard00_missing_edit_ids.txt` |
| 01 | 53 | 544 | `bench_review/default_val_test_available/missing/shard01_missing_obj_ids.txt` | `bench_review/default_val_test_available/missing/shard01_missing_edit_ids.txt` |
| 03 | 63 | 473 | `bench_review/default_val_test_available/missing/shard03_missing_obj_ids.txt` | `bench_review/default_val_test_available/missing/shard03_missing_edit_ids.txt` |

### Required Data On The Remote Machine

The remote machine must have the same logical workspace layout, or equivalent symlinks, for:

- `data/H3D_v1/manifests/all.jsonl`
- `data/H3D_v1/<edit_type>/<shard>/<obj_id>/<edit_id>/{before.png,after.png,before.npz,after.npz,meta.json}`
- `H3D_v1_hf/data/splits/{train,val,test}.obj_ids.txt`
- `outputs/partverse/shardXX/mode_e_text_align/objects/XX/<obj_id>/edits_3d/<edit_id>/`
- For flux edit types (`modification`, `scale`, `material`, `color`, `global`): `outputs/partverse/shardXX/mode_e_text_align/objects/XX/<obj_id>/edits_2d/<edit_id>_input.png` and `<edit_id>_edited.png`

`addition` and `deletion` do not need `edits_2d` images. If the machine only has one target shard, that is fine; use `--shards` to force the manifest builder to include only that shard.

### Build One Shard Package

Set the shard id and build the review manifest:

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
export SHARD=00  # use 00, 01, or 03

python bench_review/build_default_val_test_review_manifest.py \
  --repo /mnt/zsn/zsn_workspace/PartCraft3D \
  --out-dir /mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_shard${SHARD} \
  --shards ${SHARD}
```

Check `summary.md` in the output directory. The `Total available records` should match the shard's expected edit count when all required pipeline outputs are present:

- `SHARD=00`: 485 edits
- `SHARD=01`: 544 edits
- `SHARD=03`: 473 edits

Then build chunked review ZIPs with global numbering. The already uploaded main package uses `000`-`038`, so missing shards continue from `039`:

```bash
case "$SHARD" in
  00) CHUNK_OFFSET=39 ;;
  01) CHUNK_OFFSET=44 ;;
  03) CHUNK_OFFSET=50 ;;
  *) echo "unknown SHARD=$SHARD" >&2; exit 2 ;;
esac

python bench_review/build_zip_review.py \
  --manifest /mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_shard${SHARD}/h3d_default_val_test_available_manifest.jsonl \
  --out-dir /mnt/zsn/zsn_workspace/PartCraft3D/bench_review/default_val_test_available_shard${SHARD}_zip_review \
  --chunk-size 100 \
  --chunk-offset ${CHUNK_OFFSET} \
  --all
```

Expected global chunk names:

- `SHARD=00`: 5 chunks, `h3d_test_review_039_assets.zip`-`h3d_test_review_043_assets.zip`; last chunk 85 edits
- `SHARD=01`: 6 chunks, `h3d_test_review_044_assets.zip`-`h3d_test_review_049_assets.zip`; last chunk 44 edits
- `SHARD=03`: 5 chunks, `h3d_test_review_050_assets.zip`-`h3d_test_review_054_assets.zip`; last chunk 73 edits

With this convention, the full review sequence is globally unique: main package `000`-`038`, missing shards `039`-`054`.

### Upload One Shard To Hugging Face

Do not put tokens in files. Export a token only in the shell session. Upload missing shard asset ZIPs into the same top-level `assets/` directory as the existing `000`-`038` chunks; global numbering prevents collisions.

```bash
export HF_TOKEN='<your-write-token>'
export SHARD=00  # use 00, 01, or 03
export HF_REPO='Dennis0626/h3d-default-val-test-available-zip-review'
```

Upload shard metadata, then asset chunks in filename order. Reuse the already uploaded top-level `h3d_review_tool.html`:

```bash
python - <<'PYUPLOAD'
from pathlib import Path
import os
from huggingface_hub import HfApi

repo_id = os.environ['HF_REPO']
token = os.environ['HF_TOKEN']
shard = os.environ['SHARD']
base = Path('/mnt/zsn/zsn_workspace/PartCraft3D/bench_review')
meta_dir = base / f'default_val_test_available_shard{shard}'
review_dir = base / f'default_val_test_available_shard{shard}_zip_review'
api = HfApi(token=token)

for local, remote in [
    (meta_dir / 'h3d_default_val_test_available_manifest.jsonl', f'metadata/missing_shards/shard{shard}/h3d_default_val_test_available_manifest.jsonl'),
    (meta_dir / 'h3d_default_val_test_available_edit_ids.txt', f'metadata/missing_shards/shard{shard}/h3d_default_val_test_available_edit_ids.txt'),
    (meta_dir / 'summary.md', f'metadata/missing_shards/shard{shard}/summary.md'),
]:
    api.upload_file(
        repo_id=repo_id,
        repo_type='dataset',
        path_or_fileobj=str(local),
        path_in_repo=remote,
        commit_message=f'Add shard{shard} {remote}',
    )
    print(f'uploaded {remote}')

chunks = sorted(review_dir.glob('h3d_test_review_*_assets.zip'))
for idx, path in enumerate(chunks):
    remote = f'assets/{path.name}'
    api.upload_file(
        repo_id=repo_id,
        repo_type='dataset',
        path_or_fileobj=str(path),
        path_in_repo=remote,
        commit_message=f'Add shard{shard} review assets chunk {idx:03d}',
    )
    print(f'uploaded {idx + 1}/{len(chunks)} {remote}')
PYUPLOAD
```

After upload, verify the file count for that shard:

```bash
python - <<'PYCHECK'
import os
from huggingface_hub import HfApi
repo_id = os.environ['HF_REPO']
shard = os.environ['SHARD']
files = HfApi(token=os.environ['HF_TOKEN']).list_repo_files(repo_id, repo_type='dataset')
prefix = 'assets/h3d_test_review_'
expected = {
    '00': range(39, 44),
    '01': range(44, 50),
    '03': range(50, 55),
}[shard]
expected_names = [f'assets/h3d_test_review_{i:03d}_assets.zip' for i in expected]
missing = [name for name in expected_names if name not in files]
print(f'shard{shard} expected chunks: {len(expected_names)}')
print(f'shard{shard} missing chunks: {len(missing)}')
for name in expected_names:
    print(name, 'OK' if name in files else 'MISSING')
PYCHECK
```

Reviewers use the same top-level `h3d_review_tool.html` and the same top-level `assets/*.zip` sequence. The HTML tool names exports from the ZIP stem, so reviewing `h3d_test_review_044_assets.zip` exports `selected_edit_ids_h3d_test_review_044_assets.txt` and `review_results_h3d_test_review_044_assets.json`. That makes every exported decision file traceable back to exactly one asset ZIP. Exported `selected_edit_ids_*.txt` files should later be concatenated across the full `000`-`054` sequence.
