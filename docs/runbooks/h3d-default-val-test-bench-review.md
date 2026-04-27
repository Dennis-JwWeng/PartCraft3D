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
