# Default val+test Available Review Manifest

This manifest contains default val+test H3D edits whose local pipeline outputs are ready for ZIP review.
Flux edit types require both `edits_2d/<edit_id>_input.png` and `edits_2d/<edit_id>_edited.png`.

Total available records: `3855`

## Available By Shard

| shard | edits |
|---|---:|
| 02 | 714 |
| 04 | 435 |
| 05 | 574 |
| 06 | 495 |
| 07 | 659 |
| 08 | 541 |
| 09 | 437 |

## Available By Edit Type

| edit_type | edits |
|---|---:|
| deletion | 1196 |
| addition | 1196 |
| modification | 378 |
| scale | 56 |
| material | 184 |
| color | 210 |
| global | 635 |

## Source Split Mix

| split | edits |
|---|---:|
| test | 1912 |
| val | 1943 |

## Reject Counts

| reason | count |
|---|---:|
| no_pipeline_shard_dir | 1502 |
| not_val_or_test | 97347 |

## Missing Pipeline Outputs

| shard | objects | edits |
|---|---:|---:|
| 00 | 57 | 485 |
| 01 | 53 | 544 |
| 03 | 63 | 473 |

## Review Export

Open `h3d_review_tool.html`, drag in one `*_assets.zip`, mark decisions, then use `导出 selected_edit_ids_*.txt`.
Concatenate those selected-edit-id files across chunks to get the final benchmark candidate list.
