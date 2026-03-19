# PartCraft3D Data Pipeline

## Pipeline Architecture (Closed-Loop)

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    PREREQUISITE (one-time)                       │
 │  prepare_partobjaverse.py → data/partobjaverse_tiny/            │
 │  prerender.py → Blender 150 views + SLAT encoding               │
 │  Cost: GPU only, 0 API tokens                                   │
 └───────────────────────────┬──────────────────────────────────────┘
                             ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  STEP 1: SEMANTIC LABELING          [VLM API, ~4K tok/obj]      │
 │  ┌─────────────┐    ┌──────────────┐                            │
 │  │  Phase 0     │ →  │  Enrichment  │                            │
 │  │  4-view VLM  │    │  1-view VLM  │                            │
 │  │  labeling    │    │  edit prompts│                            │
 │  └─────────────┘    └──────────────┘                            │
 │  Output: semantic_labels.jsonl                                  │
 └───────────────────────────┬──────────────────────────────────────┘
                             ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  STEP 2: EDIT PLANNING              [CPU, 0 tokens]             │
 │  PartCatalog → deletion / addition / modification / global      │
 │  Output: edit_specs.jsonl                                       │
 └───────────────────────────┬──────────────────────────────────────┘
                             ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  STEP 3: 2D IMAGE EDITING           [VLM API, parallelizable]  │
 │  For each spec: plain view image + edit prompt → Gemini          │
 │  Prompt provides semantic context (no mask annotation needed)   │
 │  Output: 2d_edits/{edit_id}_edited.png                          │
 │  Cost: ~458 input tokens + 1 image output per spec              │
 └───────────────────────────┬──────────────────────────────────────┘
                             ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  STEP 4: 3D EDITING — TRELLIS       [GPU, main workload]        │
 │                                                                  │
 │  ┌─ Deletion (contact-aware) ───────────────────────────────┐    │
 │  │  Non-contact voxels: removed entirely (no S1 regen)      │    │
 │  │  Contact boundary: S1 closes seam via soft mask           │    │
 │  │  Pure floating part: skip S1/S2, direct voxel removal    │    │
 │  │  cfg_strength=3.0-5.5 (contact closure only)             │    │
 │  └──────────────────────────────────────────────────────────┘    │
 │                                                                  │
 │  ┌─ Modification ──────────────────────────────────────────┐    │
 │  │  Flow Inversion + Repaint: edit part only                │    │
 │  │  Contact-aware anisotropic soft mask at boundaries       │    │
 │  └──────────────────────────────────────────────────────────┘    │
 │                                                                  │
 │  ┌─ Global (auto-promoted if part > 40% SLAT) ────────────┐    │
 │  │  TextureOnly: S1 skipped (shape preserved)               │    │
 │  │  S2 repaint only — changes color/material/texture        │    │
 │  │  cfg_strength=5.0 controls appearance departure          │    │
 │  └──────────────────────────────────────────────────────────┘    │
 │                                                                  │
 │  ┌─ Addition ──────────────────────────────────────────────┐    │
 │  │  Swap before/after from corresponding deletion pair      │    │
 │  │  No TRELLIS inference needed                             │    │
 │  └──────────────────────────────────────────────────────────┘    │
 │                                                                  │
 │  Dynamic soft mask: σ = f(contact_ratio)                        │
 │    s1_sigma = 1.5 + contact_ratio × 4.0                        │
 │    s2_sigma = 2.0 + contact_ratio × 10.0                       │
 │                                                                  │
 │  Output: mesh_pairs/{edit_id}/before.ply, after.ply             │
 │  Cost: 0 API tokens, ~60-120s GPU per edit                      │
 └───────────────────────────┬──────────────────────────────────────┘
                             ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  STEP 5: QUALITY SCORING             [VLM API]                  │
 │  Render 4 views per model → VLM scores:                         │
 │    execution | localization | preservation | overall             │
 │  Output: vlm_scores.jsonl + quality_tier (high/medium/negative) │
 │  Cost: ~3K tokens per spec                                      │
 └───────────────────────────┬──────────────────────────────────────┘
                             ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  STEP 6: EXPORT                      [CPU, 0 tokens]            │
 │  Generate instruction variants + assemble final dataset         │
 │  Output: edit_pairs.jsonl (training-ready manifest)             │
 └──────────────────────────────────────────────────────────────────┘
```

## Scripts Directory (Refactored)

```
scripts/
├── run_pipeline.py          # Main unified pipeline (run this)
├── prepare_partobjaverse.py # Data preparation (one-time)
├── prerender.py             # SLAT pre-encoding (one-time, GPU)
├── run_2d_edit.py           # Standalone 2D editing (parallel API)
├── blender_render.py        # Blender subprocess (called internally)
│
├── standalone/              # Individual phase scripts (debugging)
│   ├── run_phase0.py
│   ├── run_phase1.py
│   ├── run_phase2.py        # Mesh assembly (demoted, fallback only)
│   ├── run_phase2_5.py      # Old TRELLIS runner (replaced by pipeline)
│   ├── run_phase3.py
│   ├── run_enrich.py
│   ├── run_all.py           # Old orchestrator (replaced by pipeline)
│   ├── encode_slat.py
│   └── test_editformer.py
│
├── tools/                   # Utility tools
└── vis/                     # Visualization scripts
```

## Token Cost Per Data Item (1 Edit Spec)

### Cost Model: Gemini 2.5 Flash

| Pricing | Rate |
|---------|------|
| Input tokens | $0.15 / 1M tokens |
| Output tokens | $0.60 / 1M tokens |
| Image output | ~$0.02 / image |
| Image input | ~258 tokens / image |

### Per-Object Costs (amortized across N edits)

| Step | Input Tokens | Output Tokens | Images | Notes |
|------|-------------|---------------|--------|-------|
| Phase 0 labeling | 1,532 | 2,000 | 0 | 4 views × 258 + 500 text |
| VLM enrichment | 1,258 | 3,000 | 0 | 1 view × 258 + 1000 text |
| **Subtotal/object** | **2,790** | **5,000** | **0** | |

With ~27 edits/object (PartObjaverse-Tiny average):
- Amortized per edit: **~103 input + ~185 output tokens**

### Per-Edit Costs

| Step | Input Tokens | Output Tokens | Images Out | USD |
|------|-------------|---------------|------------|-----|
| Step 3: 2D Edit | 458 | 0 | 1 | $0.0201 |
| Step 4: 3D Edit | 0 | 0 | 0 | $0.0000 (GPU) |
| Step 5: Quality | 2,564 | 500 | 0 | $0.0007 |
| **Subtotal/edit** | **3,022** | **500** | **1** | **$0.0208** |

### Total Per Edit (with amortization)

| Component | Input | Output | Images | USD |
|-----------|-------|--------|--------|-----|
| Amortized semantic | 103 | 185 | 0 | $0.0001 |
| 2D image editing | 458 | 0 | 1 | $0.0201 |
| Quality scoring | 2,564 | 500 | 0 | $0.0007 |
| **TOTAL per edit** | **3,125** | **685** | **1** | **$0.0209** |

### Scale Projections

| Scale | Edits | API Cost | GPU Time (est.) |
|-------|-------|----------|-----------------|
| Tiny (test) | 100 | $2.09 | ~3 hours |
| Small | 1,000 | $20.90 | ~1.5 days |
| Medium | 10,000 | $209.00 | ~2 weeks |
| Large | 100,000 | $2,090.00 | ~5 months (multi-GPU) |

> Note: 2D image editing ($0.02/image) dominates API cost (~96%).
> GPU time assumes ~90s/edit on a single A100.

### Cost Optimization Strategies

1. **Skip 2D editing for text-only guidance**: `--no-2d-edit` → saves 96% API cost
2. **Pre-generate 2D edits**: `run_2d_edit.py --workers 16` → parallel API calls
3. **Filter specs before 3D editing**: Quality prediction to skip low-potential specs
4. **Batch SLAT encoding**: Group by object to reuse SLAT (already implemented)
5. **Tag-based A/B testing**: `--tag v1` to isolate experiments

## Usage Examples

```bash
# 1. Prepare data (one-time)
python scripts/prepare_partobjaverse.py --output data/partobjaverse_tiny

# 2. Pre-render + SLAT encoding (one-time, GPU)
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/prerender.py \
    --config configs/partobjaverse.yaml --render-workers 4

# 3. Run full pipeline
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/partobjaverse.yaml --tag v1

# 4. Run specific steps
ATTN_BACKEND=xformers python scripts/run_pipeline.py --steps 4 5 --tag v1

# 5. Pre-generate 2D edits (can run on CPU machine)
python scripts/run_2d_edit.py --config configs/partobjaverse.yaml \
    --tag v1 --workers 16

# 6. Cost estimation
python scripts/run_pipeline.py --config configs/partobjaverse.yaml --dry-run

# 7. Resume after interruption (automatic)
ATTN_BACKEND=xformers python scripts/run_pipeline.py --tag v1
```

## Edit Type Routing Logic

```
Input: edit_spec (type, part_ids)
  │
  ├─ type == "addition"
  │   └─ Swap before/after from deletion pair → no inference
  │
  ├─ type == "deletion" or "modification"
  │   │
  │   ├─ build_part_mask()
  │   │   ├─ Voxelize GT parts → 64³ mask
  │   │   ├─ Align to SLAT coordinates (KNN)
  │   │   └─ Check part ratio:
  │   │       ├─ ratio > 40% → AUTO-PROMOTE to Global (TextureOnly)
  │   │       └─ ratio ≤ 40% → tight mask + dilation
  │   │
  │   ├─ [if Deletion] → Modification path, dynamic mask (radius 2-4),
  │   │                   dynamic cfg (3.0-7.0), S1 fills hole
  │   └─ [if Modification] standard edit, cfg=7.5
  │
  └─ type == "global"
      └─ TextureOnly: S1 skipped, S2 repaint only, cfg=5.0
```

## Output Structure

```
outputs/partobjaverse_tiny/
├── cache/
│   ├── phase0/semantic_labels.jsonl
│   ├── phase1/edit_specs.jsonl
│   ├── phase2_5/
│   │   ├── edit_results_{tag}.jsonl    # Step 4 results
│   │   ├── 2d_edits_{tag}/             # Step 3 edited images
│   │   ├── phase3_{tag}/               # Step 5 quality scores
│   │   └── debug_masks/               # Mask visualizations
│   └── phase3/vlm_scores.jsonl
├── mesh_pairs_{tag}/
│   └── {edit_id}/
│       ├── before.ply
│       ├── after.ply
│       ├── before_slat/
│       └── after_slat/
└── edit_pairs_{tag}.jsonl              # Final dataset (Step 6)
```
