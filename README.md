# PartCraft3D

Industrial-scale data pipeline for generating native 3D editing training pairs from part-level datasets.

## Overview

PartCraft3D takes part-segmented 3D assets and programmatically generates large-scale **(before_3D, after_3D, edit_instruction)** training triplets through part-level manipulation.

**Core idea**: Editing = combinatorial operations on parts (add, remove, modify). One disassembly yields both a deletion and an addition pair.

---

## Pipeline Architecture (6-Step)

```
Prerequisite (one-time, GPU)
    prepare_partobjaverse.py → data/partobjaverse_tiny/
    prerender.py → Blender 150 views + SLAT encoding

Step 1: Semantic Labeling (VLM, ~4K tok/obj)
    4-view VLM labeling + 1-view edit prompt enrichment
    → semantic_labels.jsonl

Step 2: Edit Planning (CPU, 0 tokens)
    PartCatalog → deletion / addition / modification / global
    → edit_specs.jsonl

Step 3: 2D Image Editing (VLM / local diffusers)
    Plain input view + constrained prompt → edited reference image
    → 2d_edits_{tag}/{edit_id}_edited.png

Step 4: 3D Editing — TRELLIS (GPU, main workload)
    ├─ Deletion:     via Modification, S1 fills hole (dynamic mask/cfg)
    ├─ Modification:  Flow Inversion + Repaint, contact-aware soft mask
    ├─ Global:        TextureOnly — S1 skipped, S2 repaint changes texture
    └─ Addition:      swap before/after from deletion pair (no inference)
    → mesh_pairs_{tag}/{edit_id}/before.ply, after.ply

Step 5: Quality Scoring (VLM)
    4-view rendering → VLM scores (execution, localization, preservation)
    → vlm_scores.jsonl + quality tier (high/medium/negative)

Step 6: Export (CPU, 0 tokens)
    Instruction variants + final dataset assembly
    → edit_pairs_{tag}.jsonl
```

> See [PIPELINE.md](PIPELINE.md) for detailed architecture diagram, token cost breakdown, and scale projections.

---

## Quick Start

### Prerequisites

1. **Data**: `data/partobjaverse_tiny/` (200 objects, HY3D-Part format)
2. **Checkpoints**: `checkpoints/TRELLIS-text-xlarge/` + `checkpoints/TRELLIS-image-large/`
3. **Vinedresser3D**: Configured in config yaml
4. **Conda env**: `vinedresser3d` (pipeline + TRELLIS), `qwen_test` (image editing)

```bash
pip install numpy trimesh tqdm pyyaml scipy pillow openai plyfile open3d scikit-learn imageio
pip install torch torchvision xformers
```

### Option A: Local Deployment (zero API cost)

Uses locally deployed Qwen3.5-27B (VLM) + Qwen-Image-Edit-2511 (image edit).

```bash
# Terminal 1: Start VLM server (SGLang, port 8000)
conda activate vinedresser3d
bash scripts/tools/launch_local_vlm.sh

# Terminal 2: Start image edit server (diffusers, port 8001)
conda activate qwen_test
python scripts/tools/image_edit_server.py --gpu 2

# Terminal 3: Run pipeline
conda activate vinedresser3d
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --tag v1
```

### Option B: API Deployment (Gemini)

Uses Gemini 2.5 Flash via API. Requires API key in config.

```bash
# Full pipeline
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/default.yaml --tag v1

# Pre-generate 2D edits in parallel (can run on CPU machine)
python scripts/run_2d_edit.py --config configs/default.yaml \
    --tag v1 --workers 16
```

### Common Operations

```bash
# Run specific steps only (reads previous steps' cache)
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --steps 3 4 --tag v1

# Cost estimation (dry run, API backend only)
python scripts/run_pipeline.py --config configs/default.yaml --dry-run

# Resume after interruption (automatic via manifest)
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --tag v1

# Run standalone 2D editing with local server
python scripts/run_2d_edit.py --config configs/local_sglang.yaml \
    --tag v1 --workers 1
```

### Tag-Based Experiment Isolation

Use `--tag` to isolate experiment outputs. Each tag creates separate directories:

```bash
# Experiment v1
python scripts/run_pipeline.py --tag v1   # → 2d_edits_v1/, mesh_pairs_v1/, edit_results_v1.jsonl

# Experiment v2 (different config, same data)
python scripts/run_pipeline.py --tag v2   # → 2d_edits_v2/, mesh_pairs_v2/, edit_results_v2.jsonl
```

Steps 1-2 (semantic labels + edit specs) are shared across tags. Steps 3-6 outputs are tag-isolated.

---

## Key Features

### Edit Type Routing

| Edit Type | S1 (Structure) | S2 (Texture) | Mask | cfg |
|-----------|---------------|-------------|------|-----|
| **Deletion** | Modification (fills hole) | Full repaint | Dynamic (radius 2-4) | 3.0-7.0 |
| **Modification** | Flow Inversion + Repaint | Full repaint | Tight + 1-voxel dilation | 7.5 |
| **Global** | Skipped (shape preserved) | S2 repaint only | Full 64^3 | 5.0 |
| **Addition** | N/A (swap from deletion) | N/A | N/A | N/A |

### Large Part Auto-Promotion

When the edit part covers >40% of SLAT voxels, Deletion/Modification is automatically promoted to Global (TextureOnly).

### Contact-Aware Soft Mask

Dynamic Gaussian blur sigma based on contact ratio between edited and preserved geometry:
- `s1_sigma = 1.5 + contact_ratio * 4.0`
- `s2_sigma = 2.0 + contact_ratio * 10.0`

---

## Cache & Data Flow

Each step reads the previous step's cache:

```
Step 1 → cache/phase0/semantic_labels.jsonl           (shared, no tag)
Step 2 → cache/phase1/edit_specs.jsonl                 (shared, no tag)
Step 3 → cache/phase2_5/2d_edits_{tag}/               (tag-isolated)
Step 4 → cache/phase2_5/edit_results_{tag}.jsonl       (tag-isolated)
         outputs/mesh_pairs_{tag}/{edit_id}/            (tag-isolated)
Step 5 → cache/phase2_5/phase3_{tag}/vlm_scores.jsonl  (tag-isolated)
Step 6 → outputs/edit_pairs_{tag}.jsonl                 (tag-isolated)
```

You can skip steps and the pipeline will read from cache:
```bash
# Only run Step 4 (reads specs from Step 2, 2D edits from Step 3)
python scripts/run_pipeline.py --steps 4 --tag v1
```

---

## Project Structure

```
partcraft/                          # Core library
├── phase0_semantic/
│   ├── labeler.py                  # VLM labeling (semantic.json + 150 views)
│   └── catalog.py                  # Global Part Catalog index
├── phase1_planning/
│   ├── planner.py                  # EditSpec generation (del/add/mod/global)
│   └── enricher.py                 # VLM enrichment
├── phase2_assembly/
│   └── trellis_refine.py           # TRELLIS Flow Inversion + Repaint
├── phase3_filter/
│   └── filter.py                   # Quality metrics
└── phase4_filter/
    └── instruction.py              # Instruction templates

scripts/                            # Pipeline scripts
├── run_pipeline.py                 # Main unified pipeline (run this)
├── run_2d_edit.py                  # Standalone parallel 2D editing
├── prerender.py                    # Blender rendering + SLAT encoding (one-time)
├── prepare_partobjaverse.py        # Data preparation (one-time)
├── tools/
│   ├── image_edit_server.py        # Qwen-Image-Edit HTTP server (qwen_test env)
│   └── launch_local_vlm.sh        # SGLang VLM launcher
└── vis/
    └── render_gs_pairs.py          # Side-by-side comparison video

configs/
├── default.yaml                    # API backend (Gemini)
└── local_sglang.yaml              # Local backend (SGLang + diffusers)
```

---

## Output Structure

```
outputs/partobjaverse_tiny/
├── cache/
│   ├── phase0/semantic_labels.jsonl       # Step 1 (shared)
│   ├── phase1/edit_specs.jsonl            # Step 2 (shared)
│   └── phase2_5/
│       ├── edit_results_{tag}.jsonl       # Step 4 results
│       ├── 2d_edits_{tag}/               # Step 3 edited images
│       ├── phase3_{tag}/                 # Step 5 quality scores
│       └── debug_masks/                  # Mask visualizations
├── mesh_pairs_{tag}/
│   └── {edit_id}/
│       ├── before.ply, after.ply         # Gaussian Splatting PLY
│       └── before_slat/, after_slat/     # SLAT (feats.pt + coords.pt)
└── edit_pairs_{tag}.jsonl                # Final dataset (Step 6)
```

---

## Configuration

### `configs/local_sglang.yaml` (Local Deployment)

```yaml
phase0:
  vlm_backend: "local"
  vlm_model: "/path/to/Qwen3.5-27B"
  vlm_base_url: "http://localhost:8000/v1"
  vlm_api_key: "dummy"

phase2_5:
  image_edit_backend: "local_diffusers"
  image_edit_base_url: "http://localhost:8001"
  num_edit_views: 8
```

### `configs/default.yaml` (API Deployment)

```yaml
phase0:
  vlm_backend: "api"
  vlm_model: "gemini-2.5-flash"
  vlm_base_url: "https://..."
  vlm_api_key: "your-key"

phase2_5:
  image_edit_model: "gemini-2.5-flash-image"
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Pre-encoded SLAT not found` | Prerender not run | Run `scripts/prerender.py` first |
| `ATTN_BACKEND` error | xformers not set | Prefix command with `ATTN_BACKEND=xformers` |
| `Image edit server not reachable` | Server not started | Start `image_edit_server.py` in `qwen_test` env |
| `BrokenPipeError` in edit server | Client timeout | Use `--workers 1` (auto-forced for local backend) |
| Empty mask | Part mesh too small | Check `debug_masks/` visualizations |
| Large part artifacts | Part >40% of object | Auto-promoted to Global (built-in) |
| `No module named 'trellis'` | Wrong Vinedresser path | Check `phase2_5.vinedresser_path` |
