# PartCraft3D

Industrial-scale data pipeline for generating native 3D editing training pairs from part-level datasets.

## Overview

PartCraft3D takes part-segmented 3D assets and programmatically generates large-scale **(before_3D, after_3D, edit_instruction)** training triplets through part-level manipulation.

**Core idea**: Editing = combinatorial operations on parts (add, remove, modify). One disassembly yields both a deletion and an addition pair.

---

## Pipeline Architecture (6-Step)

```
Prerequisite (one-time, GPU)
    pack_prerender_npz.py → data/{dataset}/images/ + mesh/
        Source mesh (GLB) aligned to VD space via transforms.json
    prerender.py → Blender 150 views + SLAT encoding

Step 1: Semantic Labeling (VLM, ~4K tok/obj)
    4-view orthogonal VLM labeling + edit prompt enrichment
    Labels derived from NPZ split_mesh (no semantic.json required)
    → cache/phase0/semantic_labels_{tag}.jsonl

Step 2: Edit Planning (CPU, 0 tokens)
    Per-part deletion / addition / modification + global edits
    → cache/phase1/edit_specs_{tag}.jsonl

Step 3: 2D Image Editing (VLM / local diffusers)
    Plain input view + constrained prompt → edited reference image
    → cache/phase2_5/2d_edits_{tag}/{edit_id}_edited.png

Step 4: 3D Editing — TRELLIS (GPU, main workload)
    ├─ Deletion:     via Modification, S1 fills hole (dynamic mask/cfg)
    ├─ Modification:  Flow Inversion + Repaint, contact-aware soft mask
    ├─ Global:        TextureOnly — S1 skipped, S2 repaint changes texture
    └─ Addition:      swap before/after from deletion pair (no inference)
    → mesh_pairs_{tag}/{edit_id}/before.ply, after.ply

Step 5: Quality Scoring (VLM)
    4-view rendering → VLM scores (execution, localization, preservation)
    → cache/phase3/vlm_scores_{tag}.jsonl

Step 6: Export (CPU, 0 tokens)
    Instruction variants + final dataset assembly
    → edit_pairs_{tag}.jsonl
```

> See [PIPELINE.md](PIPELINE.md) for detailed architecture diagram, token cost breakdown, and scale projections.

---

## Quick Start

### Prerequisites

1. **Data**: `data/{dataset}/images/` + `data/{dataset}/mesh/` (prerendered NPZ files)
2. **SLAT**: Pre-encoded in `{vinedresser_path}/outputs/slat/` (via `prerender.py`)
3. **Checkpoints**: `checkpoints/TRELLIS-text-xlarge/` + `checkpoints/TRELLIS-image-large/`
4. **Vinedresser3D**: Path configured in config yaml (`phase2_5.vinedresser_path`)
5. **Conda env**: `vinedresser3d` (pipeline + TRELLIS), `qwen_test` (VLM + image editing servers)

```bash
pip install numpy trimesh tqdm pyyaml scipy pillow openai plyfile open3d scikit-learn imageio
pip install torch torchvision xformers
```

### Data Preparation

Pack source mesh + VD prerender into NPZ format:

```bash
# One-time: pack source mesh (GLB) aligned to VD coordinate space
python scripts/pack_prerender_npz.py --config configs/local_sglang.yaml

# Force re-pack (e.g. after fixing alignment)
python scripts/pack_prerender_npz.py --config configs/local_sglang.yaml --force
```

The packing step applies the exact coordinate transform from VD's `transforms.json`:
1. **Blender GLB axis conversion**: Y-up → Z-up `(x, y, z) → (x, -z, y)`
2. **Blender normalization**: `vertex = (blender_vertex + offset) * scale`

This ensures source mesh face ordering (matching `instance_gt`) is preserved while vertex positions align with VD's coordinate space.

---

## Running Modes

### Streaming Mode (recommended)

Each object is processed through the full chain (enrich → plan → 2D edit → 3D edit) before moving to the next. Supports resume on interruption.

```bash
# Terminal 1: Start VLM server (SGLang, port 8002)
conda activate qwen_test
VLM_PORT=8002 bash scripts/tools/launch_local_vlm.sh

# Terminal 2: Start image edit server (diffusers, port 8001)
conda activate qwen_test
python scripts/tools/image_edit_server.py --gpu 2

# Terminal 3: Run pipeline (streaming mode)
conda activate vinedresser3d
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --streaming --tag v1
```

Each `--tag` produces independent outputs with fresh VLM enrichment. Resume is automatic — re-running with the same tag skips already-completed objects.

### Multi-GPU Parallel Streaming

Partition objects across workers, each on a different GPU. Workers share VLM/image-edit servers but write to separate output files (`_w0.jsonl`, `_w1.jsonl`) to avoid conflicts.

```bash
# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --streaming --tag v1 \
    --num-workers 4 --worker-id 0 &

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --streaming --tag v1 \
    --num-workers 4 --worker-id 1 &

# Worker 2 on GPU 3
CUDA_VISIBLE_DEVICES=3 ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --streaming --tag v1 \
    --num-workers 4 --worker-id 2 &

# Worker 3 on GPU 4
CUDA_VISIBLE_DEVICES=4 ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --streaming --tag v1 \
    --num-workers 4 --worker-id 3 &
```

> GPU 2 is reserved for the image edit server in this example.

### Batch Mode (step-by-step)

Traditional per-step batch processing. Useful for debugging or running steps separately:

```bash
# Full pipeline (all 6 steps sequentially)
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --tag v1

# Run specific steps only (e.g. 3D editing + quality)
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --steps 4 5 --tag v1

# Cost estimation (dry run)
python scripts/run_pipeline.py --config configs/default.yaml --dry-run
```

### API Deployment (Gemini)

Uses Gemini 2.5 Flash via API for both VLM and image editing. Requires API key in config.

```bash
python scripts/run_pipeline.py \
    --config configs/default.yaml --streaming --tag v1
```

### Hybrid Mode (local GPU + remote API)

Local image editing (Qwen-Image-Edit) + remote VLM (Gemini API):

```bash
python scripts/run_pipeline.py \
    --config configs/hybrid_streaming.yaml --streaming --tag v1
```

---

## Configuration

### `configs/local_sglang.yaml` (Full Local — Zero API Cost)

```yaml
data:
  image_npz_dir: "data/partobjaverse_tiny/images"
  mesh_npz_dir: "data/partobjaverse_tiny/mesh"
  shards: ["00"]
  output_dir: "outputs/partobjaverse_tiny"

phase0:
  vlm_backend: "local"
  vlm_model: "/path/to/Qwen3.5-27B"
  vlm_base_url: "http://localhost:8002/v1"
  vlm_api_key: "dummy"

phase2_5:
  vinedresser_path: "/path/to/Vinedresser3D"
  image_edit_backend: "local_diffusers"
  image_edit_base_url: "http://localhost:8001"
  image_edit_workers: 2          # concurrent requests to edit server
  num_edit_views: 8              # views for 2D editing

pipeline:
  attn_backend: "xformers"       # or "flash-attn"
```

### Multi-GPU Config Notes

No special config changes needed for multi-GPU. All GPU routing is via CLI args + `CUDA_VISIBLE_DEVICES`:

| Component | GPU Assignment | Notes |
|-----------|---------------|-------|
| VLM server (SGLang) | Dedicated GPU | Set in `launch_local_vlm.sh` |
| Image edit server | Dedicated GPU | `--gpu N` in `image_edit_server.py` |
| Pipeline workers | 1 GPU each | `CUDA_VISIBLE_DEVICES=N` per worker |

For remote services, update URLs in config:
```yaml
phase0:
  vlm_base_url: "http://<remote-ip>:8002/v1"
phase2_5:
  image_edit_base_url: "http://<remote-ip>:8001"
```

---

## CLI Reference

| Argument | Description |
|---|---|
| `--config PATH` | Config YAML file |
| `--streaming` | Streaming mode (per-object full chain) |
| `--tag NAME` | Experiment tag for output isolation |
| `--limit N` | Process first N objects only |
| `--num-workers N` | Multi-GPU parallel worker count |
| `--worker-id K` | This worker's ID (0-indexed) |
| `--steps 3 4 5` | Run specific steps only (batch mode) |
| `--seed N` | Random seed for TRELLIS (default: 1) |
| `--force` | Force re-run, overwrite cached results |
| `--edit-ids id1 id2` | Process specific edit IDs only |
| `--no-2d-edit` | Skip 2D image editing (text-only TRELLIS) |
| `--debug` | Save debug files (masks, views, enricher ortho images) |
| `--dry-run` | Cost estimation only |

---

## Data Flow & Paths

All inputs come from preprocessed NPZ files, all outputs go to `outputs/`:

```
data/{dataset}/
├── source/                             # Original data (used by pack_prerender_npz)
│   ├── mesh.zip                        # Source GLB meshes (correct face ordering)
│   ├── instance_gt.zip                 # Per-face part labels (matches source mesh)
│   └── semantic.json                   # Part label names
├── images/{shard}/{obj_id}.npz         # 150 prerendered views + split_mesh.json
└── mesh/{shard}/{obj_id}.npz           # Per-part meshes (PLY, in VD coordinate space)

outputs/{dataset}/
├── cache/
│   ├── phase0/semantic_labels_{tag}.jsonl    # Step 1: enriched part descriptions
│   ├── phase1/edit_specs_{tag}.jsonl         # Step 2: edit specifications
│   └── phase2_5/
│       ├── edit_results_{tag}.jsonl          # Step 4: 3D edit results
│       ├── 2d_edits_{tag}/                  # Step 3: edited reference images
│       └── phase3_{tag}/                    # Step 5: quality scores
├── mesh_pairs_{tag}/
│   └── {edit_id}/
│       ├── before.ply, after.ply            # Gaussian Splatting PLY
│       └── before_slat/, after_slat/        # SLAT (feats.pt + coords.pt)
├── vis_masks/{tag}/                         # Mask debug visualizations
└── edit_pairs_{tag}.jsonl                   # Step 6: final dataset
```

---

## Edit Types

| Edit Type | S1 (Structure) | S2 (Texture) | Mask | cfg |
|-----------|---------------|-------------|------|-----|
| **Deletion** | Modification (fills hole) | Full repaint | Dynamic (radius 2-4) | 3.0-7.0 |
| **Modification** | Flow Inversion + Repaint | Full repaint | Tight + 1-voxel dilation | 7.5 |
| **Global** | Skipped (shape preserved) | S2 repaint only | Full 64^3 | 5.0 |
| **Addition** | N/A (swap from deletion) | N/A | N/A | N/A |

**Large Part Auto-Promotion**: When the edit part covers >40% of SLAT voxels, Deletion/Modification is automatically promoted to Global (TextureOnly).

**Contact-Aware Soft Mask**: Dynamic Gaussian blur sigma based on contact ratio between edited and preserved geometry.

---

## Coordinate Space Pipeline

Source mesh (GLB, Y-up) undergoes the following transforms to reach SLAT space:

```
Source GLB (Y-up)
    │
    │ pack_prerender_npz.py: _align_source_to_vd()
    │   1. Blender axis conversion: (x, y, z) → (x, -z, y)
    │   2. Normalize: (vertex + offset) * scale    [from transforms.json]
    ▼
VD Space (Z-up, [-0.5, 0.5]³, centered at origin)
    │
    │ build_part_mask(): _voxelize_combined()
    │   1. HY3D→VD: (v - hy3d_center) * scale_factor + vd_center
    │      (identity when NPZ already in VD space)
    │   2. VD→SLAT axis reorder: (x, y, z) → (x, -z, y) → clip to [-0.5, 0.5]
    ▼
SLAT Space (64³ voxel grid)
    │
    │ _align_masks_to_slat(): KNN re-projection
    ▼
SLAT-aligned mask (64³ bool tensor)
```

Key: `transforms.json` (from VD Blender prerender) records the exact `scale` and `offset` used to normalize the source mesh. The packing step applies this same transform to the source mesh so that part meshes in the NPZ are already in VD coordinate space.

---

## Project Structure

```
partcraft/                          # Core library
├── phase0_semantic/
│   ├── labeler.py                  # VLM labeling (fallback path)
│   └── catalog.py                  # Global Part Catalog index
├── phase1_planning/
│   ├── planner.py                  # EditSpec generation (del/add/mod/global)
│   └── enricher.py                 # VLM enrichment (orthogonal 4-view)
├── phase2_assembly/
│   └── trellis_refine.py           # TRELLIS Flow Inversion + Repaint
├── phase3_filter/
│   └── vlm_filter.py              # VLM quality scoring
├── io/
│   └── partcraft_loader.py        # NPZ dataset loader + pack_prerender
└── utils/
    ├── config.py                   # Config loading + path resolution
    └── logging.py                  # Logging setup

scripts/                            # Pipeline scripts
├── run_pipeline.py                 # Main unified pipeline (batch + streaming)
├── run_2d_edit.py                  # Standalone parallel 2D editing
├── pack_prerender_npz.py           # Pack source mesh + VD prerender → NPZ
├── prerender.py                    # Blender rendering + SLAT encoding (one-time)
├── tools/
│   ├── image_edit_server.py        # Qwen-Image-Edit HTTP server
│   └── launch_local_vlm.sh        # SGLang VLM launcher
├── vis/
│   ├── visualize_masks.py          # Per-spec mask diagnostic (2D + 3D voxel)
│   ├── visualize_partobjaverse.py  # Part-level mesh visualization
│   ├── render_gs_pairs.py          # Side-by-side comparison video
│   └── visualize_edit_pair.py      # Before/after edit comparison
└── standalone/                     # Standalone per-phase scripts (for debugging)

configs/
├── default.yaml                    # API backend (Gemini)
├── local_sglang.yaml              # Full local backend (SGLang + diffusers)
└── hybrid_streaming.yaml          # Hybrid (local image edit + remote VLM)
```

---

## Visualization & Debugging

### Mask Diagnostics

Visualize the full mask chain (VLM labels → edit spec → voxel mask) for all specs:

```bash
python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml

# Single edit
python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml --edit-id del_000000

# Limit specs
python scripts/vis/visualize_masks.py --config configs/local_sglang.yaml --limit 10
```

Output: per-spec diagnostic image with rendered views (edit parts in red), 3-axis voxel projections (color-coded), and text info. HTML index at `outputs/.../vis_masks/{tag}/index.html`.

### Debug Mode

```bash
# Save mask projections, 2D edit views, enricher ortho images
python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --streaming --tag debug --limit 1 --debug
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Pre-encoded SLAT not found` | Run `scripts/prerender.py` first |
| FlashInfer `nvcc: not found` | Set `export CUDA_HOME=/usr/local/cuda` |
| `Image edit server not reachable` | Start `image_edit_server.py` in `qwen_test` env |
| Empty mask | Part mesh too small at 64³; check with `--debug` |
| Large part artifacts | Auto-promoted to Global (>40% coverage, built-in) |
| Mask misaligned with geometry | Re-run `pack_prerender_npz.py --force` to re-align |
| Port conflict on VLM server | Use `VLM_PORT=8002 bash scripts/tools/launch_local_vlm.sh` |
| Multi-GPU worker collision | Each worker writes `_wN.jsonl`; no lock needed |
