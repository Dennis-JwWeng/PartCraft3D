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

Step 4: 3D Editing (GPU, main workload)
    ├─ Deletion:     Direct GT mesh removal (no generation, trimesh PLY)
    ├─ Modification:  TRELLIS Flow Inversion + Repaint (Gaussian Splatting PLY)
    ├─ Global:        TextureOnly — S1 skipped, S2 repaint (Gaussian Splatting PLY)
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
2. **SLAT**: Pre-encoded in `data/slat/` (via `prerender.py`)
3. **Checkpoints**: `checkpoints/TRELLIS-text-xlarge/` + `checkpoints/TRELLIS-image-large/`
4. **Third-party**: `third_party/trellis/`, `third_party/encode_asset/`, `third_party/interweave_Trellis.py` (bundled)

### Installation

```bash
conda create -n partcraft3d python=3.10
conda activate partcraft3d

# 1. PyTorch + xFormers (must match your CUDA version, example for CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers

# 2. spconv (must match CUDA version)
pip install spconv-cu121>=2.3

# 3. All other dependencies
pip install -r requirements.txt

# Or install as editable package with all deps:
# pip install -e ".[trellis]"
```

For VLM / image editing servers (separate env):
```bash
conda create -n qwen_test python=3.10
conda activate qwen_test
pip install sglang[all] diffusers transformers accelerate
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

### Streaming Mode — `run_streaming.py` (recommended)

Each object is processed through the full chain (enrich → plan → 2D edit → 3D edit) before moving to the next. Supports resume on interruption. After streaming completes, run quality scoring + export via the batch script.

```bash
# Terminal 1: Start VLM server (SGLang, port 8002)
conda activate qwen_test
VLM_PORT=8002 bash scripts/tools/launch_local_vlm.sh

# Terminal 2: Start image edit server (FLUX.2-klein-9B, port 8001)
conda activate qwen_test
CUDA_VISIBLE_DEVICES=2 python scripts/tools/image_edit_server.py

# Terminal 3: Run streaming pipeline
conda activate partcraft3d
ATTN_BACKEND=xformers python scripts/run_streaming.py \
    --config configs/local_sglang.yaml --tag v1

# Terminal 3 (after streaming completes): Quality scoring + export
python scripts/run_pipeline.py \
    --config configs/local_sglang.yaml --steps 5 6 --tag v1
```

Each `--tag` produces independent outputs with fresh VLM enrichment. Resume is automatic — re-running with the same tag skips already-completed objects.

### Multi-GPU Parallel Streaming

Partition objects across workers, each on a different GPU. Workers share VLM/image-edit servers but write to separate output files (`_w0.jsonl`, `_w1.jsonl`) to avoid conflicts.

**Architecture overview**:
```
┌─────────────────────────────────────────────────────────────┐
│  Shared Services (start once, all workers connect via HTTP) │
│                                                             │
│  GPU 5: VLM Server (SGLang, port 8002)                     │
│  GPU 2: Image Edit Server (FLUX.2-klein, port 8001)        │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP API (stateless)
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│Worker 0 │   │Worker 1 │   │Worker 2 │  ...
│ GPU 0   │   │ GPU 1   │   │ GPU 3   │
│ _w0.jsonl│   │ _w1.jsonl│   │ _w2.jsonl│
└─────────┘   └─────────┘   └─────────┘
  obj 0,4,8..   obj 1,5,9..   obj 2,6,10..  (modulo partition)
```

**Step 1: Start shared services** (one-time, separate terminals):
```bash
# Terminal 1: VLM server
conda activate qwen_test
CUDA_VISIBLE_DEVICES=5 VLM_PORT=8002 bash scripts/tools/launch_local_vlm.sh

# Terminal 2: Image edit server
conda activate qwen_test
CUDA_VISIBLE_DEVICES=2 python scripts/tools/image_edit_server.py  # port 8001
```

**Step 2: Launch workers** (one per GPU):
```bash
conda activate partcraft3d

# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/run_streaming.py \
    --config configs/hybrid_streaming.yaml --tag v1 \
    --num-workers 4 --worker-id 0 &

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 ATTN_BACKEND=xformers python scripts/run_streaming.py \
    --config configs/hybrid_streaming.yaml --tag v1 \
    --num-workers 4 --worker-id 1 &

# Worker 2 on GPU 3
CUDA_VISIBLE_DEVICES=3 ATTN_BACKEND=xformers python scripts/run_streaming.py \
    --config configs/hybrid_streaming.yaml --tag v1 \
    --num-workers 4 --worker-id 2 &

# Worker 3 on GPU 4
CUDA_VISIBLE_DEVICES=4 ATTN_BACKEND=xformers python scripts/run_streaming.py \
    --config configs/hybrid_streaming.yaml --tag v1 \
    --num-workers 4 --worker-id 3 &
```

> GPU 2 and 5 are reserved for shared services in this example.

**Step 3: Merge worker outputs** (after all workers finish):

Multi-GPU streaming produces per-worker fragments (`_w0.jsonl`, `_w1.jsonl`, ...). Merge them before running quality scoring:

```bash
TAG=v1
OUT=outputs/partobjaverse_tiny

# Merge semantic labels, edit specs, and edit results
cat $OUT/cache/phase0/semantic_labels_${TAG}_w*.jsonl > $OUT/cache/phase0/semantic_labels_${TAG}.jsonl
cat $OUT/cache/phase1/edit_specs_${TAG}_w*.jsonl     > $OUT/cache/phase1/edit_specs_${TAG}.jsonl
cat $OUT/cache/phase2_5/edit_results_${TAG}_w*.jsonl > $OUT/cache/phase2_5/edit_results_${TAG}.jsonl
```

**Step 4: Quality scoring + export**:
```bash
python scripts/run_pipeline.py \
    --config configs/hybrid_streaming.yaml --steps 5 6 --tag v1
```

**Data safety guarantees** (no locks needed):
- **No data races**: Objects are deterministically partitioned by `index % num_workers == worker_id`. Each object is processed by exactly one worker.
- **Per-worker output files**: Each worker writes to its own `_w{id}.jsonl` files (labels, specs, results). No shared writes.
- **Shared 2D cache is safe**: `2d_edits_{tag}/` directory is shared, but edit IDs are unique per object (embedded obj_id hash), so workers write different files. Writes use atomic temp-file + rename.
- **Resume-safe**: Re-running with the same `--tag` and `--worker-id` skips completed objects. Safe to kill and restart any worker.
- **VLM/Image Edit servers**: Accessed via stateless HTTP API, naturally thread-safe. May become bottlenecks with >4 workers — consider deploying multiple server replicas if needed.

### Batch Mode — `run_pipeline.py` (step-by-step)

Traditional per-step batch processing. All objects go through each step before moving to the next. Supports cross-object swap modifications (addition from other objects) and selective step execution.

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

### Hybrid Mode (local GPU + remote API)

Local image editing (FLUX.2-klein-9B) + remote VLM (Gemini API):

```bash
# Streaming
ATTN_BACKEND=xformers python scripts/run_streaming.py \
    --config configs/hybrid_streaming.yaml --tag v1

# Batch
ATTN_BACKEND=xformers python scripts/run_pipeline.py \
    --config configs/hybrid_streaming.yaml --tag v1
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

### `run_streaming.py` (streaming mode)

| Argument | Description |
|---|---|
| `--config PATH` | Config YAML file |
| `--tag NAME` | Experiment tag for output isolation |
| `--limit N` | Process first N objects only |
| `--num-workers N` | Multi-GPU parallel worker count |
| `--worker-id K` | This worker's ID (0-indexed) |
| `--seed N` | Random seed for TRELLIS (default: 1) |
| `--no-2d-edit` | Skip 2D image editing (text-only TRELLIS) |
| `--debug` | Save debug files (masks, views, enricher ortho images) |

### `run_pipeline.py` (batch mode)

| Argument | Description |
|---|---|
| `--config PATH` | Config YAML file |
| `--tag NAME` | Experiment tag for output isolation |
| `--steps 3 4 5` | Run specific steps only (default: all) |
| `--limit N` | Process first N objects only |
| `--seed N` | Random seed for TRELLIS (default: 1) |
| `--workers N` | Parallel workers for 2D editing (default: 4) |
| `--force` | Force re-run, overwrite cached results |
| `--edit-ids id1 id2` | Process specific edit IDs only |
| `--no-2d-edit` | Skip 2D image editing (text-only TRELLIS) |
| `--edit-dir DIR` | Pre-generated 2D edits subdir |
| `--suffix STR` | Suffix for spec files (e.g. `_action`) |
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
│       ├── before.ply, after.ply            # Trimesh PLY (del/add) or GS PLY (mod/glb)
│       └── before_slat/, after_slat/        # SLAT features (mod/glb only, absent for del/add)
├── vis_masks/{tag}/                         # Mask debug visualizations
└── edit_pairs_{tag}.jsonl                   # Step 6: final dataset
```

---

## Edit Types

| Edit Type | Method | Output Format | Mask | Notes |
|-----------|--------|---------------|------|-------|
| **Deletion** | Direct GT mesh removal (`direct_delete_mesh`) | Trimesh PLY (vertices + faces + vertex colors), no SLAT | N/A | Assembles remaining parts from ground-truth NPZ mesh |
| **Modification** | TRELLIS Flow Inversion + Repaint | Gaussian Splatting PLY + SLAT (feats.pt + coords.pt) | Tight + 1-voxel dilation, cfg=7.5 | 2D edit image as conditioning |
| **Global** | TextureOnly (S1 skipped, S2 repaint) | Gaussian Splatting PLY + SLAT | Full 64^3, cfg=5.0 | Shape preserved, only texture changes |
| **Addition** | Swap before/after from deletion pair | Inherits deletion format (trimesh PLY, no SLAT) | N/A | No inference, copy + swap |

**Large Part Auto-Promotion**: When the edit part covers >40% of SLAT voxels, Modification is automatically promoted to Global (TextureOnly).

**Contact-Aware Soft Mask**: Dynamic Gaussian blur sigma based on contact ratio between edited and preserved geometry.

**Output format note**: Deletion/Addition pairs produce standard trimesh PLY (mesh with vertex colors). Modification/Global pairs produce Gaussian Splatting PLY (point cloud with SH coefficients + covariances) plus SLAT features. Downstream consumers should check for the presence of `before_slat/` to distinguish formats.

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
├── run_pipeline.py                 # Batch pipeline (step-by-step, all 6 steps)
├── run_streaming.py                # Streaming pipeline (per-object full chain)
├── pipeline_common.py              # Shared utilities (config, paths, constants)
├── run_2d_edit.py                  # Standalone parallel 2D editing
├── build_dataset.py                # Build training dataset JSON from pipeline outputs
├── pack_prerender_npz.py           # Pack source mesh + VD prerender → NPZ
├── prerender.py                    # Blender rendering + SLAT encoding (one-time)
├── tools/
│   ├── image_edit_server.py        # FLUX.2-klein-9B / Qwen image edit HTTP server
│   └── launch_local_vlm.sh        # SGLang VLM launcher
├── vis/
│   ├── render_gs_pairs.py          # Gaussian Splatting side-by-side comparison
│   ├── render_ply_pairs.py         # Trimesh PLY pair rendering (deletion/addition)
│   └── visualize_edit_pair.py      # Before/after edit comparison
└── standalone/                     # Standalone per-phase scripts (for debugging)

third_party/                        # Bundled external dependencies
├── trellis/                        # TRELLIS 3D generation/editing pipeline
├── interweave_Trellis.py           # Flow Inversion core algorithm
└── encode_asset/                   # Asset encoding utilities

data/                               # Input data (symlinks or actual files)
├── slat/                           # Pre-encoded SLAT ({obj_id}_feats.pt + coords.pt)
├── img_Enc/                        # Pre-rendered views + reference meshes
└── {dataset}/                      # NPZ dataset (images/ + mesh/)

checkpoints/                        # TRELLIS model weights
├── TRELLIS-text-xlarge/            # Text-conditioned 3D generation
└── TRELLIS-image-large/            # Image-conditioned 3D generation

configs/
├── default.yaml                    # API backend (Gemini)
├── local_sglang.yaml              # Full local backend (SGLang + diffusers)
└── hybrid_streaming.yaml          # Hybrid (local image edit + remote VLM)
```

---

## Building Training Dataset

After the pipeline completes, build a structured training dataset JSON:

```bash
# Build dataset (groups edits by object, pairs del/add as reverse operations)
python scripts/build_dataset.py --tag v2

# Also fix missing before_slat for mod/glb pairs (copies from data/slat/)
python scripts/build_dataset.py --tag v2 --fix-slat
```

Output: `outputs/{dataset}/dataset_{tag}.json` with structure:
```json
{
  "meta": {"total_objects": 200, "total_edits": 2131, "type_counts": {...}},
  "objects": {
    "obj_id": {
      "slat_feats": "data/slat/obj_id_feats.pt",
      "slat_coords": "data/slat/obj_id_coords.pt",
      "edits": [
        {"edit_id": "del_obj_id_000", "edit_type": "deletion", "prompt": "Remove the armrest", "reverse_id": "add_obj_id_000", ...},
        {"edit_id": "add_obj_id_000", "edit_type": "addition", "prompt": "Add the armrest", "reverse_id": "del_obj_id_000", ...},
        {"edit_id": "mod_obj_id_000", "edit_type": "modification", "prompt": "...", "has_slat_pair": true, ...}
      ]
    }
  }
}
```

---

## Visualization & Debugging

### Rendering Edit Pairs

```bash
# Render Gaussian Splatting pairs (modification/global edits)
python scripts/vis/render_gs_pairs.py --config configs/local_sglang.yaml --tag v1

# Render trimesh PLY pairs (deletion/addition edits)
python scripts/vis/render_ply_pairs.py --config configs/local_sglang.yaml --tag v1

# Before/after comparison for specific edit
python scripts/vis/visualize_edit_pair.py --edit-id mod_abc12345_001 --tag v1
```

### Debug Mode

```bash
# Save mask projections, 2D edit views, enricher ortho images
python scripts/run_streaming.py \
    --config configs/local_sglang.yaml --tag debug --limit 1 --debug
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
| `ModuleNotFoundError: spconv` | `pip install spconv-cu121` (match your CUDA version) |
| `ModuleNotFoundError: utils3d` | `pip install utils3d>=0.6` |

---

## Migration to a New Machine

The project has three external symlinks that must be resolved if migrating to a machine without the original paths:

| Symlink | Target | Size |
|---------|--------|------|
| `data/slat` | Vinedresser3D outputs | 191MB |
| `data/img_Enc` | Vinedresser3D outputs | 6.2GB |
| `checkpoints/*/ckpts/*.safetensors` | 3DEditFormer checkpoints | 7GB |

To make the project fully self-contained:
```bash
# Replace symlinks with actual files
cp -rL data/slat data/slat_real && rm data/slat && mv data/slat_real data/slat
cp -rL data/img_Enc data/img_Enc_real && rm data/img_Enc && mv data/img_Enc_real data/img_Enc

# For checkpoints, resolve all symlinks under checkpoints/
find checkpoints/ -type l -exec bash -c 'cp --remove-destination "$(readlink -f "$0")" "$0"' {} \;
```
