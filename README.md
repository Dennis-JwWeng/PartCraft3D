# PartCraft3D

Industrial-scale data pipeline for generating native 3D editing training pairs from part-level datasets.

## Overview

PartCraft3D takes part-segmented 3D assets and programmatically generates large-scale **(before_3D, after_3D, edit_instruction)** training triplets through part-level manipulation.

**Core idea**: Editing = combinatorial operations on parts (add, remove, modify). One disassembly yields both a deletion and an addition pair.

---

## Pipeline Architecture

```
Prerender (一次性, GPU)
    source/mesh.zip → Blender 150 views → Open3D voxelize → DINOv2 → SLAT
        ↓ outputs/img_Enc/{obj_id}/, outputs/slat/{obj_id}_*.pt

Phase 0: VLM Semantic Labeling (API)
    semantic.json + 150 views → VLM (Gemini) → 描述 + 编辑提示
        ↓ semantic_labels.jsonl

Phase 1: Edit Planning (CPU, 秒级)
    Part Catalog → edit specifications (del/add/mod)
        ↓ edit_specs.jsonl

Phase 2: Mesh Assembly (CPU, 多进程)
    del/add specs + watertight part meshes → before/after PLY pairs
        ↓ assembled_pairs.jsonl + mesh_pairs/

Phase 2.5: TRELLIS Generative Editing (GPU)
    mod specs + 预编码 SLAT → Ground-truth part mask
        → Multi-view 2D image editing (Gemini, 8 views)
        → Multi-view DINOv2 feature averaging
        → TRELLIS Flow Inversion + Repaint
        → Gaussian PLY + SLAT export
        ↓ edit_results.jsonl + mesh_pairs/

Visualization (GPU, 按需)
    SLAT → Gaussian → side-by-side turntable comparison video
        ↓ vis_compare/
```

---

## Quick Start

### Prerequisites

1. **数据**: `data/partobjaverse_tiny/` (200 objects, HY3D-Part format)
2. **Checkpoints**: `checkpoints/TRELLIS-text-xlarge/` + `checkpoints/TRELLIS-image-large/`
3. **Vinedresser3D**: `/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main`
4. **API Key**: `configs/partobjaverse.yaml` 中配置 Gemini API key
5. **Conda 环境**: PyTorch, xformers, open3d, trimesh 等

```bash
pip install numpy trimesh tqdm pyyaml scipy pillow openai plyfile open3d scikit-learn imageio
pip install torch torchvision xformers
```

### Full Pipeline

```bash
# 1. 统一预渲染 (GPU, 一次性, ~3-5 min/object)
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/prerender.py \
    --config configs/partobjaverse.yaml

# 2. 语义标注 (VLM API)
python scripts/run_phase0.py --config configs/partobjaverse.yaml

# 3. 编辑规划 (CPU, 秒级)
python scripts/run_phase1.py --config configs/partobjaverse.yaml

# 4. 网格组装 — deletion/addition (CPU)
python scripts/run_phase2.py --config configs/partobjaverse.yaml --workers 8

# 5. TRELLIS 3D 编辑 — modification/deletion/addition (GPU)
ATTN_BACKEND=xformers python scripts/run_phase2_5.py \
    --config configs/partobjaverse.yaml --resume

# 6. 可视化对比 (GPU, 按需)
ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
    --config configs/partobjaverse.yaml
```

---

## Current Status (200 objects)

| 阶段 | 状态 | 数量 |
|------|------|------|
| Prerender | 200/200 完成 | 150 views + SLAT per object |
| Phase 0 | 200/200 完成 | 语义标注 |
| Phase 1 | 200/200 完成 | 5463 edit specs (3157 mod + 1153 del + 1153 add) |
| Phase 2 | 待运行 | deletion + addition mesh pairs |
| Phase 2.5 | 待运行 | TRELLIS generative editing |

---

## Phase Details

### Prerender — 统一渲染与编码

一次性完成所有对象的渲染和 SLAT 编码，后续阶段直接复用。

```bash
# 完整预渲染
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/prerender.py \
    --config configs/partobjaverse.yaml

# 仅渲染 (不需要 GPU)
python scripts/prerender.py --config configs/partobjaverse.yaml --render-only

# 仅编码 SLAT
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/prerender.py \
    --config configs/partobjaverse.yaml --encode-only

# 指定对象
CUDA_VISIBLE_DEVICES=0 ATTN_BACKEND=xformers python scripts/prerender.py \
    --config configs/partobjaverse.yaml \
    --obj-ids 00aee5c2fef743d69421bb642d446a5b
```

**输出** (在 Vinedresser3D 的 `outputs/` 下):
- `img_Enc/{obj_id}/000.png..149.png` — 150 张 Blender 渲染图 (512x512)
- `img_Enc/{obj_id}/mesh.ply` — 归一化 mesh [-0.5, 0.5]^3
- `slat/{obj_id}_feats.pt, _coords.pt` — SLAT 编码

支持断点续传，自动跳过已完成的对象。

### Phase 0 — 语义标注

部件语义从 `semantic.json` 获取 (ground truth)，VLM 用预渲染 150 views 生成描述和编辑提示。

```bash
python scripts/run_phase0.py --config configs/partobjaverse.yaml

# 强制重新标注
python scripts/run_phase0.py --config configs/partobjaverse.yaml --force

# 限制数量
python scripts/run_phase0.py --config configs/partobjaverse.yaml --limit 10
```

**输出**: `data/partobjaverse_tiny/cache/phase0/semantic_labels.jsonl`

增量运行，自动跳过已标注的对象。

### Phase 1 — 编辑规划

根据语义描述生成三类编辑方案 (modification / deletion / addition)。

```bash
python scripts/run_phase1.py --config configs/partobjaverse.yaml
```

**输出**: `outputs/partobjaverse_tiny/cache/phase1/edit_specs.jsonl`

### Phase 2 — 网格组装

通过直接操作部件 mesh 生成 before/after PLY 对。仅处理 deletion 和 addition 类型。

```bash
python scripts/run_phase2.py --config configs/partobjaverse.yaml --workers 8
```

**输出**: `outputs/partobjaverse_tiny/mesh_pairs/{edit_id}/before.ply, after.ply`

### Phase 2.5 — TRELLIS 3D 编辑

纯编辑阶段，直接加载预编码 SLAT，使用多视角 DINOv2 特征平均进行结构一致的 3D 编辑。

```bash
# 全量运行
ATTN_BACKEND=xformers python scripts/run_phase2_5.py \
    --config configs/partobjaverse.yaml

# 分类型运行
ATTN_BACKEND=xformers python scripts/run_phase2_5.py \
    --config configs/partobjaverse.yaml --type modification --limit 200

# 断点续传
ATTN_BACKEND=xformers python scripts/run_phase2_5.py \
    --config configs/partobjaverse.yaml --resume

# A/B 实验: --tag 隔离输出到 mesh_pairs_{tag}/ 和 edit_results_{tag}.jsonl
ATTN_BACKEND=xformers python scripts/run_phase2_5.py \
    --config configs/partobjaverse.yaml --tag multiview8 --limit 5
```

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--type` | 全部 | 筛选编辑类型: modification / deletion / addition |
| `--limit N` | 1000 | 最多处理 N 条 edit specs |
| `--combs 0 1 ...` | 自动 | TRELLIS condition 组合索引 (默认只跑 combo 0) |
| `--seed N` | 1 | 随机种子 |
| `--no-2d-edit` | 启用 | 跳过 Gemini 2D 图像编辑 |
| `--resume` | 关闭 | 断点续传 (跳过已成功的 edit_id) |
| `--tag TAG` | 无 | 实验标签, 输出隔离到 `mesh_pairs_{TAG}/` |

**关键技术:**

- **Multi-view DINOv2 Conditioning**: 渲染 8 个视角 (赤道+仰角交替)，分别用 Gemini 编辑，DINOv2 特征平均后作为 TRELLIS image conditioning。相比单视角驱动，编辑后 3D 结构更一致。
- **单 Combination**: 默认只跑 combo 0 (complete description CFG, new↑ old↓)。之前跑 5 种但只用第一个，现在省去 4/5 的计算。Deletion 自动使用 unconditioned removal (cfg=0)。
- **输出精简**: 只导出 PLY + SLAT，视频/多视角渲染解耦到可视化工具。

**输出:**
- `mesh_pairs/{edit_id}/before.ply, after.ply` — Gaussian Splatting PLY
- `mesh_pairs/{edit_id}/before_slat/, after_slat/` — SLAT (feats.pt + coords.pt)
- `cache/phase2_5/edit_results.jsonl` — 结果清单
- `cache/phase2_5/2d_edits/` — 2D 编辑中间结果
- `cache/phase2_5/debug_masks/` — Mask 可视化

### Visualization — 对比渲染

从 SLAT 解码 Gaussian，渲染 side-by-side 转盘对比视频，附带编辑 prompt 文字叠加。

```bash
# 渲染所有 pairs 的对比视频
ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
    --config configs/partobjaverse.yaml

# 渲染指定 tag 的实验结果
ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
    --config configs/partobjaverse.yaml --tag multiview8

# 渲染指定 edit IDs
ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
    --config configs/partobjaverse.yaml --edit-ids mod_000001 del_000002

# 同时保存单独视角图片
ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
    --config configs/partobjaverse.yaml --save-views --num-views 16

# 跳过对比视频, 只渲染单独 before/after 视频
ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
    --config configs/partobjaverse.yaml --no-compare
```

**输出:**
- `vis_compare/{edit_id}.mp4` — Side-by-side 对比视频 (Before | After + prompt overlay)
- `mesh_pairs/{edit_id}/before.mp4, after.mp4` — 单独转盘视频

---

## Configuration

### `configs/partobjaverse.yaml` 关键配置

```yaml
data:
  image_npz_dir: "data/partobjaverse_tiny/images"
  mesh_npz_dir: "data/partobjaverse_tiny/mesh"
  shards: ["00"]
  output_dir: "outputs/partobjaverse_tiny"

phase0:
  vlm_model: "gemini-2.5-flash"
  vlm_base_url: "https://llm-api.mmchat.xyz/v1"
  max_workers: 8

phase1:
  core_categories: ["body", "torso", "base", "frame", "main", "head"]

phase2_5:
  vinedresser_path: "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main"
  image_edit_model: "gemini-2.5-flash-image"
  num_edit_views: 8           # Multi-view 2D editing viewpoints
  seed: 42
```

---

## Project Structure

```
partcraft/
├── io/
│   ├── hy3d_loader.py              # HY3D-Part NPZ 数据加载
│   └── export.py                   # 编辑对导出
├── phase0_semantic/
│   ├── labeler.py                  # VLM 标注 (semantic.json + 150 views)
│   └── catalog.py                  # 全局 Part Catalog 索引
├── phase1_planning/
│   ├── planner.py                  # EditSpec 生成 (del/add/mod)
│   └── enricher.py                 # VLM 增强描述
├── phase2_assembly/
│   ├── assembler.py                # 网格组装 (del/add)
│   ├── alignment.py                # 几何对齐
│   └── trellis_refine.py           # Phase 2.5: TRELLIS pipeline
├── phase3_filter/
│   └── filter.py                   # 质量 metrics
├── phase4_filter/
│   └── instruction.py              # 指令模板
└── utils/
    ├── config.py                   # YAML 配置加载
    └── logging.py                  # 日志

scripts/
├── run_all.py                      # 完整流水线
├── run_phase0.py                   # Phase 0: 语义标注
├── run_phase1.py                   # Phase 1: 编辑规划
├── run_phase2.py                   # Phase 2: 网格组装
├── run_phase2_5.py                 # Phase 2.5: TRELLIS 编辑
├── run_phase3.py                   # Phase 3: 质量筛选
├── prerender.py                    # 统一预渲染
├── encode_slat.py                  # 批量 SLAT 编码
├── prepare_partobjaverse.py        # 数据格式转换
├── run_2d_edit.py                  # 批量 2D 图像编辑
├── run_enrich.py                   # VLM 增强
└── vis/
    └── render_gs_pairs.py          # 对比视频渲染

configs/
├── default.yaml                    # 默认配置
├── partobjaverse.yaml              # PartObjaverse-Tiny 配置
├── partobjaverse_test.yaml         # 测试配置
└── local_vlm.yaml                  # 本地 VLM 配置
```

---

## Output Structure

```
outputs/partobjaverse_tiny/
├── mesh_pairs/                          # Phase 2 + 2.5 输出 (默认)
│   ├── del_000000/
│   │   ├── before.ply, after.ply        # Watertight mesh
│   │   ├── before_slat/, after_slat/    # SLAT (feats.pt + coords.pt)
│   │   └── before.mp4, after.mp4        # 转盘视频 (vis 生成)
│   └── mod_000000/
│       ├── before.ply, after.ply        # Gaussian Splatting PLY
│       └── before_slat/, after_slat/    # SLAT
├── mesh_pairs_multiview8/               # --tag multiview8 隔离输出
│   └── ...
├── vis_compare/                         # 对比视频
│   └── mod_000000.mp4                   # Side-by-side (Before | After)
├── cache/
│   ├── phase1/edit_specs.jsonl          # 5463 条编辑规划
│   ├── phase2/assembled_pairs.jsonl     # 网格组装结果
│   └── phase2_5/
│       ├── edit_results.jsonl           # TRELLIS 编辑结果
│       ├── edit_results_multiview8.jsonl # --tag 隔离结果
│       ├── 2d_edits/                    # 2D 编辑中间图 (cached)
│       └── debug_masks/                 # Mask 可视化
└── logs/
```

---

## Troubleshooting

| 问题 | 原因 | 解决 |
|------|------|------|
| `Pre-encoded SLAT not found` | 未运行 prerender | 先跑 `scripts/prerender.py` |
| `ATTN_BACKEND` 报错 | xformers 未设置 | 命令前加 `ATTN_BACKEND=xformers` |
| Phase 0 卡住 | VLM API 无响应 | API 调用已加 120s 超时，会自动重试 |
| Empty mask | Part mesh 太小或坐标变换问题 | 检查 `debug_masks/` 可视化 |
| 2D 编辑无图片 | Gemini API 配额/key | 检查 config 中 API key |
| `No module named 'trellis'` | Vinedresser3D 路径错误 | 检查 `phase2_5.vinedresser_path` |
