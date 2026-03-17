# Vinedresser3D vs PartCraft3D 对比分析报告

## 1. 项目定位对比

| | Vinedresser3D | PartCraft3D |
|---|---|---|
| **目标** | 推理时 3D 编辑（单物体交互式编辑） | 离线大规模生产 3D 编辑训练数据 |
| **输入** | 1 个 GLB/PLY 文件 + 1 条编辑指令 | 240K 物体的 part-level 数据集 (HY3D-Part) |
| **输出** | 1 个编辑后的 3D 物体 (SLAT → Gaussian/Mesh) | ~300K 组 (before_mesh, after_mesh, instruction) 训练三元组 |
| **一句话** | "用扩散模型在潜空间里编辑一个 3D 物体" | "用真实部件拼装批量制造编辑训练数据" |
| **核心方法** | Flow Inversion + Repaint (生成式) | 组合拆装 (确定性) |

---

## 2. 管线流程对比

### Vinedresser3D — 8 步推理管线

```
GLB + 编辑指令
  → Blender 150views → DINOv2 → SLAT编码
  → PartField 3D分割 (K=3..8)
  → VLM 理解: 11个文本条件 + 编辑类型分类
  → VLM 语义定位: 选K + 选颜色 → 64³掩码
  → [Addition/Modification] VLM选视角 + Gemini Flash 2D图编辑
  → 5组 Flow Inversion + Interweave Repaint
  → VLM 从5个候选中选最佳
  → 导出 SLAT + 渲染视频
```

### PartCraft3D — 5 阶段数据管线

```
HY3D-Part 数据集 (240K 物体 × 42视角 × N部件)
  → Phase 0: VLM 语义标注 (部件命名 + core/peripheral 分类)
  → Phase 1: 组合编辑规划 (deletion/addition/swap/graft)
  → Phase 2: 网格拼装 (真实部件 PLY → before/after mesh)
  → Phase 3: 渲染 (before图免费复用 + after图Blender渲染)
  → Phase 4: 质量过滤 + 指令生成 + JSONL导出
```

### 流程图对比

```
         Vinedresser3D                          PartCraft3D
         ─────────────                          ───────────

    ┌──────────────────┐               ┌──────────────────────┐
    │  1个GLB + 1条指令  │               │  240K物体 part数据集   │
    └────────┬─────────┘               └──────────┬───────────┘
             │                                    │
    ┌────────▼─────────┐               ┌──────────▼───────────┐
    │ Blender渲染150张   │               │ Phase 0: VLM标注      │
    │ DINOv2特征提取     │               │ 渲染图+掩码→部件语义    │
    │ SLAT编码          │               │ (已有42张渲染图,免渲染)  │
    └────────┬─────────┘               └──────────┬───────────┘
             │                                    │
    ┌────────▼─────────┐               ┌──────────▼───────────┐
    │ PartField 3D分割   │               │ Phase 1: 编辑规划      │
    │ K=3..8 聚类        │               │ 组合生成编辑spec       │
    │ (需要GPU推理)      │               │ (纯CPU,秒级完成)       │
    └────────┬─────────┘               └──────────┬───────────┘
             │                                    │
    ┌────────▼─────────┐               ┌──────────▼───────────┐
    │ VLM理解+定位       │               │ Phase 2: 网格拼装      │
    │ 11个文本条件编码    │               │ 加载part PLY → 组装    │
    │ 64³布尔掩码生成    │               │ (trimesh, 秒级)       │
    └────────┬─────────┘               └──────────┬───────────┘
             │                                    │
    ┌────────▼─────────┐               ┌──────────▼───────────┐
    │ Flow Inversion     │               │ Phase 3: 渲染         │
    │ + Repaint ×5组合   │               │ before: NPZ直抽(免费)  │
    │ (5×完整ODE求解)    │               │ after: Blender渲染     │
    └────────┬─────────┘               └──────────┬───────────┘
             │                                    │
    ┌────────▼─────────┐               ┌──────────▼───────────┐
    │ VLM选最佳候选      │               │ Phase 4: 过滤+导出     │
    │ 渲染5×16=80张图    │               │ 质量检查→指令模板→JSONL │
    │ 导出1个SLAT        │               │ 导出 ~300K 训练对      │
    └──────────────────┘               └──────────────────────┘
```

---

## 3. 核心思想继承与创新

### 3.1 直接继承的思想

#### (1) 部件级编辑粒度 — 最核心的继承

**Vinedresser3D**: 用 PartField 将物体分割为语义部件，基于部件选择生成编辑掩码，在部件级别执行局部编辑。

**PartCraft3D**: 直接使用数据集自带的 part-level 分割，在部件级别规划和执行所有编辑操作。

```
Vinedresser3D 的洞察:
  "3D编辑的正确粒度是部件 (part)"

PartCraft3D 的继承:
  "既然编辑粒度是部件,那训练数据也应该在部件级别构造"
```

#### (2) 三种编辑类型分类

**Vinedresser3D** (`main.py` L72-79):
```python
if "Modification" in prompts["edit_type"]:
    prompts["edit_type"] = "Modification"
elif "Deletion" in prompts["edit_type"]:
    prompts["edit_type"] = "Deletion"
elif "Addition" in prompts["edit_type"]:
    prompts["edit_type"] = "Addition"
```

**PartCraft3D** (`planner.py` L25):
```python
edit_type: str   # "deletion" | "addition" | "modification" | "graft"
```

两者共享 Deletion / Addition / Modification 三分法。PartCraft3D 额外增加了 **Graft (移植)** 类型——跨物体跨类别的部件嫁接。

#### (3) VLM 驱动的语义理解

**Vinedresser3D**: 使用 Gemini 2.5 Pro 做 5 种 VLM 任务:
- 选择分割粒度 K (`select_K`)
- 定位编辑部件 (`select_editing_parts`)
- 理解编辑意图 (`obtain_overall_prompts`)
- 选择最佳编辑视角 (`select_img_to_edit`)
- 选择最佳编辑结果 (`select_the_best_edited_object`)

**PartCraft3D**: 使用 VLM 做 1 种核心任务:
- 为每个部件生成语义标签 + 描述 + core/peripheral 分类 (`label_single_object`)

| 对比 | Vinedresser3D | PartCraft3D |
|------|---------------|-------------|
| VLM 调用次数/物体 | 5-7 次 | 1 次 |
| 单次输入图片数 | 8-88 张 | 4-6 张全景 + N 张部件裁剪 |
| VLM 输出 | 自然文本 (需正则解析) | 结构化 JSON |
| 是否支持批量 | 否 (串行) | 是 (ThreadPoolExecutor) |

#### (4) 部件掩码可视化

**Vinedresser3D**: 分割渲染用 8 种颜色标注 K 个部件，VLM 按颜色选择 (`Seg_render_multiview_imgs_voxel`)。

**PartCraft3D**: `_highlight_part_on_view()` 函数:
- 暗化非目标部件像素 (×0.3 亮度)
- 目标部件加红色边框 (binary_dilation + 红色覆盖)
- 裁剪到部件 bbox + padding

两者都通过视觉高亮帮助 VLM 理解"哪个是目标部件"。

#### (5) 多视角渲染与视角选择

**Vinedresser3D**:
- 150 视角 Hammersley 球面采样 (编码用)
- 8 视角固定 yaw/pitch (展示用)
- VLM 从 24 视角中选最佳编辑视角

**PartCraft3D**:
- `select_diverse_views()`: farthest-point sampling 在 (azi, elev) 空间中选 N 个最分散的视角
- 数据集自带 42 视角，无需重新渲染

#### (6) 缓存与断点续跑

**Vinedresser3D**: 每步检查文件是否存在，跳过已完成:
```python
if os.path.exists(f"outputs/prompts/{name}.pkl"):
    prompts = pickle.load(...)
```

**PartCraft3D**: JSONL 追加写入 + 跳过已标注 obj_id:
```python
done_ids = {json.loads(line)["obj_id"] for line in open(output_path)}
if obj.obj_id in done_ids: continue
```

#### (7) 分阶段管线架构

两者都采用分阶段设计，每个阶段独立可跑。但 PartCraft3D 的阶段划分更工业化:

| 阶段 | Vinedresser3D | PartCraft3D |
|------|---------------|-------------|
| 阶段定义 | 线性脚本中的代码块 | 独立模块 + 独立脚本 |
| 阶段间数据 | 内存变量传递 + 散落文件 | 结构化 JSONL + 规范目录 |
| 跳过阶段 | 手动注释代码 | `--phases 1 2 4` 命令行参数 |
| 错误处理 | 无 (失败即停) | 错误日志 + 跳过失败项继续 |

---

### 3.2 PartCraft3D 的关键创新 (未继承部分)

#### (1) "反向思维" — 从生成式编辑到确定性组装

这是 PartCraft3D 最根本的创新。

**Vinedresser3D 的路径**:
```
原始3D → [扩散模型生成] → 编辑后3D
  (不确定性高, 质量依赖模型能力, 计算极重)
```

**PartCraft3D 的路径**:
```
真实部件A + 真实部件B → [拼装] → 编辑后3D
  (确定性, 100%几何准确, 计算极轻)
```

| 维度 | Vinedresser3D (生成式) | PartCraft3D (组装式) |
|------|----------------------|---------------------|
| 编辑结果来源 | 扩散模型凭空生成 | 真实部件网格拼接 |
| 几何精度 | 模型能力上限 | 100% (ground truth) |
| 纹理一致性 | 可能出现伪影 | 原始纹理 |
| 单次编辑耗时 | ~10 分钟 (GPU) | ~1 秒 (CPU) |
| 是否需要 GPU | 是 (多个大模型) | 否 (仅 VLM API) |

#### (2) 双向数据对 — "一次拆解,两份数据"

**Vinedresser3D**: 1 个编辑指令 → 1 个编辑结果。

**PartCraft3D**: 拆下 1 个部件 → 同时产生:
- **Deletion 对**: (完整物体 → 缺少该部件的物体)
- **Addition 对**: (缺少该部件的物体 → 完整物体)

```python
# planner.py: plan_deletion_addition()
# 同一次拆解同时生成两个 EditSpec
specs.append(EditSpec(edit_id=f"del_{counter}", edit_type="deletion", ...))
specs.append(EditSpec(edit_id=f"add_{counter}", edit_type="addition", ...))
```

数据效率翻倍。

#### (3) Before 图像零成本复用

**Vinedresser3D**: 编辑前后都需要重新渲染。before 视角用 `render_multiview_images()` 从 Gaussian Splatting 渲染。

**PartCraft3D**: HY3D-Part 已有 42 视角高质量渲染图, before 图直接从 NPZ 提取:
```python
# renderer.py: extract_before_images()
img_bytes = obj.get_image_bytes(vid)  # 直接读取已有渲染图
```

零渲染成本, 零 GPU 开销。

#### (4) 跨物体组合编辑 (Swap / Graft)

**Vinedresser3D**: 只能在单个物体上编辑，编辑内容由 2D 图像编辑模型引导生成。

**PartCraft3D**: 支持跨物体部件操作:
- **Swap**: 用 B 物体的椅腿替换 A 物体的椅腿 (同类别跨物体)
- **Graft**: 把 B 物体的翅膀嫁接到 A 物体上 (跨类别跨物体)

```python
# planner.py: plan_swap() - 同类别跨物体替换
candidates = catalog.get_swap_candidates(entry, max_candidates=5)

# planner.py: plan_graft() - 跨类别跨物体嫁接
unique_in_b = obj_cats[b] - obj_cats[a]  # B有但A没有的部件类别
```

这使得编辑多样性远超单物体编辑。

#### (5) 语义目录 (Part Catalog) — 全局知识索引

**Vinedresser3D**: 没有全局部件知识库, 每个编辑任务独立处理。

**PartCraft3D**: 构建全局语义目录:
```python
# catalog.py
class PartCatalog:
    entries: list[CatalogEntry]    # 所有部件
    by_category: dict[str, list]   # "leg" → [所有椅腿/桌腿/...]
    by_object: dict[str, list]     # obj_id → [该物体所有部件]
    object_descs: dict[str, str]   # obj_id → "一把红色办公椅"
```

目录支持:
- **类别归一化**: `"left_front_leg"` → `"leg"`, `"rear_wheel_01"` → `"wheel"`
- **跨物体部件检索**: 按类别查找所有同类部件
- **Swap 候选生成**: 自动找到可替换的同类部件

#### (6) 工业级质量过滤

**Vinedresser3D**: 仅靠 VLM 主观选择 "看起来最好的" 候选。

**PartCraft3D**: 5 项几何质量检查:

| 检查项 | 含义 | 阈值 |
|--------|------|------|
| `volume_reasonable` | 编辑前后体积比合理 | 0.1 ~ 10.0 |
| `not_fragmented` | 编辑后不过于碎裂 | ≤ 10 连通分量 |
| `has_geometry` | 编辑后有足够几何 | > 50 顶点 |
| `edit_nontrivial` | 编辑确实改变了网格 | edit_ratio ≥ 0.01 |
| `edit_not_total` | 不是完全替换 | edit_ratio ≤ 0.95 |

#### (7) 模板化指令生成

**Vinedresser3D**: 编辑指令由用户手动输入 (`--editing_prompt`)。

**PartCraft3D**: 自动生成多样化编辑指令:
```python
_DELETION_TEMPLATES = [
    "Remove the {part} from the {obj}",
    "Delete the {part}",
    "Take away the {part} of the {obj}",
    ...
]
```

每个编辑对生成 3+ 条不同表述的指令, 增加训练数据的语言多样性。

---

## 4. 技术栈对比

| 维度 | Vinedresser3D | PartCraft3D |
|------|---------------|-------------|
| **3D 表示** | SLAT (64³ 稀疏体素特征) | Trimesh (三角网格 PLY) |
| **生成模型** | TRELLIS Flow Matching (text + image) | 无 (确定性组装) |
| **分割** | PartField (运行时推理) | 数据集预分割 (HY3D-Part) |
| **特征提取** | DINOv2 ViT-L/14 | 无需 |
| **VLM** | Gemini 2.5 Pro (5-7 次/物体) | Gemini 2.5 Pro (1 次/物体) |
| **图像编辑** | Gemini 2.5 Flash Image | 无需 |
| **渲染** | Blender 3.0 + Gaussian Splatting | Blender (仅 after) + NPZ 直抽 (before) |
| **GPU 依赖** | 必需 (TRELLIS + PartField + DINOv2) | 仅 Phase 0 VLM API (可纯 CPU) |
| **并行** | 无 | ThreadPoolExecutor (Phase 0) |

---

## 5. 性能与规模对比

| 指标 | Vinedresser3D | PartCraft3D |
|------|---------------|-------------|
| **单次编辑耗时** | ~10-15 分钟 (A100 GPU) | Phase 0: ~5秒/物体 (API), Phase 1-4: ~1秒/对 (CPU) |
| **GPU 显存** | ~40GB (TRELLIS text-xlarge + image-large) | 0 (纯 CPU + API) |
| **VLM API 调用** | 5-7 次/编辑 × 8-88 张图/次 | 1 次/物体 × ~20 张图 |
| **可处理规模** | 逐个物体 (无批量) | 240K 物体 → ~300K 训练对 |
| **数据/编辑比** | 1:1 | 1:N (一个物体产生多对) |
| **容错** | 无 (中间失败需从头) | 错误跳过 + 断点续跑 |

---

## 6. 编辑能力对比

### Vinedresser3D 能做但 PartCraft3D 不能做的

| 能力 | 说明 |
|------|------|
| **自由形态编辑** | "把椅子变成沙发" — 大幅改变结构 |
| **纹理/材质修改** | "把红色椅子变成蓝色" — PartCraft3D 无法修改纹理 |
| **生成全新部件** | "给椅子加翅膀" — PartCraft3D 只能用已有部件 |
| **连续形变** | "把腿变长" — PartCraft3D 只有离散的添加/删除 |
| **用户意图理解** | 可以理解自然语言的模糊编辑指令 |

### PartCraft3D 能做但 Vinedresser3D 不擅长的

| 能力 | 说明 |
|------|------|
| **大规模数据生产** | 一次跑产出 300K 训练对 |
| **跨物体部件组合** | 椅子A的腿 + 椅子B的靠背 |
| **几何精确编辑** | 100% ground truth, 无生成噪声 |
| **组删除** | "删除所有椅腿" (同类别部件一起操作) |
| **自动质量控制** | 5 项几何检查 + 分数排序 |
| **批量指令生成** | 每对自动生成 3+ 条多样化指令 |

---

## 7. 两者的互补关系

```
                PartCraft3D                    Vinedresser3D
            (数据生产管线)                    (推理编辑系统)
                  │                                │
                  ▼                                │
    ┌─────────────────────────┐                    │
    │ 300K 组训练三元组         │                    │
    │ (before, after, instr)  │                    │
    └────────────┬────────────┘                    │
                 │                                 │
                 ▼                                 │
    ┌─────────────────────────┐                    │
    │ 训练原生3D编辑模型        │                    │
    │ (学习: 指令→3D变换)      │                    │
    └────────────┬────────────┘                    │
                 │                                 │
                 ▼                                 │
    ┌─────────────────────────┐    ┌───────────────▼──────────┐
    │ 原生3D编辑模型 (推理)    │    │ SLAT Repaint (推理)       │
    │ 直接在3D空间编辑         │    │ 在潜空间编辑              │
    │ (PartCraft3D训练的)     │    │ (需要每次跑Inversion)     │
    │                         │    │                          │
    │ 优势: 快速, 端到端      │    │ 优势: 灵活, 无需训练数据  │
    └─────────────────────────┘    └──────────────────────────┘
```

**核心洞察**: PartCraft3D 的终极目标是替代 Vinedresser3D。

- **Vinedresser3D** 在推理时做编辑 — 慢但灵活，每次编辑需要完整的 Inversion + Repaint 流程
- **PartCraft3D** 生产训练数据 → 训练一个**原生 3D 编辑模型** → 推理时一步到位，不再需要 Inversion/Repaint

这是经典的 "test-time compute ↔ training data" 的权衡: PartCraft3D 把 Vinedresser3D 的推理时计算量前置到了数据准备阶段。

---

## 8. 总结

| 维度 | 结论 |
|------|------|
| **继承关系** | PartCraft3D 继承了 Vinedresser3D "部件级编辑"的核心哲学和三种编辑类型分类 |
| **方法论转变** | 从"推理时生成式编辑"转变为"离线确定性数据生产" |
| **关键创新** | 双向数据对、跨物体组合、全局语义目录、零成本before图、工业级质量过滤 |
| **互补关系** | PartCraft3D 生产数据 → 训练模型 → 最终替代 Vinedresser3D 的重型推理流程 |
| **核心价值** | 以 1/1000 的计算成本, 产出 1000× 的训练数据量, 精度从"模型能力上限"变为"100% ground truth" |
