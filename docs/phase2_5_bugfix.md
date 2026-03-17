# Phase 2.5 Bug 修复文档

## 修改文件

- `partcraft/phase2_assembly/trellis_refine.py`
- `scripts/run_phase2_5.py`
- `scripts/vis/render_gs_pairs.py`

---

## Bug 1: Mask 覆盖 84-94% 的 64³ 空间，导致大面积修改和模型破损

### 问题

`_compute_editing_region()` 中 `mask = mask | ~bbox_preserved` 将 preserved 部件 bounding box 之外的**所有**空间标记为可编辑。64³ = 262,144 个体素中，Final Mask 覆盖 22-25 万个（84-94%），即使只编辑一个几十体素的小按钮。

三种编辑类型均受影响：

| 类型 | 问题 |
|------|------|
| Modification | `~bbox_preserved` 占满整个网格 |
| Addition | `~preserved_parts` 覆盖 obj_vicinity 内除 preserved 外的所有空间 |
| Deletion | mask 大小正确，但坐标不对齐导致删错体素 |

### 根因

Vinedresser3D 中 PartField 直接在 SLAT 体素上聚类，`bbox_preserved` 紧密覆盖物体，`~bbox_preserved` 只是远离物体的空区域。

PartCraft3D 用 HY3D-Part mesh 体素化构建 mask，与 SLAT 坐标空间不一定对齐，导致 `bbox_preserved` 过小，`~bbox_preserved` 膨胀到几乎整个 64³。

### 修复

**1. 删除 `mask = mask | ~bbox_preserved`**

对 Modification 和 Addition 统一使用 `_compute_editing_region()`，通过 KNN 邻域扩展代替 `~bbox_preserved`，扩展范围限制在 SLAT bbox + pad 体素内。

- Modification: pad=3
- Addition: pad=5（需要更多空间生长新部件）
- Deletion: 直接使用 `edit_parts`（mask 大小本身正确）

**2. 新增 `_align_masks_to_slat()` — SLAT 坐标对齐**

将 mesh 体素化的 mask 投射到 SLAT 坐标上：
1. 查询每个 SLAT 体素在 mesh-voxelized grid 中的标签
2. 对未分配的 SLAT 体素用 KNN 找最近已标记体素赋标签
3. 从 SLAT 坐标重建 64³ mask

确保 mask 与 SLAT 体素完全对齐，与 VD 原版 PartField 行为一致。

**3. 清理无用变量**

移除不再需要的 `edit_bboxes`、`preserved_bboxes` 64³ tensor 构建。

### 预期效果

| Edit | 修复前 Final Mask | 修复后 Final Mask |
|------|------------------|------------------|
| mod_000000 (camera body) | 247,596 (94.4%) | ~6,000-8,000 (~3%) |
| mod_000002 (小按钮) | 222,551 (84.9%) | ~500-1,000 (~0.4%) |
| mod_000003 (镜头玻璃) | 221,760 (84.6%) | ~100-300 (~0.1%) |

---

## Bug 2: Structure / Appearance Prompt 未分离

### 问题

Vinedresser3D 通过 `decompose_prompt()` 调用 LLM 将描述分解为：
- S1 (structure): 去除颜色/纹理/材质形容词，只保留结构
- S2 (appearance): 去除形状形容词，只保留外观

PartCraft3D 的 `build_prompts_from_spec()` 中：
```
ori_s1_cpl = ori_s2_cpl = object_desc   # 完全相同
new_s1_cpl = new_s2_cpl = after_desc    # 完全相同
```

S1 flow model（控制结构）收到颜色/材质变化信号 → 不必要地修改结构。S2 flow model（控制外观）收到形状变化信号 → 外观被形状信号干扰。

### 修复

新增 `_decompose_local()` 和 `_strip_words()` 函数，用规则匹配代替 LLM 调用：
- `_APPEARANCE_WORDS`: 颜色、纹理、材质相关形容词集合
- `_STRUCTURE_WORDS`: 形状、尺寸、几何相关形容词集合
- S1 = 原文去除外观词 → 只保留结构描述
- S2 = 原文去除结构词 → 只保留外观描述

示例（编辑 Button: "silver → red, flat → dome-shaped"）：

| 通道 | 修复前 | 修复后 |
|------|-------|-------|
| S1 neg | 完整 object_desc | "flat, circular metal button" |
| S1 pos | 完整 after_desc | "dome-shaped button" |
| S2 neg | 完整 object_desc | "silver metal button" |
| S2 pos | 完整 after_desc | "red, plastic button" |

无需修改 Phase 0/1，无需额外 API 调用。

---

## Bug 3: 2D 图像编辑不一致影响 DINOv2 Conditioning

### 问题

VLM (Gemini) 编辑 2D 图片时：
1. 修改了整个物体而非仅目标部件（全前景 diff 高达 20+/255）
2. 部分视角添加了背景/环境（uniform area 从 100% 降到 72%）

DINOv2 编码全图特征并平均 → `img_cond` 携带非目标区域的变化 → TRELLIS image conditioning 推动 3D 全局修改。

### 修复

**1. VLM Prompt 收紧**

```diff
- "Edit the image provided with the editing prompt: {edit_prompt}"
+ "This is a 3D rendered object on a plain background.
+  Edit ONLY the {editing_part} part of this object.
+  Keep the background completely unchanged.
+  Keep all other parts of the object exactly as they are.
+  Editing instruction: {edit_prompt}"
```

**2. 背景合成保护 — `_composite_edit()`**

用原始渲染的前景 mask 做合成：
- 从原图提取前景 mask（非白色像素）
- MaxFilter 膨胀 + GaussianBlur 柔化边缘
- 编辑后的前景 + 原始背景合成

**3. DINOv2 特征残差 — `encode_multiview_cond()` 重写**

```python
feat_cond = feat_ori + edit_strength × (feat_edited - feat_ori)
```

- `edit_strength=1.0`: 直接用编辑特征（旧行为）
- `edit_strength=0.5`（默认）: 只取编辑增量的一半，抑制全局变化
- 可通过 config `phase2_5.edit_strength` 调节

接口变化：
- `obtain_edited_images()` 返回 `(original_images, edited_images)` 元组
- `encode_multiview_cond()` 新增 `original_images` 和 `edit_strength` 参数

---

## Bug 4: 小部件编辑警告

### 问题

数据集中 5.5% 的 modification 目标部件 < 100 面，在 64³ 网格下可能只有几个体素，编辑质量天然受限。

### 修复

在 `build_part_mask()` 中 SLAT 对齐后检查 edit 体素数，如果 < 10 个 SLAT 体素，输出 warning 日志：

```
WARNING: Edit part has only 3 SLAT voxels (< 10). Part may be too small for reliable editing at 64³ resolution.
```

---

## Bug 5: 输出路径双重嵌套

### 问题

`load_config()` 已将相对 `cache_dir` 解析为 `output_dir/cache/phase2_5`，但 `run_phase2_5.py` 又做了一次 join：

```
load_config:     "outputs/partobjaverse_tiny" + "cache/phase2_5"
                 → "outputs/partobjaverse_tiny/cache/phase2_5"              ✓

run_phase2_5.py: "outputs/partobjaverse_tiny" + "outputs/partobjaverse_tiny/cache/phase2_5"
                 → "outputs/partobjaverse_tiny/outputs/partobjaverse_tiny/cache/phase2_5"  ✗
```

### 修复

`run_phase2_5.py` 和 `render_gs_pairs.py` 中去掉冗余的 `output_dir / cache_dir` join，直接使用 `load_config` 已解析好的路径。

修复后路径：
```
outputs/partobjaverse_tiny/
├── cache/phase2_5/
│   ├── 2d_edits/
│   ├── debug_masks/
│   └── edit_results.jsonl
├── mesh_pairs/
└── logs/
```

---

## 测试命令

```bash
# 清理旧的双重嵌套路径
rm -rf outputs/partobjaverse_tiny/outputs

# 测试前 5 个 modification
ATTN_BACKEND=xformers python scripts/run_phase2_5.py \
  --config configs/partobjaverse.yaml \
  --type modification --limit 5 \
  --tag test_fix --no-resume

# 三种类型各测几个
ATTN_BACKEND=xformers python scripts/run_phase2_5.py \
  --config configs/partobjaverse.yaml \
  --edit-ids mod_000000 mod_000002 del_000000 del_000002 add_000001 \
  --tag test_fix --no-resume
```

### 验证要点

1. **日志检查**: `Final mask` 占 64³ 比例应从 84-94% 降到 < 5%
2. **日志检查**: `SLAT label check` 应显示 unassigned 数量和 KNN 分配结果
3. **日志检查**: `DINOv2 feature residual` 应显示 delta_norm 和 edit_strength
4. **路径检查**: 输出文件应在 `outputs/partobjaverse_tiny/cache/phase2_5/` 下，无双重嵌套
5. **结果对比**: `mesh_pairs_test_fix/` 中 after.ply 应比之前更局部化，非编辑区域保持完整
