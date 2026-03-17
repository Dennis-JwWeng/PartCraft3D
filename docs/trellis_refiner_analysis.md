# TrellisRefiner 原理详解

TrellisRefiner 复用了 Vinedresser3D 的 TRELLIS Flow Inversion + Repaint 编辑管线，但替换了两个关键模块：用 HY3D-Part 真值分割替代 PartField，用 Phase 0/1 预计算 prompt 替代 VLM 生成。

## 整体流程（6步）

```
GLB → SLAT编码 → Gaussian解码 → 部件Mask → 2D图像编辑 → TRELLIS反演+重绘 → 导出PLY
```

---

## 核心代码详解

### 1. Prompt 分解 — `build_prompts_from_spec()` (L106-158)

VD 原版用 LLM 做 7 次 API 调用来生成/分解 prompt。这里用**规则替代 LLM**：

```python
def _decompose_local(desc):
    s1 = _strip_words(desc, _APPEARANCE_WORDS)  # 去掉颜色/材质词 → 只留结构描述
    s2 = _strip_words(desc, _STRUCTURE_WORDS)    # 去掉形状/大小词 → 只留外观描述
    return s1, s2
```

TRELLIS 有两个 stage：**S1 控制结构（shape）**、**S2 控制外观（texture）**。VD 用 LLM 把完整描述拆分成这两个通道，这里用预定义的词表做规则匹配（`_APPEARANCE_WORDS` 约 100 个颜色/材质词，`_STRUCTURE_WORDS` 约 60 个形状/几何词）。

生成的 prompt dict 包含 `ori_s1_cpl`/`new_s1_cpl`（编辑前/后的完整结构描述）、`ori_s2_cpl`/`new_s2_cpl`（完整外观描述），以及 part-level 对应物。

---

### 2. SLAT 加载 — `encode_object()` (L227-253)

```python
slat = sp.SparseTensor(
    feats=torch.load(feats_path),   # [N, 8] 稀疏体素特征
    coords=torch.load(coords_path), # [N, 4] 坐标 (batch_idx, x, y, z)
)
```

直接加载 `prerender.py` 预编码的 SLAT（Sparse Latent）。SLAT 是 TRELLIS 的核心表示：在 64³ 空间中约 10000 个稀疏体素，每个 8 维特征。这些特征经过 TRELLIS encoder 从多视角 DINOv2 特征编码而来。

---

### 3. 部件 Mask 构建 — `build_part_mask()` (L264-406)

这是**最复杂的模块**，替代了 VD 的 PartField 分割。核心逻辑：

#### a) 坐标空间对齐 (L291-305)

```python
# HY3D-Part 在 [-1,1] 空间，VD 在 [-0.5,0.5] 空间
scale_factor = vd_extent / hy3d_extent
part_verts_vd = (part_verts - hy3d_center) * scale_factor + vd_center
```

#### b) 部件体素化 (L312-348)

用 Open3D 的 `VoxelGrid.create_from_triangle_mesh_within_bounds` 把每个部件 mesh 体素化到 64³ 网格，分为 `edit_parts`（要编辑的部件）和 `preserved_parts`（保留的部件）。

#### c) SLAT 对齐 — `_align_masks_to_slat()` (L408-465)

```python
# Mesh 体素化的位置可能和 SLAT 体素不完全重合
# 用 KNN 把未分配的 SLAT 体素分配给最近的已标注体素
slat_is_edit = edit_parts[sc[:, 0], sc[:, 1], sc[:, 2]]  # 直接查表
# 未命中的用 KNN 找最近邻
nbrs = NearestNeighbors(n_neighbors=1).fit(assigned_coords)
_, indices = nbrs.kneighbors(unassigned_coords)
```

VD 用 PartField 直接在 SLAT 体素上聚类，mask 天然对齐。但 mesh 体素化可能和 SLAT 坐标有偏差，所以需要 KNN 做再投影。

#### d) 编辑区域扩展 — `_compute_editing_region()` (L467-518)

```python
# 对每个空体素，统计 KNN 邻居中编辑部件的比例
# 比例 > 0.5 的空体素也加入 mask（为编辑留出扩展空间）
mask_proportions = neighbor_masks.float().mean(dim=1)
mask[...] = (mask_proportions > 0.5)
mask = mask | edit_parts       # 编辑部件自身一定在 mask 内
mask = mask & ~preserved_parts  # 保留部件一定排除
```

用 `obj_vicinity`（SLAT bbox + pad）限制扩展范围，修复了 VD 原版 `mask | ~bbox_preserved` 导致 mask 膨胀到 84-94% 的 bug。

---

### 4. 多视角 2D 编辑 — `obtain_edited_images()` + `encode_multiview_cond()` (L570-686)

```python
# 渲染 N 个视角 → VLM 编辑每个 → DINOv2 特征残差
yaws = np.linspace(0, 360, num_views, endpoint=False)
imgs = render_utils.Trellis_render_multiview_images(gaussian, yaws, pitches)

# 关键：特征残差 = 原始特征 + α × (编辑特征 - 原始特征)
feat_cond = feat_orig + edit_strength * (feat_edited - feat_orig)
```

多视角编辑 + **特征残差**是核心改进：
- 多视角保证 3D 一致性（DINOv2 特征平均）
- 残差隔离了编辑 delta，防止 VLM 对非目标区域的修改泄漏到条件信号

`_composite_edit()` 还做了前景合成：用原图的前景 mask 将编辑结果只混合到前景区域，背景保持不变。

---

### 5. TRELLIS 编辑 — `edit()` (L792-881)

```python
# 核心：调用 VD 的 interweave_Trellis_TI
slat_new = interweave_Trellis_TI(
    args, self.trellis_text, self.trellis_img,
    slat, mask, prompts, effective_img, seed=seed)
```

这是**直接复用 VD 的核心算法**。`interweave_Trellis_TI` 做 Flow Inversion + Repaint：
1. 对原始 SLAT 做 flow inversion（反演到噪声空间）
2. 在 mask 区域注入新的条件信号（text prompt 驱动 S1/S2，image 驱动外观）
3. mask 外区域保持原始反演轨迹（repaint 策略）
4. 前向去噪生成编辑后的 SLAT

**组合策略 (`combinations`)**：
- `s1_pos_cond="new_s1_cpl", s1_neg_cond="ori_s1_cpl"` → CFG 引导：增强新描述、抑制旧描述
- `cfg_strength=7.5` → 引导强度
- 删除操作用 `cfg_strength=0`（无条件去噪，让 mask 区域"消失"）

**img_cond monkey-patch (L841-851)**：当使用多视角平均 DINOv2 特征时，直接替换 `trellis_img` 的 `get_cond` 方法返回预计算结果，避免重复编码。

---

### 6. 导出 — `export_pair()` (L885-917)

```python
gaussian = self.decode_to_gaussian(slat)  # SLAT → Gaussian Splatting
gaussian.save_ply(str(ply_path))          # 保存 PLY
torch.save(slat.feats, slat_path / "feats.pt")  # 保存 SLAT 用于后续可视化
```

---

## 与 VD 原版的关键差异

| 模块 | VD 原版 | TrellisRefiner |
|---|---|---|
| 部件分割 | PartField 聚类 SLAT | HY3D-Part GT mesh → 体素化 → KNN 对齐 |
| Prompt 生成 | 7次 LLM/VLM 调用 | 规则词表分解（`_strip_words`） |
| 2D 编辑 | 单视角 | 多视角 + 特征残差 |
| Mask 区域 | `mask \| ~bbox_preserved`（膨胀） | SLAT bbox + pad 限制（修复） |
| 编辑核心 | `interweave_Trellis_TI` | **完全复用** |
