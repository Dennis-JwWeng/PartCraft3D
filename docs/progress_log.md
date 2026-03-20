# PartCraft3D 开发进展日志

## 2026-03-19: Mask 管线修复 — 面序对齐 + 坐标空间对齐

### 问题

Mask 在整个管线中与实际 3D 几何不对应：2D mask 的语义标签和 part ID 对不上，3D voxel mask 在 SLAT 空间中位置偏移。

### 根因分析

#### Bug 1: 面序不匹配（~84% 的面被分配到错误的 part）

- **文件**: `partcraft/io/partcraft_loader.py` — `_pack_one_object()`
- **问题**: `instance_gt`（per-face part labels）是按 source mesh.glb 的面顺序生成的，但 `_pack_one_object` 将其直接应用到 VD mesh.ply 上。虽然两者面数相同（35900），但面的排列顺序完全不同。
  - 面质心距离均值: 0.4376（全部 > 0.01）
  - 错误标签比例: 30247/35900 = **84.3%**
- **修复**: 使用 source mesh.glb（面序与 instance_gt 一致）代替 VD mesh.ply 进行 part 分割
- **影响**: 每个 part 的面和语义标签完全对应

#### Bug 2: 坐标空间不对齐（Y-up vs Z-up）

- **文件**: `partcraft/io/partcraft_loader.py` — `_align_source_to_vd()`
- **问题**: Source mesh (GLB) 是 Y-up 坐标系，VD mesh.ply 是 Z-up（Blender 导入时转换）。修复 Bug 1 后，NPZ 中的 part mesh 在 source 坐标空间中，但 `build_part_mask()` 期望 VD 坐标空间。
  - Source: Y extent = 1.0（最高轴），Z extent = 0.669
  - VD: Z extent = 1.0（最高轴），Y extent = 0.669
- **错误做法**: 基于 bounding box extent 猜测轴交换（遗漏了符号翻转，不够鲁棒）
- **正确做法**: 读取 VD prerender 的 `transforms.json` 中记录的精确归一化参数:
  1. Blender GLB 导入轴转换: `(x, y, z) → (x, -z, y)` (Y-up → Z-up)
  2. Blender 场景归一化: `vertex = (blender_vertex + offset) * scale`
- **验证**: 变换后所有 20784 个顶点与 VD mesh.ply 的最近邻距离 < 4.5e-7
- **影响**: 3D voxel mask 在 SLAT 空间中精确对齐

### 验证结果

使用 `scripts/vis/visualize_masks.py` 生成 per-spec 诊断图：
- 2D 渲染中红色高亮与语义标签完全匹配
- 3D voxel projection 中 edit 区域精确对应目标 part 位置
- XY/XZ/YZ 三个投影视角中 mask 形状和位置正确
- Deletion（猴子）: 1613 voxels (0.6%)，精确覆盖猴子区域
- Modification（猴子→鹦鹉）: 27285 voxels (10.4%)，edit + expansion 区域正确

### 文件变更

| 文件 | 变更 | 说明 |
|------|------|------|
| `partcraft/io/partcraft_loader.py` | 修改 | `_pack_one_object()`: 使用 source mesh + `_align_source_to_vd()` 精确坐标变换 |
| `partcraft/io/partcraft_loader.py` | 新增函数 | `_align_source_to_vd()`: 基于 transforms.json 的精确 GLB→VD 坐标变换 |
| `partcraft/io/partcraft_loader.py` | 新增函数 | `_load_source_mesh()`: 从 mesh.zip 加载原始 GLB mesh |
| `scripts/vis/visualize_masks.py` | 新增 | Per-spec mask 诊断可视化（2D highlight + 3D voxel projection + info） |
| `scripts/pack_prerender_npz.py` | 已有 | 薄包装层，调用 `PartCraftDataset.prepare_from_prerender()` |

### 当前状态

- [x] 面序对齐：source mesh 面序与 instance_gt 一致
- [x] 坐标对齐：基于 transforms.json 精确变换（最大误差 4.5e-7）
- [x] 2D mask 语义验证通过
- [x] 3D voxel mask SLAT 对齐验证通过
- [x] 可视化工具完成（visualize_masks.py）
- [ ] 端到端管线运行（streaming mode with TRELLIS）
- [ ] 大批量数据验证

---

## 2026-03-13: Phase 2.5 SLAT 编码与 TRELLIS 编辑调试

### 背景

Phase 2.5 使用 Vinedresser3D 的 Flow Inversion + Repaint 管线对 modification 类型的编辑（`mod_XXXXXX`）进行 3D 风格修改。核心流程：
1. 将 HY3D-Part 数据集中的 42 视图编码为 SLAT（Structured Latent）
2. 根据 part segmentation 构建 3D 编辑 mask
3. 使用 2D 图像编辑 API（Gemini）生成视觉引导
4. 通过 TRELLIS interweave 管线执行 3D 编辑
5. 解码为 Gaussian Splatting PLY 并渲染多视角图片/视频

### 发现并修复的 Bug

#### Bug 1: 体素化仅使用顶点（2026-03-13 修复）
- **文件**: `partcraft/phase2_assembly/trellis_refine.py` — `_voxelize_mesh()`
- **问题**: 原实现只用 mesh 顶点做体素化，仅产生约 7,375 个体素
- **修复**: 增加 `trimesh.sample.sample_surface(mesh, count=100000)` 采样三角面表面点，体素数提升至 ~12,624（增加 71%），覆盖率显著提升
- **影响**: 编码特征分辨率更高，减少模糊

#### Bug 2: 坐标空间不匹配 — 关键问题（2026-03-13 修复）
- **文件**: `partcraft/phase2_assembly/trellis_refine.py` — `encode_hy3d_object_to_slat()`
- **问题**: 体素位置在归一化空间 `[-0.5, 0.5]`，但 HY3D-Part 的相机是为原始 mesh 空间 `[-1, 1]` 设计的。DINOv2 特征投影到了错误的图像位置。
  - 修复前 UV 范围: `[-0.36, 0.39]`（严重偏离）
  - 修复后 UV 范围: `[-0.77, 0.81]`（接近正确的 `[-1, 1]`）
- **修复**: 将归一化的体素坐标变换回世界空间:
  ```python
  positions_world = positions_norm * mesh_scale + mesh_center
  ```
- **影响**: 这是导致 "所有输出都是模糊东西" 的根本原因

#### Bug 3: 编辑 mask 归一化不一致（2026-03-13 修复）
- **文件**: `partcraft/phase2_assembly/trellis_refine.py` — `build_edit_mask_from_parts()`
- **问题**: 编辑部件的体素化使用独立的 center/scale（部件自身的），而非完整 mesh 的 center/scale。导致部件体素占满整个 [0, 63] 网格，mask 完全错误。
- **修复**: 先获取完整 mesh 的 center/scale，传入 `_voxelize_mesh()`:
  ```python
  _, full_center, full_scale = _voxelize_mesh(full_mesh_bytes)
  edit_grid, _, _ = _voxelize_mesh(part_bytes, center=full_center, scale=full_scale)
  ```
- **影响**: mask 区域从 "覆盖全部" 修正为 "仅覆盖目标部件"

#### Bug 4: NameError — 变量重命名遗漏（2026-03-13 修复）
- **文件**: `partcraft/phase2_assembly/trellis_refine.py` — `_encode_views_to_features()`
- **问题**: 参数重命名为 `voxel_positions_world` 但函数体内仍引用旧名 `voxel_positions`
- **修复**: 统一使用新变量名

#### Bug 5: API Key 未找到导致 2D 编辑禁用（2026-03-13 修复）
- **文件**: `scripts/run_phase2_5.py`
- **问题**: `configs/partobjaverse.yaml` 中 `vlm_api_key` 为空，代码只从 `cfg["phase0"]` 读取
- **修复**: 增加回退逻辑，从 `configs/default.yaml` 读取 API key

### 关于 Gaussian Splatting PLY 输出格式

- `decode_slat(slat, ['gaussian'])` 输出的是 3D Gaussian Splatting 点云（位置 + 协方差 + 颜色），**不是三角网格**，这是预期行为
- PLY 文件在 MeshLab 等工具中显示为点云是正常的
- 质量应通过渲染的多视角图片和视频来评估
- 如需三角网格输出，可使用 `decode_slat(slat, ['mesh'])`，但质量取决于 SLAT 质量

### 2D 图像编辑 API 测试

- **状态**: 正常工作
- 模型: `gemini-2.5-flash-image`
- 输入/输出图片保存在: `outputs/partobjaverse_tiny/cache/phase2_5/2d_edits/`
- 测试对象: `00aee5c2fef743d69421bb642d446a5b`（装有面包棒的篮子）
- 编辑 prompt: "Replace the woven brown basket with a sleek black metal basket."
- API 返回了正确的编辑图像

### 新增工具脚本

- **`scripts/run_2d_edit.py`**: 批量 2D 图像编辑，支持多线程 API 调用、断点续传
- **`scripts/encode_slat.py`**: 使用 Vinedresser3D 原始管线预编码 SLAT（Blender 渲染 150 视图）

### 当前状态与待解决问题

#### 已完成
- [x] 4 个 SLAT 编码 bug 全部修复
- [x] API key 回退逻辑
- [x] 2D 图像编辑 API 验证通过
- [x] 输入/编辑图片保存用于检查

#### 进行中
- [ ] **Vinedresser3D 原始管线 baseline 对比**
  - 对象: `00aee5c2fef743d69421bb642d446a5b`
  - 已编码 SLAT（在 `Vinedresser3D-main/outputs/slat/`），但编辑流程尚未完成
  - 目的: 比较 PartCraft3D（42 视图, fov=60°）vs Vinedresser3D（150 视图, fov=40°）的编码质量

#### 待验证
- [ ] 修复后的 SLAT 编码质量是否足够（42 视图 vs 150 视图）
- [ ] 修复后的 before/after Gaussian Splatting 渲染质量
- [ ] 是否需要切换到三角网格输出 (`decode_slat(slat, ['mesh'])`)
- [ ] 大批量运行（200 条数据）的稳定性

### 关键参数对比

| 参数 | PartCraft3D (HY3D-Part) | Vinedresser3D |
|------|--------------------------|---------------|
| 视图数 | 42 (6×7 grid) | 150 (Hammersley) |
| FOV | 60° | 40° |
| 相机距离 | 2.5 | 2.0 |
| Mesh 归一化 | [-1, 1] | Blender 默认 |
| 体素化方法 | 顶点+表面采样 | Open3D 三角网格 |
| 渲染器 | Blender/pyrender | Blender |

### 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `partcraft/phase2_assembly/trellis_refine.py` | 大量修改 | SLAT 编码 4 个 bug 修复 |
| `scripts/run_phase2_5.py` | 修改 | API key 回退、图片保存 |
| `scripts/run_2d_edit.py` | 新增 | 批量 2D 编辑脚本 |
| `scripts/encode_slat.py` | 新增 | Vinedresser3D 管线 SLAT 预编码 |
