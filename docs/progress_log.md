# PartCraft3D 开发进展日志

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
