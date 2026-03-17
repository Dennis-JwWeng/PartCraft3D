# HY3D-Part vs PartObjaverse 数据集对比

> 基于实际数据检查生成，对比 PartCraft3D 管线所用的两个数据集。

## 1. 总览

| 特性 | HY3D-Part | PartObjaverse (converted) |
|------|-----------|---------------------------|
| 来源 | HY3D 生成管线 | Objaverse (Sketchfab 艺术家模型) |
| 对象数量 | 940 (shard 00) | 200 |
| 分类体系 | 无显式分类 | 8 大类 (Human-Shape, Animals, Daily-Used, Buildings&Outdoor, Transportations, Plants, Food, Electronics) |
| 分割标注来源 | HY3D 管线自动生成 | SAMPart3D 人工标注 |
| 平均部件数 | ~13.7 / 对象 | ~12.3 / 对象 |
| 部件语义标签 | 网格节点名 (如 `Box001`) | 人类可读语义 (如 `Body`, `Head`, `Sword`) |

---

## 2. 网格数据 (Mesh NPZ)

两个数据集均采用 NPZ 格式，内含 `full.ply` + `part_*.ply`。

| 特性 | HY3D-Part | PartObjaverse (converted) |
|------|-----------|---------------------------|
| NPZ 平均大小 | ~16.6 MB | ~0.9 MB |
| 平均面数 | ~460K | ~45K |
| 网格质量 | 高精度 watertight | 中等 (Sketchfab 原始) |
| PLY 内置顶点色 | **无** (统一灰色 `[102,102,102]`) | **有** (从 Objaverse 纹理烘焙) |
| 颜色获取方式 | 必须调用 `bake_vertex_colors()` 从渲染图投射 | PLY 自带 + 也可从渲染图投射 |

---

## 3. 渲染图与分割 (Image NPZ)

两个数据集均采用 NPZ 格式，内含 WebP 渲染图 + 分割 Mask + 相机参数。

| 特性 | HY3D-Part | PartObjaverse (converted) |
|------|-----------|---------------------------|
| NPZ 平均大小 | ~1.6 MB | ~0.4 MB |
| 视角数量 | 42 | 42 |
| 分辨率 | 518 x 518 | 518 x 518 |
| 图像模式 | RGBA | RGBA |
| 渲染彩色 | 有 (丰富彩色) | 有 (丰富彩色) |
| 纹理来源 | HY3D 管线渲染 | 原始 Objaverse 纹理 GLB 经 pyrender 渲染 |
| 分割 Mask | 有 (int16, -1 = 背景) | 有 (int16, -1 = 背景) |

---

## 4. 相机参数

| 特性 | HY3D-Part | PartObjaverse (converted) |
|------|-----------|---------------------------|
| 投影方式 | 正交投影 (`proj_type: ortho`) | 透视投影 (`PerspectiveCamera`) |
| FOV | 60.9° | 60° (camera_angle_x = pi/3) |
| 仰角范围 | [-73°, +76°] | [-20°, +80°] |
| 方位角覆盖 | 连续旋转 (0° ~ 519°) | 均匀 7 等分 x 6 层 |
| 相机距离 | ~1.5 (ortho) | 2.5 (perspective) |
| Frame 额外字段 | `fov`, `azi`, `elev`, `cam_dis`, `proj_type` | 仅 `camera_angle_x`, `transform_matrix` |

---

## 5. 管线兼容性

| 阶段 | HY3D-Part | PartObjaverse (converted) |
|------|-----------|---------------------------|
| Phase 0 (VLM 语义标注) | 需运行 VLM API | **已预填充** (来自 SAMPart3D 标注，零成本) |
| Phase 1 (编辑规划) | 完全支持 | 完全支持 |
| Phase 2 (网格组装) | 完全支持 | 完全支持 |
| Phase 2.5 (TRELLIS 生成编辑) | 完全支持 | 支持 (相机参数格式略有差异) |
| Phase 3 (质量过滤) | 完全支持 | 完全支持 |
| Phase 4 (指令生成与导出) | 完全支持 | 完全支持 |

---

## 6. 各自优劣势

| 维度 | HY3D-Part | PartObjaverse |
|------|-----------|---------------|
| 网格精度 | 高面数 (460K)，几何细节丰富 | 面数较低 (45K) |
| 语义标注质量 | 节点名无语义，依赖 VLM 二次标注 | 人类可读标签，自带 8 大类 |
| 纹理质量 | HY3D 管线生成材质 | 艺术家手工 UV 纹理贴图 |
| PLY 自带颜色 | 无，需从渲染图烘焙 | 有，预烘焙顶点色 |
| Phase 0 成本 | 需 VLM API 调用 (Gemini / 本地 Qwen) | 零成本 (预填充 cache) |
| 相机元数据丰富度 | 丰富 (含 azi/elev/cam_dis/proj_type) | 基础 (仅 transform_matrix + fov) |
| 对象多样性 | 940 对象，无分类 | 200 对象，8 大类覆盖 |

---

## 7. 文件结构

两个数据集在转换后共享相同的目录结构，可无缝切换：

```
data/<dataset>/
├── images/<shard>/<obj_id>.npz     # 42 views (WebP) + masks + transforms.json + split_mesh.json
├── mesh/<shard>/<obj_id>.npz       # full.ply + part_*.ply
├── cache/phase0/semantic_labels.jsonl  # Phase 0 预填充 (仅 PartObjaverse)
└── metadata.json                       # 数据集元信息 (仅 PartObjaverse)
```

配置切换：

```yaml
# HY3D-Part
data:
  image_npz_dir: "/Node11_nvme/zsn/test_data/part/images"
  mesh_npz_dir: "/Node11_nvme/zsn/test_data/part/mesh"

# PartObjaverse
data:
  image_npz_dir: "data/partobjaverse/images"
  mesh_npz_dir: "data/partobjaverse/mesh"
```

---

## 8. 已知差异与注意事项

1. **投影方式不同**：HY3D-Part 使用正交投影，PartObjaverse 使用透视投影。Phase 2.5 (TRELLIS) 如果依赖 `proj_type` 字段可能需要适配。
2. **PLY 颜色差异**：HY3D-Part 的 PLY 本身无色，`get_full_mesh(colored=True)` 会自动调用 `bake_vertex_colors()` 从渲染图烘焙。PartObjaverse 的 PLY 已带颜色，但 `bake_vertex_colors()` 会覆盖为从渲染图投射的颜色（二者效果一致）。
3. **相机 Frame 字段**：HY3D-Part 的 Frame 包含 `azi`, `elev`, `cam_dis`, `proj_type` 等额外字段，PartObjaverse 仅有 `camera_angle_x` 和 `transform_matrix`。如果下游代码访问这些额外字段需做兼容处理。
