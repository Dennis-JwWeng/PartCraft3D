# Vinedresser3D 代码总结报告

## 1. 项目概述

Vinedresser3D 是一个基于 **SLAT (Structured Latent)** 空间的 3D 编辑系统，核心思想是将 3D 物体编码到结构化潜空间中，然后通过 **Flow Inversion + Repaint** 的方式实现局部编辑。整个系统依赖 TRELLIS 框架（微软开源的 3D 生成模型）作为底层 3D 表示和生成引擎。

**一句话总结**：Vinedresser3D = PartField分割 + VLM理解 + 2D图像编辑引导 + SLAT空间Repaint式3D编辑

---

## 2. 文件结构与模块职责

```
Vinedresser3D-main/
├── main.py                          # 主管线入口：编码→分割→提示分解→定位→掩码→图像编辑→交织采样→VLM选择→导出
├── interweave_Trellis.py            # 核心编辑引擎：Flow Inversion + Repaint (Stage1 + Stage2)
├── encode_asset/
│   ├── render_img_for_enc.py        # Blender 渲染 150 视角 + Open3D 体素化到 64³
│   ├── encode_into_SLAT.py          # DINOv2 特征提取 + SLAT 编码器
│   ├── blender_script/render.py     # Blender 渲染脚本
│   └── utils.py                     # Hammersley 球面采样
├── VLM_LLM/
│   ├── Gemini_VLM.py                # VLM 多模态调用（5个功能函数）
│   └── Gemini_LLM.py                # LLM 纯文本调用（提示分解、部件描述提取）
├── Nano_banana.py                   # Gemini 2.5 Flash 图像编辑 API 封装
├── PartField_segmentation.py        # PartField 3D 分割的薄封装
├── utilis.py                        # 点云→体素标签映射、PLY加载、SLAT工具函数
└── trellis/                         # TRELLIS 框架（修改版）
    ├── pipelines/                   # Text-to-3D / Image-to-3D 管线
    ├── models/                      # Flow模型、VAE、编码器/解码器
    ├── modules/sparse/              # 稀疏张量操作
    ├── representations/             # Gaussian / Mesh / NeRF 表示
    ├── renderers/                   # Gaussian Splatting 渲染器
    └── utils/render_utils.py        # 多视角渲染工具
```

---

## 3. 核心管线流程 (`main.py`)

### 3.1 总体流程图

```
输入: GLB/PLY 3D文件 + 编辑指令(文本)
  │
  ├─ Step 1: 编码资产 (encode_asset/)
  │   ├─ Blender渲染 150 视角 (518×518)
  │   ├─ Open3D 体素化到 64³ 网格
  │   └─ DINOv2 特征提取 → SLAT 编码器 → (feats, coords) 潜码
  │
  ├─ Step 2: PartField 3D 分割
  │   └─ 以子进程调用 PartField，输出 K=3..8 的聚类标签
  │
  ├─ Step 3: VLM 理解 + 提示分解
  │   ├─ obtain_overall_prompts(): 生成5个文本字段
  │   │   (ori_cpl, editing_part, new_cpl, target_part, edit_type)
  │   ├─ decompose_prompt(): 结构描述(s1) / 外观描述(s2) 分解
  │   └─ identify_ori_part() / identify_new_part(): 提取部件描述
  │
  ├─ Step 4: 语义定位 (Grounding)
  │   ├─ VLM选择最佳K值和部件颜色 (select_editing_parts / select_K)
  │   ├─ 点云标签 → 体素标签映射 (pc_to_voxel)
  │   └─ 生成 edit_parts / preserved_parts / bbox 掩码 (64³ bool)
  │
  ├─ Step 5: 计算编辑区域掩码
  │   ├─ Modification: KNN邻域扩展 + bbox约束
  │   ├─ Deletion: 直接使用 edit_parts 作为掩码
  │   └─ Addition: ~preserved_parts（非保留区域都可编辑）
  │
  ├─ Step 6: 获取编辑参考图像 (Addition/Modification)
  │   ├─ 渲染 24 视角预编辑图像
  │   ├─ VLM选择最佳视角 (select_img_to_edit)
  │   └─ Gemini 2.5 Flash 编辑该视角图像 (Nano_banana_edit)
  │
  ├─ Step 7: 交织采样 × 5 组合 (interweave_Trellis_TI)
  │   ├─ 5种条件组合（正/负提示 × 完整/部件描述 × 有无CFG）
  │   ├─ 每组生成一个编辑后的 SLAT
  │   └─ 解码为 Gaussian Splatting → 渲染 16 视角
  │
  └─ Step 8: VLM 选择最佳结果 + 导出
      ├─ select_the_best_edited_object(): 从5个候选中选最佳
      └─ 保存编辑后的 SLAT + 渲染视频
```

### 3.2 关键代码位置

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 资产编码入口 | `main.py` | 266-269 | 检测缓存，按需渲染+编码 |
| 模型加载 | `main.py` | 272-278 | 加载 TRELLIS text/img 双管线 + ss_encoder |
| SLAT解码→Gaussian | `main.py` | 291, 315 | 重复调用 `decode_slat`（优化点） |
| PartField子进程 | `main.py` | 294 | `os.system("python PartField_segmentation.py")` |
| 5组交织采样 | `main.py` | 320-347 | 循环5种condition组合 |
| VLM最佳选择 | `main.py` | 359 | 从5候选中选最优 |

---

## 4. 核心编辑引擎 (`interweave_Trellis.py`)

### 4.1 Flow Inversion + Repaint 机制

Vinedresser3D 的编辑核心是在 SLAT 空间中执行 **repaint 式局部编辑**：

```
原始 SLAT → [Flow Inversion] → 噪声空间 (存储中间状态)
                                    │
                                    ↓
噪声空间 → [Repaint 采样] → 编辑后 SLAT
                │
                ├─ 编辑区域: 用新条件引导去噪
                └─ 保留区域: 每步用 Inversion 中间状态覆盖
```

**双阶段架构**：
- **Stage 1 (S1)**: 16³ 稀疏结构（控制几何拓扑，如是否存在某个部件）
- **Stage 2 (S2)**: 64³ SLAT 特征（控制细节外观、纹理）

### 4.2 `interweave_Trellis_TI()` 函数详解

```python
def interweave_Trellis_TI(args, trellis_text, trellis_img,
    slat, mask, prompts, img_new, seed):
```

**执行步骤**：

#### Step A: 编码 11 个文本条件 (L234-L245)
```python
conds = {
    "ori_cpl":     trellis_text.get_cond([prompts["ori_cpl"]]),      # 原始完整描述
    "new_cpl":     trellis_text.get_cond([prompts["new_cpl"]]),      # 编辑后完整描述
    "ori_s1_cpl":  ...,  # 原始结构描述（去除颜色/纹理）
    "ori_s2_cpl":  ...,  # 原始外观描述（去除形状）
    "ori_s1_part": ...,  # 原始部件结构描述
    "ori_s2_part": ...,  # 原始部件外观描述
    "new_s1_cpl":  ...,  # 新的结构描述
    "new_s2_cpl":  ...,  # 新的外观描述
    "new_s1_part": ...,  # 新部件结构描述
    "new_s2_part": ...,  # 新部件外观描述
    "null":        ...,  # 空条件 (用于无条件引导)
}
```

#### Step B: S2 Inversion (L275-L282)
```python
# 从 SLAT 反向 ODE 到噪声，CFG=0
# 存储每个时间步的中间状态到 inverse_dict
inverse_dict = {'s2_0.0': slat}
for t_curr, t_prev in t_pairs:
    sample = RF_sample_once(text_s2_flow_model, sample, t_curr, t_prev,
                            inverse=True, cfg_strength=0)
    inverse_dict[f's2_{t_prev}'] = sample
s2_noise = sample
```

#### Step C: S1 Inversion → S1 Repaint (L284-L326, 仅 Addition/Modification)
```python
# S1 Inversion: 16³ 结构空间反向
z_s = s1_encoder(sparse_voxels)
for t_curr, t_prev in t_pairs:
    sample = RF_sample_once(text_s1_flow_model, sample, ...)
    inverse_dict[f's1_{t_prev}'] = sample
s1_noise = sample

# S1 Repaint: 交替 text/image 条件去噪
for t_curr, t_prev in t_pairs:
    if t_curr > t_prev:  # 去噪方向
        if cnt % 2 == 0:  # 文本条件
            x_t_1 = RF_sample_once(text_s1_flow_model, x_t, ..., cond=text_cond)
        else:             # 图像条件（交织!）
            x_t_1 = RF_sample_once(img_s1_flow_model, x_t, ..., cond=img_cond)
        x_t_1[s1_mask] = inverse_dict[f's1_{t_prev}'][s1_mask]  # Repaint保留区域
    else:  # 加噪方向（Resample机制）
        x_t_1 = RF_sample_once(..., inverse=True)
```

#### Step D: S2 Repaint + 软掩码融合 (L329-L360)
```python
# 与S1类似的交织采样，但在64³空间
# 关键：保留区域使用软掩码（soft mask）进行平滑过渡
if torch.sum(mask_new) > 0:
    x_t_1.feats[mask_new] = (
        x_t_1.feats[mask_new] * s2_soft_mask +           # 编辑结果
        inverse_dict[f's2_{t_prev}'].feats[mask_ori] * (1 - s2_soft_mask)  # 原始状态
    )
```

### 4.3 关键辅助函数

| 函数 | 行号 | 功能 |
|------|------|------|
| `get_times()` | L12-L36 | 生成时间序列，支持 Resample（来回去噪/加噪） |
| `RF_sample_once()` | L72-L92 | 二阶 Runge-Kutta Flow Matching 采样器 |
| `get_s1_mask()` | L94-L108 | 64³→16³ 下采样掩码（4³块投票） |
| `get_coords_mask()` | L110-L113 | 从64³掩码中提取稀疏坐标的掩码 |
| `remove_small_components()` | L115-L172 | BFS去除小于50体素的连通分量 |
| `get_s2_noise_new()` | L174-L202 | 为新坐标构建噪声（保留已知 + 随机新增） |
| `get_soft_weights()` | L204-L215 | KNN计算软权重（距离衰减或阈值截断） |
| `get_s2_soft_mask()` | L217-L222 | 生成 S2 空间的软过渡掩码 |

### 4.4 采样器细节 (`RF_sample_once`)

```python
# 二阶 Runge-Kutta (Heun's method)
pred_vec = model(x_t, t_curr)                           # 在 t_curr 处预测速度
sample_mid = x_t + (t_prev - t_curr)/2 * pred_vec       # 走半步
pred_vec_mid = model(sample_mid, (t_curr+t_prev)/2)      # 在中点处预测速度
first_order = (pred_vec_mid - pred_vec) / (dt/2)          # 估计加速度
x_next = x_t + dt * pred_vec + 0.5 * dt² * first_order   # 二阶更新
```

---

## 5. 资产编码模块 (`encode_asset/`)

### 5.1 渲染 + 体素化 (`render_img_for_enc.py`)

```
输入 GLB/PLY → Blender 渲染 150 视角 (518×518 PNG)
                     ↓
              Open3D 体素化 → 64³ 体素网格 (voxels.ply)
```

- **Blender 版本**: 3.0.1，使用 CYCLES 引擎
- **视角分布**: Hammersley 球面序列（均匀覆盖球面）
- **体素化**: `create_from_triangle_mesh_within_bounds`，范围 [-0.5, 0.5]³

### 5.2 SLAT 编码 (`encode_into_SLAT.py`)

```
150 张渲染图 → DINOv2 (vitl14_reg) → 1024维 patch tokens (37×37)
                                          ↓
每个体素 → 投影到各视角 → grid_sample 双线性插值 → 150视角平均
                                          ↓
聚合特征 SparseTensor → SLAT Encoder (swin8_B_64l8) → (feats, coords)
```

**关键参数**：
- DINOv2: `dinov2_vitl14_reg`, 输出 1024 维特征
- Patch size: 14×14, 图像 518→37 个 patch
- 特征聚合: 150 视角简单平均
- SLAT 编码器: `slat_enc_swin8_B_64l8_fp16`

**性能瓶颈**: 150 张图像逐张处理 DINOv2（未 batch 化）

---

## 6. VLM/LLM 模块

### 6.1 VLM 多模态调用 (`Gemini_VLM.py`)

5 个功能函数，全部使用 **Gemini 2.5 Pro**：

| 函数 | 输入 | 输出 | 图片数 |
|------|------|------|--------|
| `select_K()` | 8 原始视角 + 6×8 分割视角 | 最佳 K 值 (3-8) | 56 |
| `select_editing_parts()` | 8 视角 + 6×8 分割视角 | `K&&&color1,color2` | 56 |
| `obtain_overall_prompts()` | 8 视角 + 编辑指令 | 5个字段 (ori/new/edit_type) | 8 |
| `select_img_to_edit()` | 24 预编辑视角 | 最佳视角索引 | 24 |
| `select_the_best_edited_object()` | 8 原始 + 5×16 候选视角 | 最佳候选索引 | 88 |

### 6.2 LLM 文本调用 (`Gemini_LLM.py`)

3 个函数，用于提示词工程：

| 函数 | 功能 |
|------|------|
| `decompose_prompt()` | 将描述拆分为结构描述(S1) + 外观描述(S2) |
| `identify_ori_part()` | 从原始描述中提取被编辑部件的描述 |
| `identify_new_part()` | 从新描述中提取目标部件的描述 |

### 6.3 图像编辑 (`Nano_banana.py`)

- **模型**: Gemini 2.5 Flash Image Generation
- **输入**: 原始渲染图 + 编辑指令 + 新部件描述
- **输出**: 编辑后的 2D 图像（作为 image-conditioned 生成的参考）

---

## 7. 分割与定位

### 7.1 PartField 分割 (`PartField_segmentation.py`)

- 薄封装，通过 `os.system("python PartField_segmentation.py")` 子进程调用
- 输出: `PartField/clustering_results/cluster_out/{name}_0_{K:02d}.npy`，K=3..8
- 每个文件是 shape=(N,) 的 int32 数组，N=Gaussian点数

### 7.2 语义定位 (`grounding()`, `main.py` L99-L174)

```
Modification/Deletion:
  VLM选择K + 颜色 → 点云标签 → 体素标签 → edit_parts / preserved_parts (64³ bool)
  同时生成 bbox_edit / bbox_preserved 用于区域约束

Addition:
  VLM选择K → 所有部件作为 preserved_parts → edit区域 = ~preserved_parts
```

### 7.3 工具函数 (`utilis.py`)

- `pc_to_voxel()`: 将 Gaussian 点云的部件标签映射到 64³ 体素，使用标签计数+KNN回退
- `find_nn_label()`: KNN 最近邻标签查找
- `ply_to_coords()` / `feats_to_slat()`: PLY/NPZ → SparseTensor 转换

---

## 8. 编辑类型与策略

| 编辑类型 | S1 处理 | S2 处理 | 图像引导 |
|----------|---------|---------|----------|
| **Modification** | S1 Inversion + Repaint | S2 Inversion + Repaint + 软掩码 | 需要 |
| **Addition** | S1 Inversion + Repaint | S2 Inversion + Repaint + 软掩码 | 需要 |
| **Deletion** | 不执行 S1 Repaint | S2 仅保留非掩码坐标 + 无条件去噪 | 不需要 |

**5 种条件组合** (`main.py` L320-L326)：

| 组合 | S1 正条件 | S1 负条件 | S2 正条件 | S2 负条件 | CFG |
|------|-----------|-----------|-----------|-----------|-----|
| 0 | new_s1_cpl | ori_s1_cpl | new_s2_cpl | ori_s2_cpl | 7.5 |
| 1 | new_s1_cpl | null | new_s2_cpl | null | 7.5 |
| 2 | new_s1_part | ori_s1_part | new_s2_part | ori_s2_part | 7.5 |
| 3 | new_s1_part | null | new_s2_part | null | 7.5 |
| 4 | null | null | null | null | 0 |

---

## 9. 模型依赖

| 模型 | 来源 | 用途 |
|------|------|------|
| TRELLIS Text-XLarge | `microsoft/TRELLIS-text-xlarge` | 文本条件 3D 生成 + SLAT 解码 |
| TRELLIS Image-Large | `microsoft/TRELLIS-image-large` | 图像条件 3D 生成 |
| Sparse Structure Encoder | `ckpts/ss_enc_conv3d_16l8_fp16` | 体素→S1 编码 |
| SLAT Encoder | `TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16` | DINOv2特征→SLAT |
| DINOv2 ViT-L/14 | `facebookresearch/dinov2` | 图像特征提取 |
| PartField | 本地安装 | 3D 部件分割 |
| Gemini 2.5 Pro | API | VLM/LLM 理解 |
| Gemini 2.5 Flash Image | API | 2D 图像编辑 |

---

## 10. 性能瓶颈分析

| 瓶颈 | 位置 | 影响 | 优化方向 |
|------|------|------|----------|
| DINOv2 逐张处理 | `encode_into_SLAT.py` L58-83 | 150 次前向传播 | 批处理 (batch=8~16) |
| decode_slat 重复调用 | `main.py` L291, L315 | 两次完全相同的解码 | 缓存结果 |
| PartField 子进程 | `main.py` L294 | 进程启动 + GPU初始化 | 改为库调用 |
| 5×交织采样 | `main.py` L328-347 | 5次完整 Inversion+Repaint | 共享 Inversion |
| 11次文本编码 | `interweave_Trellis.py` L234-245 | 每组合重复编码 | 外提编码 |
| VLM 串行调用 | 多处 | API 延迟叠加 | 并行/批量 |
| 56 张图片的 VLM 请求 | `Gemini_VLM.py` | 巨大 token 消耗 | 减少视角/降分辨率 |

---

## 11. 数据流总结

```
                    ┌─────────────────────────────────────────┐
                    │              输入                        │
                    │  GLB/PLY文件 + 编辑指令文本               │
                    └─────────┬───────────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────────┐
                    │  Stage 1: 3D → SLAT                     │
                    │  Blender 150views → DINOv2 → Encoder    │
                    │  输出: (feats, coords) SparseTensor      │
                    └─────────┬───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
   ┌──────▼──────┐   ┌───────▼───────┐   ┌───────▼───────┐
   │  PartField  │   │ VLM 提示理解   │   │ SLAT → Gauss  │
   │  K=3..8     │   │ 11个文本条件   │   │  → 渲染视角    │
   │  分割标签    │   │ 编辑类型分类   │   │  → 2D图像编辑  │
   └──────┬──────┘   └───────┬───────┘   └───────┬───────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                    ┌─────────▼───────────────────────────────┐
                    │  Grounding: 部件定位 → 64³ 掩码          │
                    │  编辑区域 / 保留区域 / bbox              │
                    └─────────┬───────────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────────┐
                    │  Flow Inversion: SLAT → 噪声            │
                    │  S2 Inversion → S1 Inversion            │
                    │  存储所有中间状态                         │
                    └─────────┬───────────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────────┐
                    │  Repaint 采样 ×5 组合                    │
                    │  S1 Repaint (text↔image交织)             │
                    │  S2 Repaint + 软掩码融合                 │
                    └─────────┬───────────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────────┐
                    │  VLM 选择最佳 → 导出 SLAT + 视频         │
                    └─────────────────────────────────────────┘
```

---

## 12. 缓存策略

Vinedresser3D 在多个环节使用文件缓存避免重复计算：

| 缓存内容 | 路径 | 触发条件 |
|----------|------|----------|
| 渲染图+体素 | `outputs/img_Enc/{name}/` | 检查 `voxels.ply` 是否存在 |
| SLAT 潜码 | `outputs/slat/{name}_feats.pt` | 检查文件是否存在 |
| 提示分解结果 | `outputs/prompts/{name}.pkl` | 检查 pickle 是否存在 |
| 定位结果 | `outputs/grounding/{name}.txt` | 检查文件是否存在 |
| 编辑参考图 | `outputs/img_edit/img/{name}_editedImg.png` | 检查文件是否存在 |
| 最佳视角ID | `outputs/img_edit/ID/{name}_editImgID.txt` | 检查文件是否存在 |

---

## 13. 代码质量观察

### 优点
- 清晰的模块化分工（编码/分割/VLM/编辑引擎/工具）
- 完整的缓存机制，支持断点续跑
- 灵活的条件组合策略 + VLM自动选优
- 双阶段 S1/S2 分离处理结构和外观

### 待改进
- 硬编码路径（`outputs/`, `PartField/`）
- API Key 明文写在代码中 (`main.py` L283)
- PartField 以子进程调用而非库导入
- 无错误恢复机制（中间步骤失败需从头开始）
- 无批量处理支持（只能逐个物体编辑）
- 无日志/指标记录系统

---

## 附录：关键概念解释

### A. SLAT (Structured Latent)
TRELLIS 的核心表示：64³ 稀疏体素网格上的特征向量。每个被占据的体素携带一个高维特征，可解码为 Gaussian Splatting / Mesh / NeRF 等 3D 表示。

### B. Flow Matching
一种生成模型框架，学习从噪声到数据的连续 ODE 流。相比 Diffusion Model 的离散步骤，Flow Matching 的轨迹更平滑、训练更稳定。

### C. Flow Inversion
反向运行 ODE，将数据映射回噪声空间。关键是存储中间状态，用于 Repaint 时恢复保留区域。

### D. Repaint
源自 2D 图像修复的技术：在去噪过程中，每步将保留区域的特征替换为 Inversion 得到的对应时间步的状态，只让编辑区域自由生成。

### E. Interweave (交织)
Vinedresser3D 的创新：在 Repaint 的去噪步骤中，奇偶交替使用文本条件和图像条件的 Flow 模型，融合两种引导信号。
