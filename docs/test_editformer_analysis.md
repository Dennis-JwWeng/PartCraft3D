# test_editformer.py 移植分析

将 3DEditFormer 的推理管线（原 `eval_3d_editing.py`）移植到 PartCraft3D 数据管线上，复用 VD 预处理的数据和 Phase 2.5 的 2D 编辑缓存，实现与 VD 方法在相同 test case 上的对比评测。

## 整体流程（三阶段）

```
Phase A: 加载 TRELLIS + 预处理数据
    ↓
Phase B: 加载 3DEditFormer 两阶段编辑模型
    ↓
Phase C: 逐案例执行编辑 → 渲染对比视频 → 导出 mesh
```

---

## 与原版 `eval_3d_editing.py` 的对比

| 维度 | 原版 `eval_3d_editing.py` | 移植版 `test_editformer.py` |
|---|---|---|
| 数据来源 | 3DEditVerse 数据集（alpaca/flux_edit/mixamo） | PartCraft3D Phase 2.5 test cases + VD 预处理数据 |
| 原始图像 | 数据集自带 `ori_img_path` | VD 预渲染视角 (`img_Enc/{obj_id}/000.png`) |
| SS latent | 数据集预计算 `.npz` | 从图像实时提取（缓存到 `editformer_cache/`） |
| SLAT | 数据集预计算 `.npz` | VD 预编码（`slat/{obj_id}_feats.pt`）+ 归一化转换 |
| 编辑图像 | Flux 生成 / 数据集自带 | Phase 2.5 缓存的 Gemini 2D 编辑结果 |
| 渲染对比 | 6列（ori_img + ori_voxel + ori_gs + edit_img + edit_voxel + edit_gs） | 4列（ori_img + ori_gs + edit_img + edit_gs） |
| 后处理渲染 | Blender 多线程渲染队列 | 无（仅 Gaussian Splatting 渲染） |
| 环境兼容 | 仅 3DEditFormer 环境 | 自动适配两套 `diff_gaussian_rasterization` |

---

## 核心代码详解

### Phase A: 数据准备 (L384-474)

#### 1. Gaussian 渲染器补丁 — `_patch_gaussian_renderer()` (L50-175)

**问题**：3DEditFormer 环境的 `diff_gaussian_rasterization` 是 2DGS/GOF 变体，`GaussianRasterizationSettings` 比 VD 环境多了两个必须参数：`kernel_size`（float）和 `subpixel_offset`（tensor [H,W,2]）。而 TRELLIS 的渲染器是按 VD 环境编写的，缺少这两个参数。

**解决方案**：通过 `inspect.signature()` 自动检测当前环境的 `GaussianRasterizationSettings` 是否需要这两个参数：

```python
import inspect as _insp
_raster_params = _insp.signature(GaussianRasterizationSettings).parameters
_needs_kernel = 'kernel_size' in _raster_params

if _needs_kernel:
    raster_kw['kernel_size'] = getattr(pipe, 'kernel_size', 0.0)
    raster_kw['subpixel_offset'] = torch.zeros((h, w, 2), dtype=torch.float32, device="cuda")
```

补丁分两层：
- **L116**：替换模块级 `render` 函数 → `mod.render = _patched_render`
- **L123-175**：重写 `GaussianRenderer.render` 方法 → 因为原方法通过闭包捕获了旧的 `render` 引用，仅替换模块级函数不够，需要把整个类方法也替换掉

#### 2. TRELLIS Pipeline 加载 (L387-396)

```python
pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_path)
pipeline.cuda()
slat_norm_std = torch.tensor(pipeline.slat_normalization['std'])[None].cuda()
slat_norm_mean = torch.tensor(pipeline.slat_normalization['mean'])[None].cuda()
```

加载 TRELLIS-image-large，同时提取归一化参数。这些参数在 VD SLAT ↔ 3DEditFormer SLAT 转换中至关重要。

#### 3. 逐对象数据加载 (L402-467)

对每个 `obj_id` 准备三类数据：

**a) 原始图像 (L413-431)**：
```
优先级: 本地缓存 > VD预渲染(img_Enc/{obj_id}/000.png) > 从SLAT实时渲染
```
加载后统一经过 `preprocess_image()` 做 rembg 去背景 + 前景裁剪 + resize 到 518×518。

**b) SLAT 归一化 (L433-445)**：
```python
# VD 预缓存的 SLAT 是 DENORMALIZED（TRELLIS 原始输出）
raw_feats = torch.load(vd_feats_path)  # [N, 8]
# 3DEditFormer 需要 NORMALIZED SLAT
norm_feats = (raw_feats - slat_norm_mean) / slat_norm_std
```

这是移植中**最关键的转换**。VD 存储的是 TRELLIS 原始输出（反归一化后的），而 3DEditFormer 的 denoiser 训练时使用的是归一化后的 SLAT。归一化参数：
- mean: `[-2.169, -0.004, -0.134, -0.084, -0.527, 0.724, -1.141, 1.204]`
- std: `[2.378, 2.386, 2.124, 2.175, 2.664, 2.371, 2.622, 2.685]`

**c) SS latent 提取 (L448-465)**：
```python
# SS latent (dense 8×16×16×16) 不在 VD 预缓存中，需从图像提取
cond = pipeline.get_cond([ori_img_pp])        # DINOv2 编码
fm = pipeline.models['sparse_structure_flow_model']
z_s = pipeline.sparse_structure_sampler.sample(fm, noise, **cond, ...).samples
```

SS latent 是 3DEditFormer 独有的输入——VD 的 repaint 方法不需要它。SS latent 是 TRELLIS 编码器对体素结构（哪些位置有物体）的 dense 表示，shape 为 `[1, 8, 16, 16, 16]`。

#### 4. 显存优化 (L470-474)

```python
del pipeline.models['sparse_structure_flow_model']
del pipeline.models['slat_flow_model']
gc.collect(); torch.cuda.empty_cache()
```

提取完 SS latent 后，flow model 不再需要，释放约 2-4 GB 显存给后续编辑模型。

---

### Phase B: 3DEditFormer 模型加载 (L478-523)

#### Stage 1: SS 编辑模型 (L495-507)

```python
ss_cfg = edict(json.load(open("configs/editing/ss_flow_img_dit_L_16l8_fp16.json")))
ss_cfg.load_dir = f"{ckpt_root}/img_to_voxel"
ss_cfg.load_ckpt = 40000
ss_models = {n: getattr(models, m.name)(**m.args).cuda() for n, m in ss_cfg.models.items()}
ss_trainer = getattr(trainers, ss_cfg.trainer.name)(ss_models, ds, ...)
ss_sampler = ss_trainer.get_sampler()
```

- 配置：`ss_flow_img_dit_L_16l8_fp16.json` → DiT-L 架构，16³ 分辨率，8 通道，FP16
- 检查点：`work_dirs/Editing_Training/img_to_voxel/ckpts/denoiser_step0040000.pt`
- 功能：根据编辑图像条件，生成编辑后的体素占用结构

#### Stage 2: SLAT 编辑模型 (L509-521)

```python
slat_cfg = edict(json.load(open("configs/editing/slat_flow_img_dit_L_64l8p2_fp16.json")))
slat_cfg.load_dir = f"{ckpt_root}/voxel_to_texture"
slat_cfg.load_ckpt = 40000
```

- 配置：`slat_flow_img_dit_L_64l8p2_fp16.json` → DiT-L 架构，64³ 分辨率，8 通道，patch_size=2，FP16
- 检查点：`work_dirs/Editing_Training/voxel_to_texture/ckpts/denoiser_step0040000.pt`
- 功能：在 Stage 1 确定的新体素结构上，生成编辑后的 SLAT 纹理特征

#### `_DS` 占位数据集 (L485-493)

```python
class _DS(SparseStructureLatentVisMixin):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.normalization = None
        self.loads = [100] * 100
    def __len__(self): return 100
```

trainer 初始化时需要 dataset 对象（继承 `SparseStructureLatentVisMixin` 以获取 `decode_latent()` 方法），但推理时不实际加载数据，所以用空壳实现。`decode_latent()` 用于将 SS latent 解码为体素占用网格。

---

### Phase C: 逐案例编辑 (L530-698)

#### 1. 2D 编辑图像获取 (L546-568)

```
优先级: Phase 2.5 缓存 (2d_edits/{edit_id}_edited_v0.png) > VLM 实时生成 > 跳过(--skip-vlm)
```

复用 VD 管线已经做过的 Gemini 2D 图像编辑结果，避免重复 API 调用。

#### 2. Stage 1: SS 编辑 (L572-593)

```python
data_t = {
    'ori_cond_img': ori_tensor,         # 原始图像 [1,3,518,518]
    'edited_cond_img': edited_tensor,   # 编辑图像 [1,3,518,518]
    'ori_ss_latent': z_s,               # 原始 SS latent [1,8,16,16,16]
    'edited_ss_latent': z_s,            # 同上（编辑前后结构目标相同）
}
inf_args = ss_trainer.get_inference_cond(**data_t)
inf_args['cond'] = torch.clone(inf_args['edited_cond_img'])

res_ss = ss_sampler.sample(
    ss_trainer.models['denoiser'],
    noise=ss_noise,                     # 随机噪声 [1,8,16,16,16]
    **inf_args,
    steps=25, cfg_strength=0,           # 无 CFG（编辑条件已在 cond 中）
    rescale_t=3.0,                      # 时间步缩放
    start_step=0, end_step=25,
    verbose=False,
).samples
```

**核心原理**：
- 输入：原始/编辑图像对 + 原始 SS latent
- `get_inference_cond()` 内部用 DINOv2 编码图像，计算编辑条件
- `cond = edited_cond_img`：以编辑后图像作为去噪条件
- `cfg_strength=0`：不使用 Classifier-Free Guidance（条件信号足够强）
- `rescale_t=3.0`：时间步缩放因子，控制编辑强度
- 输出：编辑后的 SS latent `[1,8,16,16,16]`

```python
voxel_edit = slat_trainer.dataset.decode_latent(res_ss) > 0
coords_edit = torch.argwhere(voxel_edit)[:, [0, 2, 3, 4]].int()
```

将 SS latent 解码为二值体素网格，提取非零体素坐标。这决定了编辑后物体的**结构**（哪些位置有体素）。

#### 3. Stage 2: SLAT 编辑 (L595-621)

```python
noise_slat = sp.SparseTensor(
    feats=torch.randn(coords_edit.shape[0], in_channels).cuda(),
    coords=coords_edit,  # Stage 1 输出的新体素坐标
)
ori_slat_sp = sp.SparseTensor(feats=ori_feats, coords=ori_c.int())

data_t['ori_ss_latent'] = ori_slat_sp      # 注意：这里换成了 SLAT
data_t['edited_ss_latent'] = ori_slat_sp
inf_args = slat_trainer.get_inference_cond(**data_t)
inf_args['cond'] = torch.clone(inf_args['edited_cond_img'])

slat_no_norm = slat_sampler.sample(
    slat_trainer.models['denoiser'],
    noise=noise_slat, **inf_args,
    steps=25, cfg_strength=0, rescale_t=3.0, ...
).samples
```

**核心原理**：
- 噪声初始化在 Stage 1 确定的新坐标上（稀疏张量）
- 条件包含：原始 SLAT（归一化后）+ 原始/编辑图像 DINOv2 特征
- 输出：编辑后的 SLAT 特征（归一化空间）

**关键细节**——字段复用 `ori_ss_latent`：
```python
data_t['ori_ss_latent'] = ori_slat_sp  # Stage 2 中此字段放的是 SLAT，不是 SS latent
```
3DEditFormer 的 trainer 在 Stage 2 用 `ori_ss_latent` 字段传递原始 SLAT 特征作为参考条件，字段名虽然叫 `ss_latent`，实际传的是 SLAT。这是原版代码的设计（L363-364），移植时保持一致。

#### 4. 反归一化 (L624)

```python
slat_edited = slat_no_norm * slat_norm_std + slat_norm_mean
```

3DEditFormer denoiser 在归一化空间中工作，输出也是归一化的。需要反归一化回 TRELLIS 原始空间才能用 TRELLIS 解码器解码。

#### 5. 解码 + 渲染 + 导出 (L627-686)

```python
# SLAT → Gaussian + Mesh
decoded = pipeline.decode_slat(slat_edited, ['gaussian', 'mesh'])

# 渲染 120 帧对比视频：[ori_img | ori_gs | edit_img | edit_gs]
comp = [np.concatenate([ori_thumb, ori_vid[i], edit_thumb, edit_vid[i]], axis=1)
        for i in range(len(ori_vid))]

# 导出 GLB mesh（简化 95% 三角面 + 1024 纹理）
glb = postprocessing_utils.to_glb(gs_edit, decoded['mesh'][0],
                                   simplify=0.95, texture_size=1024)
```

同时渲染原始和编辑后的 Gaussian Splatting，拼接成 4 列对比视频。

---

## 数据流总结

```
┌─────────────────────────────────────────────────────────────────┐
│ 输入数据（全部来自 VD/PartCraft3D 已有缓存）                       │
│                                                                 │
│  VD pre-rendered view ──→ preprocess_image() ──→ ori_img_pp     │
│  VD SLAT feats.pt ──→ (feats - mean) / std ──→ norm_feats      │
│  VD SLAT coords.pt ──→ ori_coords                              │
│  ori_img_pp ──→ TRELLIS SS sampler ──→ z_s (SS latent)         │
│  Phase 2.5 2D edit cache ──→ edited_img_pp                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: SS 编辑 (img_to_voxel, DiT-L, 16³)                    │
│                                                                 │
│  条件: DINOv2(ori_img) + DINOv2(edited_img) + z_s              │
│  输入: 随机噪声 [1,8,16,16,16]                                  │
│  输出: 编辑后 SS latent → decode → 体素坐标 coords_edit          │
│  含义: 决定编辑后物体的 3D 结构（哪里有/没有体素）                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: SLAT 编辑 (voxel_to_texture, DiT-L, 64³)              │
│                                                                 │
│  条件: DINOv2(ori_img) + DINOv2(edited_img) + ori_SLAT_norm    │
│  输入: 稀疏噪声 SparseTensor(coords=coords_edit)                │
│  输出: 编辑后 SLAT (归一化) → × std + mean → SLAT (原始空间)     │
│  含义: 在新结构上生成纹理/外观特征                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TRELLIS 解码器 (不参与训练，直接复用)                              │
│                                                                 │
│  SLAT → Gaussian Splatting → 渲染对比视频                       │
│  SLAT → Mesh → GLB 导出                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 与 VD (TrellisRefiner) 方法的核心差异

| 维度 | VD (TrellisRefiner) | 3DEditFormer |
|---|---|---|
| **编辑策略** | Flow Inversion + Repaint（反演→mask区域注入→去噪） | 条件生成（图像条件→直接生成编辑后latent） |
| **是否需要 mask** | 需要（精确的部件mask控制编辑区域） | 不需要（模型自动学习编辑区域） |
| **结构编辑** | 不改变体素结构（仅修改 SLAT 特征） | Stage 1 重新生成体素结构 |
| **文本条件** | S1/S2 prompt 分解 + CFG 引导 | 无文本条件（纯图像条件） |
| **编辑条件** | 文本 + 可选图像（DINOv2 特征残差） | 编辑前/后图像对（DINOv2 特征差异） |
| **训练依赖** | 无需训练（利用 TRELLIS 预训练 flow） | 需要在编辑数据集上训练两阶段 denoiser |
| **编辑粒度** | 受 mask 精度影响，可部件级精确控制 | 全局编辑，由模型学习哪里该变 |
| **计算量** | Flow inversion（~50步）+ Repaint（~50步） | SS采样（25步）+ SLAT采样（25步） |
