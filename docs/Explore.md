# TRELLIS Explore Notes

## 1) 本次问题总览

这次主要梳理了 TRELLIS 的以下问题：

- `encode_into_SLAT.py` 在做什么，SLAT 的形状是什么
- TRELLIS 中 encode / decode 代码位置
- 训练模型和 encode 模型可直接读取的数据格式
- `structured latents` 在代码里具体怎么定义、怎么进入/输出 pipeline
- stage1 sparse structure 与 stage2 structured latent 的关系，是否可互转
- 两套 VAE（sparse structure VAE vs structured latent VAE）差异与训练方式
- flow transformer（dense/sparse）结构、原理、训练，以及和普通 transformer 的区别
- flow matching 中 velocity 的预测机制

## 2) 关键结论

### 2.1 `structured latents` 的本质

- 在 TRELLIS 里，`structured latents` 本质是 `sp.SparseTensor`（不是单独的新数据类）。
- 由两部分组成：
  - `feats`: `[N, C_latent]`
  - `coords`: `[N, 4]`，格式 `[batch_idx, x, y, z]`

代码定义：
- `third_party/trellis/modules/sparse/basic.py` (`class SparseTensor`)

### 2.2 `encode_into_SLAT.py` 的输出形状

- 脚本先把多视角 DINO 特征投影/聚合到 voxel，再送入 `SLatEncoder`。
- 最终保存两个文件：
  - `*_feats.pt`
  - `*_coords.pt`
- 在该脚本里校验约束为：
  - feat dim = 8
  - coord dim = 4

代码：
- `third_party/encode_asset/encode_into_SLAT.py`

### 2.3 sparse structure 和 structured latent 的关系

- stage1 生成 `sparse structure`（体素占据坐标）
- stage2 在这些坐标上生成 `structured latent` 特征

pipeline 中耦合方式：
1. `sample_sparse_structure` 得到 `coords`
2. `sample_slat` 用同一 `coords` 初始化 sparse noise 并采样 `slat`
3. `decode_slat` 将 `slat` 解码成 `mesh/gaussian/radiance_field`

代码：
- `third_party/trellis/pipelines/trellis_image_to_3d.py`
- `third_party/trellis/pipelines/trellis_text_to_3d.py`

### 2.4 只保存了 SLAT，能不能恢复 structure / z_s

- `SLAT -> structure`：可以（直接用 `slat.coords`）
- `SLAT -> 原始 z_s`：不能无损恢复（信息不可逆）
- 可以用恢复的 occupancy 重新编码得到 `z_s'`，但不是原始采样时的 `z_s`

### 2.5 两套 VAE 的区别

#### A) Stage1 Sparse Structure VAE

- 目标：重建体素占据（几何结构）
- 数据形态：稠密 3D 张量 `[B,1,R,R,R]`
- 损失：BCE/L1/Dice + KL

代码：
- 模型：`third_party/trellis/models/sparse_structure_vae.py`
- 训练：`third_party/trellis/trainers/vae/sparse_structure_vae.py`
- 数据：`third_party/trellis/datasets/sparse_structure.py`

#### B) Structured Latent VAE（资产 encoding/decoding 相关）

- 目标：学习稀疏 latent 并解码到可渲染 3D 表示
- 数据形态：`SparseTensor(coords+feats)` + 渲染监督
- 损失（主线）：重建图像（L1/SSIM/LPIPS）+ KL

代码：
- 编码器：`third_party/trellis/models/structured_latent_vae/encoder.py`
- 解码器：
  - `decoder_gs.py`
  - `decoder_rf.py`
  - `decoder_mesh.py`
- 训练：
  - `third_party/trellis/trainers/vae/structured_latent_vae_gaussian.py`
  - `third_party/trellis/trainers/vae/structured_latent_vae_rf_dec.py`
  - `third_party/trellis/trainers/vae/structured_latent_vae_mesh_dec.py`

## 3) Flow Transformer 总结（stage1 vs stage2）

### 3.1 Stage1 Dense Flow Transformer

- 模型：`third_party/trellis/models/sparse_structure_flow.py` (`SparseStructureFlowModel`)
- 输入：稠密 3D latent `[B,C,R,R,R]`
- 机制：`patchify -> transformer blocks -> unpatchify`
- block：`ModulatedTransformerCrossBlock`（self-attn + cross-attn + FFN + AdaLN）

相关模块：
- `third_party/trellis/modules/transformer/modulated.py`
- `third_party/trellis/modules/transformer/blocks.py`

### 3.2 Stage2 Sparse Flow Transformer

- 模型：`third_party/trellis/models/structured_latent_flow.py` (`SLatFlowModel`)
- 输入：`SparseTensor`（稀疏坐标上的 token）
- 机制：sparse self-attn / cross-attn（支持 serialized/windowed）
- block：`ModulatedSparseTransformerCrossBlock`

相关模块：
- `third_party/trellis/modules/sparse/transformer/modulated.py`
- `third_party/trellis/modules/sparse/transformer/blocks.py`
- `third_party/trellis/modules/sparse/attention/modules.py`

## 4) Flow Matching 中 velocity 怎么预测

核心流程（训练）：

1. 采样 `t` 和 `noise`
2. 构造 `x_t = diffuse(x_0, t, noise)`
3. 用 denoiser transformer 预测 `pred_v`
4. 用解析公式算 `target_v = (1 - sigma_min) * noise - x_0`
5. MSE 对齐：`MSE(pred_v, target_v)`

代码：
- 通用 flow trainer：`third_party/trellis/trainers/flow_matching/flow_matching.py`
- sparse flow trainer：`third_party/trellis/trainers/flow_matching/sparse_flow_matching.py`
- 采样器（Euler ODE）：`third_party/trellis/pipelines/samplers/flow_euler.py`

结论：
- 训练时是“单步监督速度场”（每次一个 t 的前向）
- 推理时是“多步积分速度场”（Euler 多步更新）

## 5) `ss dataset` 来源

`SparseStructure` 数据集是离线数据读取，不是在线构建：
- `root/metadata.csv`（含 `voxelized` 标记）
- `root/voxels/<sha256>.ply`

代码：
- `third_party/trellis/datasets/components.py`
- `third_party/trellis/datasets/sparse_structure.py`

补充：
- 当前 `encode_asset/render_img_for_enc.py` 也会做 voxelize，但输出目录是 `img_Enc/<name>/voxels.ply`，与 `SparseStructure` 训练目录组织不同（概念一致，组织格式不同）。
