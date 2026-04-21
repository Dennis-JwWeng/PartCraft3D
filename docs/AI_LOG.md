# AI_LOG（Pipeline v3）

按时间倒序记录与 **pipeline_v3** 相关的决策与变更。  
实现细节以 **`partcraft/pipeline_v3/`** 与 **`docs/ARCH.md`** 为准。  
其他主题（pipeline_v2、H3D_v1、清洗、训练等）的历史条目已从此文件移除，见 git 历史。

---

## 2026-04-21 — pipeline_v3：ckpt 根目录与 Trellis worker 配置对齐

**背景**：`load_config()` 只认环境变量 **`PARTCRAFT_CKPT_ROOT`** 作为权重根覆盖；machine env 长期只写 **`TRELLIS_CKPT_ROOT`**，二者未同步时易出现 Trellis 权重路径不符合预期。另 aibox 上 `trellis_workers_per_gpu: 2` 会每 GPU fork 两个 worker，NFS 冷读权重时压力大。

**变更**：

- **`configs/machine/aibox-rd3996bf91f9-68f4cd496c-nsm56.env`**：`export PARTCRAFT_CKPT_ROOT="${PARTCRAFT_CKPT_ROOT:-${TRELLIS_CKPT_ROOT}}"`。
- **`configs/machine/aibox-re289eb56bb7-6b4f8cd8fb-xgpnz.env`**：同上（dev_hs 挂载 sister 机）。
- **`configs/pipeline_v3_shard09.yaml`**：`services.image_edit.trellis_text_ckpt` 改为相对名 `TRELLIS-text-xlarge`（在 `ckpt_root` / `PARTCRAFT_CKPT_ROOT` 下解析）；`trellis_workers_per_gpu: 1`；`ckpt_root` 注释说明可被 `PARTCRAFT_CKPT_ROOT` 覆盖。

**文档**：`docs/ARCH.md`「权重路径」补充 `PARTCRAFT_CKPT_ROOT` 与 `TRELLIS_CKPT_ROOT` 关系及相对 `trellis_text_ckpt` 的推荐写法。

---

## 2026-04-20 — pipeline_v3 color 编辑 3D 生效（双 bug 修复）

**背景**：shard08 `mode_e_text_align` 的 color 编辑 gate_e 通过率只有 **14.4%**（同配置下 `material` 60.8%、`global` 85.2%）。FLUX 已经把 2D 颜色改对了（`edits_2d/clr_*_edited.png` 显示目标色），但 TRELLIS 输出的 5 视图预览和 BEFORE 几乎像素级相同。定位到两个独立的退化路径同时存在。

**变更**：

- **Bug 1 — DINOv2 图像条件没接上 Color**  
  `partcraft/pipeline_v3/trellis_utils.py::resolve_2d_conditioning` 的白名单只列 `Modification/Scale/Material/Global`，color 编辑一律 `return None`。`TrellisRefiner.edit` 在 `repaint_mode='image'` 下 `img_cond is None and img_new is None` 触发 `effective_mode='interleaved'` 回退到一张 518×518 白图，FLUX 的 recolour **完全没有进入 S2**。白名单加入 `"Color"` 后，`encode_multiview_cond` 产出的多视角 DINOv2 特征正确注入 S2。

- **Bug 2 — S2 positive text 看到的是 BEFORE 文本**  
  `partcraft/pipeline_v3/specs.py::EditSpec.from_parsed_edit` 的 `new_parts_desc` fallback 链是 `edit["new_parts_desc"] → edit["target_part_desc"]`，但 Phase-1 VLM 的 parsed.json **从不写** `new_parts_desc` 字段，只写 `after_desc`。修复：fallback 链插入 `edit.get("after_desc")` 作为中间级（`new_parts_desc → after_desc → target_part_desc`），有 `new_parts_desc` 时行为不变。

- **Bench harness**（`scripts/tools/bench_color_fix.py`）：独立实验驱动；monkey-patch `trellis_3d.iter_flux_specs` + `preview_render.iter_all_specs` 仅产 color 的 specs。

**验证**：多组随机样本中，典型 color case 在修复后 5 视角预览出现目标色相（详见原提交说明与 bench 报告）。

**影响**：未改变 material/modification/global 的主路径语义；material 的 `new_s2_cpl` 在缺 `new_parts_desc` 时会吃到 `after_desc` 中的材质描述，预期略有正面影响。

**Commits**（节选）：
- `39f7b09` fix(pipeline_v3): route color edits through DINOv2 2D conditioning  
- `504a033` fix(pipeline_v3): use after_desc fallback when new_parts_desc absent  
- `edc2e12` tools(bench): isolated color-fix re-run harness for shard08  

---
