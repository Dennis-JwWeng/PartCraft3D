# AI_LOG（Pipeline v3）

按时间倒序记录与 **pipeline_v3** 相关的决策与变更。  
实现细节以 **`partcraft/pipeline_v3/`** 与 **`docs/ARCH.md`** 为准。  
其他主题（pipeline_v2、H3D_v1、清洗、训练等）的历史条目已从此文件移除，见 git 历史。

---

## 2026-04-22 — pipeline_v3 `preview_del`：新增 `--best-view-only` 单视角回补

**背景**：shard05 跑完整条 v3 pipeline 后发现 `edits_3d/del_*/` 里只有 `after.npz` / `after_new.glb`，没有 `preview_*.png`（上游 `preview_del` 阶段被漏掉）。`pull_deletion` 仍能入库 NPZ，但 H3D_v1 `deletion/.../after.png` 因缺少 pipeline `preview_{k}.png` 源一直为空，后续 `pull_addition` 全部 `skip=pair_after_image_missing`，`promoted=0`。

**决策**：不回跑上游整条 pipeline，也不做 `pull_deletion` 40 视角渲染结果的角度匹配 fallback；直接给 `preview_del` 加一个**单视角快速回补路径**。H3D_v1 promoter 本来就只 hardlink 一张 `preview_{best_view_index}.png`（见 `partcraft.cleaning.h3d_v1.promoter._views_block`），渲 5 张是纯浪费。

**变更**：

- **`partcraft/pipeline_v3/preview_render.py`**
  - 新增 `DEFAULT_FRONT_VIEW_INDEX = 4` 与 `_best_view_slot_for_edit(ctx, edit_id)`：按 del → paired del（针对 addition）→ 默认槽 4 的顺序解析 `edit_status.json → gates.A.vlm.best_view`，与 promoter 侧逻辑完全一致。
  - `_read_camera_views_from_npz(…, slots=…)` 支持子集槽位读取；新增 `_render_glb_views(…, view_slots=…)` 返回 `{slot: BGR_img}`；旧 `_render_glb_five_views` 改为薄 wrapper 保持向后兼容。
  - `_write_preview_images_by_slot(edit_dir, imgs_by_slot)` 按槽位写 `preview_{slot}.png`；`_slot_previews_exist(edit_dir, slots)` 做 slot-precise skip。
  - `render_del_previews_for_object(…, best_view_only=False)` 与 `render_del_previews_batch(…, best_view_only=False)`：True 时 deletion 只渲 1 张 best view，addition 只复制 paired del 的这 1 张；False 时行为与旧实现字节等价。`_count_pending_edits` 也按 slot-precise 计数。
- **`partcraft/pipeline_v3/run.py`**：新增 CLI `--best-view-only`，在 `preview_del` 分支透传；多 GPU 子进程 `spawn` 同步带上该 flag。
- **`tests/test_preview_best_view_only.py`**：新增 6 条 unit test 覆盖 best_view 解析（正常 / 缺失 / 越界 / addition 镜像 / addition 无 paired）与 slot 存在性，全部通过，`tests.test_pipeline_v3_hooks` 27/27 无回归。
- **文档**：`docs/runbooks/h3d-v1-promote.md` 新增 §2b「Backfill `preview_{k}.png` on shards that skipped `preview_del`」，含命令、耗时估算、两条 troubleshooting 行；`preview_render.py` 模块 docstring 说明 fast-path 语义。

**验证**：shard05 单 obj `5dd321dc17774aeb830876bbb588d188`（2 deletion + 2 addition）上跑：`del_000 best_view=3` → `preview_3.png`，`del_001 best_view=0` → `preview_0.png`，对应 addition 镜像拷贝。侧视对比 `image_npz` 同一槽位的原物体渲染，相机角度 + `normalize_scene(scale, offset)` 一致，保留部分尺寸/位置不会被 re-normalize 拉近。单 GPU 约 2.5–5 s/edit，8 GPU 全 shard05（~3223 条 del）≈ 17 min。

**默认行为**：未传 `--best-view-only` 时 `preview_del` 路径与旧实现完全一致（5 张）。已跑完 5 视角的 shard（06/07/08）不受影响。

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
