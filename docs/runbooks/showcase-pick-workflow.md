# Runbook: Showcase / gallery 编辑对人工挑选流程

**用途**：从已经跑完 v3 管线（`text_gen_gate_a → flux_2d → del_mesh →
trellis_preview → gate_quality`）的 shard 里，**手动挑出 N 个高质量编辑对**，
最后输出：

- `shard_showcase_hero.html` — 给 review 用的高质量 5-视角 BEFORE/AFTER 拼图（PNG 渲染）
- `gallery_picks.json` — 拿去做 GLB gallery / Blender 场景搭建的 picks 清单

> 历史背景：2026-04-19 为 H3D Dataset Overview 报告挑出 28 个 hero（shard08）；
> 2026-04-19 为多 shard gallery 挑出 50 个候选（shard07 + shard08）。

---

## 0. 前置条件

| 项 | 检查 |
|---|---|
| pipeline_v3 完成 | `outputs/partverse/<shard>/mode_e_text_align/objects/<NN>/<obj_id>/edit_status.json` 内绝大多数 edit `final_pass=true` |
| 输入 NPZ 在线 | `data/partverse/inputs/images/<NN>/<obj_id>.npz`（提供 BEFORE 5-view 帧） |
| 输入 mesh 在线 | `data/partverse/inputs/mesh/<NN>/<obj_id>.npz` 内含 `full.glb`、`part_*.glb`（gallery 阶段才用到） |
| Python 环境 | `numpy + Pillow`，无 GPU 依赖；conda `vinedresser3d` 即可 |

---

## 1. 数据来源契约

挑选脚本只读这些文件，**不会回写任何东西**：

| 字段 | 文件 |
|---|---|
| edit final_pass / VLM score / pixel_counts / best_view | `<obj>/edit_status.json` |
| edit_type / prompt / target_part_desc / selected_part_ids | `<obj>/phase1/parsed.json`（按 pipeline_v3.specs 的 seq 规则映射出 `eid`） |
| **addition** edit 的 prompt 与 source_del_id | `<obj>/edits_3d/<add_eid>/meta.json`（addition 不在 parsed.json 里） |
| BEFORE 5 视角图 | `data/partverse/inputs/images/<shard>/<obj>.npz` 取 view `[89, 90, 91, 100, 8]`，**RGBA → 白底合成** |
| AFTER 5 视角图 | `<obj>/edits_3d/<eid>/preview_{0..4}.png`（Blender 渲染） |
| **deletion 后的 GLB** | `<obj>/edits_3d/<del_eid>/after_new.glb` |

### 1.1 BEFORE/AFTER 的特殊规则

- **deletion / FLUX (mod/mat/clr/glb/scl)**：BEFORE = NPZ 5 帧（原物体），
  AFTER = `preview_{0..4}.png`（编辑后渲染）
- **addition**：是 deletion 的反向 —— BEFORE = `source_del_id` 的
  `preview_{0..4}.png`（已删除状态），AFTER = NPZ 5 帧（part 还在的原物体）。
  这点和直觉相反，**一定要走 source_del_id 的 preview，不能用 add 自己的
  preview** —— add 的 preview 是 TRELLIS 重建版，可能带伪影。

### 1.2 白底强制

NPZ 内的视角图是 RGBA，preview_*.png 也是 RGBA。所有进入卡片的图都先经
`_flatten_to_white(RGBA → RGB on white)`，与 gate_a / gate_e 看到的版本保持一致。
不做这一步会得到 hero 报告里黑底 + 透明 PNG 的怪异混排。

---

## 2. `eid` 规则（脚本里要复刻的）

`partcraft/pipeline_v3/specs.iter_all_specs` 决定 `parsed.json.edits[*]` 怎么映射到
最终 `eid`：

- **deletion** 用独立 `del_seq`（0-based），输出 `del_<obj>_<NNN>`
- **FLUX 五类** (`modification / scale / material / color / global`) 共享同一个
  `flux_seq`，按 parsed.json 里出现的顺序累加，输出 `mod_/scl_/mat_/clr_/glb_<obj>_<NNN>`
- **addition** 不在 parsed.json 里，由 s7 阶段从 deletion 反推；prompt 只能从
  `edits_3d/<add_eid>/meta.json` 拿

挑选脚本里这部分逻辑在 `_gather_candidates` / `_gather_one_shard` 里，**改 specs.py
里的 seq 规则时这里也要同步**，否则 prompt 会贴错卡。

---

## 3. 三个脚本的分工

所有脚本都在 `scripts/tools/` 下，纯 IO，不依赖 GPU：

| 脚本 | 入口 | 用途 |
|---|---|---|
| `build_showcase_candidates.py` | 单 shard | 生成候选 HTML（用于挑 hero） |
| `build_gallery_candidates.py` | 多 shard | 生成跨 shard 候选 HTML（用于挑 gallery，含 GLB-availability 标记） |
| `build_showcase_hero.py` | 单 / 多 shard 均可 | 根据 picks JSON 生成最终 hero HTML |

候选脚本的输出（HTML）含一段 vanilla JS：
- 点击星标 → 收集 picks
- 右下角 `<pre>` 实时显示 picks 列表
- 顶部 toolbar 按 `type / shard / glb / picked-only` 筛选

---

## 4. 一次完整流程（以 50 个 gallery 为例）

### 4.1 生成候选页

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python scripts/tools/build_gallery_candidates.py \
  --shards 07 08 \
  --root outputs/partverse/shard07/mode_e_text_align \
  --root outputs/partverse/shard08/mode_e_text_align \
  --images-root data/partverse/inputs/images \
  --seed 2026 \
  --target 50 \
  --quota deletion=60 modification=70 global=60 material=50 \
          scale=30 color=30 addition=50 \
  --out reports/h3d_gallery_candidates.html
```

要点：
- `--shards` 与 `--root` **数量必须相等且一一对应**
- `--seed` 决定 deletion / addition 的同分抽样次序，以及最终洗牌；
  换 seed 会得到完全不同的候选池
- `--target` 只影响 HTML 上的 counter 文案（`X / 50 picked`）
- `--quota` 总和 = 候选页总卡数；典型 ~7× 目标 picks（50 → 350）

打开 `reports/h3d_gallery_candidates.html`，按需用 toolbar 过滤
（type / shard / `glb ready vs needs decode`）后点星标。

### 4.2 拷出 picks JSON

右下角 `copy JSON` 按钮 → 粘贴成文件，例如
`reports/h3d_gallery_picks.json`。

格式（card id 由候选脚本类型决定）：

- 单 shard 候选页：`"<obj>/<eid>"`
- 多 shard 候选页：`"<shard>/<obj>/<eid>"`

`build_showcase_hero.py` 两种格式都吃。

### 4.3 生成最终 hero 报告

```bash
python scripts/tools/build_showcase_hero.py \
  --root outputs/partverse/shard08/mode_e_text_align \
  --shard 08 \
  --images-root data/partverse/inputs/images \
  --picks reports/shard08_picks.json \
  --out reports/shard08_showcase_hero.html
```

参数：
- `--shard` 当前还是单 shard 入口；如果 picks 含多 shard，需要按 shard 拆开
  跑两次再拼，或扩展脚本（TODO，多 shard hero 还没做）
- 默认产出 360px-高的 5-view 拼图，JPEG quality=92

---

## 5. GLB / Blender 场景搭建（gallery 阶段）

候选脚本会在每张卡上打 `glb ready` / `needs decode` 徽章。规则：

| edit_type | GLB 来源 | 说明 |
|---|---|---|
| `deletion` | `outputs/.../edits_3d/<del_eid>/after_new.glb` | s5b (`del_mesh`) 直接写出 |
| `addition` | source_del 的 `after_new.glb` 当 BEFORE，input mesh NPZ 内的 `full.glb` 当 AFTER | 不需要新建文件，引用即可 |
| `modification` / `material` / `color` / `global` / `scale` | **磁盘上没有 GLB**，只有 `<eid>/after.npz` 里的 TRELLIS slat tokens (`slat_feats / slat_coords / ss`) | 需要走 `partcraft/trellis/refiner.py` 解码出 PLY/GLB |

> 当前未提供 picks → GLB 解码的成品脚本。最朴素做法是仿照
> `partcraft/pipeline_v3/preview_render.py` 里 `after_new.glb`/`after.ply` 的
> 处理路径写一个解码 CLI；或对 picks 中的 FLUX 项重跑 trellis_preview 阶段以
> 复用现有产物。优先挑 `glb ready` 的卡能完全跳过这一步。

`full.glb` / `part_*.glb` 的取法（Blender 里直接 import）：

```python
import numpy as np, io
mesh = np.load("data/partverse/inputs/mesh/08/<obj>.npz")
glb_bytes = bytes(mesh["full.glb"].tobytes())   # 或 part_0.glb / part_1.glb / ...
open("/tmp/full.glb", "wb").write(glb_bytes)
```

---

## 6. 报告产物的归宿

`reports/` 里目前的命名约定：

| 文件 | 含义 |
|---|---|
| `<tag>_showcase_candidates.html` | 单 shard 候选页（旧脚本） |
| `h3d_gallery_candidates.html` / `<tag>_gallery_candidates.html` | 多 shard 候选页（新脚本） |
| `<tag>_picks.json` | 用户挑选结果 |
| `<tag>_showcase_hero.html` | 最终 hero 高质量报告 |

这些都是**自包含 HTML**（图全 base64 内嵌），可以直接拷贝走。

---

## 7. 常见坑

- **prompt 不对**：检查 specs.py 里 deletion / FLUX 的 seq 规则有没有改；
  候选脚本的 `_gather_candidates / _gather_one_shard` 必须保持同步
- **addition 卡 BEFORE/AFTER 一样** ：忘了走 source_del_id 的 preview。
  这是 add 的特殊路径，普通 FLUX 走不到
- **背景是黑的** ：忘了 `_flatten_to_white`。RGBA → RGB 不能直接 `.convert("RGB")`，
  会丢掉 alpha 信息变成黑底
- **shard07 没有 addition**：是因为该 shard 还没跑 addition s7 阶段，不是
  脚本 bug；多 shard 候选页里 add 卡全部来自 shard08
