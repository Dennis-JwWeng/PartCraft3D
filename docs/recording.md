# PartCraft3D 工作记录

## 2026-03-11 语义提示修复 + 本地 VLM 部署测试

### 问题背景

Pipeline 生成的 addition prompt 存在语义矛盾：`object_desc` 描述的是**完整物体**（包含所有零件），但 addition 编辑的 "before" 状态应该是**缺少该零件的物体**。

**典型错误示例：**
> Add doorknob to: A simple rectangular door with a light teal panel, a light blue frame, **and a small brown doorknob**.

门把手已经在描述里了，却要求"添加门把手"——逻辑矛盾。

### 解决方案：VLM 生成 `desc_without`

在 Phase 0 语义标注阶段，让 VLM 为每个非核心零件生成 `desc_without` 字段——描述**移除该零件后**物体的外观。

**修改的文件：**

| 文件 | 改动 |
|------|------|
| `partcraft/phase0_semantic/labeler.py` | VLM prompt 增加 `desc_without` 字段要求；新增 `_create_vlm_client()` 支持 local/api 双后端 |
| `partcraft/phase0_semantic/catalog.py` | `CatalogEntry` 增加 `desc_without` 字段；`save()`/`load()`/`from_phase0_output()` 同步更新 |
| `partcraft/phase1_planning/planner.py` | `EditSpec` 增加 `before_desc` 字段；deletion 用 `object_desc`，addition 用 `desc_without` |
| `partcraft/phase2_assembly/assembler.py` | `_make_result_dict()` 输出 `before_desc` |
| `partcraft/phase4_filter/instruction.py` | addition 模板使用 `before_desc` 代替 `object_desc` |
| `scripts/visualize_edit_pair.py` | addition prompt 使用 manifest 中的 `before_desc` |

**向后兼容：** 旧 Phase 0 数据缺少 `desc_without` 字段时默认为空字符串，planner 回退到 `object_desc`。

### 本地 VLM 部署方案

为节省 API token 开销，新增本地 GPU 部署选项：

**新增文件：**
- `configs/local_vlm.yaml` — 本地 VLM 专用配置，`vlm_backend: "local"`
- `scripts/launch_local_vlm.sh` — vLLM/SGLang 一键启动脚本

**支持的部署方式：**
- vLLM (`pip install vllm>=0.6.0`)
- SGLang (`pip install sglang[all]`)
- 运行环境：`qwen_test` conda 环境（已安装 SGLang 0.5.6）

### 本地模型测试：Qwen3-VL-2B-Instruct

**模型路径：** `/Node11_nvme/wjw/checkpoints/Qwen3-VL-2B-Instruct`

使用 SGLang 在 GPU 6 上部署，测试 10 条数据，结果 9/10 成功但**质量很差**：

| 问题 | 示例 |
|------|------|
| object_desc 太泛 | "A 3D object with two segmented parts"（未识别物体类型） |
| 标签重复/幻觉 | 4 个 part 全标成 `car_hood`；出现 shoulder、face、eye 等无关标签 |
| desc_without 模板化 | "A mechanical device with X but without X"（机械重复） |
| core 判断失误 | 19 个 part 全标 core=True |

**结论：2B 模型参数量不足，语义标注不可用。**

### API 测试：Gemini-2.5-Flash

同样测试 10 条数据，7/10 成功，质量显著优于本地 2B：

| 对比项 | 2B 本地 | Gemini API |
|--------|---------|-----------|
| object_desc | 泛化描述 | 具体描述（"A fantasy sword with a grey blade, a dark red gem..."） |
| 标签质量 | 瞎猜重复 | 准确语义标签（`sword_blade`, `hilt_gem`, `hilt_collar`） |
| desc_without | 模板化 | 语义正确（"A sword with a grey blade...but missing the dark red gem accent"） |
| core 判断 | 全对/全错 | 合理区分核心/外围 |

### 当前数据状态

- `outputs/cache/phase0/semantic_labels.jsonl`: 59 条（52 旧 + 7 新 API）
- 已清除 2B 模型生成的 9 条低质量数据
- 总数据集：940 objects（shard 00），尚未全量运行

### 待办

- [ ] 用 API 跑完全部 940 条 Phase 0 数据（含 `desc_without`）
- [ ] 旧 52 条数据缺少 `desc_without`，需要重新跑或补充
- [ ] 如需本地部署，考虑下载更大模型（Qwen3-VL-7B-Instruct 或以上）
- [ ] 测试 Phase 1 → Phase 4 完整流程
