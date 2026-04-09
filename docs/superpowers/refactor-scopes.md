# PartCraft3D 重构 scope 索引（对话外可检索）

本文件记录曾在「配置 / I/O / 可跑性」讨论中拆过的 **scope 字母含义**，避免只存在于某次聊天里。  
**正式设计与实现计划**仍以各自日期的 `docs/superpowers/specs/*.md` 与 `docs/superpowers/plans/*.md` 为准。

| Scope | 一句话 | 状态 |
|-------|--------|------|
| **A** | 新机器：依赖与路径对齐、`--check`、单一 onboarding 文档、跑通全 phase（打包数据已就绪） | **已有 spec + 已实现** → [`specs/2026-04-09-new-machine-onboarding-design.md`](specs/2026-04-09-new-machine-onboarding-design.md) |
| **B** | 输入/输出目录语义：`data_dir` / `output_dir` / `dataset_root` 等与文档、磁盘布局 1:1，减少隐式派生 | **已落地（首版）**：[`dataset-path-contract.md`](../dataset-path-contract.md)；`load_config` 在 pipeline 模式下同步 `images_root`↔`image_npz_dir`、`mesh_root`↔`mesh_npz_dir` |
| **C** | 代码结构：`paths` / `config` / 各 step 路径拼装集中，减少重复与分支 | **首版已落地**：`DatasetRoots`（`paths.py`）+ `run.py` 统一输入 NPZ 根路径；见 [`specs/2026-04-09-scope-c-paths-structure-design.md`](specs/2026-04-09-scope-c-paths-structure-design.md) |
| **D** | 可重复验证：小数据 smoke、`LIMIT=1` 穿全 phase 等 | **首版已落地**：`LIMIT` 在 `run.py` 生效；[`smoke-pipeline.md`](../../smoke-pipeline.md)；spec → [`specs/2026-04-09-scope-d-smoke-design.md`](specs/2026-04-09-scope-d-smoke-design.md) |
| **E** | B+C+D 都要，但 **分阶段** 做（每期一个 spec） | 未立项 |

**新开对话时怎么用：** 在第一条消息写「继续做 scope **B**（或 C）」并 `@` 本文件或对应 spec 路径即可。

**与 `docs/ARCH.md` 的关系：** 架构与数据契约以 `ARCH.md` 为权威；本文件只做 **scope 字母表**，不重复架构正文。

## Phase 2（配置/代码继续收敛）

- **Wave 1（已记录）**：Blender 路径 — [`specs/2026-04-10-phase2-config-hygiene-blender.md`](specs/2026-04-10-phase2-config-hygiene-blender.md)

