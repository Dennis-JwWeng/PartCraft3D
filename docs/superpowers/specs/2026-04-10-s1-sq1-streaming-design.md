# Spec: s1+sq1 per-object streaming — VLM 生命周期合并

**日期**: 2026-04-10  
**状态**: 待实现  
**目标**: 消除 Stage A → A_qc 之间的 VLM 重启开销，并把 sq1 与 s1 的 per-object 执行流水线化，缩短整体 A+A_qc 耗时。

---

## 背景与问题

现有管线把 `s1`（VLM 生成 parsed.json）和 `sq1`（VLM 校验 edit instruction）拆成两个独立 stage（A / A_qc），每个 stage 各自 start_vlm / stop_vlm。问题有两个：

1. **重启开销**：两次 start_vlm，SGLang 27B 加载约 5 min，每次完整 pipeline 浪费一次。
2. **批次串行**：sq1 必须等所有对象的 s1 全部完成才能开始，对 1000 个对象约浪费 T(sq1) ≈ 1–2 h 的 overlap 时间。

sq1 只依赖 s1 的 per-object 产物（`parsed.json` + `overview.png`），因此可以在每个对象 s1 完成后立即触发，无需等待整个 batch。

---

## 目标与约束

- **不改** resume 协议、validators、config 格式的向后兼容性
- **不改** sq2、sq3（留作后续迭代）
- 旧 config（仍有独立 A + A_qc stage）继续可用，只是不享受流水线加速

---

## 设计

### 1. Config 层（推荐做法）

将 A 和 A_qc 合并为单个 stage，servers 仍为 `vlm`：

```yaml
# 推荐新配置
- {name: A, desc: "phase1 VLM + QC-A", servers: vlm, steps: [s1, sq1]}
# 删除原独立的 A_qc entry
```

旧配置（A + A_qc 分开）仍然可以运行，shell 逻辑不变，只是 VLM 会重启一次（性能损失但不报错）。

### 2. Shell 层

不变。服务 start/stop 在 stage 边界，合并后 VLM 只启停一次。

### 3. Python 层

#### 3a. `s1_phase1_vlm.py` — 新增 `post_object_fn` 参数

`run_many_streaming` 增加可选参数：

```python
async def run_many_streaming(
    ctxs,
    *,
    blender, vlm_urls, vlm_model,
    n_prerender_workers=8,
    force=False,
    log_every=20,
    post_object_fn=None,   # NEW: Callable[[ObjectContext, str], Awaitable[None]]
                           # 参数: ctx, vlm_url（该 consumer 对应的 server URL）
) -> list[Phase1Result]:
```

在 `consumer(idx)` 中，`_call_one` 返回后立即调用（仅跳过 `too_many_parts`）：

```python
res = await _call_one(client, ctx, ...)
results.append(res)
if post_object_fn is not None and res.error != "too_many_parts":
    await post_object_fn(ctx, vlm_urls[idx])
```

说明：
- `too_many_parts` 对象没有 `parsed.json`，跳过 hook 避免 sq1 误报 FAIL。
- s1 FAIL 对象（VLM 调用失败）：`_process_one` 会检测 `missing_parsed_json` 并标记 FAIL，行为与独立运行 sq1 一致。
- consumer 本身已串行化（每次处理一个对象），sq1 的 `asyncio.gather`（per-edit 并发）在 consumer 内安全。

#### 3b. `run.py` — look-ahead 传递 hook

look-ahead 发生在 `main()` 的 step 循环中（`run_step` 不持有 steps 列表）。实现方式：`run_step` 增加可选 `post_object_fn=None` 参数，main() 在调用前计算 hook：

```python
# main() 的 step 循环内
for i, step in enumerate(steps):
    post_fn = None
    if step == "s1" and "sq1" in steps:
        from .sq1_qc_a import _process_one as _sq1_one
        _model = psvc.vlm_model_name(cfg)
        _force = args.force
        async def _post(ctx, vlm_url, _m=_model, _f=_force):
            await _sq1_one(ctx, vlm_url, _m, _f)
        post_fn = _post
    run_step(step, ctxs, cfg, args, post_object_fn=post_fn)

# run_step 签名加一个可选参数：
def run_step(step, ctxs, cfg, args, post_object_fn=None):
    ...
    if step == "s1":
        asyncio.run(run_many_streaming(
            ctxs, ..., post_object_fn=post_object_fn,
        ))
```

当 steps 随后执行 `sq1` 时，所有对象均已有 `step_done(ctx, "sq1_qc_A") == True`，`run` 全部跳过（**catch-up pass**，用于 resume 兜底）。

### 4. Resume 正确性

| 崩溃场景 | 重启行为 |
|---------|---------|
| s1 完成 500/1000，sq1 跟着完成同 500 | 500 对象 parsed.json 存在且 sq1_qc_A done → 两步均跳过；续跑剩余 500 |
| s1 全部完成（1000 parsed.json），sq1 一个未跑 | s1 todo=[]（全 skip）→ post_hook 不触发；catch-up pass 跑完全部 1000 sq1 |
| s1+sq1 全部完成 | s1 todo=[] + sq1 全 step_done → 两步秒跳过 |
| 单对象 s1 成功、sq1 失败 | sq1 catch-up pass 重跑该对象（force=False，其他对象 skip） |

---

## 文件改动清单

| 文件 | 改动内容 |
|------|---------|
| `partcraft/pipeline_v2/s1_phase1_vlm.py` | `run_many_streaming` 增加 `post_object_fn` 参数；consumer 调用 hook |
| `partcraft/pipeline_v2/run.py` | `run_step("s1", ...)` 检测 steps 含 sq1 时构造并传入 `post_object_fn` |
| `configs/pipeline_v2_shard00.yaml` | 合并 A + A_qc 为单 stage，`steps: [s1, sq1]` |
| 其他 shard configs（shard01/02/03） | 同上，可选（不合并也不报错） |
| `partcraft/pipeline_v2/sq1_qc_a.py` | **不改**（`_process_one` 直接调用） |

---

## 不在本次范围内

- sq2 / sq3 的流水线化（需要跨不同 service，GPU 资源设计更复杂）
- GPU 池分离（方案 C）

---

## 验收标准

1. `STAGES=A LIMIT=5` 跑 5 个对象，日志中 sq1 在 s1 完成后立即出现（同一 VLM 生命周期内）。
2. log 中不出现第二次 `[VLM] starting`。
3. 强行 kill 后重启，验证上表四种 resume 场景的行为正确。
4. 旧 config（独立 A + A_qc）运行无报错，行为不变。
