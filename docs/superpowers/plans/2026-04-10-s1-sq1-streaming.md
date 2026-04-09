# s1+sq1 Streaming Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `run_many_streaming` 中注入 per-object `post_object_fn` hook，使 sq1 在每个对象的 s1 完成后立即在同一 VLM session 内执行，消除 stage A → A_qc 之间的 VLM 重启和 batch 等待开销。

**Architecture:** `s1_phase1_vlm.run_many_streaming` 新增 `post_object_fn` 可选参数，consumer 在 `_call_one` 返回后调用 hook；`run.py` 的 `run_step` 新增同名参数透传，main() 循环在 step=s1 且 steps 含 sq1 时构造 hook；config 中 A+A_qc 合并为单 stage `steps: [s1, sq1]`。

**Tech Stack:** Python asyncio, pytest, PyYAML; 改动范围 3 个 Python 文件 + N 个 YAML。

---

## File Map

| 文件 | 操作 | 说明 |
|------|------|------|
| `partcraft/pipeline_v2/s1_phase1_vlm.py` | Modify | `run_many_streaming` + `consumer` 加 hook |
| `partcraft/pipeline_v2/run.py` | Modify | `run_step` + `main()` 加 look-ahead |
| `configs/pipeline_v2_shard00.yaml` | Modify | 合并 A + A_qc |
| `configs/pipeline_v2_shard01.yaml` | Modify | 同上 |
| `configs/pipeline_v2_shard02.yaml` | Modify | 同上 |
| `configs/pipeline_v2_shard03.yaml` | Modify | 同上 |
| `tests/test_s1_streaming.py` | Create | post_object_fn 单元测试 |

---

## Task 1: `run_many_streaming` — 新增 `post_object_fn` 参数

**Files:**
- Modify: `partcraft/pipeline_v2/s1_phase1_vlm.py`
- Create: `tests/test_s1_streaming.py`

### 1.1 先写失败测试

- [ ] 创建 `tests/test_s1_streaming.py`，内容如下：

```python
"""Unit tests for s1_phase1_vlm.run_many_streaming post_object_fn hook.

No Blender, no GPU, no VLM network calls — everything mocked.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.pipeline_v2.paths import PipelineRoot
from partcraft.pipeline_v2.s1_phase1_vlm import Phase1Result


def _make_ctx(tmp: Path, obj_id: str):
    root = PipelineRoot(tmp / "pipeline_out")
    ctx = root.context("00", obj_id)
    (ctx.dir / "phase1").mkdir(parents=True, exist_ok=True)
    ctx.mesh_npz = tmp / f"{obj_id}_mesh.npz"
    ctx.image_npz = tmp / f"{obj_id}_img.npz"
    ctx.mesh_npz.write_bytes(b"fake")
    ctx.image_npz.write_bytes(b"fake")
    return ctx


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


FAKE_PRE = (b"png", "user_msg", [0, 1], {"deletion": 1, "modification": 1,
            "scale": 1, "material": 1, "global": 1}, "menu")


class TestPostObjectFn:
    def test_hook_called_for_each_successful_object(self, tmp_path):
        """post_object_fn is called once per object after _call_one succeeds."""
        from partcraft.pipeline_v2.s1_phase1_vlm import run_many_streaming

        ctxs = [_make_ctx(tmp_path, f"obj_{i:03d}") for i in range(3)]

        async def _fake_call_one(client, ctx, png, user_msg, pids, quota,
                                  model, sem, *, part_menu=""):
            # write minimal parsed.json so resume logic is happy
            ctx.parsed_path.write_text(
                json.dumps({"parsed": {"edits": [], "object": {"parts": []}}})
            )
            from partcraft.pipeline_v2.status import update_step, STATUS_OK
            update_step(ctx, "s1_phase1", status=STATUS_OK, n_edits=0, resumed=False)
            return Phase1Result(ctx.obj_id, ok=True)

        hook_calls: list[tuple[str, str]] = []

        async def _hook(ctx, vlm_url: str):
            hook_calls.append((ctx.obj_id, vlm_url))

        vlm_urls = ["http://fake:8002/v1", "http://fake:8012/v1"]

        with patch("partcraft.pipeline_v2.s1_phase1_vlm._prerender_worker",
                   return_value=FAKE_PRE), \
             patch("partcraft.pipeline_v2.s1_phase1_vlm._call_one",
                   side_effect=_fake_call_one):
            _run(run_many_streaming(
                ctxs, blender="/fake/blender",
                vlm_urls=vlm_urls, vlm_model="fake-model",
                post_object_fn=_hook, force=True,
            ))

        assert len(hook_calls) == 3
        obj_ids = {c[0] for c in hook_calls}
        assert obj_ids == {c.obj_id for c in ctxs}
        # each hook call must use a URL from vlm_urls
        for _, url in hook_calls:
            assert url in vlm_urls

    def test_hook_not_called_for_too_many_parts(self, tmp_path):
        """post_object_fn is NOT called when s1 skips due to too_many_parts."""
        from partcraft.pipeline_v2.s1_phase1_vlm import run_many_streaming

        ctxs = [_make_ctx(tmp_path, "obj_big")]

        hook_calls: list = []

        async def _hook(ctx, vlm_url):
            hook_calls.append(ctx.obj_id)

        # _prerender_worker returning None triggers too_many_parts skip
        with patch("partcraft.pipeline_v2.s1_phase1_vlm._prerender_worker",
                   return_value=None):
            _run(run_many_streaming(
                ctxs, blender="/fake/blender",
                vlm_urls=["http://fake:8002/v1"], vlm_model="fake-model",
                post_object_fn=_hook, force=True,
            ))

        assert hook_calls == []

    def test_no_hook_does_not_break(self, tmp_path):
        """run_many_streaming without post_object_fn works as before."""
        from partcraft.pipeline_v2.s1_phase1_vlm import run_many_streaming

        ctxs = [_make_ctx(tmp_path, "obj_000")]

        async def _fake_call_one(client, ctx, png, user_msg, pids, quota,
                                  model, sem, *, part_menu=""):
            ctx.parsed_path.write_text(
                json.dumps({"parsed": {"edits": [], "object": {"parts": []}}})
            )
            from partcraft.pipeline_v2.status import update_step, STATUS_OK
            update_step(ctx, "s1_phase1", status=STATUS_OK, n_edits=0, resumed=False)
            return Phase1Result(ctx.obj_id, ok=True)

        with patch("partcraft.pipeline_v2.s1_phase1_vlm._prerender_worker",
                   return_value=FAKE_PRE), \
             patch("partcraft.pipeline_v2.s1_phase1_vlm._call_one",
                   side_effect=_fake_call_one):
            results = _run(run_many_streaming(
                ctxs, blender="/fake/blender",
                vlm_urls=["http://fake:8002/v1"], vlm_model="fake-model",
                force=True,
                # post_object_fn omitted intentionally
            ))

        assert len(results) == 1
        assert results[0].ok is True
```

- [ ] 确认测试失败（`post_object_fn` 参数还不存在）：

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python -m pytest tests/test_s1_streaming.py -v 2>&1 | head -30
```

预期：`TypeError: run_many_streaming() got an unexpected keyword argument 'post_object_fn'`

### 1.2 实现 `post_object_fn` 参数

- [ ] 修改 `partcraft/pipeline_v2/s1_phase1_vlm.py` 的 `run_many_streaming` 函数签名，在 `log_every: int = 20,` 后面加一行：

```python
    post_object_fn=None,   # Callable[[ObjectContext, str], Awaitable[None]] | None
```

完整新签名（替换现有的）：
```python
async def run_many_streaming(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    vlm_urls: list[str],
    vlm_model: str,
    n_prerender_workers: int = 8,
    force: bool = False,
    log_every: int = 20,
    post_object_fn=None,
) -> list[Phase1Result]:
```

- [ ] 在 `consumer(idx)` 内，找到 `results.append(res)` 和 `n_done += 1` 那段，在 log 语句**之后**加入 hook 调用。将现有代码段：

```python
            res = await _call_one(client, ctx, png, user_msg, pids, quota,
                                  vlm_model, sem, part_menu=menu)
            results.append(res)
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                log.info("s1 stream: %d/%d  ok_so_far=%d",
                         n_done, n_total,
                         sum(1 for r in results if r.ok))
```

替换为：

```python
            res = await _call_one(client, ctx, png, user_msg, pids, quota,
                                  vlm_model, sem, part_menu=menu)
            results.append(res)
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                log.info("s1 stream: %d/%d  ok_so_far=%d",
                         n_done, n_total,
                         sum(1 for r in results if r.ok))
            if post_object_fn is not None and res.error != "too_many_parts":
                try:
                    await post_object_fn(ctx, vlm_urls[idx])
                except Exception as _hook_exc:
                    log.warning("post_object_fn %s: %s", ctx.obj_id[:12], _hook_exc)
```

注意：hook 异常用 `try/except` 隔离，避免单个对象的 sq1 失败中断整个 producer-consumer 循环。

- [ ] 运行测试确认通过：

```bash
python -m pytest tests/test_s1_streaming.py -v
```

预期：3 个 PASS

- [ ] 提交：

```bash
git add partcraft/pipeline_v2/s1_phase1_vlm.py tests/test_s1_streaming.py
git commit -m "feat(s1): add post_object_fn hook to run_many_streaming"
```

---

## Task 2: `run.py` — `run_step` 透传 + main() look-ahead

**Files:**
- Modify: `partcraft/pipeline_v2/run.py`

### 2.1 写失败测试

- [ ] 在 `tests/test_s1_streaming.py` 末尾追加：

```python
class TestRunStepLookahead:
    def test_run_step_accepts_post_object_fn(self):
        """run_step('s1', ...) accepts post_object_fn without raising TypeError."""
        import inspect
        from partcraft.pipeline_v2.run import run_step
        sig = inspect.signature(run_step)
        assert "post_object_fn" in sig.parameters, (
            "run_step must accept post_object_fn keyword argument"
        )

    def test_post_object_fn_default_is_none(self):
        """run_step post_object_fn defaults to None (backward compat)."""
        import inspect
        from partcraft.pipeline_v2.run import run_step
        sig = inspect.signature(run_step)
        assert sig.parameters["post_object_fn"].default is None
```

- [ ] 确认测试失败：

```bash
python -m pytest tests/test_s1_streaming.py::TestRunStepLookahead -v
```

预期：`AssertionError: run_step must accept post_object_fn keyword argument`

### 2.2 修改 `run_step` 签名

- [ ] 在 `run.py` 中找到 `run_step` 函数定义，将签名从：

```python
def run_step(
    step: str,
    ctxs: list[ObjectContext],
    cfg: dict,
    args: argparse.Namespace,
) -> None:
```

改为：

```python
def run_step(
    step: str,
    ctxs: list[ObjectContext],
    cfg: dict,
    args: argparse.Namespace,
    post_object_fn=None,
) -> None:
```

- [ ] 在 `run_step` 内 `step == "s1"` 的分支中，找到 `asyncio.run(run_many_streaming(...))` 调用，在关键字参数列表末尾加 `post_object_fn=post_object_fn,`：

将现有代码：
```python
        asyncio.run(run_many_streaming(
            ctxs, blender=blender, vlm_urls=urls,
            vlm_model=model, n_prerender_workers=n_pre,
            force=args.force,
        ))
```

替换为：
```python
        asyncio.run(run_many_streaming(
            ctxs, blender=blender, vlm_urls=urls,
            vlm_model=model, n_prerender_workers=n_pre,
            force=args.force,
            post_object_fn=post_object_fn,
        ))
```

- [ ] 确认签名测试通过：

```bash
python -m pytest tests/test_s1_streaming.py::TestRunStepLookahead -v
```

预期：2 PASS

### 2.3 main() look-ahead 注入

- [ ] 在 `run.py` 的 `main()` 函数中，找到主循环：

```python
    exit_rc = 0
    for step in steps:
        # GPU dispatch only when ...
        wants_dispatch = (step in GPU_STEPS
                          and args.gpus
                          and (phase_use_gpus or not run_stage)
                          and not args.single_gpu)
        if wants_dispatch:
            rc = dispatch_gpus(step, args.config, args)
```

在 `exit_rc = 0` 之后、`for step in steps:` 循环体开头加入 look-ahead 逻辑，替换为：

```python
    exit_rc = 0
    for step in steps:
        # look-ahead: inject sq1 as post-hook when s1 and sq1 share a VLM stage
        _post_fn = None
        if step == "s1" and "sq1" in steps:
            from .sq1_qc_a import _process_one as _sq1_process_one
            _vlm_model = psvc.vlm_model_name(cfg)
            _force = args.force
            async def _sq1_hook(ctx, vlm_url,
                                _m=_vlm_model, _f=_force):
                await _sq1_process_one(ctx, vlm_url, _m, _f)
            _post_fn = _sq1_hook

        # GPU dispatch only when ...
        wants_dispatch = (step in GPU_STEPS
                          and args.gpus
                          and (phase_use_gpus or not run_stage)
                          and not args.single_gpu)
        if wants_dispatch:
            rc = dispatch_gpus(step, args.config, args)
            if rc != 0:
                LOG.error("[%s] dispatch_gpus returned rc=%d — aborting", step, rc)
                exit_rc = rc
        else:
            run_step(step, ctxs, cfg, args, post_object_fn=_post_fn)
```

注意：`run_single_gpu` 也调用 `run_step`，但它只处理单个 step，不持有完整 steps 列表，所以 look-ahead 不适用——单 GPU 子进程不会收到 `post_object_fn`（使用 catch-up pass 兜底，见 Resume 表）。这是预期行为。

- [ ] 提交：

```bash
git add partcraft/pipeline_v2/run.py
git commit -m "feat(run): inject sq1 as post-hook when s1+sq1 in same stage"
```

---

## Task 3: Config 合并 A + A_qc

**Files:**
- Modify: `configs/pipeline_v2_shard00.yaml`
- Modify: `configs/pipeline_v2_shard01.yaml`
- Modify: `configs/pipeline_v2_shard02.yaml`
- Modify: `configs/pipeline_v2_shard03.yaml`

- [ ] 在 `configs/pipeline_v2_shard00.yaml` 的 `stages:` 列表中，将：

```yaml
  - {name: A,    desc: "phase1 VLM",         servers: vlm,  steps: [s1]}
  - {name: A_qc, desc: "QC-A instruction",   servers: vlm,  steps: [sq1]}
```

替换为：

```yaml
  - {name: A,    desc: "phase1 VLM + QC-A",  servers: vlm,  steps: [s1, sq1]}
```

- [ ] 对 `shard01.yaml`、`shard02.yaml`、`shard03.yaml` 做同样改动（如果这些 config 里有独立 A + A_qc entry 的话；若原本没有 A_qc 则跳过）。

```bash
# 检查哪些 config 有独立 A_qc
grep -l "A_qc" configs/pipeline_v2_shard*.yaml
```

- [ ] 验证 YAML 可被正确解析（无语法错误）：

```bash
python -c "
import yaml, glob
for f in glob.glob('configs/pipeline_v2_shard*.yaml'):
    cfg = yaml.safe_load(open(f))
    stages = [s['name'] for s in cfg['pipeline']['stages']]
    print(f, '->', stages)
"
```

预期输出中不再出现 `A_qc`，且 `A` 的 stage 包含 `[s1, sq1]`。

- [ ] 提交：

```bash
git add configs/pipeline_v2_shard*.yaml
git commit -m "config: merge stage A+A_qc into single VLM stage"
```

---

## Task 4: Smoke 验收

- [ ] 用 `LIMIT=5` 跑 5 个对象验证功能：

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
STAGES=A LIMIT=5 bash scripts/tools/run_pipeline_v2_shard.sh shard00 \
    configs/pipeline_v2_shard00.yaml 2>&1 | tee /tmp/smoke_A.log
```

- [ ] 验证 log 中 VLM 只启动一次：

```bash
grep "\[VLM\] starting\|VLM.*ready\|VLM.*killing" /tmp/smoke_A.log
```

预期：出现 `[VLM] starting 4 servers` **恰好一次**，出现 `[VLM] all 4 servers ready` 一次，结尾出现 `[VLM] killing all servers` 一次。

- [ ] 验证 sq1 在 s1 完成后立即出现（不是等所有 s1 完成）：

```bash
grep "s1 stream\|sq1_qc_A\|sq1.*n_pass\|pipeline_v2.sq1" /tmp/smoke_A.log | head -30
```

预期：s1 和 sq1 的日志交替出现（而非所有 s1 在前、所有 sq1 在后）。

- [ ] 验证 5 个对象均有 sq1 结果：

```bash
python -c "
import json, glob
for f in sorted(glob.glob('outputs/partverse/pipeline_v2_shard00/objects/00/*/status.json'))[:5]:
    s = json.load(open(f))
    print(f.split('/')[-2][:12], '-> sq1:', s.get('sq1_qc_A', 'MISSING'))
"
```

预期：5 个对象均有 `sq1_qc_A: OK` 或 `sq1_qc_A: FAIL`（不是 MISSING）。

- [ ] 最终提交（如有任何 hotfix）：

```bash
git add -A && git commit -m "chore: smoke-verified s1+sq1 streaming pipeline"
```

---

## 自检清单（实施前对照）

- [ ] spec 的 Resume 表四个场景均有对应测试或手动验证步骤
- [ ] `too_many_parts` 不触发 hook ✓ (Task 1 test_hook_not_called_for_too_many_parts)
- [ ] hook 异常不中断主循环 ✓ (Task 1 实现中加了 try/except)
- [ ] 旧 config（独立 A + A_qc）不报错 ✓ (只改了 shard00 等新 config，旧 config 仍兼容)
- [ ] run_single_gpu 不需要 look-ahead ✓ (用 catch-up pass 兜底)
