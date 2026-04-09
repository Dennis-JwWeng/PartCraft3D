# Per-Type QC Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three semantic QC stages (A_qc/C_qc/E_qc) to pipeline_v2 that validate edit instructions and results per edit type, writing per-edit pass/fail to `qc.json`.

**Architecture:** New steps `sq1/sq2/sq3` run after stages A/C/E. Shared `qc_io.py` handles atomic `qc.json` read/write. `s4/s5/s5b` skip edits already marked `qc_fail`. VLM calls reuse `call_vlm_async` + `extract_json_object` from `s1_vlm_core.py`.

**Tech Stack:** Python asyncio, openai client, cv2 (image stitch), `cleaning/vlm_filter.call_vlm_judge` (sq3), unittest.

**Spec:** `docs/superpowers/specs/2026-04-09-qc-pipeline-design.md`

---
## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `partcraft/pipeline_v2/paths.py` | Modify | Add `ctx.qc_path` property |
| `partcraft/pipeline_v2/qc_rules.py` | Create | 7 pure rule check functions |
| `partcraft/pipeline_v2/qc_io.py` | Create | qc.json atomic IO |
| `partcraft/pipeline_v2/sq1_qc_a.py` | Create | QC-A runner |
| `partcraft/pipeline_v2/sq2_qc_c.py` | Create | QC-C runner |
| `partcraft/pipeline_v2/sq3_qc_e.py` | Create | QC-E runner |
| `partcraft/pipeline_v2/validators.py` | Modify | Add check_sq1/sq2/sq3 |
| `partcraft/pipeline_v2/run.py` | Modify | Wire sq1/sq2/sq3 |
| `partcraft/pipeline_v2/s4_flux_2d.py` | Modify | Skip qc_failed edits |
| `partcraft/pipeline_v2/s5_trellis_3d.py` | Modify | Skip qc_failed edits |
| `partcraft/pipeline_v2/s5b_deletion.py` | Modify | Skip qc_failed edits |
| `configs/pipeline_v2_shard00.yaml` | Modify | Add QC stages + qc: block |
| `configs/pipeline_v2_shard02.yaml` | Modify | Same |
| `tests/test_qc_rules.py` | Create | Unit tests for qc_rules.py |
| `tests/test_qc_io.py` | Create | Unit tests for qc_io.py |

---

## Task 1: Add ctx.qc_path to paths.py

**Files:** Modify: partcraft/pipeline_v2/paths.py

- [ ] **Step 1: Find status_path property and add qc_path after it**

Find the status_path property in paths.py and add immediately after:

```python
@property
def qc_path(self) -> Path:
    return self.dir / "qc.json"
```

- [ ] **Step 2: Verify import and attribute**

```bash
python -c "
from partcraft.pipeline_v2.paths import PipelineRoot
import pathlib
ctx = PipelineRoot(pathlib.Path('/tmp/x')).context('00','obj001')
assert str(ctx.qc_path).endswith('qc.json')
print('OK:', ctx.qc_path)
"
```

Expected: OK: .../objects/00/obj001/qc.json

- [ ] **Step 3: Commit**

```bash
git add partcraft/pipeline_v2/paths.py
git commit -m "feat(qc): add ctx.qc_path to ObjectContext"
```

---

## Task 2: qc_rules.py - Pure Rule Functions

**Files:** Create: partcraft/pipeline_v2/qc_rules.py, tests/test_qc_rules.py

- [ ] **Step 1: Write failing tests** - create tests/test_qc_rules.py

```python
from __future__ import annotations
import unittest

class TestQcRules(unittest.TestCase):
    def setUp(self):
        from partcraft.pipeline_v2.qc_rules import check_rules
        self.check = check_rules
        self.parts = {0: {"part_id": 0, "name": "leg"}, 1: {"part_id": 1, "name": "seat"}}

    def test_pass_deletion(self):
        edit = {"edit_type": "deletion", "prompt": "Remove the leg from the chair",
                "selected_part_ids": [0], "target_part_desc": "chair leg"}
        self.assertEqual(self.check(edit, self.parts), {})

    def test_prompt_too_short(self):
        edit = {"edit_type": "deletion", "prompt": "hi", "selected_part_ids": [0]}
        self.assertIn("prompt_too_short", self.check(edit, self.parts))

    def test_parts_missing(self):
        edit = {"edit_type": "deletion",
                "prompt": "Remove the leg from the chair", "selected_part_ids": []}
        self.assertIn("parts_missing", self.check(edit, self.parts))

    def test_parts_invalid(self):
        edit = {"edit_type": "deletion",
                "prompt": "Remove the leg from the chair", "selected_part_ids": [99]}
        self.assertIn("parts_invalid", self.check(edit, self.parts))

    def test_new_desc_missing_mod(self):
        edit = {"edit_type": "modification",
                "prompt": "Replace the leg with a metal rod",
                "selected_part_ids": [0], "target_part_desc": "wooden leg",
                "new_parts_desc": "", "new_parts_desc_stage1": "", "new_parts_desc_stage2": ""}
        fails = self.check(edit, self.parts)
        self.assertIn("new_desc_missing", fails)
        self.assertIn("stage_decomp_missing", fails)

    def test_target_desc_missing_scale(self):
        edit = {"edit_type": "scale", "prompt": "Make the leg taller",
                "selected_part_ids": [0], "target_part_desc": ""}
        self.assertIn("target_desc_missing", self.check(edit, self.parts))

    def test_verb_conflict_deletion(self):
        edit = {"edit_type": "deletion",
                "prompt": "Add a new leg to the chair", "selected_part_ids": [0]}
        self.assertIn("verb_conflict", self.check(edit, self.parts))

    def test_global_no_parts(self):
        edit = {"edit_type": "global", "prompt": "Change the style to industrial metal"}
        self.assertEqual(self.check(edit, {}), {})

    def test_pass_modification_full(self):
        edit = {"edit_type": "modification",
                "prompt": "Replace the wooden leg with a metal rod",
                "selected_part_ids": [0], "target_part_desc": "wooden leg",
                "new_parts_desc": "thin metal rod",
                "new_parts_desc_stage1": "metal rod geometry", "new_parts_desc_stage2": ""}
        self.assertEqual(self.check(edit, self.parts), {})

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Confirm failure**

```bash
python -m pytest tests/test_qc_rules.py -v 2>&1 | head -10
```

Expected: ModuleNotFoundError: No module named partcraft.pipeline_v2.qc_rules

- [ ] **Step 3: Implement** - create partcraft/pipeline_v2/qc_rules.py

```python
from __future__ import annotations
from typing import Any

_PART_REQUIRED = frozenset({"deletion", "modification", "scale", "material"})
_ADD_VERBS = ("add ", "insert", "attach", "place", "put ")
_REMOVE_ONLY = ("remove", "delete", "erase", "strip", "eliminate")
_REPLACE_IND = ("replace", "swap", "change", "modify", "convert")


def check_rules(edit: dict[str, Any], parts_by_id: dict[int, Any]) -> dict[str, bool]:
    """Run 7 rule checks. Returns dict of failing codes (empty = all pass)."""
    et = edit.get("edit_type", "")
    prompt = (edit.get("prompt") or "").strip()
    pids = list(edit.get("selected_part_ids") or [])
    pl = prompt.lower()
    fails: dict[str, bool] = {}

    if len(prompt) < 8:
        fails["prompt_too_short"] = True
    if et in _PART_REQUIRED:
        if not pids:
            fails["parts_missing"] = True
        elif any(p not in parts_by_id for p in pids):
            fails["parts_invalid"] = True
    if et == "modification" and not (edit.get("new_parts_desc") or "").strip():
        fails["new_desc_missing"] = True
    if et in ("modification", "scale", "material"):
        if not (edit.get("target_part_desc") or "").strip():
            fails["target_desc_missing"] = True
    if et == "modification":
        s1 = (edit.get("new_parts_desc_stage1") or "").strip()
        s2 = (edit.get("new_parts_desc_stage2") or "").strip()
        if not s1 and not s2:
            fails["stage_decomp_missing"] = True
    if et == "deletion" and any(v in pl for v in _ADD_VERBS):
        fails["verb_conflict"] = True
    elif et in ("modification", "scale", "material", "global"):
        if any(v in pl for v in _REMOVE_ONLY) and not any(v in pl for v in _REPLACE_IND):
            fails["verb_conflict"] = True
    return fails

__all__ = ["check_rules"]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_qc_rules.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add partcraft/pipeline_v2/qc_rules.py tests/test_qc_rules.py
git commit -m "feat(qc): add qc_rules rule layer with tests"
```

---

## Task 3: qc_io.py - qc.json Read/Write Helpers

**Files:** Create: partcraft/pipeline_v2/qc_io.py, tests/test_qc_io.py

- [ ] **Step 1: Write failing tests** - create tests/test_qc_io.py

```python
from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from unittest.mock import MagicMock

def _ctx(tmp, oid="obj001"):
    c = MagicMock(); c.obj_id = oid; c.shard = "00"
    c.dir = tmp / oid; c.dir.mkdir(parents=True, exist_ok=True)
    c.qc_path = c.dir / "qc.json"; return c

class TestQcIo(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory(); self.p = Path(self._t.name)
    def tearDown(self): self._t.cleanup()

    def test_load_missing_skeleton(self):
        from partcraft.pipeline_v2.qc_io import load_qc
        self.assertEqual(load_qc(_ctx(self.p))["edits"], {})

    def test_save_load_roundtrip(self):
        from partcraft.pipeline_v2.qc_io import load_qc, save_qc
        ctx = _ctx(self.p); qc = load_qc(ctx)
        qc["edits"]["del_000"] = {"final_pass": True}; save_qc(ctx, qc)
        self.assertTrue(load_qc(ctx)["edits"]["del_000"]["final_pass"])

    def test_gate_rule_fail(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, load_qc
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": False, "checks": {"prompt_too_short": True}})
        e = load_qc(ctx)["edits"]["del_000"]
        self.assertFalse(e["final_pass"])
        self.assertEqual(e["fail_gate"], "A"); self.assertEqual(e["fail_reason"], "prompt_too_short")

    def test_all_pass(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, load_qc
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": True, "checks": {}},
                         vlm_result={"pass": True, "score": 0.9, "reason": ""})
        update_edit_gate(ctx, "del_000", "deletion", "E",
                         vlm_result={"pass": True, "score": 0.85, "reason": ""})
        self.assertTrue(load_qc(ctx)["edits"]["del_000"]["final_pass"])

    def test_not_failed_before_qc(self):
        from partcraft.pipeline_v2.qc_io import is_edit_qc_failed
        self.assertFalse(is_edit_qc_failed(_ctx(self.p), "del_000"))

    def test_failed_after_fail(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_edit_qc_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": False, "checks": {"parts_missing": True}})
        self.assertTrue(is_edit_qc_failed(ctx, "del_000"))

    def test_null_c_gate_counts_as_pass(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, load_qc
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": True, "checks": {}},
                         vlm_result={"pass": True, "score": 0.9, "reason": ""})
        update_edit_gate(ctx, "del_000", "deletion", "E",
                         vlm_result={"pass": True, "score": 0.9, "reason": ""})
        self.assertTrue(load_qc(ctx)["edits"]["del_000"]["final_pass"])

if __name__ == "__main__": unittest.main()
```

- [ ] **Step 2: Confirm failure**

```bash
python -m pytest tests/test_qc_io.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement** - create partcraft/pipeline_v2/qc_io.py

```python
from __future__ import annotations
import json, os, tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from .paths import ObjectContext

def _now(): return datetime.now().isoformat(timespec="seconds")

def load_qc(ctx: ObjectContext) -> dict[str, Any]:
    if ctx.qc_path.is_file():
        try: return json.loads(ctx.qc_path.read_text())
        except json.JSONDecodeError: pass
    return {"obj_id": ctx.obj_id, "shard": ctx.shard, "updated": None, "edits": {}}

def save_qc(ctx: ObjectContext, qc: dict[str, Any]) -> None:
    qc.update({"obj_id": ctx.obj_id, "shard": ctx.shard, "updated": _now()})
    ctx.dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".qc.", suffix=".tmp", dir=str(ctx.dir))
    try:
        with os.fdopen(fd, "w") as f: json.dump(qc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ctx.qc_path)
    except Exception: Path(tmp).unlink(missing_ok=True); raise

def update_edit_gate(
    ctx: ObjectContext, edit_id: str, edit_type: str, gate: str,
    *, rule_result: dict | None = None, vlm_result: dict | None = None,
) -> None:
    qc = load_qc(ctx)
    entry = qc.setdefault("edits", {}).setdefault(edit_id, {
        "edit_type": edit_type, "gates": {"A": None, "C": None, "E": None}, "final_pass": False})
    gd: dict[str, Any] = {}
    if rule_result is not None: gd["rule"] = rule_result
    if vlm_result is not None:  gd["vlm"] = vlm_result
    entry["gates"][gate] = gd if gd else None
    entry["final_pass"] = all(_gp(entry["gates"][g]) for g in ("A", "C", "E"))
    if not entry["final_pass"]:
        for g in ("A", "C", "E"):
            gd2 = entry["gates"][g]
            if gd2 is not None and not _gp(gd2):
                entry["fail_gate"] = g
                r = gd2.get("rule"); v = gd2.get("vlm")
                if r and not r.get("pass", True):
                    entry["fail_reason"] = next(iter(r.get("checks") or {}), "rule_fail")
                elif v and not v.get("pass", True):
                    entry["fail_reason"] = (v.get("reason") or "vlm_fail")[:80]
                break
    else:
        entry.pop("fail_gate", None); entry.pop("fail_reason", None)
    save_qc(ctx, qc)

def _gp(gd: dict | None) -> bool:
    if gd is None: return True
    r = gd.get("rule"); v = gd.get("vlm")
    if r is not None and not r.get("pass", True): return False
    if v is not None and not v.get("pass", True): return False
    return True

def is_edit_qc_failed(ctx: ObjectContext, edit_id: str) -> bool:
    if not ctx.qc_path.is_file(): return False
    e = load_qc(ctx).get("edits", {}).get(edit_id)
    return e is not None and e.get("final_pass", True) is False

__all__ = ["load_qc", "save_qc", "update_edit_gate", "is_edit_qc_failed"]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_qc_io.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add partcraft/pipeline_v2/qc_io.py tests/test_qc_io.py
git commit -m "feat(qc): add qc_io atomic qc.json helpers with tests"
```

---

## Task 4: sq1_qc_a.py - QC-A Runner

**Files:** Create: partcraft/pipeline_v2/sq1_qc_a.py

- [ ] **Step 1: Create partcraft/pipeline_v2/sq1_qc_a.py**

Two-phase per object: (1) sync rule check via check_rules, (2) async VLM for rule-passing edits.
VLM prompt sends overview.png as image + edit_type/prompt/target_part_desc/part_labels as text.
VLM response: {instruction_clear, part_identifiable, type_consistent, reason}.
Pass = all three booleans true. Score = count_true / 3.0.

```python
from __future__ import annotations
import asyncio, base64, json, logging
from typing import Iterable
from openai import AsyncOpenAI
from .paths import ObjectContext
from .specs import iter_all_specs
from .status import update_step, STATUS_OK, STATUS_FAIL, step_done
from .qc_rules import check_rules
from .qc_io import update_edit_gate
from .s1_vlm_core import extract_json_object

LOG = logging.getLogger("pipeline_v2.sq1")
_SYS = "You evaluate 3D part-editing instructions. Output only valid JSON."
_TPL = ("5x2 overview (top=photos, bottom=colored part renders).\n"
        "Edit type: {edit_type}\nPrompt: \"{prompt}\"\n"
        "Target: \"{target_part_desc}\"\nParts: {part_labels}\n"
        "Reply ONLY: {{\"instruction_clear\":true/false,\"part_identifiable\":true/false,"
        "\"type_consistent\":true/false,\"reason\":\"one sentence\"}}")


async def _vlm_one(client, model, png, et, prompt, tpd, labels):
    b64 = base64.b64encode(png).decode()
    user = _TPL.format(edit_type=et, prompt=prompt, target_part_desc=tpd or "",
                       part_labels=", ".join(labels) or "(none)")
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                          {"type": "text", "text": user}]}],
            temperature=0.2, max_tokens=256, timeout=120,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        p = extract_json_object(r.choices[0].message.content or "")
        if p and "instruction_clear" in p: return p
    except Exception as e: LOG.warning("sq1 vlm: %s", e)
    return {"instruction_clear": False, "part_identifiable": False,
            "type_consistent": False, "reason": "vlm_error"}


async def _process_one(ctx, vlm_url, vlm_model, force):
    if not force and step_done(ctx, "sq1_qc_A"):
        return {"obj_id": ctx.obj_id, "skipped": True}
    if not ctx.parsed_path.is_file():
        update_step(ctx, "sq1_qc_A", status=STATUS_FAIL, error="missing_parsed_json")
        return {"obj_id": ctx.obj_id, "error": "missing_parsed_json"}
    raw = json.loads(ctx.parsed_path.read_text())
    obj = (raw.get("parsed") or {}).get("object") or {}
    parts_by_id = {p["part_id"]: p for p in (obj.get("parts") or [])
                   if isinstance(p, dict) and "part_id" in p}
    edits = (raw.get("parsed") or {}).get("edits") or []
    ov = ctx.overview_path.read_bytes() if ctx.overview_path.is_file() else None
    client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
    n_pass = n_fail = 0; vlm_q = []
    for spec in iter_all_specs(ctx):
        e = edits[spec.edit_idx] if spec.edit_idx < len(edits) else {}
        fails = check_rules(e, parts_by_id)
        rr = {"pass": not fails, "checks": fails}
        if fails:
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "A", rule_result=rr)
            n_fail += 1
        else:
            vlm_q.append((spec, rr))
    if vlm_q and ov:
        async def _c(spec, rr):
            v = await _vlm_one(client, vlm_model, ov, spec.edit_type,
                               spec.prompt, spec.target_part_desc, spec.part_labels)
            ok = bool(v.get("instruction_clear") and v.get("part_identifiable")
                      and v.get("type_consistent"))
            sc = sum(map(bool, [v.get("instruction_clear"), v.get("part_identifiable"),
                                v.get("type_consistent")])) / 3.0
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "A", rule_result=rr,
                             vlm_result={"pass": ok, "score": round(sc, 3),
                                         "reason": v.get("reason", "")})
            return ok
        res = await asyncio.gather(*[_c(s, r) for s, r in vlm_q])
        n_pass += sum(res); n_fail += len(res) - sum(res)
    else:
        for spec, rr in vlm_q:
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "A", rule_result=rr,
                             vlm_result={"pass": True, "score": 1.0, "reason": "no_overview_skip"})
            n_pass += 1
    update_step(ctx, "sq1_qc_A", status=STATUS_OK, n_pass=n_pass, n_fail=n_fail)
    return {"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail}


async def run(ctxs: Iterable[ObjectContext], *, vlm_urls, vlm_model, force=False, concurrency=4):
    ctxs = list(ctxs); sem = asyncio.Semaphore(concurrency)
    async def _o(i, c):
        async with sem: return await _process_one(c, vlm_urls[i % len(vlm_urls)], vlm_model, force)
    return await asyncio.gather(*[_o(i, c) for i, c in enumerate(ctxs)])

__all__ = ["run"]
```

- [ ] **Step 2: Import check**

```bash
python -c "from partcraft.pipeline_v2.sq1_qc_a import run; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add partcraft/pipeline_v2/sq1_qc_a.py
git commit -m "feat(qc): add sq1_qc_a QC-A runner"
```

---

## Task 5: sq2_qc_c.py - QC-C Lightweight 2D Runner

**Files:** Create: partcraft/pipeline_v2/sq2_qc_c.py

- [ ] **Step 1: Create partcraft/pipeline_v2/sq2_qc_c.py**

Stitches highlight e{idx}.png (Stage B) and FLUX result side-by-side with cv2.
Binary VLM check with max_tokens=128 for speed. Only runs for FLUX types
(iter_flux_specs returns modification/scale/material/global). deletion/addition
have C gate null. Skip edits already failed via is_edit_qc_failed.

```python
from __future__ import annotations
import asyncio, base64, logging
from typing import Iterable
import cv2, numpy as np
from openai import AsyncOpenAI
from .paths import ObjectContext
from .specs import iter_flux_specs
from .status import update_step, STATUS_OK, step_done
from .qc_io import update_edit_gate, is_edit_qc_failed
from .s1_vlm_core import extract_json_object

LOG = logging.getLogger("pipeline_v2.sq2")
_SYS = "You are a visual quality judge. Output only valid JSON."
_USR = ("LEFT=highlighted region (magenta=target part). RIGHT=2D edit result. "
        "Did the edit primarily affect the magenta region? "
        "Reply ONLY: {\"region_match\":true/false,\"reason\":\"one short phrase\"}")

def _stitch(a, b):
    h = min(a.shape[0], b.shape[0], 512)
    def _r(img): s = h / img.shape[0]; return cv2.resize(img, (int(img.shape[1] * s), h))
    ok, buf = cv2.imencode(".png", np.hstack([_r(a), _r(b)]))
    if not ok: raise RuntimeError("imencode failed")
    return buf.tobytes()

async def _vlm_one(client, model, png):
    b64 = base64.b64encode(png).decode()
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                          {"type": "text", "text": _USR}]}],
            temperature=0.1, max_tokens=128, timeout=60,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        p = extract_json_object(r.choices[0].message.content or "")
        if p and "region_match" in p: return p
    except Exception as e: LOG.warning("sq2 vlm: %s", e)
    return {"region_match": False, "reason": "vlm_error"}

async def _process_one(ctx, vlm_url, vlm_model, force):
    if not force and step_done(ctx, "sq2_qc_C"):
        return {"obj_id": ctx.obj_id, "skipped": True}
    client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
    n_pass = n_fail = n_skip = 0

    async def _check(spec):
        nonlocal n_pass, n_fail, n_skip
        if is_edit_qc_failed(ctx, spec.edit_id): n_skip += 1; return
        hl = ctx.highlight_path(spec.edit_idx); ed = ctx.edit_2d_output(spec.edit_id)
        if not hl.is_file() or not ed.is_file():
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "C",
                             vlm_result={"pass": False, "score": 0.0, "reason": "missing_artifact"})
            n_fail += 1; return
        hi = cv2.imread(str(hl)); ei = cv2.imread(str(ed))
        if hi is None or ei is None:
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "C",
                             vlm_result={"pass": False, "score": 0.0, "reason": "unreadable_image"})
            n_fail += 1; return
        raw = await _vlm_one(client, vlm_model, _stitch(hi, ei))
        ok = bool(raw.get("region_match"))
        update_edit_gate(ctx, spec.edit_id, spec.edit_type, "C",
                         vlm_result={"pass": ok, "score": 1.0 if ok else 0.0,
                                     "reason": raw.get("reason", "")})
        if ok: n_pass += 1
        else: n_fail += 1

    await asyncio.gather(*[_check(s) for s in iter_flux_specs(ctx)])
    update_step(ctx, "sq2_qc_C", status=STATUS_OK,
                n_pass=n_pass, n_fail=n_fail, n_skip=n_skip)
    return {"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail}

async def run(ctxs, *, vlm_urls, vlm_model, force=False, concurrency=4):
    ctxs = list(ctxs); sem = asyncio.Semaphore(concurrency)
    async def _o(i, c):
        async with sem: return await _process_one(c, vlm_urls[i % len(vlm_urls)], vlm_model, force)
    return await asyncio.gather(*[_o(i, c) for i, c in enumerate(ctxs)])

__all__ = ["run"]
```

- [ ] **Step 2: Import check**

```bash
python -c "from partcraft.pipeline_v2.sq2_qc_c import run; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add partcraft/pipeline_v2/sq2_qc_c.py
git commit -m "feat(qc): add sq2_qc_c QC-C 2D region check"
```

---

## Task 6: sq3_qc_e.py - QC-E Final Quality Runner

**Files:** Create: partcraft/pipeline_v2/sq3_qc_e.py

- [ ] **Step 1: Create partcraft/pipeline_v2/sq3_qc_e.py**

Synchronous runner (vlm_filter.call_vlm_judge is sync). For each edit:
- Skip is_edit_qc_failed edits (already failed A or C)
- Load before/after 3D renders via ctx.edit_3d_png(edit_id, "before"/"after")
- Build top/bottom collage (before=top, after=bottom)
- call_vlm_judge returns dict with edit_executed, correct_region, preserve_other, visual_quality, reason
- _passes() applies per-type thresholds: deletion checks correct_region, mod/scale check preserve_other
- Note: large-part modification promoted to Global in s5 will naturally fail preserve_other here - intentional

```python
from __future__ import annotations
import logging
from typing import Iterable
import cv2, numpy as np
from openai import OpenAI
from .paths import ObjectContext
from .specs import iter_all_specs
from .status import update_step, STATUS_OK, step_done
from .qc_io import update_edit_gate, is_edit_qc_failed

LOG = logging.getLogger("pipeline_v2.sq3")
_DEFS = {
    "deletion":     {"min_visual_quality": 3, "require_preserve_other": False},
    "modification": {"min_visual_quality": 3, "require_preserve_other": True},
    "scale":        {"min_visual_quality": 3, "require_preserve_other": True},
    "material":     {"min_visual_quality": 3, "require_preserve_other": False},
    "global":       {"min_visual_quality": 3, "require_preserve_other": False},
    "addition":     {"min_visual_quality": 3, "require_preserve_other": False},
}

def _collage(b, a):
    bi = cv2.imread(str(b)); ai = cv2.imread(str(a))
    if bi is None or ai is None: return None
    h = 512
    def _r(x): s = h / x.shape[0]; return cv2.resize(x, (int(x.shape[1] * s), h))
    ok, buf = cv2.imencode(".png", np.vstack([_r(bi), _r(ai)]))
    return buf.tobytes() if ok else None

def _passes(j, et, thr):
    t = thr.get(et, _DEFS.get(et, {}))
    if not j.get("edit_executed", False): return False
    if j.get("visual_quality", 0) < t.get("min_visual_quality", 3): return False
    if et == "deletion" and not j.get("correct_region", False): return False
    if t.get("require_preserve_other") and not j.get("preserve_other", False): return False
    return True

def run(ctxs, *, vlm_url, vlm_model, cfg, force=False):
    from partcraft.cleaning.vlm_filter import call_vlm_judge
    thr = (cfg.get("qc") or {}).get("thresholds_by_type") or _DEFS
    client = OpenAI(base_url=vlm_url, api_key="EMPTY")
    out = []
    for ctx in ctxs:
        if not force and step_done(ctx, "sq3_qc_E"):
            out.append({"obj_id": ctx.obj_id, "skipped": True}); continue
        n_pass = n_fail = n_skip = 0
        for spec in iter_all_specs(ctx):
            if is_edit_qc_failed(ctx, spec.edit_id): n_skip += 1; continue
            bp = ctx.edit_3d_png(spec.edit_id, "before"); ap = ctx.edit_3d_png(spec.edit_id, "after")
            if not bp.is_file() or not ap.is_file():
                update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                                 vlm_result={"pass": False, "score": 0.0, "reason": "missing_3d_renders"})
                n_fail += 1; continue
            coll = _collage(bp, ap)
            if coll is None:
                update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                                 vlm_result={"pass": False, "score": 0.0, "reason": "collage_failed"})
                n_fail += 1; continue
            j = call_vlm_judge(client, vlm_model, coll, edit_prompt=spec.prompt,
                               edit_type=spec.edit_type, object_desc=spec.object_desc,
                               part_label=", ".join(spec.part_labels))
            if j is None:
                update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                                 vlm_result={"pass": False, "score": 0.0, "reason": "vlm_no_response"})
                n_fail += 1; continue
            ok = _passes(j, spec.edit_type, thr)
            update_edit_gate(ctx, spec.edit_id, spec.edit_type, "E",
                             vlm_result={"pass": ok,
                                         "score": round(j.get("visual_quality", 0) / 5.0, 2),
                                         "reason": j.get("reason", "")})
            if ok: n_pass += 1
            else: n_fail += 1
        update_step(ctx, "sq3_qc_E", status=STATUS_OK,
                    n_pass=n_pass, n_fail=n_fail, n_skip=n_skip)
        out.append({"obj_id": ctx.obj_id, "n_pass": n_pass, "n_fail": n_fail})
    return out

__all__ = ["run"]
```

- [ ] **Step 2: Import check**

```bash
python -c "from partcraft.pipeline_v2.sq3_qc_e import run; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add partcraft/pipeline_v2/sq3_qc_e.py
git commit -m "feat(qc): add sq3_qc_e QC-E final quality runner"
```

---

## Task 7: Wire validators.py and run.py

**Files:** Modify: partcraft/pipeline_v2/validators.py, partcraft/pipeline_v2/run.py

- [ ] **Step 1: Add to validators.py** - after last check_s7 function

```python
def check_sq1(ctx: ObjectContext) -> StepCheck:
    return _check_files("sq1_qc_A", [("qc.json", ctx.qc_path)])

def check_sq2(ctx: ObjectContext) -> StepCheck:
    from .specs import iter_flux_specs
    if not any(True for _ in iter_flux_specs(ctx)):
        return StepCheck(step="sq2_qc_C", ok=True, expected=0, found=0, skip=True)
    return _check_files("sq2_qc_C", [("qc.json", ctx.qc_path)])

def check_sq3(ctx: ObjectContext) -> StepCheck:
    return _check_files("sq3_qc_E", [("qc.json", ctx.qc_path)])
```

In VALIDATORS dict add: "sq1": check_sq1, "sq2": check_sq2, "sq3": check_sq3

- [ ] **Step 2: Edit run.py** - three changes

ALL_STEPS: append "sq1", "sq2", "sq3" to tuple.

After s7 elif in run_step():

```python
    elif step == "sq1":
        from .sq1_qc_a import run as sq1_run
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        asyncio.run(sq1_run(ctxs, vlm_urls=urls,
                            vlm_model=psvc.vlm_model_name(cfg), force=args.force))

    elif step == "sq2":
        from .sq2_qc_c import run as sq2_run
        import asyncio
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        asyncio.run(sq2_run(ctxs, vlm_urls=urls,
                            vlm_model=psvc.vlm_model_name(cfg), force=args.force))

    elif step == "sq3":
        from .sq3_qc_e import run as sq3_run
        urls = ([u.strip() for u in args.vlm_url.split(",") if u.strip()]
                if getattr(args, "vlm_url", None) else sched.vlm_urls_for(cfg))
        sq3_run(ctxs, vlm_url=urls[0], vlm_model=psvc.vlm_model_name(cfg),
                cfg=cfg, force=args.force)
```

_STATUS_KEYS dict: add "sq1": "sq1_qc_A", "sq2": "sq2_qc_C", "sq3": "sq3_qc_E"

- [ ] **Step 3: Verify**

```bash
python -c "
from partcraft.pipeline_v2.run import ALL_STEPS
assert all(s in ALL_STEPS for s in ('sq1','sq2','sq3')), ALL_STEPS
print('OK:', ALL_STEPS)
"
```

- [ ] **Step 4: Commit**

```bash
git add partcraft/pipeline_v2/validators.py partcraft/pipeline_v2/run.py
git commit -m "feat(qc): wire sq1/sq2/sq3 into run.py dispatch and validators"
```

---

## Task 8: Skip qc_failed Edits in s4 / s5 / s5b

**Files:** Modify: s4_flux_2d.py, s5_trellis_3d.py, s5b_deletion.py

In each file add: from .qc_io import is_edit_qc_failed

In the loop building jobs/pending list, before each append:

```python
        if is_edit_qc_failed(ctx, spec.edit_id):
            log.info("[s4] skip %s (qc_fail)", spec.edit_id)  # adjust sX per file
            continue
```

- [ ] **Step 1: Edit s4_flux_2d.py** - find the jobs.append loop and add import + skip
- [ ] **Step 2: Edit s5_trellis_3d.py** - find the pending.append loop and add import + skip
- [ ] **Step 3: Edit s5b_deletion.py** - find deletion spec loop and add import + skip

- [ ] **Step 4: Import check**

```bash
python -c "
import partcraft.pipeline_v2.s4_flux_2d
import partcraft.pipeline_v2.s5_trellis_3d
import partcraft.pipeline_v2.s5b_deletion
print('all OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add partcraft/pipeline_v2/s4_flux_2d.py partcraft/pipeline_v2/s5_trellis_3d.py partcraft/pipeline_v2/s5b_deletion.py
git commit -m "feat(qc): skip qc_failed edits in s4/s5/s5b"
```

---

## Task 9: Update YAML Configs

**Files:** Modify: configs/pipeline_v2_shard00.yaml, configs/pipeline_v2_shard02.yaml

- [ ] **Step 1: Replace stages: block in both files** (remove optional: true on B, add 3 QC stages)

```yaml
  stages:
  - {name: A,    desc: "phase1 VLM",         servers: vlm,  steps: [s1]}
  - {name: A_qc, desc: "QC-A instruction",   servers: vlm,  steps: [sq1]}
  - {name: B,    desc: "highlights",          servers: none, steps: [s2]}
  - {name: C,    desc: "FLUX 2D",             servers: flux, steps: [s4]}
  - {name: C_qc, desc: "QC-C 2D region",     servers: vlm,  steps: [sq2]}
  - {name: D,    desc: "TRELLIS 3D edit",     servers: none, steps: [s5],      use_gpus: true}
  - {name: D2,   desc: "deletion mesh",       servers: none, steps: [s5b]}
  - {name: E,    desc: "3D rerender",         servers: none, steps: [s6, s6b], use_gpus: true}
  - {name: E_qc, desc: "QC-E final quality",  servers: vlm,  steps: [sq3]}
  - {name: F,    desc: "addition backfill",   servers: none, steps: [s7]}
```

- [ ] **Step 2: Add qc: block at end of both files**

```yaml
qc:
  vlm_score_threshold: 0.7
  thresholds_by_type:
    deletion:     {min_visual_quality: 3}
    modification: {min_visual_quality: 3, require_preserve_other: true}
    scale:        {min_visual_quality: 3, require_preserve_other: true}
    material:     {min_visual_quality: 3}
    global:       {min_visual_quality: 3}
    addition:     {min_visual_quality: 3}
```

- [ ] **Step 3: Verify scheduler reads new stages**

```bash
python -c "
import yaml
from partcraft.pipeline_v2.scheduler import stages_for
cfg = yaml.safe_load(open('configs/pipeline_v2_shard02.yaml'))
names = [s.name for s in stages_for(cfg)]
print(names)
assert names == ['A','A_qc','B','C','C_qc','D','D2','E','E_qc','F'], names
print('config OK')
"
```

- [ ] **Step 4: Full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass including test_qc_rules and test_qc_io.

- [ ] **Step 5: Commit**

```bash
git add configs/pipeline_v2_shard00.yaml configs/pipeline_v2_shard02.yaml
git commit -m "feat(qc): add A_qc/C_qc/E_qc stages to configs, remove B optional"
```

---

## Final Verification

```bash
python -m partcraft.pipeline_v2.run \
    --config configs/pipeline_v2_shard02.yaml \
    --shard 02 --all --dry-run
```

Expected: object list and manifest summary printed, no import errors.
