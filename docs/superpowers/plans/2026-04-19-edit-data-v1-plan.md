# Edit Dataset v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline-version-agnostic cleaned edit dataset (`data/partverse_edit_v1/`) by writing three CLIs (`promote_to_v1`, `encode_del_latent`, `rebuild_v1_index`) and a Gate-E adapter (`run_gate_quality_on_v2`), all under `partcraft/cleaning/v1/` + `scripts/cleaning/`, reusing existing v3 `run_gate_quality` and `migrate_slat_to_npz` Phase 5 code.

**Architecture:** v1 lives under `data/partverse_edit_v1/objects/<shard>/<obj_id>/{meta.json, before/, edits/<edit_id>/}`. Source-v2/v3 adapters read each pipeline's `edit_status.json`, normalize stage names to `gate_text_align` / `gate_quality`, and emit canonical promotion records. The promoter hardlinks/symlinks/copies artifacts into v1, leaving deletion-edit `after.npz` as a placeholder consumed later by an `encode_del_latent` CLI that wraps `scripts/tools/migrate_slat_to_npz.py --phase 5`. A separate one-off script `run_gate_quality_on_v2.py` constructs `ObjectContext` instances pointing at v2 directories and calls `partcraft.pipeline_v3.vlm_core.run_gate_quality` directly to backfill Gate E into v2's `edit_status.json`.

**Tech Stack:** Python 3, stdlib (json, pathlib, os.link), `numpy` for NPZ packing, existing `partcraft.pipeline_v3.{vlm_core,specs,paths}`, existing `scripts/tools/migrate_slat_to_npz.py` (imported as a library), `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-04-19-edit-data-v1-design.md`

---

## File Structure

### New module: `partcraft/cleaning/v1/`

```
partcraft/cleaning/v1/
├── __init__.py              # public exports
├── layout.py                # path/name conventions for the v1 root
├── canonical_record.py      # PromotionRecord dataclass + canonical pass schema
├── source_v2.py             # adapter: pipeline_v2 edit_status.json → PromotionRecord
├── source_v3.py             # adapter: pipeline_v3 edit_status.json → PromotionRecord
├── linker.py                # hardlink/symlink/copy with fallback
├── promoter.py              # apply rule + write v1 dirs (composes layout/linker)
├── pending.py               # del_latent pending-list reader/writer
└── indexer.py               # rebuild objects.jsonl + edits.jsonl
```

Each file has a single responsibility and is independently testable.

### New CLIs: `scripts/cleaning/`

```
scripts/cleaning/
├── __init__.py
├── promote_to_v1.py            # walks --source-runs, calls promoter
├── encode_del_latent.py        # reads _pending/del_latent.txt, drives migrate_slat_to_npz Phase 5
├── rebuild_v1_index.py         # calls indexer.rebuild
└── run_gate_quality_on_v2.py   # builds ObjectContext over v2 layout, calls run_gate_quality
```

### New config: `configs/cleaning/promote_v1.yaml`

Default promotion rule + v1 root path.

### New tests: `tests/cleaning/v1/`

```
tests/cleaning/v1/
├── __init__.py
├── conftest.py                  # fixtures: synthetic v2 + v3 obj dirs
├── test_layout.py
├── test_canonical_record.py
├── test_source_v2.py
├── test_source_v3.py
├── test_linker.py
├── test_promoter.py
├── test_pending.py
└── test_indexer.py
```

Integration tests for `run_gate_quality_on_v2` (requires VLM server) and `encode_del_latent` (requires Blender + Trellis weights) are deferred to manual smoke (Task 13).

---

## Task 0: Bootstrap directories and default config

**Files:**
- Create: `partcraft/cleaning/v1/__init__.py`
- Create: `scripts/cleaning/__init__.py`
- Create: `tests/cleaning/v1/__init__.py`
- Create: `configs/cleaning/promote_v1.yaml`

- [ ] **Step 1: Create empty package init files**

```bash
mkdir -p partcraft/cleaning/v1 scripts/cleaning tests/cleaning/v1 configs/cleaning
touch partcraft/cleaning/v1/__init__.py scripts/cleaning/__init__.py tests/cleaning/v1/__init__.py
```

- [ ] **Step 2: Write the default promote rule config**

Create `configs/cleaning/promote_v1.yaml`:

```yaml
v1_root: data/partverse_edit_v1

promote_rules:
  required_passes:
    - gate_text_align
    - gate_quality
  edit_types_allowed:
    - deletion
    - addition
    - modification
    - scale
    - material
    - color
    - glb

source_layouts:
  v2:
    objects_glob: "objects/*/*"
    edit_status_relpath: "edit_status.json"
  v3:
    objects_glob: "*/objects/*/*"
    edit_status_relpath: "edit_status.json"

before_assets:
  view_indices: [89, 90, 91, 100, 8]
  img_enc_root: data/partverse/img_Enc
  slat_root:    data/partverse/slat

link_mode: hardlink   # hardlink | symlink | copy
```

- [ ] **Step 3: Commit**

```bash
git add partcraft/cleaning/v1/__init__.py scripts/cleaning/__init__.py tests/cleaning/v1/__init__.py configs/cleaning/promote_v1.yaml
git commit -m "chore(cleaning/v1): scaffold package + default promote rule config"
```

---

## Task 1: `v1.layout` — path/name helpers

**Files:**
- Create: `partcraft/cleaning/v1/layout.py`
- Test: `tests/cleaning/v1/test_layout.py`

The layout module owns *all* knowledge of the v1 directory shape. Other modules call into it; nothing else hardcodes paths.

- [ ] **Step 1: Write the failing test**

Create `tests/cleaning/v1/test_layout.py`:

```python
from pathlib import Path

import pytest

from partcraft.cleaning.v1.layout import V1Layout, parse_edit_id, EDIT_TYPE_BY_PREFIX


@pytest.fixture
def v1(tmp_path):
    return V1Layout(root=tmp_path / "v1")


def test_object_dir_uses_shard_subdir(v1):
    p = v1.object_dir("05", "abc123")
    assert p == v1.root / "objects" / "05" / "abc123"


def test_before_views_paths_use_view_k_naming(v1):
    paths = v1.before_view_paths("05", "abc123")
    assert [p.name for p in paths] == [f"view_{k}.png" for k in range(5)]
    assert all(p.parent.name == "views" for p in paths)


def test_edit_dir_with_no_collision(v1):
    p = v1.edit_dir("05", "abc123", "del_abc123_000", suffix="")
    assert p == v1.object_dir("05", "abc123") / "edits" / "del_abc123_000"


def test_edit_dir_with_disambiguation_suffix(v1):
    p = v1.edit_dir("05", "abc123", "del_abc123_000", suffix="__r2")
    assert p.name == "del_abc123_000__r2"


def test_after_views_naming(v1):
    paths = v1.after_view_paths("05", "abc123", "del_abc123_000", suffix="")
    assert [p.name for p in paths] == [f"view_{k}.png" for k in range(5)]


def test_after_npz_path(v1):
    p = v1.after_npz_path("05", "abc123", "del_abc123_000", suffix="")
    assert p.name == "after.npz"


def test_pending_del_latent_file(v1):
    p = v1.pending_del_latent_file()
    assert p == v1.root / "_pending" / "del_latent.txt"


def test_index_files(v1):
    assert v1.objects_jsonl() == v1.root / "index" / "objects.jsonl"
    assert v1.edits_jsonl() == v1.root / "index" / "edits.jsonl"


@pytest.mark.parametrize("edit_id, expected_type", [
    ("del_abc_000", "deletion"),
    ("add_abc_001", "addition"),
    ("mod_abc_002", "modification"),
    ("scl_abc_003", "scale"),
    ("mat_abc_004", "material"),
    ("clr_abc_005", "color"),
    ("glb_abc_006", "glb"),
])
def test_parse_edit_id_prefix(edit_id, expected_type):
    typ, _obj, _idx = parse_edit_id(edit_id)
    assert typ == expected_type


def test_parse_edit_id_returns_obj_id_and_index():
    typ, obj_id, idx = parse_edit_id("del_5dc4ca7d607c495bb82eca3d0153cc2c_007")
    assert typ == "deletion"
    assert obj_id == "5dc4ca7d607c495bb82eca3d0153cc2c"
    assert idx == 7


def test_parse_edit_id_rejects_unknown_prefix():
    with pytest.raises(ValueError, match="unknown edit type prefix"):
        parse_edit_id("xyz_abc_000")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_layout.py -v
```

Expected: ImportError for `partcraft.cleaning.v1.layout`.

- [ ] **Step 3: Write the implementation**

Create `partcraft/cleaning/v1/layout.py`:

```python
"""Path and naming conventions for the edit dataset v1 layout.

All v1-side path construction goes through ``V1Layout``.  This is the only
module that knows the v1 directory shape; downstream code stays decoupled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EDIT_TYPE_BY_PREFIX: dict[str, str] = {
    "del": "deletion", "add": "addition", "mod": "modification",
    "scl": "scale", "mat": "material", "clr": "color", "glb": "glb",
}

N_VIEWS: int = 5


def parse_edit_id(edit_id: str) -> tuple[str, str, int]:
    parts = edit_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"malformed edit_id: {edit_id!r}")
    prefix, *middle, idx_str = parts
    if prefix not in EDIT_TYPE_BY_PREFIX:
        raise ValueError(f"unknown edit type prefix {prefix!r} in {edit_id!r}")
    obj_id = "_".join(middle)
    try:
        idx = int(idx_str)
    except ValueError as e:
        raise ValueError(f"non-integer suffix in {edit_id!r}") from e
    return EDIT_TYPE_BY_PREFIX[prefix], obj_id, idx


@dataclass(frozen=True)
class V1Layout:
    root: Path

    def object_dir(self, shard: str, obj_id: str) -> Path:
        return self.root / "objects" / shard / obj_id

    def meta_json(self, shard: str, obj_id: str) -> Path:
        return self.object_dir(shard, obj_id) / "meta.json"

    def before_dir(self, shard: str, obj_id: str) -> Path:
        return self.object_dir(shard, obj_id) / "before"

    def before_ss_npz(self, shard: str, obj_id: str) -> Path:
        return self.before_dir(shard, obj_id) / "ss.npz"

    def before_slat_npz(self, shard: str, obj_id: str) -> Path:
        return self.before_dir(shard, obj_id) / "slat.npz"

    def before_view_paths(self, shard: str, obj_id: str) -> list[Path]:
        d = self.before_dir(shard, obj_id) / "views"
        return [d / f"view_{k}.png" for k in range(N_VIEWS)]

    def edit_dir(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.object_dir(shard, obj_id) / "edits" / f"{edit_id}{suffix}"

    def spec_json(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "spec.json"

    def qc_json(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "qc.json"

    def after_npz_path(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "after.npz"

    def after_pending_marker(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "_after_pending.json"

    def after_view_paths(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> list[Path]:
        d = self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "views"
        return [d / f"view_{k}.png" for k in range(N_VIEWS)]

    def pending_del_latent_file(self) -> Path:
        return self.root / "_pending" / "del_latent.txt"

    def objects_jsonl(self) -> Path:
        return self.root / "index" / "objects.jsonl"

    def edits_jsonl(self) -> Path:
        return self.root / "index" / "edits.jsonl"

    def last_rebuild_json(self) -> Path:
        return self.root / "index" / "_last_rebuild.json"

    def iter_object_dirs(self) -> Iterable[Path]:
        objects_root = self.root / "objects"
        if not objects_root.is_dir():
            return
        for shard_dir in sorted(objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            for obj_dir in sorted(shard_dir.iterdir()):
                if obj_dir.is_dir():
                    yield obj_dir
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/cleaning/v1/test_layout.py -v
```

Expected: 17 PASS (10 distinct test functions; `test_parse_edit_id_prefix` is parametrized over 7 edit-type prefixes, so pytest reports each as a separate case).

- [ ] **Step 5: Commit**

```bash
git add partcraft/cleaning/v1/layout.py tests/cleaning/v1/test_layout.py
git commit -m "feat(cleaning/v1): add V1Layout path helpers and edit_id parser"
```

---

## Task 2: `v1.canonical_record` — PromotionRecord dataclass

**Files:**
- Create: `partcraft/cleaning/v1/canonical_record.py`
- Test: `tests/cleaning/v1/test_canonical_record.py`

`PromotionRecord` is the pipeline-version-agnostic intermediate produced by source adapters and consumed by the promoter. It carries absolute source paths needed to materialize v1 files plus the canonical pass dict.

- [ ] **Step 1: Write the failing test**

Create `tests/cleaning/v1/test_canonical_record.py`:

```python
from pathlib import Path

import pytest

from partcraft.cleaning.v1.canonical_record import (
    PromotionRecord, PassResult, evaluate_rule,
)


def _ok_passes() -> dict[str, PassResult]:
    return {
        "gate_text_align": PassResult(passed=True, score=1.0,
                                      producer="v3.gate_text_align@x", reason="", ts="t"),
        "gate_quality":    PassResult(passed=True, score=0.9,
                                      producer="v3.gate_quality@x", reason="", ts="t"),
    }


def test_record_holds_source_paths():
    rec = PromotionRecord(
        obj_id="abc", shard="05", edit_id="del_abc_000", edit_type="deletion",
        source_pipeline="v2", source_run_tag="pipeline_v2_shard05",
        source_run_dir=Path("/tmp/run/abc"),
        spec={"prompt": "Remove handle", "selected_part_ids": [0]},
        passes=_ok_passes(),
        after_glb=Path("/tmp/run/abc/edits_3d/del_abc_000/after_new.glb"),
        after_npz=None,
        preview_pngs=[Path(f"/tmp/run/abc/edits_3d/del_abc_000/preview_{k}.png") for k in range(5)],
    )
    assert rec.is_deletion() is True
    assert rec.is_flux_branch() is False


def test_evaluate_rule_passes_when_all_required_pass():
    rule = {"required_passes": ["gate_text_align", "gate_quality"]}
    assert evaluate_rule(_ok_passes(), rule) == (True, "")


def test_evaluate_rule_fails_when_required_pass_failed():
    p = _ok_passes()
    p["gate_quality"] = PassResult(passed=False, score=0.1,
                                   producer="x", reason="bad", ts="t")
    ok, reason = evaluate_rule(p, {"required_passes": ["gate_text_align", "gate_quality"]})
    assert ok is False
    assert "gate_quality" in reason


def test_evaluate_rule_defers_when_required_pass_missing():
    rule = {"required_passes": ["gate_text_align", "gate_quality", "future_pass"]}
    ok, reason = evaluate_rule(_ok_passes(), rule)
    assert ok is False
    assert "future_pass" in reason
    assert "missing" in reason


def test_pass_result_to_json_roundtrip():
    pr = PassResult(passed=True, score=0.5, producer="x", reason="ok", ts="t",
                    extra={"metric_a": 0.3})
    d = pr.to_json()
    pr2 = PassResult.from_json(d)
    assert pr2 == pr


def test_record_qc_dict_contains_source_and_passes():
    rec = PromotionRecord(
        obj_id="abc", shard="05", edit_id="del_abc_000", edit_type="deletion",
        source_pipeline="v2", source_run_tag="pipeline_v2_shard05",
        source_run_dir=Path("/tmp/run/abc"),
        spec={}, passes=_ok_passes(),
        after_glb=Path("/x"), after_npz=None, preview_pngs=[],
    )
    qc = rec.to_qc_json(promoted_at="2026-04-19T12:00:00Z")
    assert qc["edit_id"] == "del_abc_000"
    assert qc["source"]["pipeline_version"] == "v2"
    assert qc["source"]["run_tag"] == "pipeline_v2_shard05"
    assert qc["source"]["promoted_at"] == "2026-04-19T12:00:00Z"
    assert set(qc["passes"].keys()) == {"gate_text_align", "gate_quality"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_canonical_record.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `partcraft/cleaning/v1/canonical_record.py`:

```python
"""Canonical, pipeline-version-agnostic representation of one promotable edit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PassResult:
    passed: bool
    score: float | None = None
    producer: str = ""
    reason: str = ""
    ts: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "pass": self.passed, "score": self.score,
            "producer": self.producer, "reason": self.reason, "ts": self.ts,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PassResult":
        return cls(
            passed=bool(d["pass"]),
            score=d.get("score"),
            producer=d.get("producer", ""),
            reason=d.get("reason", ""),
            ts=d.get("ts", ""),
            extra=dict(d.get("extra") or {}),
        )


@dataclass
class PromotionRecord:
    obj_id: str
    shard: str
    edit_id: str
    edit_type: str
    source_pipeline: str           # "v2" | "v3"
    source_run_tag: str
    source_run_dir: Path
    spec: dict[str, Any]
    passes: dict[str, PassResult]
    after_glb: Path | None
    after_npz: Path | None
    preview_pngs: list[Path]

    def is_deletion(self) -> bool:
        return self.edit_type == "deletion"

    def is_flux_branch(self) -> bool:
        return self.edit_type in {"modification", "scale", "material",
                                   "color", "glb", "addition"}

    def to_qc_json(self, *, promoted_at: str) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "source": {
                "pipeline_version": self.source_pipeline,
                "run_tag": self.source_run_tag,
                "run_dir": str(self.source_run_dir),
                "promoted_at": promoted_at,
            },
            "passes": {name: pr.to_json() for name, pr in self.passes.items()},
        }


def evaluate_rule(
    passes: dict[str, PassResult], rule: dict[str, Any],
) -> tuple[bool, str]:
    required = list(rule.get("required_passes", []))
    missing = [name for name in required if name not in passes]
    if missing:
        return False, f"missing: {','.join(missing)}"
    failing = [name for name in required if not passes[name].passed]
    if failing:
        return False, f"failed: {','.join(failing)}"
    return True, ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/cleaning/v1/test_canonical_record.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/cleaning/v1/canonical_record.py tests/cleaning/v1/test_canonical_record.py
git commit -m "feat(cleaning/v1): add PromotionRecord + canonical pass schema + rule evaluator"
```

---

## Task 3: `v1.source_v2` — adapter for pipeline_v2 outputs

**Files:**
- Create: `partcraft/cleaning/v1/source_v2.py`
- Test: `tests/cleaning/v1/test_source_v2.py`
- Create: `tests/cleaning/v1/conftest.py`

The adapter reads `<run>/objects/<NN>/<obj_id>/edit_status.json` (v2 schema) plus `phase1/parsed.json` (for the EditSpec subset) and emits `PromotionRecord` per edit.

- [ ] **Step 1: Create the conftest with synthetic v2 fixture**

Create `tests/cleaning/v1/conftest.py`:

```python
"""Shared fixtures for cleaning/v1 tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _touch(p: Path, content: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


@pytest.fixture
def v2_obj_dir(tmp_path: Path) -> Path:
    obj = tmp_path / "pipeline_v2_shard05" / "objects" / "05" / "objA"
    _write_json(obj / "phase1" / "parsed.json", {
        "object": {"full_desc_stage1": "An object", "full_desc_stage2": ""},
        "parts": [{"id": 0, "name": "handle"}, {"id": 1, "name": "body"}],
        "edits": [
            {"edit_type": "deletion", "selected_part_ids": [0],
             "prompt": "Remove the handle.", "target_part_desc": "handle on right",
             "view_index": 0, "edit_params": {}},
            {"edit_type": "modification", "selected_part_ids": [1],
             "prompt": "Make the body wooden.", "target_part_desc": "the body",
             "view_index": 1, "edit_params": {"new_part_desc": "wooden body"}},
        ],
    })
    _write_json(obj / "edit_status.json", {
        "obj_id": "objA", "shard": "05", "schema_version": 1,
        "edits": {
            "del_objA_000": {
                "edit_type": "deletion",
                "stages": {"gate_a": {"status": "pass"}},
                "gates": {
                    "A": {"rule": {"pass": True}, "vlm": {"pass": True, "score": 1.0, "reason": "ok"}},
                    "C": None,
                    "E": {"vlm": {"pass": True, "score": 0.92, "reason": "good"}},
                },
                "final_pass": True,
            },
            "mod_objA_001": {
                "edit_type": "modification",
                "stages": {"gate_a": {"status": "pass"}},
                "gates": {
                    "A": {"rule": {"pass": True}, "vlm": {"pass": True, "score": 1.0, "reason": "ok"}},
                    "C": None,
                    "E": {"vlm": {"pass": False, "score": 0.2, "reason": "blurry"}},
                },
                "final_pass": False,
            },
        },
    })
    _touch(obj / "edits_3d" / "del_objA_000" / "after_new.glb")
    for k in range(5):
        _touch(obj / "edits_3d" / "del_objA_000" / f"preview_{k}.png")
    _touch(obj / "edits_3d" / "mod_objA_001" / "after.npz")
    for k in range(5):
        _touch(obj / "edits_3d" / "mod_objA_001" / f"preview_{k}.png")
    return obj


@pytest.fixture
def v3_obj_dir(tmp_path: Path) -> Path:
    obj = tmp_path / "shard08_run" / "mode_e_text_align" / "objects" / "08" / "objB"
    _write_json(obj / "phase1" / "parsed.json", {
        "object": {"full_desc_stage1": "Another object"},
        "parts": [{"id": 0, "name": "leg"}],
        "edits": [
            {"edit_type": "deletion", "selected_part_ids": [0],
             "prompt": "Remove the leg.", "target_part_desc": "the leg",
             "view_index": 0, "edit_params": {}},
        ],
    })
    _write_json(obj / "edit_status.json", {
        "obj_id": "objB", "shard": "08", "schema_version": 1,
        "edits": {
            "del_objB_000": {
                "edit_type": "deletion",
                "stages": {"sq3_qc_E": {"status": "ok"}},
                "gates": {
                    "A": {"rule": {"pass": True}, "vlm": {"pass": True, "score": 1.0, "reason": "ok"}},
                    "C": None,
                    "E": {"vlm": {"pass": True, "score": 0.88, "reason": "ok"}},
                },
                "final_pass": True,
            },
        },
    })
    _touch(obj / "edits_3d" / "del_objB_000" / "after_new.glb")
    for k in range(5):
        _touch(obj / "edits_3d" / "del_objB_000" / f"preview_{k}.png")
    return obj
```

- [ ] **Step 2: Write the failing test**

Create `tests/cleaning/v1/test_source_v2.py`:

```python
import json
from pathlib import Path

import pytest

from partcraft.cleaning.v1.canonical_record import PassResult
from partcraft.cleaning.v1.source_v2 import iter_records_from_v2_obj


def test_v2_adapter_yields_one_record_per_edit(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    assert len(recs) == 2
    by_id = {r.edit_id: r for r in recs}
    assert set(by_id.keys()) == {"del_objA_000", "mod_objA_001"}


def test_v2_adapter_extracts_canonical_passes(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    by_id = {r.edit_id: r for r in recs}
    rec = by_id["del_objA_000"]
    assert rec.passes["gate_text_align"].passed is True
    assert rec.passes["gate_quality"].passed is True
    assert rec.passes["gate_quality"].score == pytest.approx(0.92)


def test_v2_adapter_records_failing_gate_e(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    by_id = {r.edit_id: r for r in recs}
    rec = by_id["mod_objA_001"]
    assert rec.passes["gate_quality"].passed is False
    assert "blurry" in rec.passes["gate_quality"].reason


def test_v2_adapter_omits_pass_when_gate_e_is_null(v2_obj_dir: Path):
    es_path = v2_obj_dir / "edit_status.json"
    es = json.loads(es_path.read_text())
    es["edits"]["del_objA_000"]["gates"]["E"] = None
    es_path.write_text(json.dumps(es))
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    rec = next(r for r in recs if r.edit_id == "del_objA_000")
    assert "gate_quality" not in rec.passes


def test_v2_adapter_attaches_after_paths_per_branch(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    by_id = {r.edit_id: r for r in recs}
    del_rec = by_id["del_objA_000"]
    mod_rec = by_id["mod_objA_001"]
    assert del_rec.after_glb is not None and del_rec.after_glb.name == "after_new.glb"
    assert del_rec.after_npz is None
    assert mod_rec.after_glb is None
    assert mod_rec.after_npz is not None and mod_rec.after_npz.name == "after.npz"
    assert len(del_rec.preview_pngs) == 5


def test_v2_adapter_extracts_spec_subset(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    rec = next(r for r in recs if r.edit_id == "del_objA_000")
    assert rec.spec["prompt"] == "Remove the handle."
    assert rec.spec["edit_type"] == "deletion"
    assert rec.spec["selected_part_ids"] == [0]
    assert rec.spec["part_labels"] == ["handle"]


def test_v2_adapter_uses_run_tag_as_provenance(v2_obj_dir: Path):
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    for r in recs:
        assert r.source_pipeline == "v2"
        assert r.source_run_tag == "pipeline_v2_shard05"
        assert r.source_run_dir == v2_obj_dir
        assert r.shard == "05"
        assert r.obj_id == "objA"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_source_v2.py -v
```

Expected: ImportError.

- [ ] **Step 4: Write the implementation**

Create `partcraft/cleaning/v1/source_v2.py`:

```python
"""Adapter: pipeline_v2 ``edit_status.json`` -> ``PromotionRecord`` stream."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .canonical_record import PromotionRecord, PassResult
from .layout import parse_edit_id


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _gate_e_to_pass(gate_e: dict[str, Any] | None) -> PassResult | None:
    if not gate_e:
        return None
    vlm = gate_e.get("vlm") or {}
    if not vlm:
        return None
    extra: dict[str, Any] = {}
    if isinstance(vlm.get("metrics"), dict):
        extra["metrics"] = vlm["metrics"]
    return PassResult(
        passed=bool(vlm.get("pass")),
        score=vlm.get("score"),
        producer="v3.gate_quality (backfilled into v2)",
        reason=vlm.get("reason", ""),
        ts=vlm.get("ts", ""),
        extra=extra,
    )


def _gate_a_to_pass(gate_a: dict[str, Any] | None) -> PassResult | None:
    if not gate_a:
        return None
    vlm = gate_a.get("vlm") or {}
    rule = gate_a.get("rule") or {}
    rule_pass = bool(rule.get("pass", False))
    vlm_pass = bool(vlm.get("pass", False)) if vlm else rule_pass
    return PassResult(
        passed=rule_pass and vlm_pass,
        score=vlm.get("score") if vlm else None,
        producer="v2.s1_phase1 / sq1_qc_A",
        reason=vlm.get("reason", "") if vlm else "",
        ts=vlm.get("ts", "") if vlm else "",
    )


def _spec_subset_for_edit(
    edit_id: str, edit_type: str, parsed_edits: list[dict[str, Any]],
    parts_by_id: dict[int, str],
) -> dict[str, Any]:
    _, _, idx = parse_edit_id(edit_id)
    same_type = [e for e in parsed_edits if e.get("edit_type") == edit_type]
    src = same_type[idx] if idx < len(same_type) else {}
    pids = list(src.get("selected_part_ids") or [])
    return {
        "edit_id": edit_id,
        "edit_type": edit_type,
        "prompt": src.get("prompt", ""),
        "selected_part_ids": pids,
        "part_labels": [parts_by_id.get(p, "") for p in pids],
        "target_part_desc": src.get("target_part_desc", ""),
        "new_parts_desc": (src.get("new_parts_desc") or src.get("target_part_desc", "")),
        "edit_params": dict(src.get("edit_params") or {}),
    }


def iter_records_from_v2_obj(
    obj_dir: Path, *, run_tag: str,
) -> Iterator[PromotionRecord]:
    es = _read_json(obj_dir / "edit_status.json")
    obj_id = es.get("obj_id") or obj_dir.name
    shard = es.get("shard") or obj_dir.parent.name

    parsed_p = obj_dir / "phase1" / "parsed.json"
    parsed = _read_json(parsed_p) if parsed_p.is_file() else {"edits": [], "parts": []}
    parsed_edits = parsed.get("edits") or []
    parts_by_id = {int(p["id"]): p.get("name", "") for p in (parsed.get("parts") or [])}

    edits = es.get("edits") or {}
    for edit_id, entry in edits.items():
        edit_type = entry.get("edit_type", "")
        gates = entry.get("gates") or {}
        passes: dict[str, PassResult] = {}
        gta = _gate_a_to_pass(gates.get("A"))
        if gta is not None:
            passes["gate_text_align"] = gta
        gte = _gate_e_to_pass(gates.get("E"))
        if gte is not None:
            passes["gate_quality"] = gte

        edits_3d_dir = obj_dir / "edits_3d" / edit_id
        after_glb_p = edits_3d_dir / "after_new.glb"
        after_npz_p = edits_3d_dir / "after.npz"
        preview_pngs = [edits_3d_dir / f"preview_{k}.png" for k in range(5)]

        spec = _spec_subset_for_edit(edit_id, edit_type, parsed_edits, parts_by_id)

        yield PromotionRecord(
            obj_id=obj_id, shard=shard,
            edit_id=edit_id, edit_type=edit_type,
            source_pipeline="v2", source_run_tag=run_tag,
            source_run_dir=obj_dir,
            spec=spec, passes=passes,
            after_glb=after_glb_p if after_glb_p.is_file() else None,
            after_npz=after_npz_p if after_npz_p.is_file() else None,
            preview_pngs=preview_pngs,
        )


def iter_records_from_v2_run(
    run_root: Path, *, run_tag: str | None = None,
) -> Iterator[PromotionRecord]:
    tag = run_tag or run_root.name
    objects_root = run_root / "objects"
    if not objects_root.is_dir():
        return
    for shard_dir in sorted(objects_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        for obj_dir in sorted(shard_dir.iterdir()):
            if not (obj_dir / "edit_status.json").is_file():
                continue
            yield from iter_records_from_v2_obj(obj_dir, run_tag=tag)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/cleaning/v1/test_source_v2.py -v
```

Expected: 7 PASS.

- [ ] **Step 6: Commit**

```bash
git add partcraft/cleaning/v1/source_v2.py tests/cleaning/v1/test_source_v2.py tests/cleaning/v1/conftest.py
git commit -m "feat(cleaning/v1): add pipeline_v2 source adapter + shared test fixtures"
```

---

## Task 4: `v1.source_v3` — adapter for pipeline_v3 outputs

**Files:**
- Create: `partcraft/cleaning/v1/source_v3.py`
- Test: `tests/cleaning/v1/test_source_v3.py`

The v3 schema differs from v2 only in run dir layout (extra `mode_e_text_align/` segment) and producer string. We share most logic with `source_v2` but expose a separate entry to keep schemas decoupled.

- [ ] **Step 1: Write the failing test**

Create `tests/cleaning/v1/test_source_v3.py`:

```python
from pathlib import Path

from partcraft.cleaning.v1.source_v3 import (
    iter_records_from_v3_obj, iter_records_from_v3_run,
)


def test_v3_adapter_extracts_record(v3_obj_dir: Path):
    recs = list(iter_records_from_v3_obj(v3_obj_dir, run_tag="shard08_run"))
    assert len(recs) == 1
    rec = recs[0]
    assert rec.source_pipeline == "v3"
    assert rec.source_run_tag == "shard08_run"
    assert rec.shard == "08"
    assert rec.obj_id == "objB"
    assert rec.passes["gate_text_align"].passed is True
    assert rec.passes["gate_quality"].passed is True
    assert "v3.gate_quality" in rec.passes["gate_quality"].producer


def test_v3_run_walker_finds_obj(v3_obj_dir: Path):
    run_root = v3_obj_dir.parents[3]
    recs = list(iter_records_from_v3_run(run_root, run_tag="shard08_run"))
    assert len(recs) == 1
    assert recs[0].edit_id == "del_objB_000"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_source_v3.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `partcraft/cleaning/v1/source_v3.py`:

```python
"""Adapter: pipeline_v3 ``edit_status.json`` -> ``PromotionRecord`` stream."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .canonical_record import PromotionRecord, PassResult
from .source_v2 import _read_json, _gate_a_to_pass, _spec_subset_for_edit


def _gate_e_v3_to_pass(gate_e: dict | None) -> PassResult | None:
    if not gate_e:
        return None
    vlm = gate_e.get("vlm") or {}
    if not vlm:
        return None
    extra: dict = {}
    if isinstance(vlm.get("metrics"), dict):
        extra["metrics"] = vlm["metrics"]
    return PassResult(
        passed=bool(vlm.get("pass")),
        score=vlm.get("score"),
        producer="v3.gate_quality (native)",
        reason=vlm.get("reason", ""),
        ts=vlm.get("ts", ""),
        extra=extra,
    )


def iter_records_from_v3_obj(
    obj_dir: Path, *, run_tag: str,
) -> Iterator[PromotionRecord]:
    es = _read_json(obj_dir / "edit_status.json")
    obj_id = es.get("obj_id") or obj_dir.name
    shard = es.get("shard") or obj_dir.parent.name

    parsed_p = obj_dir / "phase1" / "parsed.json"
    parsed = _read_json(parsed_p) if parsed_p.is_file() else {"edits": [], "parts": []}
    parsed_edits = parsed.get("edits") or []
    parts_by_id = {int(p["id"]): p.get("name", "") for p in (parsed.get("parts") or [])}

    edits = es.get("edits") or {}
    for edit_id, entry in edits.items():
        edit_type = entry.get("edit_type", "")
        gates = entry.get("gates") or {}
        passes: dict[str, PassResult] = {}
        gta = _gate_a_to_pass(gates.get("A"))
        if gta is not None:
            passes["gate_text_align"] = gta
        gte = _gate_e_v3_to_pass(gates.get("E"))
        if gte is not None:
            passes["gate_quality"] = gte

        edits_3d_dir = obj_dir / "edits_3d" / edit_id
        after_glb_p = edits_3d_dir / "after_new.glb"
        after_npz_p = edits_3d_dir / "after.npz"
        preview_pngs = [edits_3d_dir / f"preview_{k}.png" for k in range(5)]

        spec = _spec_subset_for_edit(edit_id, edit_type, parsed_edits, parts_by_id)

        yield PromotionRecord(
            obj_id=obj_id, shard=shard,
            edit_id=edit_id, edit_type=edit_type,
            source_pipeline="v3", source_run_tag=run_tag,
            source_run_dir=obj_dir,
            spec=spec, passes=passes,
            after_glb=after_glb_p if after_glb_p.is_file() else None,
            after_npz=after_npz_p if after_npz_p.is_file() else None,
            preview_pngs=preview_pngs,
        )


def iter_records_from_v3_run(
    run_root: Path, *, run_tag: str | None = None,
) -> Iterator[PromotionRecord]:
    """Walk ``<run_root>/[<mode>/]objects/<NN>/<obj_id>/`` and yield records."""
    tag = run_tag or run_root.name
    candidates: list[Path] = []
    if (run_root / "objects").is_dir():
        candidates = [run_root]
    else:
        for child in sorted(run_root.iterdir()):
            if child.is_dir() and (child / "objects").is_dir():
                candidates.append(child)
    for mode_dir in candidates:
        objects_root = mode_dir / "objects"
        for shard_dir in sorted(objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            for obj_dir in sorted(shard_dir.iterdir()):
                if not (obj_dir / "edit_status.json").is_file():
                    continue
                yield from iter_records_from_v3_obj(obj_dir, run_tag=tag)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/cleaning/v1/test_source_v3.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/cleaning/v1/source_v3.py tests/cleaning/v1/test_source_v3.py
git commit -m "feat(cleaning/v1): add pipeline_v3 source adapter"
```

---

## Task 5: `v1.linker` — hardlink/symlink/copy with fallback

**Files:**
- Create: `partcraft/cleaning/v1/linker.py`
- Test: `tests/cleaning/v1/test_linker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cleaning/v1/test_linker.py`:

```python
from pathlib import Path

import pytest

from partcraft.cleaning.v1.linker import LinkMode, link_one


@pytest.fixture
def src(tmp_path: Path) -> Path:
    p = tmp_path / "src" / "a.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"hello")
    return p


def test_hardlink_same_fs(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    res = link_one(src, dst, mode=LinkMode.HARDLINK)
    assert dst.read_bytes() == b"hello"
    assert dst.stat().st_ino == src.stat().st_ino
    assert res.mode_used == LinkMode.HARDLINK


def test_symlink_mode(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    res = link_one(src, dst, mode=LinkMode.SYMLINK)
    assert dst.is_symlink()
    assert dst.read_bytes() == b"hello"
    assert res.mode_used == LinkMode.SYMLINK


def test_copy_mode(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    res = link_one(src, dst, mode=LinkMode.COPY)
    assert not dst.is_symlink()
    assert dst.stat().st_ino != src.stat().st_ino
    assert res.mode_used == LinkMode.COPY


def test_existing_dst_is_skipped(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"existing")
    res = link_one(src, dst, mode=LinkMode.HARDLINK)
    assert res.skipped is True
    assert dst.read_bytes() == b"existing"


def test_force_overwrites_existing(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"existing")
    res = link_one(src, dst, mode=LinkMode.HARDLINK, force=True)
    assert res.skipped is False
    assert dst.read_bytes() == b"hello"


def test_missing_src_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        link_one(tmp_path / "nope.bin", tmp_path / "dst.bin", mode=LinkMode.HARDLINK)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_linker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `partcraft/cleaning/v1/linker.py`:

```python
"""Materialize a source file into the v1 dataset by hardlink/symlink/copy."""
from __future__ import annotations

import enum
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


class LinkMode(str, enum.Enum):
    HARDLINK = "hardlink"
    SYMLINK = "symlink"
    COPY = "copy"


@dataclass(frozen=True)
class LinkResult:
    src: Path
    dst: Path
    mode_used: LinkMode
    skipped: bool = False
    fell_back: bool = False


def link_one(
    src: Path, dst: Path, *, mode: LinkMode, force: bool = False,
) -> LinkResult:
    src = Path(src); dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"link_one: source missing: {src}")
    if dst.exists() and not force:
        return LinkResult(src=src, dst=dst, mode_used=mode, skipped=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    fell_back = False
    actual_mode = mode
    try:
        if mode is LinkMode.HARDLINK:
            os.link(src, dst)
        elif mode is LinkMode.SYMLINK:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
    except OSError:
        if mode is LinkMode.HARDLINK:
            shutil.copy2(src, dst)
            fell_back = True
            actual_mode = LinkMode.COPY
        else:
            raise
    return LinkResult(src=src, dst=dst, mode_used=actual_mode, fell_back=fell_back)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/cleaning/v1/test_linker.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/cleaning/v1/linker.py tests/cleaning/v1/test_linker.py
git commit -m "feat(cleaning/v1): add linker with hardlink/symlink/copy + fallback"
```

---

## Task 6: `v1.pending` — del_latent pending list

**Files:**
- Create: `partcraft/cleaning/v1/pending.py`
- Test: `tests/cleaning/v1/test_pending.py`

The pending list is a TSV file (`<shard>\t<obj_id>\t<edit_id>\t<suffix>` per line) with `fcntl.flock` for safe concurrent append.

- [ ] **Step 1: Write the failing test**

Create `tests/cleaning/v1/test_pending.py`:

```python
from pathlib import Path

from partcraft.cleaning.v1.pending import DelLatentPending, PendingEntry


def test_append_and_iter(tmp_path: Path):
    pending = DelLatentPending(tmp_path / "del_latent.txt")
    pending.append(PendingEntry("05", "objA", "del_objA_000", suffix=""))
    pending.append(PendingEntry("08", "objB", "del_objB_000", suffix="__r2"))
    entries = list(pending.iter_entries())
    assert entries == [
        PendingEntry("05", "objA", "del_objA_000", suffix=""),
        PendingEntry("08", "objB", "del_objB_000", suffix="__r2"),
    ]


def test_remove_keeps_others(tmp_path: Path):
    pending = DelLatentPending(tmp_path / "del_latent.txt")
    e1 = PendingEntry("05", "objA", "del_objA_000", suffix="")
    e2 = PendingEntry("08", "objB", "del_objB_000", suffix="")
    pending.append(e1); pending.append(e2)
    pending.remove(e1)
    assert list(pending.iter_entries()) == [e2]


def test_dedup_on_append(tmp_path: Path):
    pending = DelLatentPending(tmp_path / "del_latent.txt")
    e = PendingEntry("05", "objA", "del_objA_000", suffix="")
    pending.append(e); pending.append(e)
    assert len(list(pending.iter_entries())) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_pending.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `partcraft/cleaning/v1/pending.py`:

```python
"""Pending-list manager for deletion edits awaiting latent encoding."""
from __future__ import annotations

import fcntl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PendingEntry:
    shard: str
    obj_id: str
    edit_id: str
    suffix: str = ""

    def to_line(self) -> str:
        return f"{self.shard}\t{self.obj_id}\t{self.edit_id}\t{self.suffix}"

    @classmethod
    def from_line(cls, line: str) -> "PendingEntry":
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 4:
            raise ValueError(f"malformed pending line: {line!r}")
        return cls(parts[0], parts[1], parts[2], parts[3])


class DelLatentPending:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def iter_entries(self) -> Iterator[PendingEntry]:
        if not self.path.is_file():
            return
        for line in self.path.read_text().splitlines():
            if line.strip():
                yield PendingEntry.from_line(line)

    def append(self, entry: PendingEntry) -> None:
        self._ensure()
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                existing = {PendingEntry.from_line(ln) for ln in f.read().splitlines() if ln.strip()}
                if entry in existing:
                    return
                f.seek(0, 2)
                f.write(entry.to_line() + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def remove(self, entry: PendingEntry) -> None:
        if not self.path.is_file():
            return
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                kept = [ln for ln in f.read().splitlines()
                        if ln.strip() and PendingEntry.from_line(ln) != entry]
                f.seek(0); f.truncate()
                if kept:
                    f.write("\n".join(kept) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/cleaning/v1/test_pending.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/cleaning/v1/pending.py tests/cleaning/v1/test_pending.py
git commit -m "feat(cleaning/v1): add del_latent pending-list manager"
```

---

## Task 7: `v1.promoter` — apply rule + materialize v1 directory

**Files:**
- Create: `partcraft/cleaning/v1/promoter.py`
- Test: `tests/cleaning/v1/test_promoter.py`

The promoter composes layout + linker + pending. It materializes shared `before/` per-obj on first sight, one edit subdir per record, applies the `__r2`/`__r3` collision suffix per spec §10.3, and appends del edits to the pending list.

For SS+SLAT packing of before assets: the promoter loads `data/partverse/slat/<shard>/<obj_id>_{feats,coords,ss}.pt` (existing), packs them into `slat.npz` (keys: `feats`, `coords`) and `ss.npz` (key: `data` or whatever the .pt yields). No re-encoding — pure tensor->numpy.

If `<obj>_ss.pt` is absent (older runs) we leave a `ss.missing.json` placeholder beside `ss.npz`.

- [ ] **Step 1: Write the failing test**

Create `tests/cleaning/v1/test_promoter.py`:

```python
import json
from pathlib import Path

import pytest
import torch

from partcraft.cleaning.v1.layout import V1Layout
from partcraft.cleaning.v1.linker import LinkMode
from partcraft.cleaning.v1.promoter import PromoterConfig, promote_records
from partcraft.cleaning.v1.source_v2 import iter_records_from_v2_obj
from partcraft.cleaning.v1.pending import DelLatentPending


def _fake_before_assets(tmp_path: Path, obj_id: str = "objA", shard: str = "05") -> Path:
    data = tmp_path / "data_partverse"
    img_enc = data / "img_Enc" / obj_id
    img_enc.mkdir(parents=True)
    for i in [8, 89, 90, 91, 100]:
        (img_enc / f"{i:03d}.png").write_bytes(b"png")
    slat_dir = data / "slat" / shard
    slat_dir.mkdir(parents=True)
    torch.save(torch.zeros(1, 8), slat_dir / f"{obj_id}_feats.pt")
    torch.save(torch.zeros(1, 3, dtype=torch.int32), slat_dir / f"{obj_id}_coords.pt")
    torch.save(torch.zeros(1, 4), slat_dir / f"{obj_id}_ss.pt")
    return data


def _make_cfg(data_root: Path) -> PromoterConfig:
    return PromoterConfig(
        rule={"required_passes": ["gate_text_align", "gate_quality"]},
        link_mode=LinkMode.HARDLINK,
        img_enc_root=data_root / "img_Enc",
        slat_root=data_root / "slat",
        view_indices=[89, 90, 91, 100, 8],
    )


def test_promote_one_v2_obj_creates_layout(tmp_path: Path, v2_obj_dir: Path):
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = _make_cfg(data_root)
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    pending = DelLatentPending(v1.pending_del_latent_file())

    summary = promote_records(recs, layout=v1, cfg=cfg, pending=pending)

    # del_objA_000 passes (gate_quality.pass=True); mod_objA_001 fails (pass=False).
    assert summary.promoted == 1
    assert summary.deferred == 0
    assert summary.failed == 1

    assert v1.before_ss_npz("05", "objA").is_file()
    assert v1.before_slat_npz("05", "objA").is_file()
    for p in v1.before_view_paths("05", "objA"):
        assert p.is_file()

    edit_dir = v1.edit_dir("05", "objA", "del_objA_000")
    assert (edit_dir / "spec.json").is_file()
    assert (edit_dir / "qc.json").is_file()
    assert v1.after_pending_marker("05", "objA", "del_objA_000").is_file()
    assert not v1.after_npz_path("05", "objA", "del_objA_000").exists()
    for p in v1.after_view_paths("05", "objA", "del_objA_000"):
        assert p.is_file()

    assert any(e.edit_id == "del_objA_000" for e in pending.iter_entries())


def test_rerunning_same_source_skips_existing(tmp_path: Path, v2_obj_dir: Path):
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = _make_cfg(data_root)
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    pending = DelLatentPending(v1.pending_del_latent_file())
    promote_records(recs, layout=v1, cfg=cfg, pending=pending)
    summary2 = promote_records(recs, layout=v1, cfg=cfg, pending=pending)
    assert summary2.promoted == 0
    assert summary2.skipped_existing == 1


def test_collision_from_different_run_appends_r2_suffix(tmp_path: Path, v2_obj_dir: Path):
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = _make_cfg(data_root)
    recs1 = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    pending = DelLatentPending(v1.pending_del_latent_file())
    promote_records(recs1, layout=v1, cfg=cfg, pending=pending)

    recs2 = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05_rerun"))
    summary = promote_records(recs2, layout=v1, cfg=cfg, pending=pending)

    assert summary.promoted == 1
    edit_dir_r2 = v1.edit_dir("05", "objA", "del_objA_000", suffix="__r2")
    assert edit_dir_r2.is_dir()
    qc = json.loads(v1.qc_json("05", "objA", "del_objA_000", suffix="__r2").read_text())
    assert qc["source"]["run_tag"] == "pipeline_v2_shard05_rerun"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_promoter.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `partcraft/cleaning/v1/promoter.py`:

```python
"""Apply promotion rule + materialize a v1 directory tree."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .canonical_record import PromotionRecord, evaluate_rule
from .layout import V1Layout
from .linker import LinkMode, link_one
from .pending import DelLatentPending, PendingEntry


@dataclass
class PromoterConfig:
    rule: dict
    link_mode: LinkMode
    img_enc_root: Path
    slat_root: Path
    view_indices: list[int]
    force: bool = False
    promoter_version: str = "1.0.0"


@dataclass
class PromotionSummary:
    promoted: int = 0
    skipped_existing: int = 0
    deferred: int = 0
    failed: int = 0
    fallback_count: int = 0
    notes: list[str] = field(default_factory=list)


def _now_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _pack_pt_to_npz(pt_path: Path, npz_path: Path) -> None:
    import numpy as np
    import torch
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        arrs = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in obj.items()}
    elif hasattr(obj, "cpu"):
        arrs = {"data": obj.cpu().numpy()}
    else:
        arrs = {"data": obj}
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrs)


def _materialize_before(
    *, shard: str, obj_id: str, layout: V1Layout, cfg: PromoterConfig,
    summary: PromotionSummary,
) -> bool:
    img_enc_dir = cfg.img_enc_root / obj_id
    if not img_enc_dir.is_dir():
        summary.notes.append(f"missing img_Enc dir: {img_enc_dir}")
        return False
    for k, idx in enumerate(cfg.view_indices):
        src = img_enc_dir / f"{idx:03d}.png"
        dst = layout.before_view_paths(shard, obj_id)[k]
        if not src.is_file():
            summary.notes.append(f"missing view {src}")
            return False
        res = link_one(src, dst, mode=cfg.link_mode, force=cfg.force)
        if res.fell_back:
            summary.fallback_count += 1
    feats_pt = cfg.slat_root / shard / f"{obj_id}_feats.pt"
    coords_pt = cfg.slat_root / shard / f"{obj_id}_coords.pt"
    ss_pt = cfg.slat_root / shard / f"{obj_id}_ss.pt"
    if not feats_pt.is_file() or not coords_pt.is_file():
        summary.notes.append(f"missing SLAT .pt for {obj_id}")
        return False
    slat_dst = layout.before_slat_npz(shard, obj_id)
    if cfg.force or not slat_dst.is_file():
        import numpy as np
        import torch
        feats = torch.load(feats_pt, map_location="cpu", weights_only=False)
        coords = torch.load(coords_pt, map_location="cpu", weights_only=False)
        feats_arr = feats.cpu().numpy() if hasattr(feats, "cpu") else feats
        coords_arr = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
        slat_dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(slat_dst, feats=feats_arr, coords=coords_arr)
    ss_dst = layout.before_ss_npz(shard, obj_id)
    if cfg.force or not ss_dst.is_file():
        if ss_pt.is_file():
            _pack_pt_to_npz(ss_pt, ss_dst)
        else:
            ss_dst.parent.mkdir(parents=True, exist_ok=True)
            ss_dst.with_suffix(".missing.json").write_text(json.dumps({
                "reason": "ss_pt_not_found", "expected": str(ss_pt), "ts": _now_z(),
            }))
    return True


def _write_meta_json(layout: V1Layout, rec: PromotionRecord, cfg: PromoterConfig) -> None:
    p = layout.meta_json(rec.shard, rec.obj_id)
    if p.is_file() and not cfg.force:
        return
    parsed_p = rec.source_run_dir / "phase1" / "parsed.json"
    parsed = json.loads(parsed_p.read_text()) if parsed_p.is_file() else {}
    raw_caption = ""
    raw_p = rec.source_run_dir / "phase1" / "raw.txt"
    if raw_p.is_file():
        raw_caption = raw_p.read_text(errors="replace").strip()
    meta = {
        "obj_id": rec.obj_id, "shard": rec.shard,
        "source_dataset": "partverse",
        "caption": raw_caption,
        "part_list": [
            {"part_id": int(p["id"]), "name": p.get("name", "")}
            for p in (parsed.get("parts") or [])
        ],
        "promoted_at": _now_z(),
        "promoter_version": cfg.promoter_version,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))


def _resolve_suffix(layout: V1Layout, rec: PromotionRecord) -> str | None:
    base = layout.edit_dir(rec.shard, rec.obj_id, rec.edit_id, suffix="")
    if not base.exists():
        return ""
    qc_p = layout.qc_json(rec.shard, rec.obj_id, rec.edit_id, suffix="")
    if qc_p.is_file():
        try:
            qc = json.loads(qc_p.read_text())
            if qc.get("source", {}).get("run_tag") == rec.source_run_tag:
                return None
        except Exception:
            pass
    n = 2
    while layout.edit_dir(rec.shard, rec.obj_id, rec.edit_id, suffix=f"__r{n}").exists():
        n += 1
    return f"__r{n}"


def _materialize_edit(
    rec: PromotionRecord, suffix: str, *, layout: V1Layout, cfg: PromoterConfig,
    summary: PromotionSummary, pending: DelLatentPending,
) -> None:
    edit_dir = layout.edit_dir(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)
    edit_dir.mkdir(parents=True, exist_ok=True)
    layout.spec_json(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix).write_text(
        json.dumps(rec.spec, indent=2))
    layout.qc_json(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix).write_text(
        json.dumps(rec.to_qc_json(promoted_at=_now_z()), indent=2))
    for k, src in enumerate(rec.preview_pngs):
        dst = layout.after_view_paths(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)[k]
        if src.is_file():
            res = link_one(src, dst, mode=cfg.link_mode, force=cfg.force)
            if res.fell_back:
                summary.fallback_count += 1
    if rec.is_deletion():
        marker = layout.after_pending_marker(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)
        marker.write_text(json.dumps({
            "edit_id": rec.edit_id, "suffix": suffix,
            "after_glb": str(rec.after_glb) if rec.after_glb else None,
            "ts": _now_z(),
        }))
        pending.append(PendingEntry(rec.shard, rec.obj_id, rec.edit_id, suffix))
    else:
        if rec.after_npz and rec.after_npz.is_file():
            dst = layout.after_npz_path(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)
            res = link_one(rec.after_npz, dst, mode=cfg.link_mode, force=cfg.force)
            if res.fell_back:
                summary.fallback_count += 1
        else:
            summary.notes.append(f"missing after.npz for {rec.edit_id} ({rec.source_run_tag})")


def promote_records(
    recs: Iterable[PromotionRecord], *,
    layout: V1Layout, cfg: PromoterConfig, pending: DelLatentPending,
) -> PromotionSummary:
    summary = PromotionSummary()
    seen_objs: set[tuple[str, str]] = set()
    for rec in recs:
        ok, reason = evaluate_rule(rec.passes, cfg.rule)
        if not ok:
            if reason.startswith("missing"):
                summary.deferred += 1
            else:
                summary.failed += 1
            continue
        if (rec.shard, rec.obj_id) not in seen_objs:
            need_before = (cfg.force or
                           not layout.before_slat_npz(rec.shard, rec.obj_id).is_file())
            if need_before:
                if not _materialize_before(shard=rec.shard, obj_id=rec.obj_id,
                                            layout=layout, cfg=cfg, summary=summary):
                    summary.failed += 1
                    continue
            _write_meta_json(layout, rec, cfg)
            seen_objs.add((rec.shard, rec.obj_id))
        suffix = _resolve_suffix(layout, rec)
        if suffix is None:
            summary.skipped_existing += 1
            continue
        _materialize_edit(rec, suffix, layout=layout, cfg=cfg,
                          summary=summary, pending=pending)
        summary.promoted += 1
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/cleaning/v1/test_promoter.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/cleaning/v1/promoter.py tests/cleaning/v1/test_promoter.py
git commit -m "feat(cleaning/v1): add promoter (rule eval + linker + collision suffix)"
```

---

## Task 8: `v1.indexer` — rebuild objects.jsonl + edits.jsonl

**Files:**
- Create: `partcraft/cleaning/v1/indexer.py`
- Test: `tests/cleaning/v1/test_indexer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cleaning/v1/test_indexer.py`:

```python
import json
from pathlib import Path

from partcraft.cleaning.v1.indexer import rebuild_index
from partcraft.cleaning.v1.layout import V1Layout


def _fake_v1(tmp_path: Path) -> V1Layout:
    layout = V1Layout(root=tmp_path / "v1")
    obj = layout.object_dir("05", "objA")
    obj.mkdir(parents=True)
    layout.before_dir("05", "objA").mkdir(parents=True)
    layout.before_ss_npz("05", "objA").write_bytes(b"x")
    layout.before_slat_npz("05", "objA").write_bytes(b"x")
    for p in layout.before_view_paths("05", "objA"):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
    layout.meta_json("05", "objA").write_text(json.dumps({
        "obj_id": "objA", "shard": "05", "part_list": []
    }))
    for eid in ("del_objA_000", "mod_objA_001"):
        ed = layout.edit_dir("05", "objA", eid); ed.mkdir(parents=True)
        layout.spec_json("05", "objA", eid).write_text(json.dumps({
            "edit_id": eid,
            "edit_type": ("deletion" if eid.startswith("del_") else "modification"),
        }))
        layout.qc_json("05", "objA", eid).write_text(json.dumps({
            "edit_id": eid,
            "source": {"pipeline_version": "v2", "run_tag": "rt"},
            "passes": {"gate_text_align": {"pass": True}, "gate_quality": {"pass": True}},
        }))
        for p in layout.after_view_paths("05", "objA", eid):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
        if eid.startswith("mod_"):
            layout.after_npz_path("05", "objA", eid).write_bytes(b"x")
    return layout


def test_rebuild_writes_jsonl(tmp_path: Path):
    layout = _fake_v1(tmp_path)
    summary = rebuild_index(layout)
    assert summary["n_objects"] == 1
    assert summary["n_edits"] == 2
    edits_lines = layout.edits_jsonl().read_text().splitlines()
    assert len(edits_lines) == 2
    parsed = [json.loads(ln) for ln in edits_lines]
    by_id = {p["edit_id"]: p for p in parsed}
    assert by_id["mod_objA_001"]["after_npz"].endswith("after.npz")
    assert by_id["del_objA_000"]["after_npz"] is None
    assert by_id["del_objA_000"]["source_pipeline"] == "v2"


def test_rebuild_records_disambiguation_suffix(tmp_path: Path):
    layout = _fake_v1(tmp_path)
    eid = "del_objA_000"; suffix = "__r2"
    ed = layout.edit_dir("05", "objA", eid, suffix=suffix); ed.mkdir(parents=True)
    layout.spec_json("05", "objA", eid, suffix=suffix).write_text(json.dumps({
        "edit_id": eid, "edit_type": "deletion"
    }))
    layout.qc_json("05", "objA", eid, suffix=suffix).write_text(json.dumps({
        "edit_id": eid, "source": {"pipeline_version": "v3", "run_tag": "rt2"},
        "passes": {"gate_text_align": {"pass": True}, "gate_quality": {"pass": True}},
    }))
    rebuild_index(layout)
    rows = [json.loads(ln) for ln in layout.edits_jsonl().read_text().splitlines()]
    suffixed = [r for r in rows if r["edit_dir_suffix"] == "__r2"]
    assert len(suffixed) == 1
    assert suffixed[0]["source_pipeline"] == "v3"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/cleaning/v1/test_indexer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `partcraft/cleaning/v1/indexer.py`:

```python
"""Rebuild ``index/{objects,edits}.jsonl`` by scanning the v1 tree."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .layout import V1Layout


def _rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root))


def _scan_edit(edit_dir: Path, *, base_edit_id: str, suffix: str,
               obj_id: str, shard: str, layout: V1Layout) -> dict[str, Any]:
    spec_p = edit_dir / "spec.json"
    qc_p = edit_dir / "qc.json"
    spec = json.loads(spec_p.read_text()) if spec_p.is_file() else {}
    qc = json.loads(qc_p.read_text()) if qc_p.is_file() else {}
    after_npz = layout.after_npz_path(shard, obj_id, base_edit_id, suffix=suffix)
    after_views = layout.after_view_paths(shard, obj_id, base_edit_id, suffix=suffix)
    return {
        "obj_id": obj_id, "shard": shard,
        "edit_id": base_edit_id, "edit_dir_suffix": suffix,
        "edit_type": spec.get("edit_type", ""),
        "before_ss":   _rel(layout.before_ss_npz(shard, obj_id), layout.root),
        "before_slat": _rel(layout.before_slat_npz(shard, obj_id), layout.root),
        "before_views": [_rel(p, layout.root) for p in layout.before_view_paths(shard, obj_id)],
        "after_npz":   _rel(after_npz, layout.root) if after_npz.is_file() else None,
        "after_views": [_rel(p, layout.root) for p in after_views],
        "spec":        _rel(spec_p, layout.root) if spec_p.is_file() else None,
        "qc":          _rel(qc_p, layout.root) if qc_p.is_file() else None,
        "source_pipeline": qc.get("source", {}).get("pipeline_version", ""),
        "source_run_tag":  qc.get("source", {}).get("run_tag", ""),
    }


def _split_suffix(dir_name: str, *, edit_ids: set[str]) -> tuple[str, str]:
    if dir_name in edit_ids:
        return dir_name, ""
    if "__r" in dir_name:
        base, _, n = dir_name.rpartition("__r")
        if n.isdigit():
            return base, f"__r{n}"
    return dir_name, ""


def rebuild_index(layout: V1Layout) -> dict[str, int]:
    objects_rows: list[dict[str, Any]] = []
    edits_rows: list[dict[str, Any]] = []
    for obj_dir in layout.iter_object_dirs():
        shard = obj_dir.parent.name
        obj_id = obj_dir.name
        edits_root = obj_dir / "edits"
        if not edits_root.is_dir():
            continue
        edit_dirs = sorted([d for d in edits_root.iterdir() if d.is_dir()])
        canonical_ids: set[str] = set()
        for d in edit_dirs:
            spec_p = d / "spec.json"
            if spec_p.is_file():
                try:
                    canonical_ids.add(json.loads(spec_p.read_text()).get("edit_id", d.name))
                except Exception:
                    canonical_ids.add(d.name)
        type_counts: Counter[str] = Counter()
        for d in edit_dirs:
            base_id, suffix = _split_suffix(d.name, edit_ids=canonical_ids)
            row = _scan_edit(d, base_edit_id=base_id, suffix=suffix,
                             obj_id=obj_id, shard=shard, layout=layout)
            edits_rows.append(row)
            if row["edit_type"]:
                type_counts[row["edit_type"]] += 1
        objects_rows.append({
            "obj_id": obj_id, "shard": shard,
            "n_edits": len(edit_dirs),
            "edit_types": dict(type_counts),
        })
    layout.objects_jsonl().parent.mkdir(parents=True, exist_ok=True)
    layout.objects_jsonl().write_text(
        ("\n".join(json.dumps(r) for r in objects_rows) + "\n") if objects_rows else "")
    layout.edits_jsonl().write_text(
        ("\n".join(json.dumps(r) for r in edits_rows) + "\n") if edits_rows else "")
    summary = {
        "n_objects": len(objects_rows),
        "n_edits": len(edits_rows),
        "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    layout.last_rebuild_json().write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/cleaning/v1/test_indexer.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/cleaning/v1/indexer.py tests/cleaning/v1/test_indexer.py
git commit -m "feat(cleaning/v1): add JSONL indexer (objects + edits, idempotent)"
```

---

## Task 9: CLI — `scripts/cleaning/promote_to_v1.py`

**Files:**
- Create: `scripts/cleaning/promote_to_v1.py`

Thin wrapper. Modules under `partcraft.cleaning.v1` are unit-tested above; this layer is exercised by the smoke test in Task 13.

- [ ] **Step 1: Write the CLI**

Create `scripts/cleaning/promote_to_v1.py`:

```python
#!/usr/bin/env python3
"""Promote cleaned edits from one or more pipeline runs into ``data/partverse_edit_v1/``.

Usage::

    python -m scripts.cleaning.promote_to_v1 \
        --source-runs outputs/partverse/pipeline_v2_shard05 \
                       outputs/partverse/shard08/mode_e_text_align \
        --rules configs/cleaning/promote_v1.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from partcraft.cleaning.v1.layout import V1Layout
from partcraft.cleaning.v1.linker import LinkMode
from partcraft.cleaning.v1.promoter import (
    PromoterConfig, promote_records, PromotionSummary,
)
from partcraft.cleaning.v1.pending import DelLatentPending
from partcraft.cleaning.v1.source_v2 import iter_records_from_v2_run
from partcraft.cleaning.v1.source_v3 import iter_records_from_v3_run

LOG = logging.getLogger("promote_to_v1")


def _detect_pipeline_version(run_root: Path) -> str:
    if (run_root / "objects").is_dir():
        return "v2"
    for child in run_root.iterdir():
        if child.is_dir() and child.name.startswith("mode_") and (child / "objects").is_dir():
            return "v3"
    return "v2"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-runs", nargs="+", type=Path, required=True)
    ap.add_argument("--rules", type=Path,
                    default=Path("configs/cleaning/promote_v1.yaml"))
    ap.add_argument("--v1-root", type=Path, default=None)
    ap.add_argument("--link-mode", choices=[m.value for m in LinkMode], default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-pipeline", choices=["v2", "v3"], default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    rules = yaml.safe_load(args.rules.read_text())
    v1_root = args.v1_root or Path(rules["v1_root"])
    link_mode = LinkMode(args.link_mode or rules.get("link_mode", "hardlink"))
    layout = V1Layout(root=v1_root)
    cfg = PromoterConfig(
        rule=rules["promote_rules"],
        link_mode=link_mode,
        img_enc_root=Path(rules["before_assets"]["img_enc_root"]),
        slat_root=Path(rules["before_assets"]["slat_root"]),
        view_indices=list(rules["before_assets"]["view_indices"]),
        force=args.force,
    )
    pending = DelLatentPending(layout.pending_del_latent_file())

    overall = PromotionSummary()
    for run_root in args.source_runs:
        run_root = run_root.resolve()
        version = args.force_pipeline or _detect_pipeline_version(run_root)
        LOG.info("processing %s as pipeline %s", run_root, version)
        if version == "v2":
            recs = iter_records_from_v2_run(run_root, run_tag=run_root.name)
        else:
            recs = iter_records_from_v3_run(run_root, run_tag=run_root.name)
        s = promote_records(recs, layout=layout, cfg=cfg, pending=pending)
        LOG.info("  promoted=%d skipped=%d deferred=%d failed=%d fallback=%d",
                 s.promoted, s.skipped_existing, s.deferred, s.failed, s.fallback_count)
        for n in s.notes[:20]:
            LOG.info("  note: %s", n)
        overall.promoted += s.promoted
        overall.skipped_existing += s.skipped_existing
        overall.deferred += s.deferred
        overall.failed += s.failed
        overall.fallback_count += s.fallback_count

    LOG.info("TOTAL promoted=%d skipped=%d deferred=%d failed=%d fallback=%d",
             overall.promoted, overall.skipped_existing,
             overall.deferred, overall.failed, overall.fallback_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke**

```bash
python -m scripts.cleaning.promote_to_v1 --help
```

Expected: argparse usage prints; no import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/cleaning/promote_to_v1.py
git commit -m "feat(cleaning/v1): add promote_to_v1 CLI"
```

---

## Task 10: CLI — `scripts/cleaning/rebuild_v1_index.py`

**Files:**
- Create: `scripts/cleaning/rebuild_v1_index.py`

- [ ] **Step 1: Write the CLI**

Create `scripts/cleaning/rebuild_v1_index.py`:

```python
#!/usr/bin/env python3
"""Rebuild ``data/partverse_edit_v1/index/{objects,edits}.jsonl``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from partcraft.cleaning.v1.indexer import rebuild_index
from partcraft.cleaning.v1.layout import V1Layout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", type=Path,
                    default=Path("configs/cleaning/promote_v1.yaml"))
    ap.add_argument("--v1-root", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    if args.v1_root:
        v1_root = args.v1_root
    else:
        v1_root = Path(yaml.safe_load(args.rules.read_text())["v1_root"])
    summary = rebuild_index(V1Layout(root=v1_root))
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke**

```bash
python -m scripts.cleaning.rebuild_v1_index --help
```

Expected: argparse usage prints.

- [ ] **Step 3: Commit**

```bash
git add scripts/cleaning/rebuild_v1_index.py
git commit -m "feat(cleaning/v1): add rebuild_v1_index CLI"
```

---

## Task 11: CLI — `scripts/cleaning/encode_del_latent.py` (wraps migrate_slat_to_npz)

**Files:**
- Create: `scripts/cleaning/encode_del_latent.py`

The deletion encoder reuses `scripts/tools/migrate_slat_to_npz.py`'s `_render_and_full_encode` (defined at line 388) which renders 40 views via Blender, runs DINOv2, and encodes SS+SLAT into a single NPZ.

Multi-GPU is via subprocess fan-out: `--num-gpus N` spawns N children, each with `CUDA_VISIBLE_DEVICES` set, processing a round-robin slice of the pending list.

- [ ] **Step 1: Verify Phase 5 helper signature**

Run:

```bash
python3 - <<'PY'
import inspect
from scripts.tools.migrate_slat_to_npz import _render_and_full_encode
print(inspect.signature(_render_and_full_encode))
PY
```

Note the actual parameter names. The CLI below assumes `(ply_or_glb, out_npz, n_views=...)`. **If the real signature differs, adjust `_process_one`'s call site and re-commit before merging.**

- [ ] **Step 2: Write the CLI**

Create `scripts/cleaning/encode_del_latent.py`:

```python
#!/usr/bin/env python3
"""Encode deletion edits' ``after_new.glb`` into ``after.npz`` (SS + SLAT + DINOv2).

Drives ``scripts/tools/migrate_slat_to_npz._render_and_full_encode`` for
each entry in ``data/partverse_edit_v1/_pending/del_latent.txt``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from partcraft.cleaning.v1.layout import V1Layout
from partcraft.cleaning.v1.pending import DelLatentPending, PendingEntry

LOG = logging.getLogger("encode_del_latent")


def _now_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _process_one(entry: PendingEntry, *, layout: V1Layout, dino_views: int) -> bool:
    marker_p = layout.after_pending_marker(entry.shard, entry.obj_id, entry.edit_id,
                                            suffix=entry.suffix)
    if not marker_p.is_file():
        LOG.warning("no marker for %s; skipping", entry.edit_id)
        return False
    marker = json.loads(marker_p.read_text())
    glb = Path(marker["after_glb"]) if marker.get("after_glb") else None
    if glb is None or not glb.is_file():
        LOG.warning("missing after_new.glb for %s (%s)", entry.edit_id, glb)
        return False
    out_npz = layout.after_npz_path(entry.shard, entry.obj_id, entry.edit_id,
                                     suffix=entry.suffix)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    from scripts.tools.migrate_slat_to_npz import _render_and_full_encode
    _render_and_full_encode(ply_or_glb=glb, out_npz=out_npz, n_views=dino_views)

    if not out_npz.is_file():
        LOG.error("encode produced no output for %s", entry.edit_id)
        return False
    qc_p = layout.qc_json(entry.shard, entry.obj_id, entry.edit_id, suffix=entry.suffix)
    qc = json.loads(qc_p.read_text())
    qc.setdefault("passes", {})["del_latent_encode"] = {
        "pass": True, "score": None,
        "producer": "encode_del_latent.py@1.0.0",
        "reason": "", "ts": _now_z(),
    }
    qc_p.write_text(json.dumps(qc, indent=2))
    marker_p.unlink(missing_ok=True)
    return True


def _run_single(entries: list[PendingEntry], *, layout: V1Layout,
                pending: DelLatentPending, dino_views: int) -> tuple[int, int]:
    ok = fail = 0
    for e in entries:
        try:
            if _process_one(e, layout=layout, dino_views=dino_views):
                pending.remove(e); ok += 1
            else:
                fail += 1
        except Exception as exc:
            LOG.exception("encode failed for %s: %s", e.edit_id, exc); fail += 1
    return ok, fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", type=Path,
                    default=Path("configs/cleaning/promote_v1.yaml"))
    ap.add_argument("--v1-root", type=Path, default=None)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--gpu-shard", type=str, default="",
                    help="internal: e.g. '1/4' to process slice 1 of 4")
    ap.add_argument("--dino-views", type=int, default=40)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    rules = yaml.safe_load(args.rules.read_text())
    v1_root = args.v1_root or Path(rules["v1_root"])
    layout = V1Layout(root=v1_root)
    pending = DelLatentPending(layout.pending_del_latent_file())
    entries = list(pending.iter_entries())
    if not entries:
        LOG.info("nothing pending"); return 0

    if args.gpu_shard:
        i, n = (int(x) for x in args.gpu_shard.split("/"))
        entries = [e for j, e in enumerate(entries) if j % n == i]
        LOG.info("shard %d/%d: %d entries", i, n, len(entries))
        ok, fail = _run_single(entries, layout=layout, pending=pending,
                                dino_views=args.dino_views)
        LOG.info("ok=%d fail=%d", ok, fail)
        return 0

    if args.num_gpus <= 1:
        ok, fail = _run_single(entries, layout=layout, pending=pending,
                                dino_views=args.dino_views)
        LOG.info("ok=%d fail=%d", ok, fail)
        return 0

    children: list[subprocess.Popen] = []
    for i in range(args.num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [sys.executable, "-m", "scripts.cleaning.encode_del_latent",
               "--rules", str(args.rules),
               "--v1-root", str(v1_root),
               "--gpu-shard", f"{i}/{args.num_gpus}",
               "--dino-views", str(args.dino_views)]
        LOG.info("spawning GPU %d: %s", i, " ".join(cmd))
        children.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for ch in children:
        rc = max(rc, ch.wait())
    return rc


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke**

```bash
python -m scripts.cleaning.encode_del_latent --help
```

Expected: argparse usage prints.

- [ ] **Step 4: Commit**

```bash
git add scripts/cleaning/encode_del_latent.py
git commit -m "feat(cleaning/v1): add encode_del_latent CLI wrapping migrate_slat_to_npz Phase 5"
```

---

## Task 12: CLI — `scripts/cleaning/run_gate_quality_on_v2.py`

**Files:**
- Create: `scripts/cleaning/run_gate_quality_on_v2.py`

Adapter that constructs `partcraft.pipeline_v3.paths.ObjectContext` instances pointing at v2 directory layout, then calls `partcraft.pipeline_v3.vlm_core.run_gate_quality(ctxs, ...)`. Verdicts are written back into v2's `edit_status.json` under `gates.E` (see `vlm_core.py:2084` `_step_done(ctx, "sq3_qc_E")`). v2 module code is **not** modified.

- [ ] **Step 1: Inspect ObjectContext + run_gate_quality signatures**

```bash
python3 - <<'PY'
import inspect
from partcraft.pipeline_v3.paths import ObjectContext
from partcraft.pipeline_v3.vlm_core import run_gate_quality
print("ObjectContext fields:", list(ObjectContext.__dataclass_fields__.keys()))
print("run_gate_quality sig:", inspect.signature(run_gate_quality))
PY
```

Confirm `ObjectContext` has at least `dir, shard, obj_id, image_npz`. Confirm `run_gate_quality` accepts `(ctxs, *, vlm_urls, vlm_model, cfg, ...)`. **If field names or kwargs differ, fix the CLI below and re-commit.**

- [ ] **Step 2: Write the CLI**

Create `scripts/cleaning/run_gate_quality_on_v2.py`:

```python
#!/usr/bin/env python3
"""Run pipeline_v3 ``gate_quality`` (Gate E) over a pipeline_v2 run directory.

Constructs ObjectContext instances pointing at v2's
``<run>/objects/<NN>/<obj_id>/`` and invokes
``partcraft.pipeline_v3.vlm_core.run_gate_quality`` directly.
Verdicts are written back to v2's ``edit_status.json`` under
``gates.E`` (currently null on every v2 run).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from partcraft.pipeline_v3.paths import ObjectContext
from partcraft.pipeline_v3.vlm_core import run_gate_quality

LOG = logging.getLogger("run_gate_quality_on_v2")


def _build_ctx(obj_dir: Path, *, image_npz_root: Path) -> ObjectContext:
    shard = obj_dir.parent.name
    obj_id = obj_dir.name
    image_npz = image_npz_root / shard / f"{obj_id}.npz"
    return ObjectContext(
        dir=obj_dir, shard=shard, obj_id=obj_id, image_npz=image_npz,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-run", type=Path, required=True,
                    help="path like outputs/partverse/pipeline_v2_shard05")
    ap.add_argument("--v3-config", type=Path, required=True,
                    help="a v3 config YAML (used for services.vlm)")
    ap.add_argument("--shards", nargs="+", default=None)
    ap.add_argument("--obj-ids", type=Path, default=None,
                    help="optional file with one obj_id per line")
    ap.add_argument("--image-npz-root", type=Path,
                    default=Path("data/partverse/images"),
                    help="dir holding <shard>/<obj_id>.npz")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.v3_config.read_text())

    objects_root = args.v2_run / "objects"
    if not objects_root.is_dir():
        LOG.error("not a v2 run dir: %s", args.v2_run); return 2

    obj_filter: set[str] | None = None
    if args.obj_ids:
        obj_filter = {ln.strip() for ln in args.obj_ids.read_text().splitlines()
                      if ln.strip() and not ln.startswith("#")}

    ctxs: list[ObjectContext] = []
    for shard_dir in sorted(objects_root.iterdir()):
        if not shard_dir.is_dir(): continue
        if args.shards and shard_dir.name not in args.shards: continue
        for obj_dir in sorted(shard_dir.iterdir()):
            if not obj_dir.is_dir(): continue
            if obj_filter and obj_dir.name not in obj_filter: continue
            ctxs.append(_build_ctx(obj_dir, image_npz_root=args.image_npz_root))
    if args.limit > 0:
        ctxs = ctxs[: args.limit]
    LOG.info("running gate_quality on %d objects from %s", len(ctxs), args.v2_run)

    vlm_urls = cfg["services"]["vlm"]["urls"]
    vlm_model = cfg["services"]["vlm"].get("model", "")
    run_gate_quality(ctxs, vlm_urls=vlm_urls, vlm_model=vlm_model,
                     cfg=cfg, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke**

```bash
python -m scripts.cleaning.run_gate_quality_on_v2 --help
```

Expected: argparse usage prints.

- [ ] **Step 4: Commit**

```bash
git add scripts/cleaning/run_gate_quality_on_v2.py
git commit -m "feat(cleaning/v1): add run_gate_quality_on_v2 adapter (v3 gate, v2 layout)"
```

---

## Task 13: End-to-end smoke on real data

**Files:** none new (manual run + brief runbook commit).

- [ ] **Step 1: Pick a small target subset**

```bash
mkdir -p /tmp/v1_smoke
ls outputs/partverse/pipeline_v2_shard05/objects/05 | head -2 > /tmp/v1_smoke/v2_ids.txt
ls outputs/partverse/shard08/mode_e_text_align/objects/08 | head -2 > /tmp/v1_smoke/v3_ids.txt
cat /tmp/v1_smoke/v2_ids.txt /tmp/v1_smoke/v3_ids.txt
```

Expected: two non-empty obj_id lists.

- [ ] **Step 2: Backfill Gate E on v2 subset**

```bash
python -m scripts.cleaning.run_gate_quality_on_v2 \
    --v2-run outputs/partverse/pipeline_v2_shard05 \
    --v3-config configs/pipeline_v3_shard08_test20_gateQ.yaml \
    --shards 05 \
    --obj-ids /tmp/v1_smoke/v2_ids.txt
```

Expected: `gates.E` populated in those objects' `edit_status.json` (verify with a one-line python json.load print).

- [ ] **Step 3: Promote both runs into v1**

```bash
python -m scripts.cleaning.promote_to_v1 \
    --source-runs outputs/partverse/pipeline_v2_shard05 \
                   outputs/partverse/shard08/mode_e_text_align \
    --link-mode hardlink
```

Expected: log shows `promoted > 0`. `data/partverse_edit_v1/objects/{05,08}/...` directories exist.

- [ ] **Step 4: Encode pending del latents**

```bash
python -m scripts.cleaning.encode_del_latent --num-gpus 1
```

Expected: pending list shrinks; `after.npz` appears under each `del_*` dir.

- [ ] **Step 5: Rebuild index**

```bash
python -m scripts.cleaning.rebuild_v1_index
cat data/partverse_edit_v1/index/_last_rebuild.json
head -2 data/partverse_edit_v1/index/edits.jsonl
```

Expected: non-empty JSONL; `n_objects > 0`.

- [ ] **Step 6: Commit a short runbook**

Create `docs/superpowers/runbooks/2026-04-19-edit-data-v1-smoke.md` with a brief reproduce note. Commit:

```bash
git add docs/superpowers/runbooks/2026-04-19-edit-data-v1-smoke.md
git commit -m "docs(cleaning/v1): add smoke runbook"
```

---

## Self-Review Checklist (run after all tasks)

```bash
pytest tests/cleaning/v1/ -v
ls data/partverse_edit_v1/objects/05/*/before/ 2>/dev/null | head
ls data/partverse_edit_v1/objects/05/*/edits/del_*/ 2>/dev/null | head
wc -l data/partverse_edit_v1/index/edits.jsonl
```

### Spec coverage map

| Spec section | Implemented in |
|---|---|
| §3 layout | Task 1 (V1Layout) |
| §4.1 meta.json | Task 7 (`_write_meta_json`) |
| §4.2 spec.json | Tasks 3/4 (`_spec_subset_for_edit`) + Task 7 |
| §4.3 qc.json | Task 2 (`PromotionRecord.to_qc_json`) + Task 7 |
| §4.4–4.5 indexes | Task 8 (indexer) |
| §5 promote rule | Task 2 (`evaluate_rule`) + Task 7 |
| §6.1 promote CLI | Task 9 |
| §6.2 encode del CLI | Task 11 |
| §6.3 rebuild CLI | Task 10 |
| §7 data flow | Task 13 (smoke) |
| §8 extensibility | Task 2 open `passes` dict, Task 7 suffix logic |
| §9 disk footprint | Task 5 link mode default + Task 13 verifies |
| §10.1 split ss/slat | Task 7 (`_materialize_before`) |
| §10.2 Gate E one-off | Task 12 |
| §10.3 cross-version diversity | Task 7 (`_resolve_suffix`) + Task 8 (`_split_suffix`) |
