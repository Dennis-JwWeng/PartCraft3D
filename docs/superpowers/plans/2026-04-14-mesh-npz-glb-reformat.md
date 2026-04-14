# Mesh NPZ GLB Reformat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace geometry-only PLY files in mesh NPZ with UV-textured GLB files from `textured_part_glbs`, and simplify `s5b` deletion to use per-part GLBs directly instead of KD-tree matching.

**Architecture:** The mesh NPZ stores raw bytes keyed as `full.glb` / `part_N.glb` (PartVerse) or `full.ply` / `part_N.ply` (PartObjaverse, unchanged). All loaders probe `full.glb` presence to detect format. The s5b deletion path concatenates remaining part GLBs directly from NPZ, eliminating `normalized_glb_dir`/`anno_dir` as hard requirements.

**Tech Stack:** trimesh, numpy, open3d, Blender Python API (bpy), pytest

**Spec:** `docs/superpowers/specs/2026-04-14-mesh-npz-glb-reformat-design.md`

---

## Task 0: Verify trimesh GLB export preserves TextureVisuals

This is the critical assumption of the entire spec. Verify before writing any production code.

- [ ] **Step 1: Run verification**

```bash
/mnt/zsn/3dobject/envs/trellis2/bin/python - << 'PYEOF'
import trimesh, io, numpy as np
OBJ = "0008dc75fb3648f2af4ca8c4d711e53e"
part_path = f"/mnt/zsn/data/partverse/source/textured_part_glbs/{OBJ}/0.glb"
norm_path  = f"/mnt/zsn/data/partverse/source/normalized_glbs/{OBJ}.glb"
transforms = {"scale": 0.5005004940503051, "offset": [0.0, 0.0, 0.0]}

def align_vd(mesh, t):
    sv = np.array(mesh.vertices)
    b = np.empty_like(sv)
    b[:,0]=sv[:,0]; b[:,1]=-sv[:,2]; b[:,2]=sv[:,1]
    mesh.vertices = (b + np.array(t["offset"])) * t["scale"]
    return mesh

scene = trimesh.load(part_path, force="scene")
m = list(scene.geometry.values())[0]
uv_before = m.visual.uv.copy()
m = align_vd(m, transforms)
buf = io.BytesIO(); m.export(buf, file_type="glb"); buf.seek(0)
scene2 = trimesh.load(buf, file_type="glb", force="scene")
m2 = list(scene2.geometry.values())[0]
assert m2.visual.uv is not None, "UV lost after round-trip!"
assert m2.visual.uv.shape == uv_before.shape
print(f"PASS part GLB: v={m2.vertices.shape[0]}, UV={m2.visual.uv.shape}")

scene_n = trimesh.load(norm_path, force="scene")
mn = list(scene_n.geometry.values())[0]
mn = align_vd(mn, transforms)
buf_n = io.BytesIO(); mn.export(buf_n, file_type="glb"); buf_n.seek(0)
scene_n2 = trimesh.load(buf_n, file_type="glb", force="scene")
mn2 = list(scene_n2.geometry.values())[0]
assert mn2.visual.uv is not None
print(f"PASS full GLB: v={mn2.vertices.shape[0]}, UV={mn2.visual.uv.shape}")
print("ALL CHECKS PASSED")
PYEOF
```

Expected:
```
PASS part GLB: v=5105, UV=(5105, 2)
PASS full GLB: v=45073, UV=(45073, 2)
ALL CHECKS PASSED
```

Stop and investigate if any assertion fails.

---

## Task 1: Write all failing tests

**Files:**
- Create: `tests/test_mesh_npz_glb.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_mesh_npz_glb.py
"""Tests for mesh NPZ GLB format (PartVerse).
Uses real source data; skipped when data not present (CI).
"""
import io, json, os, tempfile
from pathlib import Path
import numpy as np
import pytest

SOURCE_ROOT  = Path("/mnt/zsn/data/partverse/source")
MESH_ROOT    = Path("/mnt/zsn/data/partverse/inputs/mesh")
IMG_ROOT     = Path("/mnt/zsn/data/partverse/inputs/images")
TEST_OBJ     = "0008dc75fb3648f2af4ca8c4d711e53e"
TEST_SHARD   = "00"
TPGLB_DIR    = SOURCE_ROOT / "textured_part_glbs"
NORM_GLB_DIR = SOURCE_ROOT / "normalized_glbs"
TRANSFORMS   = {"scale": 0.5005004940503051, "offset": [0.0, 0.0, 0.0]}

needs_source = pytest.mark.skipif(
    not TPGLB_DIR.is_dir(), reason="PartVerse source data not available")


def _align_vd(mesh):
    import numpy as np
    sv = np.array(mesh.vertices)
    b = np.empty_like(sv)
    b[:,0]=sv[:,0]; b[:,1]=-sv[:,2]; b[:,2]=sv[:,1]
    mesh.vertices = (b + np.array(TRANSFORMS["offset"])) * TRANSFORMS["scale"]
    return mesh


def _build_glb_npz(obj_id: str, out_path: Path) -> int:
    """Pack one object into GLB NPZ. Returns number of parts."""
    import trimesh
    tpglb = TPGLB_DIR / obj_id
    part_files = sorted(tpglb.glob("*.glb"))
    mesh_data: dict = {}
    for pf in part_files:
        pid = int(pf.stem)
        scene = trimesh.load(str(pf), force="scene")
        m = list(scene.geometry.values())[0]
        m = _align_vd(m)
        buf = io.BytesIO(); m.export(buf, file_type="glb")
        mesh_data[f"part_{pid}.glb"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    norm = trimesh.load(str(NORM_GLB_DIR / f"{obj_id}.glb"), force="scene")
    mn = list(norm.geometry.values())[0]
    mn = _align_vd(mn)
    buf_n = io.BytesIO(); mn.export(buf_n, file_type="glb")
    mesh_data["full.glb"] = np.frombuffer(buf_n.getvalue(), dtype=np.uint8)
    np.savez_compressed(str(out_path), **mesh_data)
    return len(part_files)


@needs_source
def test_glb_pack_roundtrip_uv():
    import trimesh
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / f"{TEST_OBJ}.npz"
        n = _build_glb_npz(TEST_OBJ, npz_path)
        d = np.load(str(npz_path), allow_pickle=True)
        assert "full.glb" in d.files
        scene = trimesh.load(io.BytesIO(bytes(d["part_0.glb"])), file_type="glb", force="scene")
        m = list(scene.geometry.values())[0]
        assert m.visual.uv is not None, "UV lost after round-trip"
        assert m.vertices.shape[0] > 1000
        scene_f = trimesh.load(io.BytesIO(bytes(d["full.glb"])), file_type="glb", force="scene")
        mf = list(scene_f.geometry.values())[0]
        assert mf.vertices.shape[0] > 40000, f"full.glb: {mf.vertices.shape[0]} verts"


@needs_source
def test_loader_detects_ply_format():
    import sys; sys.path.insert(0, str(Path(__file__).parents[1]))
    from partcraft.io.partcraft_loader import PartCraftDataset
    old_npz = MESH_ROOT / TEST_SHARD / f"{TEST_OBJ}.npz"
    if not old_npz.exists():
        pytest.skip("PLY NPZ not available")
    ds = PartCraftDataset(str(IMG_ROOT), str(MESH_ROOT), [TEST_SHARD])
    rec = ds.load_object(TEST_SHARD, TEST_OBJ)
    assert rec._mesh_fmt() == "ply"
    rec.close()


@needs_source
def test_loader_detects_glb_format():
    import sys; sys.path.insert(0, str(Path(__file__).parents[1]))
    from partcraft.io.partcraft_loader import ObjectRecord
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / f"{TEST_OBJ}.npz"
        _build_glb_npz(TEST_OBJ, npz_path)
        img_npz = IMG_ROOT / TEST_SHARD / f"{TEST_OBJ}.npz"
        rec = ObjectRecord(obj_id=TEST_OBJ, shard=TEST_SHARD,
                           render_npz_path=str(img_npz) if img_npz.exists() else str(npz_path),
                           mesh_npz_path=str(npz_path))
        assert rec._mesh_fmt() == "glb"
        rec.close()


@needs_source
def test_get_part_mesh_from_glb():
    import sys; sys.path.insert(0, str(Path(__file__).parents[1]))
    import trimesh
    from partcraft.io.partcraft_loader import ObjectRecord
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / f"{TEST_OBJ}.npz"
        _build_glb_npz(TEST_OBJ, npz_path)
        img_npz = IMG_ROOT / TEST_SHARD / f"{TEST_OBJ}.npz"
        rec = ObjectRecord(obj_id=TEST_OBJ, shard=TEST_SHARD,
                           render_npz_path=str(img_npz) if img_npz.exists() else str(npz_path),
                           mesh_npz_path=str(npz_path))
        m = rec.get_part_mesh(0, colored=False)
        assert isinstance(m, trimesh.Trimesh)
        assert m.vertices.shape[0] > 1000
        rec.close()


@needs_source
def test_build_deletion_from_npz():
    import sys; sys.path.insert(0, str(Path(__file__).parents[1]))
    import trimesh
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz_path = tmp_path / f"{TEST_OBJ}.npz"
        n = _build_glb_npz(TEST_OBJ, npz_path)
        pair_dir = tmp_path / "del_edit"
        from partcraft.pipeline_v2.s5b_deletion import _build_deletion_from_npz
        ok = _build_deletion_from_npz(mesh_npz=npz_path, selected_part_ids=[0],
                                      pair_dir=pair_dir)
        assert ok
        after = pair_dir / "after_new.glb"
        assert after.is_file()
        scene = trimesh.load(str(after), force="scene")
        total_v = sum(m.vertices.shape[0] for m in scene.geometry.values())
        assert total_v > 0


@needs_source
def test_extract_parts_glb():
    import sys; sys.path.insert(0, str(Path(__file__).parents[1]))
    from partcraft.render.overview import extract_parts
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz_path = tmp_path / f"{TEST_OBJ}.npz"
        n = _build_glb_npz(TEST_OBJ, npz_path)
        out_dir = tmp_path / "parts"; out_dir.mkdir()
        pids = extract_parts(npz_path, out_dir)
        assert len(pids) == n
        for pid in pids:
            assert (out_dir / f"part_{pid}.glb").is_file()


@needs_source
def test_build_part_menu_glb():
    import sys; sys.path.insert(0, str(Path(__file__).parents[1]))
    from partcraft.pipeline_v2.s1_vlm_core import build_part_menu
    img_npz = IMG_ROOT / TEST_SHARD / f"{TEST_OBJ}.npz"
    if not img_npz.exists():
        pytest.skip("images NPZ not available")
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / f"{TEST_OBJ}.npz"
        n = _build_glb_npz(TEST_OBJ, npz_path)
        pids, menu = build_part_menu(npz_path, img_npz)
        assert len(pids) == n
        assert "part_0" in menu
```

- [ ] **Step 2: Run to confirm all fail**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python -m pytest tests/test_mesh_npz_glb.py -v --tb=line 2>&1 | tail -20
```

Expected: All tests FAIL (functions not yet implemented).

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_mesh_npz_glb.py
git commit -m "test: failing tests for mesh NPZ GLB reformat"
```

---

## Task 2: Update `partcraft_loader.py`

**Files:**
- Modify: `partcraft/io/partcraft_loader.py`

- [ ] **Step 1: Add `_mesh_fmt()` after `_ensure_mesh_npz` (~line 104)**

```python
    def _mesh_fmt(self) -> str:
        """Return 'glb' for PartVerse NPZ, 'ply' for PartObjaverse NPZ."""
        self._ensure_mesh_npz()
        return "glb" if "full.glb" in self._mesh_npz.files else "ply"
```

- [ ] **Step 2: Update `_ensure_mask_renderer` (~line 190)**

Find `key = f"part_{pid}.ply"` and the trimesh.load call below it. Replace those ~4 lines with:

```python
            fmt = self._mesh_fmt()
            key = f"part_{pid}.{fmt}"
            if key not in self._mesh_npz:
                continue
            raw = trimesh.load(io.BytesIO(self._mesh_npz[key].tobytes()), file_type=fmt)
            part_mesh = (trimesh.util.concatenate(list(raw.geometry.values()))
                         if isinstance(raw, trimesh.Scene) else raw)
```

- [ ] **Step 3: Update `get_full_mesh` (~line 348)**

```python
    def get_full_mesh(self, colored: bool = True) -> "trimesh.Trimesh":
        assert trimesh is not None, "trimesh is required"
        self._ensure_mesh_npz()
        fmt = self._mesh_fmt()
        raw = trimesh.load(
            io.BytesIO(self._mesh_npz[f"full.{fmt}"].tobytes()), file_type=fmt)
        mesh = (trimesh.util.concatenate(list(raw.geometry.values()))
                if isinstance(raw, trimesh.Scene) else raw)
        if colored:
            self.bake_vertex_colors(mesh)
        return mesh
```

- [ ] **Step 4: Update `get_part_mesh` (~line 357)**

```python
    def get_part_mesh(self, part_id: int,
                      colored: bool = True) -> "trimesh.Trimesh":
        assert trimesh is not None, "trimesh is required"
        self._ensure_mesh_npz()
        fmt = self._mesh_fmt()
        key = f"part_{part_id}.{fmt}"
        if key not in self._mesh_npz:
            raise KeyError(f"'{key}' not found in {self.mesh_npz_path}")
        raw = trimesh.load(io.BytesIO(self._mesh_npz[key].tobytes()), file_type=fmt)
        mesh = (trimesh.util.concatenate(list(raw.geometry.values()))
                if isinstance(raw, trimesh.Scene) else raw)
        if colored:
            self.bake_vertex_colors(mesh)
        return mesh
```

- [ ] **Step 5: Run loader tests**

```bash
python -m pytest tests/test_mesh_npz_glb.py::test_loader_detects_ply_format \
                 tests/test_mesh_npz_glb.py::test_loader_detects_glb_format \
                 tests/test_mesh_npz_glb.py::test_get_part_mesh_from_glb -v --tb=short
```

Expected: All 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add partcraft/io/partcraft_loader.py
git commit -m "feat: partcraft_loader supports GLB mesh NPZ format"
```

---

## Task 3: Update `pack_npz.py`

**Files:**
- Modify: `scripts/datasets/partverse/pack_npz.py`
- Modify: `scripts/datasets/partverse/prerender.py`

- [ ] **Step 1: Add globals and `_pack_mesh_glb` to `pack_npz.py`**

After existing globals (`_ANNO_DIR` etc.), add:

```python
_TEXTURED_PART_GLBS_DIR = _PARTVERSE_DIR / "source" / "textured_part_glbs"
_NORMALIZED_GLB_DIR     = _PARTVERSE_DIR / "source" / "normalized_glbs"
```

After `_load_source_mesh`, add new function:

```python
def _pack_mesh_glb(obj_id, textured_part_glbs_dir, normalized_glb_dir,
                   transforms) -> "dict | None":
    """Pack per-part GLBs into mesh_data. Returns None if source missing."""
    try:
        import trimesh as _tm
    except ImportError:
        return None
    part_dir = textured_part_glbs_dir / obj_id
    norm_glb  = normalized_glb_dir / f"{obj_id}.glb"
    if not part_dir.is_dir() or not norm_glb.is_file():
        return None
    part_files = sorted(part_dir.glob("*.glb"), key=lambda p: int(p.stem))
    if not part_files:
        return None
    mesh_data = {}
    for pf in part_files:
        pid = int(pf.stem)
        scene = _tm.load(str(pf), force="scene")
        m = _align_source_to_vd(list(scene.geometry.values())[0], transforms)
        buf = io.BytesIO(); m.export(buf, file_type="glb")
        mesh_data[f"part_{pid}.glb"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    scene_n = _tm.load(str(norm_glb), force="scene")
    mn = _align_source_to_vd(list(scene_n.geometry.values())[0], transforms)
    buf_n = io.BytesIO(); mn.export(buf_n, file_type="glb")
    mesh_data["full.glb"] = np.frombuffer(buf_n.getvalue(), dtype=np.uint8)
    return mesh_data
```

- [ ] **Step 2: Update `_pack_one` signature**

Add two optional params at the end:

```python
def _pack_one(obj_id, img_enc_dir, render_out, mesh_out, captions,
              keep_views=None, anno_dir=None,
              textured_part_glbs_dir=None, normalized_glb_dir=None):
```

- [ ] **Step 3: Replace mesh_data building in `_pack_one`**

Find the block that creates `mesh_data = {"full.ply": ...}`. Replace it and all related PLY code with:

```python
    # ---- Mesh data ----
    use_glb = textured_part_glbs_dir is not None and normalized_glb_dir is not None
    mesh_data = {}
    if use_glb:
        mesh_data = _pack_mesh_glb(obj_id, textured_part_glbs_dir,
                                   normalized_glb_dir, transforms) or {}

    if not mesh_data:
        # PLY fallback (PartObjaverse or missing textured_part_glbs)
        if instance_gt is None or source_mesh is None:
            return {"status": "skip", "reason": "no geometry source for PLY path"}
        parts, split_mesh_json = _split_mesh(source_mesh, instance_gt, labels)
        mesh_data = {"full.ply": np.frombuffer(_to_ply(source_mesh), dtype=np.uint8)}
        for pid, label, sub in parts:
            mesh_data[f"part_{pid}.ply"] = np.frombuffer(_to_ply(sub), dtype=np.uint8)
```

- [ ] **Step 4: Update `_pack_worker` in `prerender.py` to pass new params**

In `prerender.py`, update `_pack_ctx` and `_pack_worker` to pass through `textured_part_glbs_dir` and `normalized_glb_dir`, and update `_run_pack` signature to accept and pass them along.

- [ ] **Step 5: Run pack test**

```bash
python -m pytest tests/test_mesh_npz_glb.py::test_glb_pack_roundtrip_uv -v --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/datasets/partverse/pack_npz.py scripts/datasets/partverse/prerender.py
git commit -m "feat: pack_npz GLB path from textured_part_glbs + normalized_glb"
```

---

## Task 4: Update `overview.py` and `blender_render_parts.py`

**Files:**
- Modify: `partcraft/render/overview.py`
- Modify: `scripts/blender_render_parts.py`

- [ ] **Step 1: Replace `extract_parts` in `overview.py`**

```python
def extract_parts(npz_path: Path, out_dir: Path) -> list[int]:
    """Extract part meshes from mesh NPZ. Supports GLB (PartVerse) and PLY (PartObjaverse)."""
    z = np.load(npz_path, allow_pickle=True)
    ext = "glb" if any(k.endswith(".glb") for k in z.files) else "ply"
    pids = []
    for k in z.files:
        if k.startswith("part_") and k.endswith(f".{ext}"):
            pid = int(k.replace("part_", "").replace(f".{ext}", ""))
            (out_dir / f"part_{pid}.{ext}").write_bytes(bytes(z[k]))
            pids.append(pid)
    pids.sort()
    return pids
```

- [ ] **Step 2: Update `blender_render_parts.py` — file discovery and pid parsing**

Change file discovery (~line 202):
```python
    parts = sorted(
        f for f in os.listdir(args.parts_dir)
        if f.startswith("part_") and (f.endswith(".ply") or f.endswith(".glb"))
    )
```

Change pid parsing (~line 207):
```python
        pid = int(fname.replace("part_", "").replace(".ply", "").replace(".glb", ""))
```

- [ ] **Step 3: Update `blender_render_parts.py` — import dispatch**

Replace `new_objs = import_ply(path)` with:
```python
        if fname.endswith(".glb"):
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=path)
            new_objs = [o for o in bpy.data.objects if o not in before]
        else:
            new_objs = import_ply(path)
```

- [ ] **Step 4: Run extract_parts test**

```bash
python -m pytest tests/test_mesh_npz_glb.py::test_extract_parts_glb -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add partcraft/render/overview.py scripts/blender_render_parts.py
git commit -m "feat: overview and blender support GLB part files"
```

---

## Task 5: Update `refiner.py` and `s1_vlm_core.py`

**Files:**
- Modify: `partcraft/trellis/refiner.py`
- Modify: `partcraft/pipeline_v2/s1_vlm_core.py`

- [ ] **Step 1: Update `encode_object` full mesh loading in `refiner.py` (~line 426)**

Replace:
```python
            npz = np.load(obj_record.mesh_npz_path, allow_pickle=False)
            if "full.ply" not in npz:
                raise FileNotFoundError(...)
            _tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
            _tmp.write(npz["full.ply"].tobytes())
```

With:
```python
            npz = np.load(obj_record.mesh_npz_path, allow_pickle=True)
            fmt_key = "full.glb" if "full.glb" in npz.files else "full.ply"
            if fmt_key not in npz.files:
                raise FileNotFoundError(
                    f"VD mesh not found at {vd_mesh_path} and "
                    f"neither full.glb nor full.ply found in "
                    f"{obj_record.mesh_npz_path}.")
            tmp_sfx = ".glb" if fmt_key == "full.glb" else ".ply"
            _tmp = tempfile.NamedTemporaryFile(suffix=tmp_sfx, delete=False)
            _tmp.write(npz[fmt_key].tobytes())
```

- [ ] **Step 2: Update `build_part_menu` in `s1_vlm_core.py` (~line 349)**

```python
    pids = sorted(
        int(k.replace("part_", "").replace(".glb", "").replace(".ply", ""))
        for k in z2.files
        if k.startswith("part_") and (k.endswith(".glb") or k.endswith(".ply"))
    )
```

- [ ] **Step 3: Run build_part_menu test**

```bash
python -m pytest tests/test_mesh_npz_glb.py::test_build_part_menu_glb -v --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add partcraft/trellis/refiner.py partcraft/pipeline_v2/s1_vlm_core.py
git commit -m "feat: refiner and s1_vlm_core support GLB mesh NPZ"
```

---

## Task 6: Update `s5b_deletion.py`

**Files:**
- Modify: `partcraft/pipeline_v2/s5b_deletion.py`

- [ ] **Step 1: Add `_build_deletion_from_npz` after `_build_deletion_glb`**

```python
def _build_deletion_from_npz(
    mesh_npz: "Path",
    selected_part_ids: list[int],
    pair_dir: "Path",
    *,
    force: bool = False,
    logger: "logging.Logger | None" = None,
) -> bool:
    """Concatenate non-deleted part GLBs from mesh NPZ into after_new.glb.

    Returns False for PLY-format NPZ (caller falls back to _build_deletion_glb).
    """
    import io as _io
    import trimesh as _tm
    import numpy as _np

    log = logger or logging.getLogger("pipeline_v2.s5b")
    after_glb  = pair_dir / "after_new.glb"
    before_glb = pair_dir / "before_new.glb"

    if after_glb.is_file() and not force:
        return True

    try:
        npz = _np.load(str(mesh_npz), allow_pickle=True)
    except Exception as exc:
        log.warning("[s5b] cannot load mesh NPZ %s: %s", mesh_npz, exc)
        return False

    if "full.glb" not in npz.files:
        return False  # PLY NPZ — caller uses KD-tree fallback

    all_pids = sorted(
        int(k.replace("part_", "").replace(".glb", ""))
        for k in npz.files if k.startswith("part_") and k.endswith(".glb")
    )
    keep_pids = [p for p in all_pids if p not in set(selected_part_ids)]
    if not keep_pids:
        log.warning("[s5b] NPZ del: all parts removed for %s", mesh_npz)
        return False

    try:
        meshes = []
        for pid in keep_pids:
            scene = _tm.load(_io.BytesIO(bytes(npz[f"part_{pid}.glb"])),
                             file_type="glb")
            meshes.append(list(scene.geometry.values())[0])
        after = _tm.util.concatenate(meshes)
        pair_dir.mkdir(parents=True, exist_ok=True)
        after.export(str(after_glb))

        if not before_glb.exists():
            scene_f = _tm.load(_io.BytesIO(bytes(npz["full.glb"])), file_type="glb")
            mf = list(scene_f.geometry.values())[0]
            mf.export(str(before_glb))

        log.info("[s5b] NPZ del: keep=%d remove=%d -> %s",
                 len(keep_pids), len(all_pids) - len(keep_pids), after_glb.name)
        return True
    except Exception as exc:
        log.warning("[s5b] _build_deletion_from_npz: %s", exc)
        return False
```

- [ ] **Step 2: Add NPZ path as primary in `run_mesh_delete_for_object`**

Inside the `use_glb` branch, before the `_build_deletion_glb` call, add:

```python
            # Primary: use per-part GLBs from mesh NPZ (no KD-tree needed)
            ok = False
            if ctx.mesh_npz is not None:
                ok = _build_deletion_from_npz(
                    ctx.mesh_npz, list(spec.selected_part_ids),
                    pair_dir, force=force, logger=log,
                )
            # Fallback: KD-tree path (PLY NPZ or missing mesh_npz)
            if not ok:
                ok = _build_deletion_glb(
                    ctx.obj_id, list(spec.selected_part_ids),
                    pair_dir, normalized_glb_dir, anno_dir,
                    force=force, logger=log,
                )
```

- [ ] **Step 3: Run deletion test**

```bash
python -m pytest tests/test_mesh_npz_glb.py::test_build_deletion_from_npz -v --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add partcraft/pipeline_v2/s5b_deletion.py
git commit -m "feat: s5b deletion uses NPZ part GLBs directly, KD-tree as fallback"
```

---

## Task 7: Update prerender configs

**Files:**
- Modify: `configs/prerender_partverse_H200.yaml`
- Modify: `configs/prerender_partverse_node39.yaml`
- Modify: `configs/prerender_partverse_wm1A800.yaml`

- [ ] **Step 1: Add paths to each config under `paths:`**

```yaml
  textured_part_glbs_dir: "source/textured_part_glbs"
  normalized_glb_dir:     "source/normalized_glbs"
```

- [ ] **Step 2: Commit**

```bash
git add configs/prerender_partverse_*.yaml
git commit -m "config: add textured_part_glbs_dir for GLB pack path"
```

---

## Task 8: Full regression — repack one object and verify

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/test_mesh_npz_glb.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 2: Repack one object**

```bash
/mnt/zsn/3dobject/envs/trellis2/bin/python \
    scripts/datasets/partverse/pack_npz.py \
    --obj-ids 0008dc75fb3648f2af4ca8c4d711e53e \
    --shard 00 --force
```

- [ ] **Step 3: Verify NPZ format**

```bash
python3 -c "
import numpy as np
d = np.load('/mnt/zsn/data/partverse/inputs/mesh/00/0008dc75fb3648f2af4ca8c4d711e53e.npz', allow_pickle=True)
print('Keys:', sorted(d.files))
print('full.glb KB:', len(d['full.glb'])//1024)
print('part_0.glb KB:', len(d['part_0.glb'])//1024)
"
```

Expected: Keys include `full.glb`, `part_0.glb` ... `part_9.glb`.

- [ ] **Step 4: Verify s5b del produces after_new.glb from NPZ**

```bash
/mnt/zsn/3dobject/envs/trellis2/bin/python -c "
import tempfile
from pathlib import Path
from partcraft.pipeline_v2.s5b_deletion import _build_deletion_from_npz
npz = Path('/mnt/zsn/data/partverse/inputs/mesh/00/0008dc75fb3648f2af4ca8c4d711e53e.npz')
with tempfile.TemporaryDirectory() as tmp:
    pair_dir = Path(tmp)
    ok = _build_deletion_from_npz(npz, [0], pair_dir)
    after = pair_dir / 'after_new.glb'
    print('ok:', ok, '  size:', after.stat().st_size if after.exists() else 'MISSING')
"
```

Expected: `ok: True  size: >100000`

- [ ] **Step 5: Verify build_part_menu reads correct part IDs**

```bash
/mnt/zsn/3dobject/envs/trellis2/bin/python -c "
from pathlib import Path
from partcraft.pipeline_v2.s1_vlm_core import build_part_menu
mesh = Path('/mnt/zsn/data/partverse/inputs/mesh/00/0008dc75fb3648f2af4ca8c4d711e53e.npz')
imgs  = Path('/mnt/zsn/data/partverse/inputs/images/00/0008dc75fb3648f2af4ca8c4d711e53e.npz')
pids, menu = build_part_menu(mesh, imgs)
print('pids:', pids)
"
```

Expected: `pids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: verify GLB reformat end-to-end on test object"
```
