# GLB Deletion Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task.

**Goal:** Replace PLY-based deletion/addition preview rendering with high-quality normalized GLB rendering (UV textures, Cycles GPU, white background compositing).

**Architecture:** (1) `paths.py` gains `normalized_glb_dir`/`anno_dir` wired through YAML. (2) `s5b_deletion.py` gains `_build_deletion_glb()` — KD-tree face-centroid matching segmented→normalized GLB, outputs `after_new.glb` + `before_new.glb` symlink. (3) `s6_preview.py` gains `_render_glb_views()` — calls `encode_asset/blender_script/render.py`, alpha-composites RGBA→white; deletion/addition branches prefer GLB, fall back to PLY.

**Tech Stack:** Python 3.11, trimesh, scipy.cKDTree, Blender 3.5.1, PIL/Pillow, encode_asset/blender_script/render.py

---

## File Map

| Action | File |
|--------|------|
| Modify | `partcraft/pipeline_v2/paths.py` |
| Modify | `configs/pipeline_v2_shard*.yaml` |
| Modify | `partcraft/pipeline_v2/s5b_deletion.py` |
| Modify | `partcraft/pipeline_v2/s6_preview.py` |

---

## Task 1: DatasetRoots — add normalized_glb_dir and anno_dir

**Files:** Modify `partcraft/pipeline_v2/paths.py`

- [ ] **Step 1: Update `DatasetRoots` dataclass**

In `paths.py` around line 43, replace the `DatasetRoots` class with:

```python
@dataclass(frozen=True)
class DatasetRoots:
    images_root: Path
    mesh_root: Path
    normalized_glb_dir: Path | None = None
    anno_dir: Path | None = None

    @classmethod
    def from_pipeline_cfg(cls, cfg: dict) -> "DatasetRoots":
        data = cfg.get("data") or {}
        raw_glb  = data.get("normalized_glb_dir")
        raw_anno = data.get("anno_dir")
        return cls(
            images_root=Path(data.get("images_root", DEFAULT_IMAGES_ROOT)),
            mesh_root=Path(data.get("mesh_root", DEFAULT_MESH_ROOT)),
            normalized_glb_dir=Path(raw_glb)  if raw_glb  else None,
            anno_dir=Path(raw_anno) if raw_anno else None,
        )

    def input_npz_paths(self, shard: str | int, obj_id: str) -> tuple[Path, Path]:
        s = normalize_shard(shard)
        return (
            self.mesh_root / s / f"{obj_id}.npz",
            self.images_root / s / f"{obj_id}.npz",
        )

    def normalized_glb_path(self, obj_id: str) -> "Path | None":
        if self.normalized_glb_dir is None:
            return None
        return self.normalized_glb_dir / f"{obj_id}.glb"

    def anno_object_dir(self, obj_id: str) -> "Path | None":
        if self.anno_dir is None:
            return None
        return self.anno_dir / obj_id
```

- [ ] **Step 2: Add to all shard YAML configs**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
for f in configs/pipeline_v2_shard*.yaml; do
  python3 -c "
import yaml
path = '$f'
cfg = yaml.safe_load(open(path))
cfg.setdefault('data', {})['normalized_glb_dir'] = '/mnt/zsn/data/partverse/source/normalized_glbs'
cfg.setdefault('data', {})['anno_dir'] = '/mnt/zsn/data/partverse/source/anno_infos/anno_infos'
yaml.dump(cfg, open(path,'w'), allow_unicode=True, sort_keys=False)
print('Updated', path)
"
done
```

- [ ] **Step 3: Verify**

```bash
python3 -c "
import yaml
from partcraft.pipeline_v2.paths import DatasetRoots
cfg = yaml.safe_load(open('configs/pipeline_v2_shard02.yaml'))
roots = DatasetRoots.from_pipeline_cfg(cfg)
print('normalized_glb_dir:', roots.normalized_glb_dir)
p = roots.normalized_glb_path('20642f7304da440cb17fb2c77c73eda5')
print('glb exists:', p and p.is_file())
"
```
Expected: `normalized_glb_dir: /mnt/zsn/data/partverse/source/normalized_glbs` and `glb exists: True`

- [ ] **Step 4: Commit**

```bash
git add partcraft/pipeline_v2/paths.py configs/pipeline_v2_shard*.yaml
git commit -m "feat(paths): add normalized_glb_dir and anno_dir to DatasetRoots"
```

---

## Task 2: s5b_deletion.py — add `_build_deletion_glb()`

**Files:** Modify `partcraft/pipeline_v2/s5b_deletion.py`

- [ ] **Step 1: Add `_build_deletion_glb()` function (before `run_mesh_delete_for_object`, line ~85)**

```python
def _build_deletion_glb(
    obj_id: str,
    selected_part_ids: list[int],
    pair_dir: Path,
    normalized_glb_dir: Path,
    anno_dir: Path,
    *,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> bool:
    """Build after_new.glb by masking selected parts from the normalized GLB.

    Uses KD-tree face-centroid matching: segmented.glb (annotated, coarse) →
    normalized GLB (high-quality UV). Returns True on success, False on error.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    log = logger or logging.getLogger("pipeline_v2.s5b")

    norm_path    = normalized_glb_dir / f"{obj_id}.glb"
    anno_obj_dir = anno_dir / obj_id
    seg_path     = anno_obj_dir / f"{obj_id}_segmented.glb"
    f2l_path     = anno_obj_dir / f"{obj_id}_face2label.json"

    for p in (norm_path, seg_path, f2l_path):
        if not p.is_file():
            log.warning("[s5b] _build_deletion_glb: missing %s", p)
            return False

    after_glb  = pair_dir / "after_new.glb"
    before_glb = pair_dir / "before_new.glb"

    if after_glb.is_file() and not force:
        return True

    try:
        norm_scene = trimesh.load(str(norm_path), force='scene')
        norm_mesh  = list(norm_scene.geometry.values())[0]
        seg_mesh   = trimesh.load(str(seg_path), force='mesh')
        f2l        = {int(k): int(v)
                      for k, v in json.load(open(f2l_path)).items()}

        seg_centroids  = seg_mesh.vertices[seg_mesh.faces].mean(axis=1)
        norm_centroids = norm_mesh.vertices[norm_mesh.faces].mean(axis=1)
        _, nn_idxs     = cKDTree(seg_centroids).query(norm_centroids, k=1)

        face_labels = np.array([f2l.get(int(i), -1) for i in nn_idxs])
        mask_keep   = ~np.isin(face_labels, selected_part_ids)

        masked = trimesh.Trimesh(
            vertices=norm_mesh.vertices,
            faces=norm_mesh.faces[mask_keep],
            visual=norm_mesh.visual,
            process=False,
        )
        masked.remove_unreferenced_vertices()
        masked.export(str(after_glb))

        if not before_glb.exists():
            try:
                before_glb.symlink_to(norm_path)
            except OSError:
                pass

        log.info("[s5b] GLB del=%d keep=%d → %s",
                 (~mask_keep).sum(), mask_keep.sum(), after_glb.name)
        return True
    except Exception as exc:
        log.warning("[s5b] _build_deletion_glb %s: %s", obj_id, exc)
        return False
```

- [ ] **Step 2: Update `run_mesh_delete_for_object` signature and PLY-skip path**

Add two new kwargs:

```python
def run_mesh_delete_for_object(
    ctx: ObjectContext,
    *,
    dataset,
    normalized_glb_dir: Path | None = None,
    anno_dir: Path | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> DelMeshResult:
```

In the skip branch (PLY already exists), add GLB build:

```python
        if a_ply.is_file() and not force:
            if normalized_glb_dir and anno_dir:
                _build_deletion_glb(
                    ctx.obj_id, list(spec.selected_part_ids),
                    pair_dir, normalized_glb_dir, anno_dir,
                    force=False, logger=log,
                )
            _backfill_add(ctx, spec, add_seq, force=False, logger=log)
            res.n_skip += 1; add_seq += 1
            continue
```

In the success branch (after `TrellisRefiner.direct_delete_mesh`):

```python
            if normalized_glb_dir and anno_dir:
                _build_deletion_glb(
                    ctx.obj_id, list(spec.selected_part_ids),
                    pair_dir, normalized_glb_dir, anno_dir,
                    force=force, logger=log,
                )
```

- [ ] **Step 3: Update `run_mesh_delete()` to pass through kwargs**

```python
def run_mesh_delete(
    ctxs: Iterable[ObjectContext],
    *,
    dataset,
    normalized_glb_dir: Path | None = None,
    anno_dir: Path | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[DelMeshResult]:
    ...  # existing loop body, pass new kwargs to run_mesh_delete_for_object
```

- [ ] **Step 4: Thread through `run.py` call site**

Find where `run_mesh_delete` is called in `partcraft/pipeline_v2/run.py`. Add:

```python
results = run_mesh_delete(
    ctxs, dataset=dataset,
    normalized_glb_dir=roots.normalized_glb_dir,
    anno_dir=roots.anno_dir,
    force=force, logger=logger,
)
```

- [ ] **Step 5: Smoke test**

```bash
python3 -c "
import yaml, logging
from pathlib import Path
from partcraft.pipeline_v2.paths import DatasetRoots
from partcraft.pipeline_v2.s5b_deletion import _build_deletion_glb

cfg   = yaml.safe_load(open('configs/pipeline_v2_shard02.yaml'))
roots = DatasetRoots.from_pipeline_cfg(cfg)
pair_dir = Path(cfg['data']['output_dir']) / 'objects/02/2064e86f4b984c2a8c04d24187a168c0/edits_3d/del_2064e86f4b984c2a8c04d24187a168c0_003'
ok = _build_deletion_glb(
    '2064e86f4b984c2a8c04d24187a168c0', [5],
    pair_dir, roots.normalized_glb_dir, roots.anno_dir,
    force=True, logger=logging.getLogger(),
)
print('success:', ok, '| after_new.glb exists:', (pair_dir/'after_new.glb').is_file())
"
```
Expected: `success: True | after_new.glb exists: True`

- [ ] **Step 6: Commit**

```bash
git add partcraft/pipeline_v2/s5b_deletion.py partcraft/pipeline_v2/run.py
git commit -m "feat(s5b): build after_new.glb via KD-tree GLB masking alongside PLY"
```

---

## Task 3: s6_preview.py — GLB rendering for deletion / addition

**Files:** Modify `partcraft/pipeline_v2/s6_preview.py`

- [ ] **Step 1: Ensure `import json` is present** (add to stdlib imports if missing)

- [ ] **Step 2: Add `_encode_asset_script()` helper (near top of file, after imports)**

```python
def _encode_asset_script() -> str:
    """Return path to encode_asset/blender_script/render.py."""
    p = _ROOT / "third_party" / "encode_asset" / "blender_script" / "render.py"
    if not p.is_file():
        raise FileNotFoundError(f"encode_asset render script not found: {p}")
    return str(p)
```

- [ ] **Step 3: Add `_extract_yaw_pitch_views()` helper (after `_render_ply_views`, before `_render_slat_views`)**

```python
def _extract_yaw_pitch_views(image_npz: Path) -> list[dict]:
    """Extract yaw/pitch/radius/fov for VIEW_INDICES from image NPZ transforms.json."""
    import math
    npz    = np.load(str(image_npz), allow_pickle=True)
    frames = json.loads(bytes(npz["transforms.json"]))["frames"]
    views  = []
    for vi in VIEW_INDICES:
        if vi >= len(frames):
            continue
        frame = frames[vi]
        m     = frame["transform_matrix"]
        c     = [m[0][3], m[1][3], m[2][3]]
        r     = math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
        views.append({
            "yaw":    math.atan2(c[0], c[1]),
            "pitch":  math.asin(max(-1.0, min(1.0, c[2] / r))),
            "radius": r,
            "fov":    frame.get("camera_angle_x", math.radians(40)),
        })
    return views
```

- [ ] **Step 4: Add `_render_glb_views()` helper (after `_extract_yaw_pitch_views`)**

```python
def _render_glb_views(
    glb_path: Path,
    image_npz: Path,
    encode_script: str,
    blender: str,
    resolution: int,
) -> list[np.ndarray]:
    """Render GLB at VIEW_INDICES cameras using encode_asset Cycles renderer.

    Returns list of BGR numpy arrays (cv2 convention) ready for _save_previews().
    Alpha-composites RGBA output onto white background (matches overview.py).
    """
    import subprocess, cv2 as _cv2
    from PIL import Image as _PIL

    views = _extract_yaw_pitch_views(image_npz)
    if len(views) != len(VIEW_INDICES):
        raise RuntimeError(
            f"Expected {len(VIEW_INDICES)} camera views, got {len(views)}"
        )

    with tempfile.TemporaryDirectory(prefix="pcv2_s6p_glb_") as tmp:
        tmp_path = Path(tmp)
        result = subprocess.run(
            [blender, "-b", "-P", encode_script, "--",
             "--object",        str(glb_path),
             "--output_folder", str(tmp_path),
             "--views",         json.dumps(views),
             "--resolution",    str(resolution)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"encode_asset Blender failed:\n{result.stderr[-400:]}"
            )

        imgs = []
        for i in range(len(VIEW_INDICES)):
            png = tmp_path / f"{i:03d}.png"
            if not png.is_file():
                raise RuntimeError(f"Missing render output: {png}")
            arr = np.array(_PIL.open(str(png)).convert("RGBA"), dtype=np.float32) / 255.0
            r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            rgb = np.stack([
                (r * a + (1 - a)) * 255,
                (g * a + (1 - a)) * 255,
                (b * a + (1 - a)) * 255,
            ], axis=-1).clip(0, 255).astype(np.uint8)
            imgs.append(_cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR))
        return imgs
```

- [ ] **Step 5: Modify deletion branch in `run_preview_for_object()`** (around line 186)

Replace the `try` block that calls `_render_ply_views` for deletion:

```python
        try:
            after_glb = edit_dir / "after_new.glb"
            if after_glb.is_file():
                imgs = _render_glb_views(
                    after_glb, ctx.image_npz, _encode_asset_script(),
                    blender, resolution,
                )
            else:
                log.debug("[s6p] del %s: no after_new.glb, PLY fallback", spec.edit_id)
                imgs = _render_ply_views(a_ply, frames, blender, resolution, samples=32)
            _save_previews(edit_dir, imgs)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] del %s: %s", spec.edit_id, e)
            res.n_fail += 1
```

- [ ] **Step 6: Modify addition branch** (around line 213)

Replace the `try` block that calls `_render_ply_views` for addition:

```python
        try:
            before_glb = ctx.edit_3d_dir(source_del_id) / "before_new.glb"
            if before_glb.is_file():
                imgs = _render_glb_views(
                    before_glb, ctx.image_npz, _encode_asset_script(),
                    blender, resolution,
                )
            else:
                log.debug("[s6p] add %s: no before_new.glb, PLY fallback", add_id)
                imgs = _render_ply_views(before_ply, frames, blender, resolution, samples=32)
            _save_previews(add_dir, imgs)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] add %s: %s", add_id, e)
            res.n_fail += 1
```

- [ ] **Step 7: Smoke test**

```bash
python3 -c "
import yaml, cv2
from pathlib import Path
from partcraft.pipeline_v2.paths import DatasetRoots
from partcraft.pipeline_v2.s6_preview import _render_glb_views, _encode_asset_script

cfg   = yaml.safe_load(open('configs/pipeline_v2_shard02.yaml'))
roots = DatasetRoots.from_pipeline_cfg(cfg)
_, image_npz = roots.input_npz_paths('02', '2064e86f4b984c2a8c04d24187a168c0')
glb = Path(cfg['data']['output_dir']) / 'objects/02/2064e86f4b984c2a8c04d24187a168c0/edits_3d/del_2064e86f4b984c2a8c04d24187a168c0_003/after_new.glb'

imgs = _render_glb_views(glb, image_npz, _encode_asset_script(), cfg['blender'], 518)
print('views:', len(imgs), 'shape:', imgs[0].shape)
for i, img in enumerate(imgs):
    cv2.imwrite(f'/tmp/smoke_{i}.png', img)
print('white bg pixel[0,0]:', imgs[0][0,0])  # should be ~[255,255,255]
"
```
Expected: 5 views, shape `(518, 518, 3)`, corner pixel close to `[255, 255, 255]`.

- [ ] **Step 8: Commit**

```bash
git add partcraft/pipeline_v2/s6_preview.py
git commit -m "feat(s6p): render deletion/addition previews from GLB via encode_asset with white-bg composite"
```

---

## Task 4: End-to-end integration test

- [ ] **Step 1: Run full s5b + s6p on test shard (force)**

```bash
python -m partcraft.pipeline_v2.run \
  --config configs/pipeline_v2_shard02.yaml \
  --shard 02 --all --phase D
python -m partcraft.pipeline_v2.run \
  --config configs/pipeline_v2_shard02.yaml \
  --shard 02 --all --phase E
```

- [ ] **Step 2: Verify GLB used (not PLY fallback)**

```bash
grep "\[s6p\].*PLY fallback" <log_file> | wc -l  # expect 0
```

- [ ] **Step 3: Spot-check one file size (GLB preview should be ~200KB, PLY ~140KB)**

```bash
ls -lh outputs/partverse/pipeline_v2_shard02*/objects/02/*/edits_3d/del_*/preview_0.png | awk '{print $5, $9}' | head -10
```

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "chore: integrate GLB preview pipeline end-to-end on shard02_test"
```
