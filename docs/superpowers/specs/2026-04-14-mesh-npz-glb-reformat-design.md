# Mesh NPZ GLB Reformat Design

**Date:** 2026-04-14  
**Status:** Approved for implementation  
**Scope:** `scripts/datasets/partverse/pack_npz.py`, `partcraft/io/partcraft_loader.py`, `partcraft/render/overview.py`, `scripts/blender_render_parts.py`, `partcraft/trellis/refiner.py`, `partcraft/pipeline_v2/s1_vlm_core.py`, `partcraft/pipeline_v2/s5b_deletion.py`

---

## Problem

The current mesh NPZ format (`mesh/{shard}/{obj_id}.npz`) stores geometry-only PLY files derived from `segmented.glb` — a coarse annotation mesh with no texture information:

```
mesh/{shard}/{uuid}.npz
  ├── full.ply       ← geometry only, xyz+faces, from segmented.glb (~6500 verts)
  └── part_N.ply     ← geometry only, from segmented.glb
```

This creates three compounding problems:

**P1: Color/texture lost at pack time.**  
`segmented.glb` has no UV textures. `_split_mesh()` and `_to_ply()` export bare geometry. The `textured_part_glbs/` source (per-part UV-textured GLBs) and `normalized_glbs/` (full UV-textured GLB) are never read during pack. All texture information is discarded before the pipeline even starts.

**P2: Coarse mesh degrades SLAT edit mask accuracy.**  
`segmented.glb` (~6496 verts) and `normalized_glb` (~45073 verts) share the same bounds but have different tessellation. The SLAT voxelization in `encode_object` (s5) uses the coarse mesh for edit mask computation, producing less accurate part boundaries than the high-quality source.

**P3: del (s5b) requires complex KD-tree matching as a workaround.**  
Because the mesh NPZ has no texture and wrong tessellation, `_build_deletion_glb` must load `normalized_glb_dir` + `anno_dir` at runtime and bridge the two meshes via KD-tree face centroid matching (~70 lines of approximation logic). This requires two extra mandatory config paths (`normalized_glb_dir`, `anno_dir`) and introduces matching noise.

**P4: `textured_part_glbs/` is dead data.**  
`data/partverse/source/textured_part_glbs/{uuid}/{pid}.glb` holds pre-split UV-textured per-part GLBs. Part numbering matches `face2label` IDs exactly (verified: 10 parts ↔ 10 GLBs, IDs 0–9). Coordinate space matches `normalized_glb` (bounds identical). These are never referenced anywhere in the pipeline.

---

## Design Decisions

**D1: Change mesh NPZ to store GLB bytes, keyed as `full.glb` and `part_N.glb`.**  
The NPZ format is transparent to key names and byte contents. GLB bytes are stored as `uint8` arrays, same as PLY currently. Loaders and writers change together as a matched pair.

**D2: Source for `part_N.glb` = `textured_part_glbs/{uuid}/{N}.glb` with `_align_source_to_vd` applied.**  
`textured_part_glbs` are in the same coordinate space as `normalized_glb` (verified). Apply `_align_source_to_vd(scale, offset)` from `transforms.json` to bring vertices into VD space, then re-export as GLB. UV coordinates and materials are preserved through the vertex-only transform.

**D3: Source for `full.glb` = `normalized_glbs/{uuid}.glb` with `_align_source_to_vd` applied.**  
Using the complete high-quality textured mesh for `full.glb`. The comment in `refiner.py` ("mesh.ply is equivalent to img_Enc/mesh.ply") becomes accurate — both now derive from the same normalized_glb source.

**D4: Downstream loaders change `file_type="ply"` → `file_type="glb"` and key names accordingly. No coordinate changes.**  
The VD-space GLB is loaded identically to the VD-space PLY from the consumer's perspective. The `scale_factor` computation in `encode_object` (vd_extent / hy3d_extent) remains valid since both reads come from the same `full.glb`.

**D5: s5b deletion uses part GLBs from mesh NPZ directly — no KD-tree, no extra config.**  
With per-part GLBs already split and labeled by ID, deletion becomes: load NPZ → filter out deleted part IDs → concatenate remaining part meshes → export `after_new.glb`. `normalized_glb_dir` and `anno_dir` are no longer required config fields for s5b.

**D6: Blender rendering (s1 overview, s2 highlight) uses GLB import with palette color override.**  
Blender imports GLB via `bpy.ops.import_scene.gltf()`. In solid-palette mode (s1/s2), imported PBR materials are stripped and replaced with flat palette emission shaders — same visual result as current PLY import. The `--use_vertex_colors` path is unchanged.

**D7: PartObjaverse dataset (`prepare.py`) is out of scope for this change.**  
PartObjaverse has no `textured_part_glbs` equivalent, so its mesh NPZ remains PLY format. `partcraft_loader.py` detects format by probing key names (`full.glb` vs `full.ply`) to maintain backward compatibility.

---

## Architecture

### Data Flow (after)

```
source/textured_part_glbs/{uuid}/{pid}.glb
        ↓ _align_source_to_vd (vertices only, UV preserved)
        ↓ trimesh.export(file_type="glb")
        → part_{pid}.glb  ┐
                          ├─ mesh/{shard}/{uuid}.npz
source/normalized_glbs/{uuid}.glb           │
        ↓ _align_source_to_vd               │
        ↓ trimesh.export(file_type="glb")   │
        → full.glb        ┘
```

### Consumer Impact

| Consumer | Current | After |
|---|---|---|
| `partcraft_loader.get_full_mesh()` | key `full.ply`, `file_type="ply"` | key `full.glb`, `file_type="glb"` |
| `partcraft_loader.get_part_mesh()` | key `part_N.ply`, `file_type="ply"` | key `part_N.glb`, `file_type="glb"` |
| `overview.extract_parts()` | writes `part_N.ply`, discovers `*.ply` | writes `part_N.glb`, discovers `*.glb` |
| `blender_render_parts.py` | `import_mesh.ply`, discovers `*.ply` | `import_scene.gltf`, discovers `*.glb` |
| `refiner.encode_object` | reads `full.ply`, temp `.ply` suffix | reads `full.glb`, temp `.glb` suffix |
| `s1_vlm_core.build_part_menu` | key suffix `.ply` | key suffix `.glb` |
| `s5b._build_deletion_glb` | KD-tree match, ~70 lines | concatenate NPZ parts, ~15 lines |

### s5b Deletion (after)

```python
def _build_deletion_from_npz(ctx, selected_part_ids, pair_dir,
                             *, force=False, logger=None) -> bool:
    import trimesh, io, numpy as np
    after_glb = pair_dir / "after_new.glb"
    if after_glb.is_file() and not force:
        return True
    npz = np.load(ctx.mesh_npz, allow_pickle=True)
    all_pids = sorted(
        int(k.replace("part_", "").replace(".glb", ""))
        for k in npz.files if k.startswith("part_") and k.endswith(".glb")
    )
    keep_pids = [p for p in all_pids if p not in set(selected_part_ids)]
    if not keep_pids:
        return False
    meshes = []
    for pid in keep_pids:
        scene = trimesh.load(io.BytesIO(bytes(npz[f"part_{pid}.glb"])),
                             file_type="glb")
        meshes.append(list(scene.geometry.values())[0])
    after = trimesh.util.concatenate(meshes)
    pair_dir.mkdir(parents=True, exist_ok=True)
    after.export(str(after_glb))
    return True
```

`run_mesh_delete` no longer instantiates `HY3DPartDataset` or requires `normalized_glb_dir` / `anno_dir` for the primary path.

### partcraft_loader Format Detection

```python
def _mesh_fmt(self) -> str:
    """Return 'glb' for PartVerse NPZ, 'ply' for PartObjaverse NPZ."""
    self._ensure_mesh_npz()
    return "glb" if "full.glb" in self._mesh_npz.files else "ply"
```

`get_full_mesh`, `get_part_mesh`, `get_assembled_mesh` branch on `_mesh_fmt()` for key name and `file_type`.

---

## Storage Impact

| | Per-object | 12030 objects |
|---|---|---|
| Current PLY NPZ | ~477 KB | ~5.6 GB |
| GLB NPZ (textured) | ~1500 KB | ~17.6 GB |
| Net increase | 3.2× | +12 GB |

The increase comes from embedded texture images (~1 MB per object). Texture data is already PNG-compressed inside GLB and gains little from NPZ's additional zip pass. Disk space should be confirmed available before running full repack.

---

## Pack Step Changes (`pack_npz.py`)

### New required inputs

| Parameter | Source path |
|---|---|
| `textured_part_glbs_dir` | `source/textured_part_glbs/` |
| `normalized_glb_dir` | `source/normalized_glbs/` |

Both paths already exist in the PartVerse source data. The prerender config files (`prerender_partverse_*.yaml`) add these two keys under `paths`.

### `_pack_one` logic replacement

```
OLD:
  1. load segmented.glb  (_load_source_mesh)
  2. _align_source_to_vd → modifies vertices only
  3. _split_mesh → creates geometry-only sub-meshes (drops TextureVisuals)
  4. _to_ply → export bytes, store as part_N.ply / full.ply

NEW:
  1. for pid in 0..N-1:
       load textured_part_glbs/{uuid}/{pid}.glb (single mesh, TextureVisuals)
       _align_source_to_vd → modifies vertices only, UV coords unchanged
       export to GLB bytes → store as part_{pid}.glb
  2. load normalized_glbs/{uuid}.glb
     _align_source_to_vd
     export to GLB bytes → store as full.glb
```

`_load_source_mesh`, `_split_mesh`, and `_to_ply` are no longer called for PartVerse objects. `face2label.json` and `segmented.glb` are still read for `split_mesh.json` metadata written into the images NPZ (part labels and cluster sizes), but not for geometry.

---

## Error Handling

- Missing `textured_part_glbs/{uuid}/{pid}.glb` for any pid: log warning, skip object.
- Missing `normalized_glbs/{uuid}.glb`: log warning, skip object.
- Part count mismatch (GLB file count vs `face2label` part count): log warning, skip object.
- If `ctx.mesh_npz` contains `full.ply` keys (old format) and s5b is called: fall back to existing `_build_deletion_glb` KD-tree path with a deprecation warning. Remove fallback after all shards are repacked.

---

## Backward Compatibility

- Existing PLY NPZs (PartObjaverse) continue to work via `_mesh_fmt()` probe.
- PartVerse PLY NPZs (old pack): still loadable; s5b falls back to KD-tree path until repack completes.
- Config keys `normalized_glb_dir` and `anno_dir` remain in the pipeline YAML but become optional for s5b once all shards are repacked.
- `--force` flag on `pack_npz.py` overwrites existing NPZs for repack.

---

## Files Changed

| File | Change | Net lines Δ |
|---|---|---|
| `scripts/datasets/partverse/pack_npz.py` | replace `_pack_one` geometry section | +40 / −20 |
| `partcraft/io/partcraft_loader.py` | `_mesh_fmt()` probe + key/file_type updates | +20 / −8 |
| `partcraft/render/overview.py` | `extract_parts`: `.ply` → `.glb` | +3 / −3 |
| `scripts/blender_render_parts.py` | file discovery + GLB import | +8 / −5 |
| `partcraft/trellis/refiner.py` | `full.ply` → `full.glb`, temp suffix | +4 / −4 |
| `partcraft/pipeline_v2/s1_vlm_core.py` | key suffix in `build_part_menu` | +1 / −1 |
| `partcraft/pipeline_v2/s5b_deletion.py` | replace `_build_deletion_glb` with NPZ concat | +20 / −70 |
| `configs/prerender_partverse_*.yaml` | add `textured_part_glbs_dir` | +1 each |

**Net: ~−40 lines** (s5b simplification dominates).

---

## Testing

1. **Unit — round-trip pack/load:** Pack one object's GLB NPZ, load all parts via `get_part_mesh`, verify vertex count matches source GLBs and `TextureVisuals` is intact.
2. **Unit — s5b del from NPZ:** Given a GLB NPZ, run `_build_deletion_from_npz` with one selected part ID, verify `after_new.glb` contains geometry from only the non-deleted parts.
3. **Integration — s1 overview render:** Run `render_overview_png` on a GLB NPZ object, verify Blender produces a valid PNG with N colored parts (palette mode).
4. **Integration — encode_object:** Run SLAT voxelization on one GLB NPZ object, verify mask shape is `(64, 64, 64)` and part mask is non-empty.
5. **Regression — PartObjaverse PLY NPZ:** Confirm existing PLY NPZ loads correctly via `_mesh_fmt()` probe, no behavior change.
