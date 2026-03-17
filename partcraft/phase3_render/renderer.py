"""Phase 3: Render edited meshes.

Key optimization: before-images are extracted directly from the original NPZ (zero cost).
Only after-meshes need to be rendered via Blender.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from partcraft.io.hy3d_loader import HY3DPartDataset, ObjectRecord
from partcraft.phase2_assembly.assembler import AssembledPair

try:
    import trimesh
except ImportError:
    trimesh = None


def select_diverse_views(frames: list[dict], n: int = 12) -> list[dict]:
    """Select n diverse views from the available camera frames.

    Tries to spread across azimuth and elevation for good coverage.
    """
    if len(frames) <= n:
        for i, f in enumerate(frames):
            f["original_index"] = i
        return frames

    # Sort by (elevation_bucket, azimuth) for diversity
    indexed = []
    for i, f in enumerate(frames):
        azi = f.get("azi", 0.0)
        elev = f.get("elev", 0.0)
        indexed.append((azi, elev, i, f))

    # Greedy farthest-point sampling in (azi, elev) space
    selected = [indexed[0]]
    remaining = list(indexed[1:])

    while len(selected) < n and remaining:
        best_idx = 0
        best_min_dist = -1
        for ri, (azi_r, elev_r, _, _) in enumerate(remaining):
            min_dist = float("inf")
            for azi_s, elev_s, _, _ in selected:
                d = (azi_r - azi_s) ** 2 + (elev_r - elev_s) ** 2
                min_dist = min(min_dist, d)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = ri
        selected.append(remaining.pop(best_idx))

    result = []
    for azi, elev, orig_idx, frame in selected:
        frame["original_index"] = orig_idx
        result.append(frame)
    return result


def extract_before_images(obj: ObjectRecord, view_indices: list[int],
                          output_dir: Path):
    """Extract before-images from existing NPZ data (zero render cost)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, vid in enumerate(view_indices):
        # Image
        img_bytes = obj.get_image_bytes(vid)
        with open(output_dir / f"{i:03d}.webp", "wb") as f:
            f.write(img_bytes)
        # Mask
        mask = obj.get_mask(vid)
        np.save(str(output_dir / f"{i:03d}_mask.npy"), mask)


def render_mesh_blender(mesh_path: str, output_dir: str, views: list[dict],
                        resolution: int = 518, blender_path: str = "blender") -> bool:
    """Render a mesh using Blender with specified camera parameters.

    Returns True if rendering succeeded.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write a minimal Blender Python script (compatible with Blender 4.x)
    script = f'''
import bpy
import json
import math
import mathutils
import os

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh (Blender 4.x API)
filepath = "{mesh_path}"
if filepath.endswith('.glb') or filepath.endswith('.gltf'):
    bpy.ops.import_scene.gltf(filepath=filepath)
elif filepath.endswith('.ply'):
    bpy.ops.wm.ply_import(filepath=filepath)
elif filepath.endswith('.obj'):
    bpy.ops.wm.obj_import(filepath=filepath)

# Setup vertex color material for all mesh objects
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    mesh = obj.data
    if not mesh.color_attributes:
        continue
    mat = bpy.data.materials.new(name="VertexColorMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    vcol_node = nodes.new(type='ShaderNodeVertexColor')
    vcol_node.layer_name = mesh.color_attributes[0].name
    links.new(vcol_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# Setup camera
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

# Setup rendering (use EEVEE for speed)
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.render.resolution_x = {resolution}
bpy.context.scene.render.resolution_y = {resolution}
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Three-point lighting for good coverage
for name, loc, energy in [("Key", (3, 2, 4), 3.0),
                            ("Fill", (-2, 3, 2), 1.5),
                            ("Rim", (0, -3, 3), 2.0)]:
    ld = bpy.data.lights.new(name, type='SUN')
    ld.energy = energy
    lo = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = loc
    direction = mathutils.Vector((0,0,0)) - mathutils.Vector(loc)
    lo.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

views = json.loads("""{json.dumps(views)}""")

for i, view in enumerate(views):
    azi = view.get("azi", 0)
    elev = view.get("elev", 0)
    cam_dis = view.get("cam_dis", 2.0)
    fov = view.get("fov", 0.7)

    # Spherical to cartesian
    x = cam_dis * math.cos(elev) * math.cos(azi)
    y = cam_dis * math.cos(elev) * math.sin(azi)
    z = cam_dis * math.sin(elev)
    cam.location = (x, y, z)

    # Point at origin
    direction = mathutils.Vector((0,0,0)) - mathutils.Vector(cam.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    cam_data.angle = fov

    bpy.context.scene.render.filepath = os.path.join("{output_dir}", f"{{i:03d}}.png")
    bpy.ops.render.render(write_still=True)
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [blender_path, "-b", "-P", script_path],
            capture_output=True, timeout=300,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(script_path)


def run_phase3(cfg: dict, assembled_pairs: list[AssembledPair],
               dataset: HY3DPartDataset | None = None):
    """Run Phase 3: render all edit pairs.

    For each pair:
      - Extract before-images from original NPZ (free)
      - Render after-mesh with Blender (or skip if cached)
    """
    cache_dir = Path(cfg["phase3"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    render_out = Path(cfg["data"]["output_dir"]) / "renders"
    render_out.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = HY3DPartDataset(
            cfg["data"]["image_npz_dir"],
            cfg["data"]["mesh_npz_dir"],
            cfg["data"]["shards"],
        )

    num_views = cfg["phase3"]["num_views"]
    resolution = cfg["phase3"]["resolution"]
    blender_path = cfg["phase3"]["blender_path"]

    print(f"Phase 3: Rendering {len(assembled_pairs)} pairs ({num_views} views each)")

    rendered = 0
    skipped = 0

    for pair in tqdm(assembled_pairs, desc="Phase 3: Render"):
        spec = pair.edit_spec
        eid = spec.edit_id

        before_dir = render_out / f"{eid}_before"
        after_dir = render_out / f"{eid}_after"

        # --- Before images: extract from NPZ (zero cost) ---
        if not before_dir.exists():
            obj = dataset.load_object(spec.shard, spec.obj_id)
            transforms = obj.get_transforms()
            views = select_diverse_views(transforms["frames"], n=num_views)
            view_indices = [v["original_index"] for v in views]
            extract_before_images(obj, view_indices, before_dir)

            # Save selected camera params for after-mesh rendering
            with open(cache_dir / f"{eid}_views.json", "w") as f:
                json.dump(views, f)
            obj.close()
        else:
            skipped += 1

        # --- After images: render via Blender ---
        if not after_dir.exists() and pair.after_mesh is not None:
            # Export after mesh to temp file
            mesh_path = str(cache_dir / f"{eid}_after.ply")
            pair.after_mesh.export(mesh_path)

            views_path = cache_dir / f"{eid}_views.json"
            if views_path.exists():
                with open(views_path) as f:
                    views = json.load(f)
            else:
                views = [{"azi": a * 3.14159 / 180, "elev": 0.3, "cam_dis": 2.0, "fov": 0.7}
                         for a in range(0, 360, 360 // num_views)]

            ok = render_mesh_blender(mesh_path, str(after_dir), views,
                                     resolution=resolution, blender_path=blender_path)
            if ok:
                rendered += 1
            # Cleanup temp mesh
            if os.path.exists(mesh_path):
                os.unlink(mesh_path)

    print(f"Phase 3 complete: {rendered} rendered, {skipped} cached")
