#!/usr/bin/env python3
"""Download PartObjaverse-Tiny + original textured Objaverse GLBs
and convert to HY3D-Part format for the PartCraft3D pipeline.

Usage (in vinedresser3d environment):
    PYOPENGL_PLATFORM=egl python scripts/prepare_partobjaverse.py [--workers 4] [--limit 10]

Output structure:
    data/partobjaverse/
    ├── images/00/{obj_id}.npz    # 42 views + masks + transforms + split_mesh
    ├── mesh/00/{obj_id}.npz      # full.ply + part_*.ply
    └── metadata.json             # category info, label mappings
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
from partcraft.utils.config import load_config

# ---------------------------------------------------------------------------
# Lazy imports (so --help works without GPU)
# ---------------------------------------------------------------------------
trimesh = None
pyrender = None
Image = None
objaverse = None


def _lazy_imports(no_render: bool = False):
    global trimesh, pyrender, Image, objaverse
    import trimesh as _trimesh
    trimesh = _trimesh

    if not no_render:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender as _pyrender
        pyrender = _pyrender

        from PIL import Image as _Image
        Image = _Image

    try:
        import objaverse as _objaverse
        objaverse = _objaverse
    except ImportError:
        objaverse = None  # not needed if using --local-source


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_REPO = "yhyang-myron/PartObjaverse-Tiny"
NUM_VIEWS = 42
RENDER_RES = 518  # match HY3D-Part
SHARD = "00"
BLENDER_PATH = "blender"
BLENDER_SCRIPT = str(_PROJECT_ROOT / "scripts" / "blender_render.py")


# ---------------------------------------------------------------------------
# Camera generation  (6 elevations × 7 azimuths = 42 views)
# ---------------------------------------------------------------------------
def generate_camera_poses(
    num_views: int = 42,
    radius: float = 2.5,
) -> list[np.ndarray]:
    """Generate orbital camera-to-world matrices (OpenGL convention)."""
    n_azim = 7
    n_elev = num_views // n_azim  # 6
    elevations = np.linspace(-20, 80, n_elev)  # degrees
    azimuths = np.linspace(0, 360, n_azim, endpoint=False)

    poses = []
    for elev in elevations:
        for azim in azimuths:
            e = math.radians(elev)
            a = math.radians(azim)
            # Camera position on sphere
            cx = radius * math.cos(e) * math.sin(a)
            cy = radius * math.sin(e)
            cz = radius * math.cos(e) * math.cos(a)
            cam_pos = np.array([cx, cy, cz])
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)  # camera -Z
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                up = np.array([0.0, 0.0, 1.0])
                right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            up /= np.linalg.norm(up)
            # c2w: columns are right, up, -forward (OpenGL)
            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward
            c2w[:3, 3] = cam_pos
            poses.append(c2w)
    return poses


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _concat_scene(scene):
    """Concatenate a Scene into a single Trimesh."""
    if hasattr(scene, 'to_geometry'):
        return scene.to_geometry()
    return scene.dump(concatenate=True)


def _fix_texture(geom):
    """Fix problematic textures and materials before pyrender.

    Handles:
    - 2-channel/grayscale textures → convert to RGBA
    - No texture but has baseColorFactor → convert to flat vertex colors
      (pyrender sometimes fails to render PBR materials without a texture)
    """
    if not hasattr(geom, 'visual') or not hasattr(geom.visual, 'material'):
        return
    mat = geom.visual.material

    # Fix texture mode issues
    has_base_texture = False
    for attr in ('baseColorTexture', 'emissiveTexture', 'normalTexture',
                 'metallicRoughnessTexture', 'occlusionTexture'):
        tex = getattr(mat, attr, None)
        if tex is not None and hasattr(tex, 'mode'):
            if tex.mode not in ('RGB', 'RGBA'):
                try:
                    setattr(mat, attr, tex.convert('RGBA'))
                except Exception:
                    setattr(mat, attr, None)
            if attr == 'baseColorTexture':
                has_base_texture = True

    # If no base texture but has baseColorFactor, convert to vertex colors
    # so pyrender renders the flat color correctly
    if not has_base_texture:
        bcf = getattr(mat, 'baseColorFactor', None)
        if bcf is not None:
            color = np.array(bcf, dtype=np.uint8)
            if len(color) == 3:
                color = np.append(color, 255)
            face_colors = np.tile(color, (len(geom.faces), 1))
            geom.visual = trimesh.visual.ColorVisuals(
                mesh=geom, face_colors=face_colors
            )


def _normalize_scene(mesh_or_scene, center=True) -> tuple:
    """Normalize mesh to fit in [-1, 1]^3, return (scale, centroid)."""
    if isinstance(mesh_or_scene, trimesh.Scene):
        mesh = _concat_scene(mesh_or_scene)
    else:
        mesh = mesh_or_scene
    centroid = mesh.bounding_box.centroid
    extent = mesh.bounding_box.extents.max()
    scale = 2.0 / extent if extent > 0 else 1.0
    return scale, centroid


def _flatten_scene_to_vertex_colored_mesh(
    scene_tm,
    scale: float,
    centroid: np.ndarray,
) -> trimesh.Trimesh:
    """Flatten a trimesh Scene into a single vertex-colored Trimesh.

    Bakes per-submesh textures / baseColorFactor into vertex colors,
    then concatenates all sub-meshes into one.  This eliminates Z-fighting
    caused by overlapping sub-meshes with different materials.
    """
    all_verts, all_faces, all_vc = [], [], []
    offset = 0

    geom_items = []
    if isinstance(scene_tm, trimesh.Scene):
        for node_name in scene_tm.graph.nodes_geometry:
            transform, geom_name = scene_tm.graph[node_name]
            geom = scene_tm.geometry[geom_name].copy()
            geom.apply_transform(transform)
            geom_items.append(geom)
    else:
        geom_items.append(scene_tm.copy())

    for geom in geom_items:
        geom.vertices = (geom.vertices - centroid) * scale
        n_verts = len(geom.vertices)

        # Extract vertex colors from whatever visual the geometry has
        vc = np.full((n_verts, 4), 180, dtype=np.uint8)  # default gray
        vc[:, 3] = 255
        try:
            if geom.visual.kind == "texture":
                mat = geom.visual.material
                tex = getattr(mat, "baseColorTexture", None)
                if tex is not None and hasattr(geom.visual, "uv") and geom.visual.uv is not None:
                    # Sample texture at UV coordinates
                    uv = np.array(geom.visual.uv)
                    tex_arr = np.array(tex.convert("RGB"))
                    h, w = tex_arr.shape[:2]
                    u = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
                    v = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
                    vc[:, :3] = tex_arr[v, u]
                else:
                    bcf = getattr(mat, "baseColorFactor", None)
                    if bcf is not None:
                        c = np.array(bcf[:3], dtype=np.uint8)
                        vc[:, :3] = c
            elif geom.visual.kind == "vertex":
                raw = np.array(geom.visual.vertex_colors)
                if len(raw) == n_verts:
                    vc[:, :raw.shape[1]] = raw[:, :4]
            elif geom.visual.kind == "face":
                fc = np.array(geom.visual.face_colors)[:, :3]
                # Average face colors to vertices
                v_colors = np.zeros((n_verts, 3), dtype=np.float64)
                v_count = np.zeros(n_verts, dtype=np.float64)
                for fi, face in enumerate(geom.faces):
                    v_colors[face] += fc[fi]
                    v_count[face] += 1
                mask = v_count > 0
                v_colors[mask] /= v_count[mask, None]
                vc[mask, :3] = np.clip(v_colors[mask], 0, 255).astype(np.uint8)
        except Exception:
            pass  # keep default gray

        all_verts.append(geom.vertices)
        all_faces.append(geom.faces + offset)
        all_vc.append(vc)
        offset += n_verts

    merged = trimesh.Trimesh(
        vertices=np.concatenate(all_verts),
        faces=np.concatenate(all_faces),
        process=False,
    )
    merged.visual = trimesh.visual.ColorVisuals(
        mesh=merged,
        vertex_colors=np.concatenate(all_vc),
    )
    return merged


def render_views_from_scene(
    scene_tm,
    scale: float,
    centroid: np.ndarray,
    poses: list[np.ndarray],
    fov: float,
    resolution: int = RENDER_RES,
    glb_path: str | None = None,
    ssaa: int = 2,
) -> list[np.ndarray]:
    """Render RGBA views using Blender Cycles (same as Vinedresser3D).

    Falls back to pyrender if Blender is unavailable.
    """
    blender_ready = (
        os.path.isabs(BLENDER_PATH) and os.path.exists(BLENDER_PATH)
    ) or (not os.path.isabs(BLENDER_PATH) and shutil.which(BLENDER_PATH))
    if glb_path and blender_ready:
        return _render_blender(glb_path, poses, fov, resolution)
    return _render_pyrender(scene_tm, scale, centroid, poses, fov, resolution, ssaa)


def _render_blender(
    glb_path: str,
    poses: list[np.ndarray],
    fov: float,
    resolution: int,
) -> list[np.ndarray]:
    """Render via Blender Cycles subprocess — faithful PBR with Vinedresser lighting."""
    import subprocess

    render_dir = tempfile.mkdtemp(prefix="blender_render_")

    # Convert OpenGL c2w poses to Blender {yaw, pitch, radius, fov}
    views = []
    for c2w in poses:
        pos = c2w[:3, 3]
        radius = float(np.linalg.norm(pos))
        # Blender coordinate system: X-right, Y-forward, Z-up
        # Our poses: camera position in OpenGL (Y-up)
        # cx = r*cos(e)*sin(a), cy = r*sin(e), cz = r*cos(e)*cos(a)
        # Blender expects: x = r*cos(yaw)*cos(pitch), y = r*sin(yaw)*cos(pitch), z = r*sin(pitch)
        pitch = float(np.arcsin(np.clip(pos[1] / radius, -1, 1)))  # our Y = Blender Z
        horiz = np.sqrt(pos[0] ** 2 + pos[2] ** 2)
        yaw = float(np.arctan2(pos[0], pos[2])) if horiz > 1e-6 else 0.0
        views.append({
            "yaw": yaw,
            "pitch": pitch,
            "radius": radius,
            "fov": float(fov),
        })

    cmd = [
        BLENDER_PATH, "-b", "-P", BLENDER_SCRIPT, "--",
        "--object", os.path.abspath(glb_path),
        "--output_folder", render_dir,
        "--views", json.dumps(views),
        "--resolution", str(resolution),
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors='replace')[-500:]
        print(f"    [WARN] Blender failed, falling back to pyrender: {stderr}")
        return None  # caller should fall back

    # Read rendered PNGs
    images = []
    for i in range(len(poses)):
        png_path = os.path.join(render_dir, f"{i:03d}.png")
        if os.path.exists(png_path):
            img = Image.open(png_path).convert("RGBA")
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.LANCZOS)
            images.append(np.array(img))
        else:
            # Missing frame — create blank
            images.append(np.zeros((resolution, resolution, 4), dtype=np.uint8))

    # Clean up
    shutil.rmtree(render_dir, ignore_errors=True)
    return images


def _render_pyrender(
    scene_tm,
    scale: float,
    centroid: np.ndarray,
    poses: list[np.ndarray],
    fov: float,
    resolution: int = RENDER_RES,
    ssaa: int = 2,
) -> list[np.ndarray]:
    """Fallback: render via pyrender with original PBR materials."""
    render_res = resolution * ssaa

    pr_scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[0.35, 0.35, 0.35],
    )

    if isinstance(scene_tm, trimesh.Scene):
        for node_name in scene_tm.graph.nodes_geometry:
            transform, geom_name = scene_tm.graph[node_name]
            geom = scene_tm.geometry[geom_name].copy()
            geom.apply_transform(transform)
            geom.vertices = (geom.vertices - centroid) * scale
            _fix_texture(geom)
            pr_scene.add(pyrender.Mesh.from_trimesh(geom, smooth=False))
    else:
        geom = scene_tm.copy()
        geom.vertices = (geom.vertices - centroid) * scale
        _fix_texture(geom)
        pr_scene.add(pyrender.Mesh.from_trimesh(geom, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=fov)
    cam_node = pr_scene.add(camera, pose=np.eye(4))
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    key_node = pr_scene.add(key_light, pose=np.eye(4))
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    fill_pose = np.eye(4)
    fill_pose[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    pr_scene.add(fill_light, pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(render_res, render_res)
    images = []
    for pose in poses:
        pr_scene.set_pose(cam_node, pose)
        pr_scene.set_pose(key_node, pose)
        color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
        color = _fill_alpha_holes(color)
        if ssaa > 1:
            color = _downsample_rgba(color, resolution)
        images.append(color)
    renderer.delete()
    return images


def _downsample_rgba(rgba: np.ndarray, target_res: int) -> np.ndarray:
    """Downsample RGBA with premultiplied alpha to avoid dark edge fringes."""
    from PIL import Image as _Img
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    premul = np.concatenate([rgb * alpha, rgba[:, :, 3:4].astype(np.float32)], axis=2)
    premul_u8 = np.clip(premul, 0, 255).astype(np.uint8)
    img = _Img.fromarray(premul_u8, 'RGBA')
    small = np.array(img.resize((target_res, target_res), _Img.LANCZOS), dtype=np.float32)
    a_small = small[:, :, 3:4]
    result = np.zeros((target_res, target_res, 4), dtype=np.uint8)
    mask = a_small > 1.0
    safe_a = np.where(mask, a_small / 255.0, 1.0)
    result[:, :, :3] = np.clip(small[:, :, :3] / safe_a, 0, 255).astype(np.uint8)
    result[:, :, :3][~mask[:, :, 0]] = 0
    result[:, :, 3] = np.clip(small[:, :, 3], 0, 255).astype(np.uint8)
    return result


def _fill_alpha_holes(rgba: np.ndarray) -> np.ndarray:
    """Fill transparent holes inside the object silhouette using OpenCV inpaint.

    Falls back to iterative dilation if OpenCV is not available.
    """
    alpha = rgba[:, :, 3]
    opaque = alpha > 0
    if opaque.all():
        return rgba

    # Build inpaint mask: transparent pixels that are *inside* the silhouette
    # (surrounded by opaque pixels), not the background.
    # Dilate the opaque region, then the mask = dilated & ~opaque
    try:
        import cv2
        # Dilate opaque mask generously to find the "hull" of the object
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(opaque.astype(np.uint8), kernel, iterations=3)
        # Holes = inside dilated region but currently transparent
        hole_mask = (dilated > 0) & (~opaque)
        if not hole_mask.any():
            return rgba

        out = rgba.copy()
        # Use Telea inpainting on RGB channels
        inpaint_mask = hole_mask.astype(np.uint8) * 255
        rgb_inpainted = cv2.inpaint(
            out[:, :, :3], inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
        )
        out[:, :, :3] = rgb_inpainted
        out[hole_mask, 3] = 255
        return out

    except ImportError:
        # Fallback: iterative dilation (slower, less quality)
        return _fill_alpha_holes_fallback(rgba)


def _fill_alpha_holes_fallback(rgba: np.ndarray, iterations: int = 15) -> np.ndarray:
    """Fill transparent holes via iterative neighbor-average dilation."""
    out = rgba.copy()
    h, w = out.shape[:2]

    for _ in range(iterations):
        alpha = out[:, :, 3]
        opaque = alpha > 0
        if opaque.all():
            break

        padded = np.pad(opaque, 1, mode='constant', constant_values=False)
        has_neighbor = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] |
            padded[1:-1, :-2] | padded[1:-1, 2:]
        )
        fill_mask = ~opaque & has_neighbor
        if not fill_mask.any():
            break

        ys, xs = np.where(fill_mask)
        for y, x in zip(ys, xs):
            colors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] > 0:
                    colors.append(out[ny, nx, :3].astype(np.float32))
            if colors:
                out[y, x, :3] = np.mean(colors, axis=0).astype(np.uint8)
                out[y, x, 3] = 255

    return out


def render_masks(
    tiny_mesh: trimesh.Trimesh,
    instance_gt: np.ndarray,
    poses: list[np.ndarray],
    fov: float,
    scale: float,
    centroid: np.ndarray,
    resolution: int = RENDER_RES,
) -> list[np.ndarray]:
    """Render per-view segmentation masks using flat-colored faces."""
    num_instances = int(instance_gt.max()) + 1

    # Assign vertex colors based on face instance IDs.
    # Each face gets a unique color encoding its instance ID.
    mesh = tiny_mesh.copy()
    mesh.vertices = (mesh.vertices - centroid) * scale

    # Create per-face colors: encode instance_id in R,G channels
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for i in range(len(mesh.faces)):
        iid = int(instance_gt[i])
        # Encode instance ID: R = iid % 256, G = iid // 256, B = 0
        face_colors[i] = [iid % 256, iid // 256, 0, 255]
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=face_colors
    )

    pr_scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[1.0, 1.0, 1.0],  # full ambient, no shading
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    pr_scene.add(pr_mesh)

    camera = pyrender.PerspectiveCamera(yfov=fov)
    cam_node = pr_scene.add(camera, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    masks = []

    for pose in poses:
        pr_scene.set_pose(cam_node, pose)
        color, _ = renderer.render(
            pr_scene,
            flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA,
        )
        # Decode instance ID from R,G channels
        mask = np.full((resolution, resolution), -1, dtype=np.int16)
        alpha = color[:, :, 3]
        fg = alpha > 0
        mask[fg] = color[fg, 0].astype(np.int16) + color[fg, 1].astype(np.int16) * 256
        masks.append(mask)

    renderer.delete()
    return masks


# ---------------------------------------------------------------------------
# Mesh splitting
# ---------------------------------------------------------------------------
def split_mesh_by_instances(
    mesh: trimesh.Trimesh,
    instance_gt: np.ndarray,
    semantic_gt: np.ndarray,
    labels: list[str],
) -> dict:
    """Split mesh into per-instance parts.

    Returns dict with:
        parts: list of (part_id, label, submesh)
        split_mesh_json: the split_mesh.json content
    """
    instance_ids = np.unique(instance_gt)
    parts = []
    part_id_to_name = []
    valid_clusters = {}

    for inst_id in sorted(instance_ids):
        face_mask = instance_gt == inst_id
        face_indices = np.where(face_mask)[0]

        # Determine semantic label for this instance (majority vote)
        sem_ids = semantic_gt[face_indices]
        sem_id = int(np.bincount(sem_ids.astype(int)).argmax())
        label = labels[sem_id] if sem_id < len(labels) else f"part_{inst_id}"

        # Extract sub-mesh
        sub_faces = mesh.faces[face_indices]
        used_verts = np.unique(sub_faces)
        vert_map = np.full(len(mesh.vertices), -1, dtype=int)
        vert_map[used_verts] = np.arange(len(used_verts))
        new_faces = vert_map[sub_faces]
        new_verts = mesh.vertices[used_verts]

        sub_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

        # Transfer vertex colors if available
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vc = np.array(mesh.visual.vertex_colors)
            if len(vc) == len(mesh.vertices):
                sub_mesh.visual.vertex_colors = vc[used_verts]

        part_id = int(inst_id)
        name = f"{label}_{inst_id}"
        part_id_to_name.append(name)

        cluster_name = f"part_{part_id}"
        valid_clusters[cluster_name] = {
            "cluster_size": int(len(face_indices)),
            "part_ids": [part_id],
        }

        parts.append((part_id, label, sub_mesh))

    split_mesh_json = {
        "part_id_to_name": part_id_to_name,
        "valid_clusters": valid_clusters,
    }

    return {"parts": parts, "split_mesh_json": split_mesh_json}


# ---------------------------------------------------------------------------
# PLY export (with vertex colors)
# ---------------------------------------------------------------------------
def mesh_to_ply_bytes(mesh: trimesh.Trimesh) -> bytes:
    """Export trimesh to PLY bytes (binary, with vertex colors if present)."""
    buf = io.BytesIO()
    mesh.export(buf, file_type="ply")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Color transfer: from rendered views to geometry-only mesh
# ---------------------------------------------------------------------------
def bake_colors_from_views(
    mesh: trimesh.Trimesh,
    views: list[np.ndarray],
    poses: list[np.ndarray],
    fov: float,
    resolution: int = RENDER_RES,
) -> trimesh.Trimesh:
    """Project colors from rendered RGBA views onto mesh vertices."""
    vertices = np.array(mesh.vertices, dtype=np.float64)
    n_verts = len(vertices)
    W = H = resolution

    color_sum = np.zeros((n_verts, 3), dtype=np.float64)
    color_count = np.zeros(n_verts, dtype=np.float64)
    verts_h = np.hstack([vertices, np.ones((n_verts, 1))])

    focal = (W / 2.0) / math.tan(fov / 2.0)

    for i, (img_arr, c2w) in enumerate(zip(views, poses)):
        alpha = img_arr[:, :, 3].astype(np.float32) / 255.0
        rgb = img_arr[:, :, :3].astype(np.float32)

        w2c = np.linalg.inv(c2w)
        verts_cam = (w2c @ verts_h.T).T

        z = verts_cam[:, 2]
        in_front = z < 0  # OpenGL: camera looks down -Z
        z_safe = np.where(in_front, z, -1e-8)

        u = focal * verts_cam[:, 0] / (-z_safe) + W / 2.0
        v = focal * (-verts_cam[:, 1]) / (-z_safe) + H / 2.0

        u_int = np.clip(np.round(u).astype(np.int32), 0, W - 1)
        v_int = np.clip(np.round(v).astype(np.int32), 0, H - 1)

        in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        visible = in_bounds & (alpha[v_int, u_int] > 0.5)

        color_sum[visible] += rgb[v_int[visible], u_int[visible]]
        color_count[visible] += 1

    has_color = color_count > 0
    result = np.full((n_verts, 4), 128, dtype=np.uint8)
    result[:, 3] = 255
    if has_color.any():
        result[has_color, :3] = np.clip(
            color_sum[has_color] / color_count[has_color, None], 0, 255
        ).astype(np.uint8)

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=result)
    return mesh


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_partobjaverse_tiny(cache_dir: Path) -> dict:
    """Download PartObjaverse-Tiny dataset files. Returns paths dict."""
    from huggingface_hub import hf_hub_download

    files = {}
    for fname in [
        "PartObjaverse-Tiny_mesh.zip",
        "PartObjaverse-Tiny_semantic.json",
        "PartObjaverse-Tiny_semantic_gt.zip",
        "PartObjaverse-Tiny_instance_gt.zip",
    ]:
        p = hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset")
        files[fname] = p
        print(f"  Downloaded: {fname}")
    return files


def download_objaverse_glbs(uids: list[str]) -> dict[str, str]:
    """Download original textured GLBs from Objaverse."""
    print(f"  Downloading {len(uids)} GLBs from Objaverse...")
    objects = objaverse.load_objects(uids=uids)
    print(f"  Downloaded {len(objects)} objects")
    return objects


# ---------------------------------------------------------------------------
# Per-object conversion
# ---------------------------------------------------------------------------
def process_object(
    uid: str,
    category: str,
    labels: list[str],
    tiny_glb_dir: str,
    seg_zip_path: str,
    inst_zip_path: str,
    output_dir: Path,
    no_render: bool = False,
) -> dict | None:
    """Convert one object to HY3D-Part format.

    Args:
        no_render: If True, skip 42-view rendering and mask rendering.
            Only produces mesh NPZ (full.ply + part_*.ply) and minimal
            image NPZ (split_mesh.json only). Use prerender.py for
            high-quality 150-view rendering separately.

    Returns metadata dict or None on failure.
    """
    try:
        # --- Load PartObjaverse-Tiny mesh (with textures) ---
        tiny_path = os.path.join(tiny_glb_dir, f"{uid}.glb")
        if not os.path.exists(tiny_path):
            print(f"    [SKIP] Tiny GLB not found: {uid}")
            return None

        tiny_scene = trimesh.load(tiny_path)
        # Flatten to single mesh for segmentation / splitting
        if isinstance(tiny_scene, trimesh.Scene):
            tiny_mesh = _concat_scene(tiny_scene)
        else:
            tiny_mesh = tiny_scene

        # --- Load segmentation GTs ---
        with zipfile.ZipFile(seg_zip_path) as z:
            with z.open(f"PartObjaverse-Tiny_semantic_gt/{uid}.npy") as f:
                semantic_gt = np.load(f)
        with zipfile.ZipFile(inst_zip_path) as z:
            with z.open(f"PartObjaverse-Tiny_instance_gt/{uid}.npy") as f:
                instance_gt = np.load(f)

        if len(semantic_gt) != len(tiny_mesh.faces):
            print(f"    [SKIP] Face count mismatch: mesh={len(tiny_mesh.faces)} seg={len(semantic_gt)}")
            return None

        # --- Normalize mesh ---
        scale, centroid = _normalize_scene(tiny_mesh)
        norm_mesh = tiny_mesh.copy()
        norm_mesh.vertices = (norm_mesh.vertices - centroid) * scale

        if no_render:
            # --- No-render mode: geometry only ---
            # Split mesh (no vertex colors, geometry only)
            split_result = split_mesh_by_instances(
                norm_mesh, instance_gt, semantic_gt, labels
            )
            parts = split_result["parts"]
            split_mesh_json = split_result["split_mesh_json"]

            # Pack image NPZ (metadata only, no views)
            img_dir = output_dir / "images" / SHARD
            img_dir.mkdir(parents=True, exist_ok=True)
            npz_data = {
                "split_mesh.json": np.frombuffer(
                    json.dumps(split_mesh_json).encode("utf-8"), dtype=np.uint8
                ),
            }
            np.savez_compressed(str(img_dir / f"{uid}.npz"), **npz_data)

            # Pack mesh NPZ
            mesh_dir = output_dir / "mesh" / SHARD
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_npz_data = {
                "full.ply": np.frombuffer(
                    mesh_to_ply_bytes(norm_mesh), dtype=np.uint8
                ),
            }
            for part_id, label, sub_mesh in parts:
                mesh_npz_data[f"part_{part_id}.ply"] = np.frombuffer(
                    mesh_to_ply_bytes(sub_mesh), dtype=np.uint8
                )
            np.savez_compressed(str(mesh_dir / f"{uid}.npz"), **mesh_npz_data)

        else:
            # --- Full mode: render 42 views + masks + bake colors ---
            fov = math.pi / 3.0  # 60 degrees
            poses = generate_camera_poses(num_views=NUM_VIEWS, radius=2.5)

            views = render_views_from_scene(
                tiny_scene, scale, centroid, poses, fov, RENDER_RES,
                glb_path=tiny_path,
            )
            if views is None:
                views = render_views_from_scene(
                    tiny_scene, scale, centroid, poses, fov, RENDER_RES,
                    glb_path=None,
                )

            masks = render_masks(
                tiny_mesh, instance_gt, poses, fov, scale, centroid, RENDER_RES
            )

            colored_mesh = bake_colors_from_views(
                norm_mesh, views, poses, fov, RENDER_RES
            )

            split_result = split_mesh_by_instances(
                colored_mesh, instance_gt, semantic_gt, labels
            )
            parts = split_result["parts"]
            split_mesh_json = split_result["split_mesh_json"]

            transforms = {
                "camera_angle_x": float(fov),
                "frames": [],
            }
            for i, pose in enumerate(poses):
                transforms["frames"].append({
                    "file_path": f"{i:03d}",
                    "transform_matrix": pose.tolist(),
                    "camera_angle_x": float(fov),
                })

            img_dir = output_dir / "images" / SHARD
            img_dir.mkdir(parents=True, exist_ok=True)
            npz_data = {}
            for i, (view, mask) in enumerate(zip(views, masks)):
                img = Image.fromarray(view, "RGBA")
                buf = io.BytesIO()
                img.save(buf, format="WEBP", quality=90)
                npz_data[f"{i:03d}.webp"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                npz_data[f"{i:03d}_mask.npy"] = mask
            npz_data["transforms.json"] = np.frombuffer(
                json.dumps(transforms).encode("utf-8"), dtype=np.uint8
            )
            npz_data["split_mesh.json"] = np.frombuffer(
                json.dumps(split_mesh_json).encode("utf-8"), dtype=np.uint8
            )
            np.savez_compressed(str(img_dir / f"{uid}.npz"), **npz_data)

            mesh_dir = output_dir / "mesh" / SHARD
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_npz_data = {
                "full.ply": np.frombuffer(
                    mesh_to_ply_bytes(colored_mesh), dtype=np.uint8
                ),
            }
            for part_id, label, sub_mesh in parts:
                mesh_npz_data[f"part_{part_id}.ply"] = np.frombuffer(
                    mesh_to_ply_bytes(sub_mesh), dtype=np.uint8
                )
            np.savez_compressed(str(mesh_dir / f"{uid}.npz"), **mesh_npz_data)

        return {
            "obj_id": uid,
            "category": category,
            "num_parts": len(parts),
            "num_faces": int(len(tiny_mesh.faces)),
            "has_texture": not no_render,
            "labels": labels,
            "parts": [
                {"part_id": pid, "label": lbl, "faces": int(len(sm.faces))}
                for pid, lbl, sm in parts
            ],
        }

    except Exception as e:
        print(f"    [ERROR] {uid}: {e}")
        traceback.print_exc()
        return None


def _render_untextured_fallback(
    mesh: trimesh.Trimesh,
    poses: list[np.ndarray],
    fov: float,
    scale: float,
    centroid: np.ndarray,
) -> list[np.ndarray]:
    """Render views of an untextured mesh (gray) as fallback."""
    m = mesh.copy()
    m.vertices = (m.vertices - centroid) * scale
    m.visual = trimesh.visual.ColorVisuals(
        mesh=m,
        face_colors=np.full((len(m.faces), 4), [180, 180, 180, 255], dtype=np.uint8),
    )

    pr_scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[0.3, 0.3, 0.3],
    )
    pr_scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    camera = pyrender.PerspectiveCamera(yfov=fov)
    cam_node = pr_scene.add(camera, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = pr_scene.add(light, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(RENDER_RES, RENDER_RES)
    images = []
    for pose in poses:
        pr_scene.set_pose(cam_node, pose)
        pr_scene.set_pose(light_node, pose)
        color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
        images.append(color)
    renderer.delete()
    return images


# ---------------------------------------------------------------------------
# Phase 0 cache generation
# ---------------------------------------------------------------------------
def generate_phase0_cache(
    metadata: list[dict],
    semantic_json: dict,
    output_dir: Path,
):
    """Pre-generate Phase 0 semantic_labels.jsonl from PartObjaverse-Tiny annotations."""
    cache_dir = output_dir / "cache" / "phase0"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Flatten semantic_json: uid -> (category, labels)
    uid_info = {}
    for category, objects in semantic_json.items():
        for uid, labels in objects.items():
            uid_info[uid] = (category, labels)

    outpath = cache_dir / "semantic_labels.jsonl"
    count = 0
    with open(outpath, "w") as f:
        for meta in metadata:
            uid = meta["obj_id"]
            if uid not in uid_info:
                continue
            category, labels = uid_info[uid]
            parts_data = meta["parts"]

            # Build Phase 0 output format
            phase0_parts = []
            for p in parts_data:
                pid = p["part_id"]
                label = p["label"]
                is_core = label.lower() in {
                    "body", "torso", "base", "frame", "main",
                    "head", "wall", "floor",
                }
                # Generate basic edit instructions
                edits = []
                if not is_core:
                    edits.append({
                        "type": "deletion",
                        "prompt": f"Remove the {label.lower()} from the object",
                        "after_desc": f"The object without the {label.lower()}",
                        "before_part_desc": label,
                        "after_part_desc": "",
                    })
                    edits.append({
                        "type": "addition",
                        "prompt": f"Add a {label.lower()} to the object",
                        "after_desc": f"The object with a {label.lower()} added",
                        "before_part_desc": "",
                        "after_part_desc": label,
                    })
                edits.append({
                    "type": "modification",
                    "prompt": f"Change the style of the {label.lower()}",
                    "after_desc": f"The object with a restyled {label.lower()}",
                    "before_part_desc": label,
                    "after_part_desc": f"restyled {label}",
                })

                phase0_parts.append({
                    "part_id": pid,
                    "label": label.lower().replace(" ", "_"),
                    "core": is_core,
                    "desc": label,
                    "desc_without": f"The object without its {label.lower()}" if not is_core else "",
                    "edits": edits,
                })

            record = {
                "obj_id": uid,
                "shard": SHARD,
                "num_parts": len(parts_data),
                "object_desc": f"A 3D {category.lower().replace('&&', 'and')} object",
                "parts": phase0_parts,
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"  Phase 0 cache: {count} objects → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global BLENDER_PATH, BLENDER_SCRIPT
    parser = argparse.ArgumentParser(description="Prepare PartObjaverse-Tiny dataset")
    parser.add_argument("--config", type=str,
                        default="configs/prerender_partobjaverse.yaml",
                        help="Prerender config path")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: paths.dataset_root from config)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N objects (0=all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (rendering needs GPU, keep 1)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if data already cached")
    parser.add_argument("--local-source", default="",
                        help="Path to local source dir (with objaverse_mapping.json, mesh.zip, etc.)")
    parser.add_argument("--blender-path", default=None,
                        help="Override tools.blender_path for this run")
    parser.add_argument("--blender-script", default=None,
                        help="Override tools.blender_script for this run")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip 42-view rendering. Only produce mesh NPZ "
                             "(full.ply + part_*.ply) and minimal image NPZ "
                             "(split_mesh.json). Use prerender.py for 150-view "
                             "rendering separately. Much faster (~10x).")
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        for_prerender=True,
        prerender_mode="partobjaverse_prepare",
    )
    paths = cfg["paths"]
    tools = cfg["tools"]
    BLENDER_PATH = args.blender_path or tools["blender_path"]
    BLENDER_SCRIPT = args.blender_script or tools["blender_script"]
    if not args.no_render:
        if not Path(BLENDER_SCRIPT).exists():
            raise FileNotFoundError(
                f"Missing Blender script at tools.blender_script: {BLENDER_SCRIPT}"
            )
        blender_exists = (
            os.path.isabs(BLENDER_PATH) and os.path.exists(BLENDER_PATH)
        ) or (not os.path.isabs(BLENDER_PATH) and shutil.which(BLENDER_PATH))
        if not blender_exists:
            raise FileNotFoundError(
                f"Blender executable not found from tools.blender_path: {BLENDER_PATH}"
            )

    _lazy_imports(no_render=args.no_render)

    output_dir = Path(args.output or paths["dataset_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Get PartObjaverse-Tiny data ----
    print("=" * 60)
    print("Step 1: Loading PartObjaverse-Tiny data...")
    print("=" * 60)

    local_src = Path(args.local_source) if args.local_source else None
    if local_src is None:
        source_dir = Path(paths["dataset_root"]) / "source"
        req = ("mesh.zip", "semantic.json", "semantic_gt.zip", "instance_gt.zip")
        if source_dir.is_dir() and all((source_dir / name).exists() for name in req):
            local_src = source_dir
    if local_src and local_src.exists():
        # Use pre-organized local source directory
        pot_files = {
            "PartObjaverse-Tiny_mesh.zip": str(local_src / "mesh.zip"),
            "PartObjaverse-Tiny_semantic.json": str(local_src / "semantic.json"),
            "PartObjaverse-Tiny_semantic_gt.zip": str(local_src / "semantic_gt.zip"),
            "PartObjaverse-Tiny_instance_gt.zip": str(local_src / "instance_gt.zip"),
        }
        # Resolve symlinks
        for k, v in pot_files.items():
            pot_files[k] = str(Path(v).resolve())
        print(f"  Using local source: {local_src}")
    else:
        pot_files = download_partobjaverse_tiny(output_dir)

    # ---- Step 2: Parse semantic labels ----
    with open(pot_files["PartObjaverse-Tiny_semantic.json"]) as f:
        semantic_json = json.load(f)

    # Flatten: uid -> (category, labels)
    uid_to_info: dict[str, tuple[str, list[str]]] = {}
    for category, objects in semantic_json.items():
        for uid, labels in objects.items():
            uid_to_info[uid] = (category, labels)

    all_uids = sorted(uid_to_info.keys())
    print(f"  Total objects: {len(all_uids)}")

    if args.limit > 0:
        all_uids = all_uids[: args.limit]
        print(f"  Limited to: {len(all_uids)}")

    # ---- Step 3: Extract PartObjaverse-Tiny meshes ----
    print("=" * 60)
    print("Step 3: Extracting PartObjaverse-Tiny meshes...")
    print("=" * 60)
    tiny_extract_dir = tempfile.mkdtemp(prefix="pot_mesh_")
    with zipfile.ZipFile(pot_files["PartObjaverse-Tiny_mesh.zip"]) as z:
        z.extractall(tiny_extract_dir)
    tiny_glb_dir = os.path.join(tiny_extract_dir, "PartObjaverse-Tiny_mesh")
    print(f"  Extracted to: {tiny_glb_dir}")

    # ---- Step 4: Process each object ----
    print("=" * 60)
    print("Step 3: Converting objects to HY3D-Part format...")
    print("=" * 60)

    metadata_all = []
    for idx, uid in enumerate(all_uids):
        category, labels = uid_to_info[uid]
        print(f"  [{idx + 1}/{len(all_uids)}] {uid} ({category})")

        # Check if already done
        img_npz = output_dir / "images" / SHARD / f"{uid}.npz"
        mesh_npz = output_dir / "mesh" / SHARD / f"{uid}.npz"
        if img_npz.exists() and mesh_npz.exists():
            print(f"    [SKIP] Already processed")
            # Load existing metadata
            try:
                existing = np.load(str(img_npz), allow_pickle=True)
                sm = json.loads(existing["split_mesh.json"].tobytes().decode())
                metadata_all.append({
                    "obj_id": uid,
                    "category": category,
                    "num_parts": len(sm["valid_clusters"]),
                    "num_faces": 0,
                    "has_texture": True,
                    "labels": labels,
                    "parts": [
                        {"part_id": int(cn.split("_")[-1]),
                         "label": labels[0] if labels else "unknown",
                         "faces": ci["cluster_size"]}
                        for cn, ci in sm["valid_clusters"].items()
                    ],
                })
                existing.close()
            except Exception:
                pass
            continue

        result = process_object(
            uid=uid,
            category=category,
            labels=labels,
            tiny_glb_dir=tiny_glb_dir,
            seg_zip_path=pot_files["PartObjaverse-Tiny_semantic_gt.zip"],
            inst_zip_path=pot_files["PartObjaverse-Tiny_instance_gt.zip"],
            output_dir=output_dir,
            no_render=args.no_render,
        )
        if result:
            metadata_all.append(result)
            print(f"    OK: {result['num_parts']} parts, texture={result['has_texture']}")
        else:
            print(f"    FAILED")

    # ---- Step 6: Save metadata ----
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "total": len(metadata_all),
            "shard": SHARD,
            "source": "PartObjaverse-Tiny",
            "objects": metadata_all,
        }, f, indent=2)
    print(f"\n  Metadata: {meta_path}")

    # ---- Step 7: Generate Phase 0 cache ----
    print("=" * 60)
    print("Step 4: Generating Phase 0 cache (pre-populated labels)...")
    print("=" * 60)
    generate_phase0_cache(metadata_all, semantic_json, output_dir)

    # ---- Cleanup ----
    shutil.rmtree(tiny_extract_dir, ignore_errors=True)

    # ---- Summary ----
    print("=" * 60)
    print("Done!")
    print(f"  Output: {output_dir}")
    print(f"  Objects processed: {len(metadata_all)}/{len(all_uids)}")
    print(f"  Image NPZs: {output_dir}/images/{SHARD}/")
    print(f"  Mesh  NPZs: {output_dir}/mesh/{SHARD}/")
    print(f"  Phase 0 cache: {output_dir}/cache/phase0/semantic_labels.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
