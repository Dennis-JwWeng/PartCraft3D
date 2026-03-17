#!/usr/bin/env python3
"""Visualize a PLY/GLB mesh from multiple viewpoints using matplotlib.

Renders a contact sheet of multi-view images showing geometry + colors.
Supports: PLY (vertex colors), GLB/GLTF (PBR textures), OBJ, STL.
No GPU or special rendering libraries needed (pure matplotlib).

Usage:
    python scripts/visualize_mesh.py /path/to/mesh.glb
    python scripts/visualize_mesh.py /path/to/mesh.ply --save output.png
    python scripts/visualize_mesh.py /path/to/mesh.glb --views 12 --res 512
"""
import argparse
import math
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw


def load_mesh_with_colors(mesh_path: str):
    """Load a mesh and extract per-face RGB colors (float64, 0-1).

    Handles:
      - GLB/GLTF with PBR textures (samples texture via UV)
      - PLY/OBJ with vertex colors
      - Untextured meshes (uniform gray)

    Returns:
        (vertices, faces, face_colors, info_str)
    """
    raw = trimesh.load(mesh_path)

    # --- GLB/GLTF: usually a Scene with textured sub-meshes ---
    if isinstance(raw, trimesh.Scene):
        all_verts, all_faces, all_fcolors = [], [], []
        offset = 0
        for name, geom in raw.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            fc = _extract_face_colors(geom)
            all_verts.append(geom.vertices)
            all_faces.append(geom.faces + offset)
            all_fcolors.append(fc)
            offset += len(geom.vertices)

        if not all_verts:
            raise ValueError(f"No geometry found in {mesh_path}")

        vertices = np.concatenate(all_verts, axis=0)
        faces = np.concatenate(all_faces, axis=0)
        face_colors = np.concatenate(all_fcolors, axis=0)

        info = f"{len(raw.geometry)} sub-meshes, textured"
        return vertices, faces, face_colors, info

    # --- Single Trimesh ---
    tm = raw
    face_colors = _extract_face_colors(tm)
    info = f"visual={tm.visual.kind}"
    return tm.vertices.copy(), tm.faces.copy(), face_colors, info


def _extract_face_colors(tm: trimesh.Trimesh) -> np.ndarray:
    """Extract per-face RGB colors (float64, 0-1) from a single Trimesh."""
    n_faces = len(tm.faces)

    # --- Try texture (GLB/GLTF PBR material) ---
    if tm.visual.kind == 'texture':
        try:
            tex_img = _get_texture_image(tm)
            if tex_img is not None:
                uv = tm.visual.uv
                if uv is not None:
                    return _sample_texture_at_faces(tex_img, uv, tm.faces)
        except Exception:
            pass

    # --- Try vertex colors ---
    try:
        vc = tm.visual.vertex_colors
        if vc is not None and len(vc) > 0:
            vc_float = vc[:, :3].astype(np.float64) / 255.0
            return vc_float[tm.faces].mean(axis=1)
    except Exception:
        pass

    # --- Try face colors ---
    try:
        fc = tm.visual.face_colors
        if fc is not None and len(fc) == n_faces:
            return fc[:, :3].astype(np.float64) / 255.0
    except Exception:
        pass

    # Fallback: gray
    return np.full((n_faces, 3), 0.6)


def _get_texture_image(tm: trimesh.Trimesh):
    """Extract the base color texture image from various material types."""
    mat = tm.visual.material

    # PBRMaterial (GLB standard)
    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
        return mat.baseColorTexture
    # SimpleMaterial
    if hasattr(mat, 'image') and mat.image is not None:
        return mat.image
    # MetallicRoughnessMaterial
    if hasattr(mat, 'baseColorTexture'):
        bct = mat.baseColorTexture
        if hasattr(bct, 'source') and bct.source is not None:
            return bct.source
    # Try to_color_image() (trimesh unified API)
    if hasattr(tm.visual, 'to_color'):
        try:
            color_vis = tm.visual.to_color()
            if color_vis.vertex_colors is not None:
                return None  # will be handled by vertex color path
        except Exception:
            pass

    return None


def _sample_texture_at_faces(tex_img, uv, faces) -> np.ndarray:
    """Sample texture at face centroid UVs."""
    tex_arr = np.array(tex_img.convert("RGB"))
    th, tw = tex_arr.shape[:2]

    # UV might be per-vertex (V,2) or per-face-vertex with different indexing
    if len(uv) >= faces.max() + 1:
        face_uv = uv[faces].mean(axis=1)  # (F, 2) - average of 3 vertices
    else:
        # Fallback: assume uv is already per-face
        face_uv = uv[:len(faces)]

    u_px = np.clip((face_uv[:, 0] % 1.0 * tw).astype(int), 0, tw - 1)
    v_px = np.clip(((1 - face_uv[:, 1] % 1.0) * th).astype(int), 0, th - 1)
    return tex_arr[v_px, u_px].astype(np.float64) / 255.0


def render_view_matplotlib(vertices, faces, face_colors, elev, azim,
                           resolution=512, bg_color="white"):
    """Render one view of a colored mesh using matplotlib's 3D projection."""
    dpi = 100
    figsize = resolution / dpi
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    poly3d = vertices[faces]  # (F, 3, 3)
    collection = Poly3DCollection(poly3d, linewidths=0.0, edgecolors='none')
    collection.set_facecolor(face_colors)
    ax.add_collection3d(collection)

    mx = vertices.max(axis=0)
    mn = vertices.min(axis=0)
    center = (mx + mn) / 2
    extent = (mx - mn).max() / 2 * 1.1
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)

    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(bg_color)
    ax.axis("off")
    ax.set_box_aspect([1, 1, 1])
    fig.patch.set_facecolor(bg_color)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def make_contact_sheet(images: list, title: str, cols: int = 4) -> Image.Image:
    """Arrange rendered views into a labeled contact sheet."""
    n = len(images)
    w, h = images[0].size
    rows = (n + cols - 1) // cols
    gap = 4
    header = 30

    canvas_w = cols * w + (cols + 1) * gap
    canvas_h = rows * h + (rows + 1) * gap + header
    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 5), title, fill=(220, 220, 220))

    for i, img in enumerate(images):
        row, col = i // cols, i % cols
        x = col * w + (col + 1) * gap
        y = row * h + (row + 1) * gap + header
        canvas.paste(img, (x, y))
        angle = 360 * i // n
        elev = 25 if i % 2 == 0 else -10
        draw.text((x + 4, y + 2), f"az={angle}° el={elev}°",
                  fill=(255, 255, 100))

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 3D mesh (PLY/GLB/OBJ) from multiple viewpoints")
    parser.add_argument("mesh_path", type=str, help="Path to mesh file")
    parser.add_argument("--save", type=str, default=None, help="Save path (PNG)")
    parser.add_argument("--views", type=int, default=8, help="Number of viewpoints")
    parser.add_argument("--res", type=int, default=512, help="Render resolution")
    parser.add_argument("--cols", type=int, default=4, help="Columns in contact sheet")
    parser.add_argument("--max-faces", type=int, default=50000,
                        help="Max faces to render (downsample larger meshes)")
    args = parser.parse_args()

    mesh_path = args.mesh_path
    if not Path(mesh_path).exists():
        print(f"File not found: {mesh_path}")
        sys.exit(1)

    # Load mesh + colors
    print(f"Loading: {Path(mesh_path).name}")
    vertices, faces, face_colors, info = load_mesh_with_colors(mesh_path)

    n_verts = len(vertices)
    n_faces = len(faces)
    print(f"  Vertices: {n_verts:,}, Faces: {n_faces:,}")
    print(f"  Info: {info}")

    # Color stats
    rgb_min = face_colors.min(axis=0)
    rgb_max = face_colors.max(axis=0)
    is_uniform = np.allclose(face_colors, face_colors[0])
    print(f"  Face color range: [{rgb_min[0]:.2f},{rgb_min[1]:.2f},{rgb_min[2]:.2f}] — "
          f"[{rgb_max[0]:.2f},{rgb_max[1]:.2f},{rgb_max[2]:.2f}]"
          f"{' (UNIFORM — no texture/color)' if is_uniform else ''}")

    # Downsample if too many faces
    if n_faces > args.max_faces:
        step = max(1, n_faces // args.max_faces)
        idx = np.arange(0, n_faces, step)
        faces = faces[idx]
        face_colors = face_colors[idx]
        print(f"  Downsampled: {n_faces:,} → {len(idx):,} faces for rendering")

    # Center and normalize to [-1, 1]
    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid
    scale = np.abs(vertices).max()
    if scale > 0:
        vertices = vertices / scale

    # Render multiple views
    print(f"Rendering {args.views} views at {args.res}x{args.res}...")
    images = []
    for i in range(args.views):
        azim = 360 * i / args.views
        elev = 25 if i % 2 == 0 else -10
        img = render_view_matplotlib(vertices, faces, face_colors,
                                     elev=elev, azim=azim,
                                     resolution=args.res)
        images.append(img)
        print(f"  View {i+1}/{args.views}: azim={azim:.0f}°, elev={elev}°")

    # Build contact sheet
    title = (f"{Path(mesh_path).name}  |  "
             f"{n_verts:,} verts, {n_faces:,} faces  |  {info}")
    sheet = make_contact_sheet(images, title, cols=args.cols)

    save_path = args.save or f"outputs/{Path(mesh_path).stem}_multiview.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
