"""Blender script: render multiple PLY parts colored by a fixed palette,
using camera poses supplied in NeRF/OpenGL convention (transform_matrix).

Usage (called as a Blender subprocess):
    blender -b -P scripts/blender_render_parts.py -- \
        --parts_dir /tmp/parts \
        --palette '[[230,25,75],[60,180,75],...]' \
        --output_folder /tmp/render_out \
        --frames '[{"transform_matrix":[[..]], "camera_angle_x":0.7}, ...]' \
        --resolution 512

Each ``part_<id>.ply`` is imported as one Blender object and painted with
``palette[id]``. The mesh is consumed AS-IS — no recentering / rescaling — so
the partverse-aligned coordinate frame is preserved and the camera matrices
from ``transforms.json`` produce renders that overlay the original views.

Render engine: CYCLES with emission shaders (4 samples, CPU).  This is more
reliable than BLENDER_WORKBENCH in headless (-b) mode: Workbench's
``film_transparent`` does not output a valid alpha channel in headless mode
(all-zero alpha => pure white composite), while Cycles+emission does.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix


def init_render(resolution=512):
    """Cycles engine — flat emission rendering, 4 samples, transparent background."""
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 4
    sc.cycles.use_denoising = False
    sc.cycles.device = 'CPU'
    sc.render.resolution_x = sc.render.resolution_y = resolution
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.film_transparent = True


def init_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m, do_unlink=True)


def import_ply(path):
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_mesh.ply(filepath=path)
    except (AttributeError, RuntimeError):
        bpy.ops.wm.ply_import(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def _srgb_to_linear(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_solid_material(name, rgb_255):
    """Emission shader: flat unlit color, correct sRGB to linear conversion."""
    r, g, b = [_srgb_to_linear(c) for c in rgb_255]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value = (r, g, b, 1.0)
    emit.inputs['Strength'].default_value = 1.0
    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat


def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_width = cam.data.sensor_height = 32
    cam.data.clip_start = 0.001
    cam.data.clip_end = 10000
    return cam


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    init_render(resolution=args.resolution)
    init_scene()

    palette = json.loads(args.palette)
    parts = sorted(
        f for f in os.listdir(args.parts_dir)
        if f.startswith("part_") and f.endswith(".ply")
    )
    print(f'[INFO] Found {len(parts)} part files in {args.parts_dir}')

    for fname in parts:
        pid = int(fname.replace("part_", "").replace(".ply", ""))
        path = os.path.join(args.parts_dir, fname)
        new_objs = import_ply(path)
        if not new_objs:
            print(f'[WARN] part_{pid}: import returned 0 new objects')
            continue
        rgb = palette[pid] if pid < len(palette) else [200, 200, 200]
        mat = make_solid_material(f'mat_{pid}', rgb)
        n_meshes = 0
        for obj in new_objs:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            n_meshes += 1
            try:
                while obj.data.color_attributes:
                    obj.data.color_attributes.remove(obj.data.color_attributes[0])
            except AttributeError:
                while obj.data.vertex_colors:
                    obj.data.vertex_colors.remove(obj.data.vertex_colors[0])
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            for poly in obj.data.polygons:
                poly.material_index = 0
        print(f'[INFO]  part_{pid} -> rgb={rgb}  '
              f'(new_objs={len(new_objs)}, meshes={n_meshes})')

    print(f'[INFO] scene now has {len(bpy.data.objects)} objects total')

    cam = init_camera()

    frames = json.loads(args.frames)
    for i, frame in enumerate(frames):
        c2w = Matrix(frame["transform_matrix"])
        cam.matrix_world = c2w
        fov = frame["camera_angle_x"]
        cam.data.lens = (cam.data.sensor_width / 2.0) / math.tan(fov / 2.0)
        bpy.context.view_layer.update()
        bpy.context.scene.render.filepath = os.path.join(
            args.output_folder, f'{i:03d}.png')
        bpy.ops.render.render(write_still=True)
    print(f'[INFO] Rendered {len(frames)} views.')


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument('--parts_dir', required=True)
    p.add_argument('--palette', required=True)
    p.add_argument('--output_folder', required=True)
    p.add_argument('--frames', required=True,
                   help='JSON list of {transform_matrix, camera_angle_x}')
    p.add_argument('--resolution', type=int, default=512)
    main(p.parse_args(argv))
