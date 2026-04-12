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

Solid-palette mode (default):
  CYCLES, CPU, 4 samples, emission shaders — fast flat rendering for VLM
  part-labeling overview images.

Vertex-color mode (--use_vertex_colors):
  CYCLES, GPU, 32 samples + denoising, Principled BSDF + 3-point lighting
  matching the dataset prerender setup (key 1000W + top-area 10000W + bottom
  1000W).  The color attribute name is detected dynamically from the imported
  mesh so the script works correctly on Blender 3.x and 4.x.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix


def init_render(resolution=512, *, use_vertex_colors=False):
    """Configure Cycles render settings.

    Solid-palette mode: CPU, 4 samples, no denoising (fast).
    Vertex-color mode:  GPU if available, 32 samples, denoising on (quality).
    """
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.render.resolution_x = sc.render.resolution_y = resolution
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.film_transparent = True

    if use_vertex_colors:
        sc.cycles.samples = 32
        sc.cycles.use_denoising = True
        sc.cycles.device = 'GPU'
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.get_devices()
            prefs.compute_device_type = 'CUDA'
            for device in prefs.devices:
                device.use = device.type != 'CPU'
        except Exception:
            sc.cycles.device = 'CPU'
    else:
        sc.cycles.samples = 4
        sc.cycles.use_denoising = False
        sc.cycles.device = 'CPU'


def init_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m, do_unlink=True)


def init_lighting():
    """3-point lighting matching the dataset prerender pipeline.

    Key point (1000 W) + top area (10000 W) + bottom area (1000 W).
    Same setup as third_party/encode_asset/blender_script/render.py.
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    key = bpy.data.objects.new("Key_Light", bpy.data.lights.new("Key_Light", type="POINT"))
    bpy.context.collection.objects.link(key)
    key.data.energy = 1000
    key.location = (4, 1, 6)

    top = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top)
    top.data.energy = 10000
    top.location = (0, 0, 10)
    top.scale = (100, 100, 100)

    bottom = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom)
    bottom.data.energy = 1000
    bottom.location = (0, 0, -10)


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


def _detect_color_layer(objs):
    """Return the first vertex color attribute name found on any mesh object.

    Blender 3.x PLY import stores colors as 'Col'; Blender 4.x may use a
    different name.  We detect dynamically so both versions work.
    """
    for obj in objs:
        if obj.type != 'MESH':
            continue
        mesh = obj.data
        if hasattr(mesh, 'color_attributes') and mesh.color_attributes:
            return mesh.color_attributes[0].name
        if mesh.vertex_colors:
            return mesh.vertex_colors[0].name
    return 'Col'   # safe fallback


def make_vertex_color_material(name, color_layer_name):
    """Principled BSDF reading PLY vertex colors — matches prerender PBR style.

    Uses ShaderNodeVertexColor with the dynamically detected attribute name
    so it works on both Blender 3.x ('Col') and Blender 4.x (e.g. 'Color').
    Roughness=0.7, Specular=0.3 mirrors blender_render.py.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    vc_node = nodes.new('ShaderNodeVertexColor')
    vc_node.layer_name = color_layer_name

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.7
    spec_key = ('Specular IOR Level'
                if 'Specular IOR Level' in bsdf.inputs else 'Specular')
    bsdf.inputs[spec_key].default_value = 0.3

    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(vc_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
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
    init_render(resolution=args.resolution, use_vertex_colors=args.use_vertex_colors)
    init_scene()

    if args.use_vertex_colors:
        init_lighting()

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
        n_meshes = 0
        if args.use_vertex_colors:
            color_layer_name = _detect_color_layer(new_objs)
            mat = make_vertex_color_material(f'mat_{pid}', color_layer_name)
        else:
            rgb = palette[pid] if pid < len(palette) else [200, 200, 200]
            mat = make_solid_material(f'mat_{pid}', rgb)
        for obj in new_objs:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            n_meshes += 1
            if not args.use_vertex_colors:
                # Strip vertex colors so the solid palette color shows cleanly.
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
        if args.use_vertex_colors:
            mode = f"vertex_colors(layer={color_layer_name})"
        else:
            mode = str(palette[pid] if pid < len(palette) else [200, 200, 200])
        print(f'[INFO]  part_{pid} -> {mode}  (new_objs={len(new_objs)}, meshes={n_meshes})')

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
    p.add_argument('--use_vertex_colors', action='store_true',
                   help='Use PLY vertex colors with PBR shading instead of solid palette')
    main(p.parse_args(argv))
