"""Blender rendering script for PartCraft3D — mirrors Vinedresser3D's setup.

Called as a subprocess:
    blender -b -P scripts/blender_render.py -- \
        --object /path/to/model.glb \
        --output_folder /tmp/render_out \
        --views '[{"yaw":0,"pitch":0,"radius":2.5,"fov":1.047}]' \
        --resolution 518
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def init_render(resolution=518):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True

    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.use_denoising = True

    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = device.type != 'CPU'


def init_scene():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam


def init_lighting():
    """Vinedresser3D three-point lighting setup."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Key light (point)
    key = bpy.data.objects.new("Key_Light", bpy.data.lights.new("Key_Light", type="POINT"))
    bpy.context.collection.objects.link(key)
    key.data.energy = 1000
    key.location = (4, 1, 6)

    # Top light (area) — large soft fill
    top = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top)
    top.data.energy = 10000
    top.location = (0, 0, 10)
    top.scale = (100, 100, 100)

    # Bottom light (area)
    bottom = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom)
    bottom.data.energy = 1000
    bottom.location = (0, 0, -10)


def scene_bbox():
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        for coord in obj.bound_box:
            coord = obj.matrix_world @ Vector(coord)
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    return Vector(bbox_min), Vector(bbox_max)


def _scene_root():
    roots = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(roots) > 1:
        parent = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent)
        for obj in roots:
            obj.parent = parent
        return parent
    return roots[0]


def normalize_scene():
    scene = _scene_root()
    bbox_min, bbox_max = scene_bbox()
    scale = 2 / max(bbox_max - bbox_min)  # match pyrender: fit in [-1,1]^3
    scene.scale = scene.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return float(scale), tuple(offset)


def apply_scale_offset(scale, offset):
    """Apply a precomputed (scale, offset) — same convention as normalize_scene."""
    scene = _scene_root()
    scene.scale = scene.scale * scale
    bpy.context.view_layer.update()
    scene.matrix_world.translation += Vector(offset)
    bpy.ops.object.select_all(action="DESELECT")


def get_transform_matrix(obj):
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = [rt[ii][jj] for jj in range(3)]
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


def import_mesh(filepath):
    """Import a mesh file, supporting GLB/GLTF, PLY, OBJ, FBX."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=filepath, merge_vertices=True,
                                   import_shading='NORMALS')
    elif ext == '.ply':
        # Blender 3.6+ uses wm.ply_import; older versions use import_mesh.ply
        try:
            bpy.ops.import_mesh.ply(filepath=filepath)
        except (AttributeError, RuntimeError):
            bpy.ops.wm.ply_import(filepath=filepath)
        # PLY vertex colors — create a material so Cycles renders them.
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            # Detect vertex color layer (API differs by Blender version)
            color_layer_name = None
            if hasattr(mesh, 'color_attributes') and mesh.color_attributes:
                color_layer_name = mesh.color_attributes[0].name
            elif mesh.vertex_colors:
                color_layer_name = mesh.vertex_colors[0].name
            if not color_layer_name:
                continue
            mat = bpy.data.materials.new(name="VertexColorMat")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            attr_node = nodes.new('ShaderNodeVertexColor')
            attr_node.layer_name = color_layer_name
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.inputs['Roughness'].default_value = 0.7
            # 'Specular' in Blender <=3.x, 'Specular IOR Level' in 4.x
            spec_key = 'Specular IOR Level' if 'Specular IOR Level' in bsdf.inputs else 'Specular'
            bsdf.inputs[spec_key].default_value = 0.3
            output = nodes.new('ShaderNodeOutputMaterial')
            links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    elif ext == '.obj':
        try:
            bpy.ops.import_scene.obj(filepath=filepath)
        except (AttributeError, RuntimeError):
            bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    init_render(resolution=args.resolution)
    init_scene()

    if args.ref_object:
        # Compute scale/offset from reference mesh, then re-import the real
        # object and apply the same transform so before/after share a frame.
        import_mesh(args.ref_object)
        ref_scale, ref_offset = normalize_scene()
        init_scene()
        import_mesh(args.object)
        apply_scale_offset(ref_scale, ref_offset)
        print(f'[INFO] Object loaded; using ref normalization '
              f'scale={ref_scale:.6f} offset={tuple(round(x,4) for x in ref_offset)}')
    else:
        import_mesh(args.object)
        print('[INFO] Object loaded.')
        normalize_scene()
        print('[INFO] Scene normalized.')

    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting ready.')

    views = json.loads(args.views)
    transforms = {"frames": []}

    for i, view in enumerate(views):
        # OpenGL Y-up → Blender Z-up coordinate conversion:
        #   blender_x =  opengl_x = r·sin(yaw)·cos(pitch)
        #   blender_y = -opengl_z = -r·cos(yaw)·cos(pitch)
        #   blender_z =  opengl_y = r·sin(pitch)
        cam.location = (
            view['radius'] * math.sin(view['yaw']) * math.cos(view['pitch']),
            -view['radius'] * math.cos(view['yaw']) * math.cos(view['pitch']),
            view['radius'] * math.sin(view['pitch']),
        )
        cam.data.lens = 16 / math.tan(view['fov'] / 2)

        bpy.context.scene.render.filepath = os.path.join(
            args.output_folder, f'{i:03d}.png')
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        transforms["frames"].append({
            "file_path": f"{i:03d}.png",
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam),
        })

    with open(os.path.join(args.output_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f'[INFO] Rendered {len(views)} views.')


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True)
    parser.add_argument('--ref_object', default=None,
                        help='Optional reference mesh — its bbox is used to '
                             'compute the scale/offset, which is then applied '
                             'to --object. Use to keep before/after at the '
                             'same scale.')
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--views', required=True)
    parser.add_argument('--resolution', type=int, default=518)
    main(parser.parse_args(argv))
