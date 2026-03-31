import os
import json
import logging
import re
import numpy as np
from subprocess import DEVNULL, Popen, PIPE, STDOUT
from .utils import sphere_hammersley_sequence
from .dataset_root import img_enc_root
import open3d as o3d
import utils3d

logger = logging.getLogger(__name__)

BLENDER_PATH = os.environ.get(
    'BLENDER_PATH',
    '/usr/local/bin/blender'
)
BLENDER_THREADS = int(os.environ.get('BLENDER_THREADS', '0'))

_ERROR_RE = re.compile(
    r"Error|Traceback|FAILED|WARNING|WARN|CUDA|out of memory", re.IGNORECASE
)
_SAVED_RE = re.compile(r"^(?:Saved|Cached): '.*?/(\d+)\.png'")

def render(file_path, name, output_dir, num_views=150):
    """Returns the Blender process exit code (0 = success)."""
    output_folder = os.path.join(output_dir, name)

    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    args = [
        BLENDER_PATH, '-b',
        *((['-t', str(BLENDER_THREADS)] if BLENDER_THREADS > 0 else [])),
        '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.abspath(os.path.expanduser(file_path)),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    saved_count = 0
    proc = Popen(args, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip('\n')
        m = _SAVED_RE.search(line)
        if m:
            saved_count += 1
            if saved_count == 1 or saved_count % 50 == 0 or saved_count == num_views:
                print(f"  {name}: view {saved_count}/{num_views}", flush=True)
            continue
        if _ERROR_RE.search(line):
            print(line, flush=True)
    return proc.wait()

def voxelize(file, name, output_dir):
    if not os.path.isfile(file):
        raise FileNotFoundError(
            f"mesh.ply not found at '{file}' — Blender likely crashed before "
            "finishing all views. Check GPU resource contention / CUDA errors."
        )
    mesh = o3d.io.read_triangle_mesh(file)
    if len(mesh.vertices) == 0:
        raise RuntimeError(
            f"Open3D read '{file}' but got an empty mesh. "
            "The PLY header may contain unsupported custom attributes "
            "(common with Blender 4.x if export_attributes=True)."
        )
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    out_path = os.path.join(output_dir, 'voxels.ply')
    try:
        utils3d.io.write_ply(out_path, vertices)
    except (AttributeError, TypeError):
        from utils3d.numpy.io.ply import write_ply as _write_ply
        _write_ply(out_path, {
            "vertex": {
                "x": vertices[:, 0].astype(np.float32),
                "y": vertices[:, 1].astype(np.float32),
                "z": vertices[:, 2].astype(np.float32),
            }
        })

def renderImg_voxelize(input_file):
    name = os.path.splitext(os.path.basename(input_file))[0]
    enc_root = img_enc_root()
    os.makedirs(enc_root, exist_ok=True)
    ret = render(input_file, name, enc_root + os.sep)
    mesh_path = os.path.join(enc_root, name, "mesh.ply")
    if ret != 0:
        logger.error("Blender exited with code %d for %s — skipping voxelize", ret, name)
        raise RuntimeError(f"Blender failed (exit {ret}) for {name}")
    if not os.path.isfile(mesh_path):
        logger.error("Blender exited 0 but mesh.ply missing for %s — skipping voxelize", name)
        raise FileNotFoundError(f"mesh.ply not produced for {name}")
    voxelize(mesh_path, name, os.path.join(enc_root, name))

if __name__ == '__main__':

    # renderImg_voxelize("ancientFighter.glb")
    # renderImg_voxelize("BATHROOM_CLASSIC.glb")
    # renderImg_voxelize("CAR_CARRIER_TRAIN.glb")
    # renderImg_voxelize("castle.glb")
    # renderImg_voxelize("elephant.glb")
    # renderImg_voxelize("foodCartTwo.glb")
    # renderImg_voxelize("horseCart.glb")

    renderImg_voxelize("KITCHEN_FURNITURE_SET.glb")
    renderImg_voxelize("PartObjaverseTiny_Eight.glb")
    renderImg_voxelize("PartObjaverseTiny_Five.glb")
    renderImg_voxelize("PartObjaverseTiny_Seventeen.glb")
    renderImg_voxelize("RJ_Rabbit_Easter_Basket_Blue.glb")
    renderImg_voxelize("Sonny_School_Bus.glb")
    renderImg_voxelize("telephone.glb")