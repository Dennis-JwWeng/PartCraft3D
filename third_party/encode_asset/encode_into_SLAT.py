import os
import tempfile

# Before trellis / spconv import (see spconv docs).
os.environ.setdefault("SPCONV_ALGO", "native")

from plyfile import PlyData
import torch
import numpy as np
from PIL import Image
import utils3d

def _intrinsics_from_fov_xy(fov_x, fov_y):
    """Compat shim: utils3d <=0.x had intrinsics_from_fov_xy; v1.7+ uses intrinsics_from_fov."""
    try:
        return utils3d.torch.intrinsics_from_fov_xy(fov_x, fov_y)
    except AttributeError:
        from utils3d.torch.transforms import intrinsics_from_fov
        return intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)

_NEW_PROJECT_CV = not hasattr(utils3d.torch, 'intrinsics_from_fov_xy')

def _project_cv(points, extrinsics, intrinsics):
    """Compat shim: old utils3d.torch.project_cv(pts, extr, intr); v1.7 swapped arg order."""
    if _NEW_PROJECT_CV:
        from utils3d.torch.transforms import project_cv
        return project_cv(points, intrinsics, extrinsics)
    return utils3d.torch.project_cv(points, extrinsics, intrinsics)
import math
import torch.nn.functional as F
from torchvision import transforms
import json

from trellis.modules import sparse as sp
import trellis.models as models

from .dataset_root import img_enc_root, slat_flat_root
from .dinov2_hub import get_dinov2_vitl14_reg

# Local tree: {path}.json + {path}.safetensors (see trellis.models.from_pretrained).
# Example: PARTCRAFT_SLAT_ENC_CKPT=/mnt/zsn/ckpts/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16
_SLAT_ENC_CKPT = os.environ.get(
    "PARTCRAFT_SLAT_ENC_CKPT",
    "JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16",
).strip()

_MIN_VOXELS = 50
_EXPECTED_FEAT_DIM = 8
_EXPECTED_COORD_DIM = 4

_cached_encoder = None


def _get_slat_encoder():
    """Return the cached SLAT encoder (loaded once per process)."""
    global _cached_encoder
    if _cached_encoder is None:
        _cached_encoder = models.from_pretrained(_SLAT_ENC_CKPT).eval().cuda()
    return _cached_encoder


def _atomic_save(tensor, path):
    """Write tensor to a temp file then atomically rename to avoid corruption."""
    dir_name = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(tensor, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_ply_to_numpy(filename):
    """Load a PLY file and extract the point cloud as a (N, 3) NumPy array."""
    ply_data = PlyData.read(filename)
    vertex_data = ply_data["vertex"]
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    return points


def validate_slat(feats: torch.Tensor, coords: torch.Tensor, name: str):
    """Validate SLAT tensors before saving. Raises ValueError on problems."""
    n_voxels = feats.shape[0]
    if n_voxels < _MIN_VOXELS:
        raise ValueError(
            f"{name}: too few voxels ({n_voxels} < {_MIN_VOXELS}), "
            "source mesh likely degenerate")
    if feats.shape[1] != _EXPECTED_FEAT_DIM:
        raise ValueError(
            f"{name}: unexpected feat dim {feats.shape[1]}, "
            f"expected {_EXPECTED_FEAT_DIM}")
    if coords.shape[1] != _EXPECTED_COORD_DIM:
        raise ValueError(
            f"{name}: unexpected coord dim {coords.shape[1]}, "
            f"expected {_EXPECTED_COORD_DIM}")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{name}: feats/coords row mismatch "
            f"({feats.shape[0]} vs {coords.shape[0]})")
    if not torch.isfinite(feats).all():
        raise ValueError(f"{name}: non-finite values in feats")
    if (feats == 0).all():
        raise ValueError(f"{name}: all-zero feats (degenerate encoding)")


def encode_into_SLAT(name):

    num_views = 150

    indices = load_ply_to_numpy(os.path.join(img_enc_root(), name, "voxels.ply"))
    indices = torch.from_numpy((indices+0.5)*64).long().cuda()
    positions = (indices.to(torch.float32)/64.0 - 0.5)

    dinov2_model = get_dinov2_vitl14_reg()
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    patchtokens_lst = []
    uv_lst = []
    fov = math.radians(40)
    with open(os.path.join(img_enc_root(), name, "transforms.json")) as tf:
        views = json.load(tf)["frames"]
    for i in range(num_views):

        img = Image.open(os.path.join(img_enc_root(), name, f"{i:03d}.png"))
        img = img.resize((518, 518), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 255
        if img.shape[2] == 4:
            img = img[:, :, :3] * img[:, :, 3:]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = transform(img)

        batch_images = torch.stack([img])
        batch_images = batch_images.cuda()

        c2w = torch.tensor(views[i]['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsic = torch.inverse(c2w)
        fov = views[i]['camera_angle_x']
        intrinsic = _intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        batch_extrinsics = extrinsic.unsqueeze(0).cuda()
        batch_intrinsics = intrinsic.unsqueeze(0).cuda()

        features = dinov2_model(batch_images, is_training=True)
        uv = _project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(1, 1024, n_patch, n_patch)
        patchtokens_lst.append(patchtokens.detach().cpu())
        uv_lst.append(uv.detach().cpu())

    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)
    feats = F.grid_sample(
        patchtokens,
        uv.unsqueeze(1),
        mode='bilinear',
        align_corners=False,
    ).squeeze(2).permute(0, 2, 1).detach().cpu().numpy()
    feats = np.mean(feats, axis=0).astype(np.float16)

    encoder = _get_slat_encoder()
    aggregated_features = sp.SparseTensor(
        feats = torch.from_numpy(feats).float(),
        coords = torch.cat([
            torch.zeros(feats.shape[0], 1).int(),
            indices.cpu().int(),
        ], dim=1),
    ).cuda()
    latent = encoder(aggregated_features, sample_posterior=False)

    validate_slat(latent.feats, latent.coords, name)

    slat_dir = slat_flat_root()
    os.makedirs(slat_dir, exist_ok=True)
    _atomic_save(latent.feats, os.path.join(slat_dir, f"{name}_feats.pt"))
    _atomic_save(latent.coords, os.path.join(slat_dir, f"{name}_coords.pt"))

    print(f"finish encoding {name}")
