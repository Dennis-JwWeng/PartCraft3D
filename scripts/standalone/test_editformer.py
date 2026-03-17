#!/usr/bin/env python3
"""Test 3DEditFormer on the same cases as Phase 2.5 VD-based method.

Runs 3DEditFormer two-stage inference on the test cases from
vis_compare_test_fix, producing comparable outputs for side-by-side evaluation.

Usage:
    # Single-view mode (no merge, full replacement)
    ATTN_BACKEND=xformers python scripts/test_editformer.py \
        --config configs/partobjaverse.yaml --skip-vlm

    # Part-aware merge: replace only edited part's voxels (recommended)
    ATTN_BACKEND=xformers python scripts/test_editformer.py \
        --config configs/partobjaverse.yaml --skip-vlm \
        --merge --output-tag partmerge

    # Larger pad = more generous edit boundary (default: 2)
    ATTN_BACKEND=xformers python scripts/test_editformer.py \
        --config configs/partobjaverse.yaml --skip-vlm \
        --merge --pad-voxels 3 --output-tag partmerge_pad3

    # Run specific edit IDs
    ATTN_BACKEND=xformers python scripts/test_editformer.py \
        --config configs/partobjaverse.yaml --skip-vlm \
        --merge --edit-ids mod_000000 mod_000002
"""

import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import argparse
import base64
import gc
import io
import json
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

EDITFORMER_ROOT = Path("/Node11_nvme/wjw/3D_Editing/3DEditFormer")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(EDITFORMER_ROOT))

# Monkey-patch: the installed diff_gaussian_rasterization lacks kernel_size/
# subpixel_offset that 3DEditFormer's renderer expects.  We patch the
# module-level ``render`` function *and* update the GaussianRenderer class
# so its method sees the patched version (it captures ``render`` via closure
# at import time).
def _patch_gaussian_renderer():
    import importlib, math, types

    mod = importlib.import_module("trellis.renderers.gaussian_render")

    def _patched_render(viewpoint_camera, pc, pipe, bg_color,
                        override_color=None, scaling_modifier=1.0):
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings, GaussianRasterizer)

        screenspace_points = torch.zeros_like(
            pc.get_xyz, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Build kwargs — auto-detect whether this build of
        # diff_gaussian_rasterization needs kernel_size / subpixel_offset
        import inspect as _insp
        _raster_params = _insp.signature(GaussianRasterizationSettings).parameters
        _needs_kernel = 'kernel_size' in _raster_params

        raster_kw = dict(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False, debug=pipe.debug,
        )
        if _needs_kernel:
            h = int(viewpoint_camera.image_height)
            w = int(viewpoint_camera.image_width)
            raster_kw['kernel_size'] = getattr(pipe, 'kernel_size', 0.0)
            raster_kw['subpixel_offset'] = torch.zeros(
                (h, w, 2), dtype=torch.float32, device="cuda")

        raster_settings = GaussianRasterizationSettings(**raster_kw)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features

        rendered_image, radii = rasterizer(
            means3D=means3D, means2D=means2D,
            shs=shs if override_color is None else None,
            colors_precomp=override_color,
            opacities=opacity, scales=scales,
            rotations=rotations, cov3D_precomp=None,
        )
        return {"render": rendered_image, "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0, "radii": radii}

    # 1) Replace module-level function
    mod.render = _patched_render

    # 2) Re-bind inside GaussianRenderer.render method so it picks up the
    #    patched version (the original method captures module-level ``render``).
    _RendererCls = mod.GaussianRenderer
    _orig_method = _RendererCls.render

    def _new_renderer_render(self, gausssian, extrinsics, intrinsics,
                             colors_overwrite=None):
        import torch.nn.functional as F
        from easydict import EasyDict as edict

        resolution = self.rendering_options.resolution
        ssaa = self.rendering_options.ssaa
        near = self.rendering_options.near
        far = self.rendering_options.far

        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(
                self.rendering_options["bg_color"],
                dtype=torch.float32, device="cuda")

        view = extrinsics
        perspective = mod.intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)

        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx, "FoVy": fovy,
            "znear": near, "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera,
        })

        render_ret = _patched_render(
            camera_dict, gausssian, self.pipe, self.bg_color,
            override_color=colors_overwrite,
            scaling_modifier=self.pipe.scale_modifier)

        if ssaa > 1:
            render_ret['render'] = F.interpolate(
                render_ret['render'][None],
                size=(resolution, resolution),
                mode='bilinear', align_corners=False,
                antialias=True).squeeze()

        return edict({'color': render_ret['render']})

    _RendererCls.render = _new_renderer_render

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def preprocess_image(img: Image.Image, no_crop=False) -> Image.Image:
    """3DEditFormer's standard preprocessing: rembg + crop + resize to 518."""
    import rembg
    has_alpha = False
    if img.mode == 'RGBA':
        alpha = np.array(img)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = img
    else:
        img = img.convert('RGB')
        mx = max(img.size)
        s = min(1, 1024 / mx)
        if s < 1:
            img = img.resize((int(img.width * s), int(img.height * s)),
                             Image.Resampling.LANCZOS)
        output = rembg.remove(img, session=rembg.new_session('u2net'))
    out_np = np.array(output)
    alpha = out_np[:, :, 3]
    if not no_crop:
        pts = np.argwhere(alpha > 0.8 * 255)
        y0, x0, y1, x1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sz = int(max(x1 - x0, y1 - y0) * 1.2)
        output = output.crop((cx - sz // 2, cy - sz // 2, cx + sz // 2, cy + sz // 2))
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    out_np = np.array(output).astype(np.float32) / 255
    out_np = out_np[:, :, :3] * out_np[:, :, 3:4]
    return Image.fromarray((out_np * 255).astype(np.uint8))


def img_to_tensor(img: Image.Image) -> torch.Tensor:
    """Image -> [1,3,H,W] float cuda tensor."""
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0


def render_single_view(pipeline, slat_dir: Path) -> Image.Image:
    """Render front-view from saved SLAT feats/coords."""
    from trellis.modules import sparse as sp
    from trellis.utils import render_utils

    feats = torch.load(slat_dir / "feats.pt", weights_only=True).cuda()
    coords = torch.load(slat_dir / "coords.pt", weights_only=True).cuda()
    slat = sp.SparseTensor(feats=feats, coords=coords)
    gs = pipeline.decode_slat(slat, ['gaussian'])['gaussian'][0]

    frames = render_utils.render_video(
        gs, resolution=518, num_frames=1, bg_color=(0, 0, 0), verbose=False)
    return Image.fromarray(frames['color'][0])


# ---------------------------------------------------------------------------
# VLM 2D editing (fallback for cases without cached edits)
# ---------------------------------------------------------------------------

def vlm_edit_image(ori_img: Image.Image, prompt: str, cfg: dict) -> Image.Image:
    """Call VLM to generate 2D edited image."""
    from openai import OpenAI
    import yaml

    p0 = cfg.get("phase0", {})
    p25 = cfg.get("phase2_5", {})
    api_key = p0.get("vlm_api_key", "")
    if not api_key:
        dp = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        if dp.exists():
            api_key = yaml.safe_load(open(dp)).get("phase0", {}).get("vlm_api_key", "")
    if not api_key:
        raise RuntimeError("No VLM API key")

    client = OpenAI(base_url=p0.get("vlm_base_url", ""), api_key=api_key)
    model = p25.get("image_edit_model", "gemini-2.5-flash-image")

    buf = io.BytesIO()
    ori_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                f"This is a 3D rendered object on a plain background. "
                f"Edit the object: {prompt}. "
                f"Keep background unchanged. Keep other parts exactly as they are.")},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    for ch in resp.choices:
        if hasattr(ch.message, 'content') and ch.message.content:
            for part in ch.message.content:
                if hasattr(part, 'type') and part.type == 'image':
                    return Image.open(io.BytesIO(
                        base64.b64decode(part.image.data))).convert("RGB")
    raise RuntimeError("VLM returned no image")


# ---------------------------------------------------------------------------
# Latent extraction using TRELLIS (matches run_custom_edit.py)
# ---------------------------------------------------------------------------

def extract_latents(pipeline, ori_img: Image.Image, cache_dir: Path):
    """Extract SS latent + SLAT from image. Cached to disk."""
    ss_npz = cache_dir / "ori_ss_latent.npz"
    slat_npz = cache_dir / "ori_slat.npz"

    if ss_npz.exists() and slat_npz.exists():
        print("  Loading cached latents...")
        z_s = torch.from_numpy(np.load(ss_npz)['mean']).cuda()[None]
        slat_data = np.load(slat_npz)
        feats = torch.from_numpy(slat_data['feats']).cuda()
        coords = torch.from_numpy(slat_data['coords']).cuda()
        return z_s, feats, coords

    cache_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        cond = pipeline.get_cond([ori_img])

        # SS latent (dense 8x16x16x16)
        print("  Extracting SS latent...")
        coords_for_slat = pipeline.sample_sparse_structure(cond, num_samples=1)
        fm = pipeline.models['sparse_structure_flow_model']
        reso = fm.resolution
        noise = torch.randn(1, fm.in_channels, reso, reso, reso).to(pipeline.device)
        z_s = pipeline.sparse_structure_sampler.sample(
            fm, noise, **cond, **pipeline.sparse_structure_sampler_params,
            verbose=False).samples
        np.savez(ss_npz, mean=z_s[0].cpu().numpy())

        # SLAT (sparse, normalized)
        print("  Extracting SLAT...")
        slat = pipeline.sample_slat(cond, coords_for_slat)
        std = torch.tensor(pipeline.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(slat.device)
        feats = (slat.feats - mean) / std
        coords = slat.coords
        np.savez(slat_npz, feats=feats.cpu().numpy(),
                 coords=coords.cpu().numpy())

    return z_s, feats, coords


# ---------------------------------------------------------------------------
# Multi-view weighted conditioning
# ---------------------------------------------------------------------------

def load_multiview_edits(edits_2d_dir: Path, edit_id: str, max_views: int | None = None):
    """Load cached multi-view input/edited image pairs.

    Supports two naming conventions:
      - Multi-view: {edit_id}_input_v0.png, _v1.png, ... (from run_phase2_5)
      - Single-view: {edit_id}_input.png, {edit_id}_edited.png (from run_2d_edit)

    Returns:
        ori_views:    list of PIL images
        edited_views: list of PIL images
        n_views:      number of views found
    """
    ori_views, edited_views = [], []
    # Try multi-view format first: _v0, _v1, ...
    v = 0
    while True:
        inp_path = edits_2d_dir / f"{edit_id}_input_v{v}.png"
        edt_path = edits_2d_dir / f"{edit_id}_edited_v{v}.png"
        if not inp_path.exists() or not edt_path.exists():
            break
        ori_views.append(Image.open(str(inp_path)).convert("RGB"))
        edited_views.append(Image.open(str(edt_path)).convert("RGB"))
        v += 1
        if max_views is not None and v >= max_views:
            break

    if ori_views:
        return ori_views, edited_views, len(ori_views)

    # Fallback: single-view format (no _v0 suffix, from run_2d_edit.py)
    inp_path = edits_2d_dir / f"{edit_id}_input.png"
    edt_path = edits_2d_dir / f"{edit_id}_edited.png"
    if inp_path.exists() and edt_path.exists():
        ori_views.append(Image.open(str(inp_path)).convert("RGB"))
        edited_views.append(Image.open(str(edt_path)).convert("RGB"))
        return ori_views, edited_views, 1

    return ori_views, edited_views, 0


def select_best_view(
    ori_views: list[Image.Image],
    edited_views: list[Image.Image],
) -> tuple[int, list[float]]:
    """Select the view with the largest edit difference.

    Computes per-view L1 pixel difference (foreground-only), returns
    the index of the best view and the score for each view.

    For large-area edits the best view shows the most changed surface.
    For small-area edits the best view is where the part is most visible.
    """
    scores = []
    for ori, edt in zip(ori_views, edited_views):
        ori_arr = np.array(ori).astype(np.float32)
        edt_arr = np.array(edt).astype(np.float32)
        # Foreground mask: pixels where either ori or edt is non-black
        fg_mask = (ori_arr.sum(axis=2) > 10) | (edt_arr.sum(axis=2) > 10)
        if fg_mask.sum() == 0:
            scores.append(0.0)
            continue
        # L1 diff only on foreground pixels
        diff = np.abs(ori_arr - edt_arr).mean(axis=2)  # [H, W]
        score = diff[fg_mask].mean()
        scores.append(float(score))

    best_idx = int(np.argmax(scores))
    return best_idx, scores


def build_multiview_cond_weighted(
    trainer,
    ori_views: list[Image.Image],
    edited_views: list[Image.Image],
    edit_strength: float = 0.7,
):
    """Build multi-view weighted conditioning features.

    For each view independently:
      delta_i = DINOv2(edited_i) - DINOv2(ori_i)

    Weighted average by per-patch delta norm (views with stronger edits
    contribute more at each spatial position):
      avg_delta = sum(w_i * delta_i)  where w_i = ||delta_i|| / sum(||delta_j||)

    Final condition:
      feat_cond = feat_ori_mv + edit_strength * avg_delta

    This ensures:
      - Background / unedited regions: dominated by multi-view original average
      - Edited regions: weighted toward views where the edit is most visible
      - Cross-view inconsistencies: smoothed out by weighted averaging in
        DINOv2 feature space
    """
    n = len(ori_views)

    # Encode all views through trainer's DINOv2
    ori_tensors = torch.cat([img_to_tensor(v) for v in ori_views], dim=0)   # [N,3,H,W]
    edt_tensors = torch.cat([img_to_tensor(v) for v in edited_views], dim=0)

    with torch.no_grad():
        feat_ori_all = trainer.encode_image(ori_tensors)    # [N, 1374, 1024]
        feat_edt_all = trainer.encode_image(edt_tensors)    # [N, 1374, 1024]

    # Per-view deltas
    deltas = feat_edt_all - feat_ori_all                    # [N, 1374, 1024]

    # Confidence weights: per-patch delta norm
    # shape: [N, 1374, 1]
    weights = deltas.norm(dim=-1, keepdim=True)
    weight_sum = weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
    weights = weights / weight_sum                          # [N, 1374, 1]

    # Weighted average delta
    avg_delta = (deltas * weights).sum(dim=0, keepdim=True) # [1, 1374, 1024]

    # Multi-view original average
    feat_ori_mv = feat_ori_all.mean(dim=0, keepdim=True)    # [1, 1374, 1024]

    # Final: multi-view original + scaled edit delta
    feat_cond = feat_ori_mv + edit_strength * avg_delta

    return feat_ori_mv, feat_cond


# ---------------------------------------------------------------------------
# Part-aware SLAT merge: use GT part meshes for precise edit mask
# ---------------------------------------------------------------------------

def build_part_mask_on_slat(
    obj_id: str,
    edit_spec: dict,
    ori_coords: torch.Tensor,
    data_root: str = "data/partobjaverse_tiny",
    vd_mesh_dir: str = "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main/outputs/img_Enc",
    pad_voxels: int = 2,
) -> torch.Tensor:
    """Build a boolean mask on SLAT voxels using GT part meshes.

    Returns a boolean tensor [N_slat] where True = this voxel belongs to
    the part being edited (should be replaced with 3DEditFormer output).

    Steps:
      1. Load GT part meshes from dataset
      2. Align coordinate space: HY3D [-1,1] → VD [-0.5,0.5] → SLAT [0,63]
      3. Voxelize the target part mesh into 64³ grid
      4. Look up which SLAT voxels fall in the part's voxels
      5. KNN assign unmatched SLAT voxels near the part boundary
    """
    import open3d as o3d
    import trimesh
    from sklearn.neighbors import NearestNeighbors

    device = ori_coords.device
    shard = edit_spec.get("shard", "00")
    edit_type = edit_spec["edit_type"]

    # Determine which part IDs are being edited
    if edit_type == "modification":
        edit_part_ids = [edit_spec["old_part_id"]] if edit_spec.get("old_part_id", -1) >= 0 else []
    elif edit_type == "deletion":
        edit_part_ids = edit_spec.get("remove_part_ids", [])
    elif edit_type == "addition":
        edit_part_ids = edit_spec.get("add_part_ids", [])
    else:
        edit_part_ids = []

    if not edit_part_ids:
        # Fallback: mark all voxels as edit (full replacement)
        return torch.ones(ori_coords.shape[0], dtype=torch.bool, device=device)

    # Load meshes
    mesh_npz_path = Path(data_root) / "mesh" / shard / f"{obj_id}.npz"
    if not mesh_npz_path.exists():
        raise FileNotFoundError(f"Part mesh not found: {mesh_npz_path}")

    mesh_data = np.load(mesh_npz_path, allow_pickle=True)

    # Full mesh for bounding box alignment
    full_ply = trimesh.load(io.BytesIO(mesh_data['full.ply'].tobytes()), file_type='ply')
    hy3d_verts = np.array(full_ply.vertices)
    hy3d_center = (hy3d_verts.max(0) + hy3d_verts.min(0)) / 2
    hy3d_extent = (hy3d_verts.max(0) - hy3d_verts.min(0)).max()

    # VD mesh for target coordinate space
    vd_mesh_path = Path(vd_mesh_dir) / obj_id / "mesh.ply"
    if vd_mesh_path.exists():
        vd_mesh = o3d.io.read_triangle_mesh(str(vd_mesh_path))
        vd_verts = np.asarray(vd_mesh.vertices)
        vd_center = (vd_verts.max(0) + vd_verts.min(0)) / 2
        vd_extent = (vd_verts.max(0) - vd_verts.min(0)).max()
    else:
        # Fallback: assume centered at origin, extent 1.0
        vd_center = np.zeros(3)
        vd_extent = 1.0

    scale_factor = vd_extent / hy3d_extent if hy3d_extent > 0 else 1.0

    # Voxelize each edit part into 64³ grid
    edit_grid = torch.zeros(64, 64, 64, device=device, dtype=torch.bool)

    for pid in edit_part_ids:
        part_key = f"part_{pid}.ply"
        if part_key not in mesh_data:
            continue

        part_mesh = trimesh.load(
            io.BytesIO(mesh_data[part_key].tobytes()), file_type='ply')
        part_verts = np.array(part_mesh.vertices)

        # HY3D → VD space
        part_verts_vd = (part_verts - hy3d_center) * scale_factor + vd_center

        # VD space → SLAT space: swap Y/Z axes and negate new Y
        # VD mesh coords (x,y,z) map to SLAT grid as (x, -z, y)
        part_verts_slat = part_verts_vd[:, [0, 2, 1]].copy()
        part_verts_slat[:, 1] = -part_verts_slat[:, 1]

        # SLAT space [-0.5, 0.5] → grid [0, 63]
        grid_coords = ((part_verts_slat + 0.5) * 64).astype(np.int32)
        grid_coords = np.clip(grid_coords, 0, 63)

        # Also try Open3D voxelization for better coverage (fills faces)
        try:
            o3d_part = o3d.geometry.TriangleMesh()
            part_verts_slat_clipped = np.clip(part_verts_slat, -0.5 + 1e-6, 0.5 - 1e-6)
            o3d_part.vertices = o3d.utility.Vector3dVector(part_verts_slat_clipped)
            o3d_part.triangles = o3d.utility.Vector3iVector(
                np.array(part_mesh.faces))
            vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                o3d_part, voxel_size=1/64,
                min_bound=(-0.5, -0.5, -0.5),
                max_bound=(0.5, 0.5, 0.5))
            voxels = np.array([v.grid_index for v in vg.get_voxels()])
            if len(voxels) > 0:
                vt = torch.from_numpy(voxels).long().to(device)
                edit_grid[vt[:, 0], vt[:, 1], vt[:, 2]] = True
        except Exception:
            pass

        # Vertex-based voxelization (always add)
        grid_unique = np.unique(grid_coords, axis=0)
        vt = torch.from_numpy(grid_unique).long().to(device)
        edit_grid[vt[:, 0], vt[:, 1], vt[:, 2]] = True

    # Optional: dilate the edit grid by pad_voxels for boundary coverage
    if pad_voxels > 0:
        import torch.nn.functional as F
        kernel_size = 2 * pad_voxels + 1
        grid_f = edit_grid.float().unsqueeze(0).unsqueeze(0)  # [1,1,64,64,64]
        padded = F.max_pool3d(grid_f, kernel_size=kernel_size,
                              stride=1, padding=pad_voxels)
        edit_grid = padded.squeeze().bool()

    # Map SLAT voxels to grid: direct lookup
    sc = ori_coords[:, 1:].long()  # [N, 3]
    slat_in_edit = edit_grid[sc[:, 0], sc[:, 1], sc[:, 2]]  # [N] bool

    return slat_in_edit, edit_grid, edit_part_ids


def save_mask_debug(
    ori_coords: torch.Tensor,
    part_mask: torch.Tensor,
    edit_grid: torch.Tensor,
    save_path: Path,
    edit_id: str,
    edit_part_ids: list[int],
):
    """Save mask debug visualization: 3 orthographic projections + stats.

    Creates a single image with:
      - Row 1: XY, XZ, YZ projections of edit_grid (64³ voxel mask, red)
      - Row 2: XY, XZ, YZ projections of SLAT voxels colored by mask
               (green=preserved, red=edit)
    """
    from PIL import Image, ImageDraw, ImageFont

    save_path.mkdir(parents=True, exist_ok=True)
    sc = ori_coords[:, 1:].cpu().numpy()  # [N, 3]
    mask_np = part_mask.cpu().numpy()      # [N]
    grid_np = edit_grid.cpu().numpy()      # [64,64,64]

    cell = 128  # pixels per projection
    img = Image.new('RGB', (cell * 3, cell * 2 + 30), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Row 1: edit_grid projections (max projection along each axis)
    projs_grid = [
        grid_np.max(axis=2),  # XY (looking down Z)
        grid_np.max(axis=1),  # XZ (looking down Y)
        grid_np.max(axis=0),  # YZ (looking down X)
    ]
    labels = ['XY (top)', 'XZ (front)', 'YZ (side)']
    for i, (proj, label) in enumerate(zip(projs_grid, labels)):
        proj_img = np.zeros((64, 64, 3), dtype=np.uint8)
        proj_img[proj > 0] = [180, 40, 40]  # red for edit region
        proj_pil = Image.fromarray(proj_img).resize(
            (cell, cell), Image.Resampling.NEAREST)
        img.paste(proj_pil, (i * cell, 0))
        draw.text((i * cell + 2, 2), label, fill=(255, 255, 255))

    # Row 2: SLAT voxels colored by mask
    proj_slat = [
        (sc[:, 0], sc[:, 1]),  # XY
        (sc[:, 0], sc[:, 2]),  # XZ
        (sc[:, 1], sc[:, 2]),  # YZ
    ]
    for i, (ax0, ax1) in enumerate(proj_slat):
        slat_img = np.zeros((64, 64, 3), dtype=np.uint8)
        slat_img[slat_img[:, :, 0] == 0] = [30, 30, 30]  # dark bg
        # Draw preserved voxels (green) first, then edit (red) on top
        for j in range(len(ax0)):
            x, y = int(ax0[j]), int(ax1[j])
            if 0 <= x < 64 and 0 <= y < 64:
                if mask_np[j]:
                    slat_img[x, y] = [220, 50, 50]   # red = edit
                else:
                    slat_img[x, y] = [50, 200, 50]    # green = preserved

        slat_pil = Image.fromarray(slat_img).resize(
            (cell, cell), Image.Resampling.NEAREST)
        img.paste(slat_pil, (i * cell, cell))

    # Stats text
    n_edit = int(part_mask.sum())
    n_total = len(part_mask)
    n_grid = int(edit_grid.sum())
    info = (f"{edit_id} parts={edit_part_ids} | "
            f"grid={n_grid}/262144 | "
            f"SLAT mask={n_edit}/{n_total} "
            f"({n_edit/n_total*100:.1f}%)")
    draw.text((4, cell * 2 + 4), info, fill=(255, 255, 200))

    out_path = save_path / f"{edit_id}_mask_debug.png"
    img.save(str(out_path))
    return out_path


def merge_slat_partaware(
    ori_feats: torch.Tensor,
    ori_coords: torch.Tensor,
    edt_feats: torch.Tensor,
    edt_coords: torch.Tensor,
    part_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Part-aware SLAT merge using GT part mask.

    Strategy:
      - Non-edit SLAT voxels: ALWAYS use original features (untouched)
      - Edit SLAT voxels: use 3DEditFormer's output if available at
        the same coordinate, otherwise drop (the edit removed them)
      - edt-only voxels in the edit region: add (new geometry from edit)
      - edt-only voxels outside edit region: drop (hallucinated changes)

    Args:
        ori_feats:  [N_ori, 8]
        ori_coords: [N_ori, 4] (batch_idx, x, y, z)
        edt_feats:  [N_edt, 8]
        edt_coords: [N_edt, 4]
        part_mask:  [N_ori] bool — True for voxels in the edited part

    Returns:
        merged_feats, merged_coords, stats
    """
    device = ori_feats.device

    def coords_to_keys(c):
        return c[:, 1] * 64 * 64 + c[:, 2] * 64 + c[:, 3]

    ori_keys = coords_to_keys(ori_coords)
    edt_keys = coords_to_keys(edt_coords)

    ori_key2idx = {k: i for i, k in enumerate(ori_keys.cpu().tolist())}
    edt_key2idx = {k: i for i, k in enumerate(edt_keys.cpu().tolist())}

    ori_key_set = set(ori_keys.cpu().tolist())
    edt_key_set = set(edt_keys.cpu().tolist())

    # Build 64³ grid of the part mask for checking edt-only voxels
    edit_grid = torch.zeros(64, 64, 64, device=device, dtype=torch.bool)
    edit_ori_sc = ori_coords[part_mask][:, 1:].long()
    if edit_ori_sc.shape[0] > 0:
        edit_grid[edit_ori_sc[:, 0], edit_ori_sc[:, 1], edit_ori_sc[:, 2]] = True

    merged_feats_list = []
    merged_coords_list = []

    n_ori_kept = 0       # non-edit ori voxels preserved
    n_ori_replaced = 0   # edit ori voxels replaced by edt
    n_ori_dropped = 0    # edit ori voxels not in edt (removed by edit)
    n_edt_new = 0        # edt-only voxels in edit region (new geometry)
    n_edt_dropped = 0    # edt-only voxels outside edit region (discarded)

    # 1) Process original voxels
    for key, idx in ori_key2idx.items():
        is_edit_part = part_mask[idx].item()

        if not is_edit_part:
            # Non-edit part: always keep original
            merged_feats_list.append(ori_feats[idx])
            merged_coords_list.append(ori_coords[idx])
            n_ori_kept += 1
        else:
            # Edit part: use edt version if exists, else drop
            if key in edt_key2idx:
                ei = edt_key2idx[key]
                merged_feats_list.append(edt_feats[ei])
                merged_coords_list.append(edt_coords[ei])
                n_ori_replaced += 1
            else:
                n_ori_dropped += 1

    # 2) Process edt-only voxels (not in original)
    edt_only_keys = edt_key_set - ori_key_set
    for key in edt_only_keys:
        ei = edt_key2idx[key]
        ec = edt_coords[ei]
        # Check if this edt-only voxel is near the edit region
        x, y, z = ec[1].item(), ec[2].item(), ec[3].item()
        if (0 <= x < 64 and 0 <= y < 64 and 0 <= z < 64
                and edit_grid[x, y, z].item()):
            # In edit region: new geometry from the edit
            merged_feats_list.append(edt_feats[ei])
            merged_coords_list.append(edt_coords[ei])
            n_edt_new += 1
        else:
            # Outside edit region: discard (hallucinated)
            n_edt_dropped += 1

    if not merged_feats_list:
        # Edge case: nothing left — return original
        return ori_feats, ori_coords, {"error": "empty merge"}

    merged_feats = torch.stack(merged_feats_list)
    merged_coords = torch.stack(merged_coords_list)

    stats = {
        "ori_voxels": len(ori_key_set),
        "edt_voxels": len(edt_key_set),
        "part_mask_edit": int(part_mask.sum()),
        "part_mask_preserved": int((~part_mask).sum()),
        "ori_kept": n_ori_kept,
        "ori_replaced": n_ori_replaced,
        "ori_dropped": n_ori_dropped,
        "edt_new": n_edt_new,
        "edt_dropped": n_edt_dropped,
        "merged_total": merged_feats.shape[0],
    }
    return merged_feats, merged_coords, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test 3DEditFormer on Phase 2.5 test cases")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tag", type=str, default="test_fix")
    parser.add_argument("--edit-ids", nargs="*", default=None)
    parser.add_argument("--specs", type=str, default=None,
                        help="Path to edit_specs JSONL "
                             "(default: {phase1.cache_dir}/edit_specs.jsonl)")
    parser.add_argument("--edit-dir", type=str, default=None,
                        help="2D edits subdir name under cache_dir "
                             "(default: '2d_edits'). Use '2d_edits_action' "
                             "to read action-style edits")
    parser.add_argument("--ckpt-root", type=str,
                        default=str(EDITFORMER_ROOT / "work_dirs" / "Editing_Training"))
    parser.add_argument("--trellis-path", type=str,
                        default=str(EDITFORMER_ROOT / "checkpoints" / "TRELLIS-image-large"))
    parser.add_argument("--total-steps", type=int, default=25)
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip cases without cached 2D edits")
    parser.add_argument("--merge", action="store_true",
                        help="Part-aware merge: replace only edited part's voxels, "
                             "keep everything else from original")
    parser.add_argument("--pad-voxels", type=int, default=2,
                        help="Dilation padding around part mask in voxels (default: 2)")
    parser.add_argument("--data-root", type=str, default="data/partobjaverse_tiny",
                        help="Dataset root with mesh/ subdir for GT part meshes")
    parser.add_argument("--edit-strength", type=float, default=0.7,
                        help="Edit delta strength for multi-view fusion (default: 0.7)")
    parser.add_argument("--num-views", type=int, default=None,
                        help="Number of views to use (default: all available)")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Separate output tag (default: same as --tag)")
    parser.add_argument("--vd-slat-dir", type=str,
                        default="/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main/outputs/slat",
                        help="Dir with pre-encoded SLAT from Vinedresser3D preprocess")
    parser.add_argument("--vd-img-dir", type=str,
                        default="/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main/outputs/img_Enc",
                        help="Dir with pre-rendered views from Vinedresser3D preprocess")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "test_editformer")

    output_base = Path(cfg["data"]["output_dir"])
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_tag = args.output_tag or args.tag
    out_tag_suffix = f"_{out_tag}" if out_tag else ""
    cache_dir = Path(cfg["phase2_5"]["cache_dir"])
    edits_2d_dir = cache_dir / (args.edit_dir or "2d_edits")

    # Build case list from edit_specs + available 2D edits
    specs_path = Path(args.specs) if args.specs else (
        Path(cfg["phase1"]["cache_dir"]) / "edit_specs.jsonl")
    if not specs_path.exists():
        logger.error(f"No edit specs: {specs_path}")
        sys.exit(1)

    # Load all edit specs
    all_specs = {}
    with open(specs_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                all_specs[d["edit_id"]] = d

    # Discover which edit_ids have cached 2D edits
    # Supports both multi-view (_edited_v0.png) and single-view (_edited.png)
    available_ids = set()
    if edits_2d_dir.exists():
        for p in edits_2d_dir.glob("*_edited_v0.png"):
            eid = p.name.replace("_edited_v0.png", "")
            if eid in all_specs:
                available_ids.add(eid)
        for p in edits_2d_dir.glob("*_edited.png"):
            eid = p.name.replace("_edited.png", "")
            if eid in all_specs and eid not in available_ids:
                available_ids.add(eid)

    # Filter by --edit-ids if specified
    if args.edit_ids:
        available_ids = available_ids & set(args.edit_ids)

    # Build edit_cases from specs (include part IDs for merge)
    edit_cases = []
    for eid in sorted(available_ids):
        spec = all_specs[eid]
        edit_cases.append({
            "edit_id": eid,
            "edit_type": spec["edit_type"],
            "obj_id": spec["obj_id"],
            "shard": spec.get("shard", "00"),
            "edit_prompt": spec["edit_prompt"],
            "object_desc": spec.get("object_desc", ""),
            "after_desc": spec.get("after_desc", ""),
            # Part IDs for part-aware merge
            "old_part_id": spec.get("old_part_id", -1),
            "remove_part_ids": spec.get("remove_part_ids", []),
            "add_part_ids": spec.get("add_part_ids", []),
        })

    logger.info(f"{len(edit_cases)} cases with 2D edits "
                f"(from {len(all_specs)} total specs)")

    ef_output = output_base / f"editformer_results{out_tag_suffix}"
    ef_output.mkdir(parents=True, exist_ok=True)
    ef_cache = cache_dir / "editformer_cache"

    # ================================================================
    # Phase A: Load TRELLIS pipeline, load pre-processed data
    # ================================================================
    _patch_gaussian_renderer()
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.modules import sparse as sp

    logger.info("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_path)
    pipeline.cuda()

    slat_norm_std = torch.tensor(pipeline.slat_normalization['std'])[None].cuda()
    slat_norm_mean = torch.tensor(pipeline.slat_normalization['mean'])[None].cuda()

    vd_slat_dir = Path(args.vd_slat_dir)
    vd_img_dir = Path(args.vd_img_dir)

    # Group by obj_id
    obj_groups: OrderedDict[str, list] = OrderedDict()
    for rec in edit_cases:
        obj_groups.setdefault(rec["obj_id"], []).append(rec)

    # Load per-object data from VD preprocess
    obj_data = {}  # obj_id -> (z_s, norm_feats, coords, ori_img_raw, ori_img_pp)
    for obj_id, recs in obj_groups.items():
        logger.info(f"\nObject: {obj_id}")
        obj_cache = ef_cache / obj_id
        obj_cache.mkdir(parents=True, exist_ok=True)

        # 1) Original image: use pre-rendered view from VD img_Enc
        ori_render_path = obj_cache / "ori_render.png"
        vd_view_path = vd_img_dir / obj_id / "000.png"  # front view
        if ori_render_path.exists():
            ori_img_raw = Image.open(ori_render_path)
            logger.info(f"  Cached ori render")
        elif vd_view_path.exists():
            ori_img_raw = Image.open(vd_view_path)
            ori_img_raw.save(str(ori_render_path))
            logger.info(f"  Using VD pre-rendered view: {vd_view_path}")
        else:
            # Fallback: render from existing SLAT
            logger.info(f"  Rendering from SLAT (no VD pre-render found)...")
            before_slat_dir = (output_base / f"mesh_pairs{tag_suffix}"
                               / recs[0]["edit_id"] / "before_slat")
            ori_img_raw = render_single_view(pipeline, before_slat_dir)
            ori_img_raw.save(str(ori_render_path))

        ori_img_pp = preprocess_image(ori_img_raw, no_crop=False)

        # 2) SLAT: prioritize VD 150-view SLAT (ground truth),
        #    fallback to cache, last resort: extract from image.
        #    Store both raw (denormalized) for "before" rendering and
        #    normalized for 3DEditFormer editing.
        cached_slat_npz = obj_cache / "ori_slat.npz"
        vd_feats_path = vd_slat_dir / f"{obj_id}_feats.pt"
        vd_coords_path = vd_slat_dir / f"{obj_id}_coords.pt"
        if vd_feats_path.exists():
            # Ground truth: VD 150-view aggregated SLAT (denormalized)
            raw_feats = torch.load(vd_feats_path, weights_only=True).cuda()
            coords = torch.load(vd_coords_path, weights_only=True).cuda()
            norm_feats = (raw_feats - slat_norm_mean.squeeze()) / slat_norm_std.squeeze()
            logger.info(f"  VD SLAT (150-view GT): {raw_feats.shape[0]} voxels")
            # Update cache with correct VD SLAT
            np.savez(cached_slat_npz, feats=norm_feats.detach().cpu().numpy(),
                     coords=coords.detach().cpu().numpy())
        elif cached_slat_npz.exists():
            slat_data = np.load(cached_slat_npz)
            norm_feats = torch.from_numpy(slat_data['feats']).cuda()
            coords = torch.from_numpy(slat_data['coords']).cuda()
            raw_feats = norm_feats * slat_norm_std.squeeze() + slat_norm_mean.squeeze()
            logger.warning(f"  Using cached SLAT (no VD GT): {norm_feats.shape[0]} voxels")
        else:
            logger.warning(f"  No VD SLAT found, extracting from image (single-view)...")
            _, norm_feats, coords = extract_latents(
                pipeline, ori_img_pp, obj_cache)
            raw_feats = norm_feats * slat_norm_std.squeeze() + slat_norm_mean.squeeze()

        # 3) SS latent: extract if not cached (fast, dense 16³)
        ss_npz = obj_cache / "ori_ss_latent.npz"
        if ss_npz.exists():
            z_s = torch.from_numpy(np.load(ss_npz)['mean']).cuda()[None]
            logger.info(f"  Cached SS latent: {z_s.shape}")
        else:
            logger.info(f"  Extracting SS latent from image...")
            with torch.no_grad():
                cond = pipeline.get_cond([ori_img_pp])
                fm = pipeline.models['sparse_structure_flow_model']
                reso = fm.resolution
                noise = torch.randn(1, fm.in_channels,
                                    reso, reso, reso).to(pipeline.device)
                z_s = pipeline.sparse_structure_sampler.sample(
                    fm, noise, **cond,
                    **pipeline.sparse_structure_sampler_params,
                    verbose=False).samples
            np.savez(ss_npz, mean=z_s[0].cpu().numpy())
            logger.info(f"  SS latent: {z_s.shape}")

        obj_data[obj_id] = (z_s, norm_feats, raw_feats, coords, ori_img_raw, ori_img_pp)

    # Free pipeline flow models, keep decoders
    if 'sparse_structure_flow_model' in pipeline.models:
        del pipeline.models['sparse_structure_flow_model']
    if 'slat_flow_model' in pipeline.models:
        del pipeline.models['slat_flow_model']
    gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Phase B: Load 3DEditFormer editing models
    # ================================================================
    logger.info("\nLoading 3DEditFormer editing models...")
    from easydict import EasyDict as edict
    from trellis import models, trainers
    from trellis.modules import sparse as sp
    from trellis.datasets.sparse_structure_latent import SparseStructureLatentVisMixin

    class _DS(SparseStructureLatentVisMixin):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.normalization = None
            self.loads = [100] * 100
        def __len__(self): return 100
        def collate_fn(self, batch): return batch

    ds = _DS()

    # Stage 1: SS editing
    ss_cfg = edict(json.load(open(
        EDITFORMER_ROOT / "configs/editing/ss_flow_img_dit_L_16l8_fp16.json")))
    ss_cfg.output_dir = '/tmp/ef_ss'
    ss_cfg.load_dir = f"{args.ckpt_root}/img_to_voxel"
    ss_cfg.load_ckpt = 40000
    ss_models = {n: getattr(models, m.name)(**m.args).cuda()
                 for n, m in ss_cfg.models.items()}
    ss_trainer = getattr(trainers, ss_cfg.trainer.name)(
        ss_models, ds, **ss_cfg.trainer.args,
        output_dir=ss_cfg.output_dir, load_dir=ss_cfg.load_dir,
        step=ss_cfg.load_ckpt)
    ss_sampler = ss_trainer.get_sampler()

    # Stage 2: SLAT editing
    slat_cfg = edict(json.load(open(
        EDITFORMER_ROOT / "configs/editing/slat_flow_img_dit_L_64l8p2_fp16.json")))
    slat_cfg.output_dir = '/tmp/ef_slat'
    slat_cfg.load_dir = f"{args.ckpt_root}/voxel_to_texture"
    slat_cfg.load_ckpt = 40000
    slat_models = {n: getattr(models, m.name)(**m.args).cuda()
                   for n, m in slat_cfg.models.items()}
    slat_trainer = getattr(trainers, slat_cfg.trainer.name)(
        slat_models, ds, **slat_cfg.trainer.args,
        output_dir=slat_cfg.output_dir, load_dir=slat_cfg.load_dir,
        step=slat_cfg.load_ckpt)
    slat_sampler = ss_trainer.get_sampler()  # reuse ss_trainer's sampler type

    logger.info("3DEditFormer models loaded!")

    # ================================================================
    # Phase C: Run editing for each case
    # ================================================================
    from trellis.utils import render_utils, postprocessing_utils

    results = []
    total_steps = args.total_steps

    for obj_id, recs in obj_groups.items():
        z_s, ori_feats, ori_raw_feats, ori_coords, ori_img_raw, ori_img_pp = obj_data[obj_id]
        ori_tensor = img_to_tensor(ori_img_pp)

        for rec in recs:
            edit_id = rec["edit_id"]
            edit_prompt = rec["edit_prompt"]
            edit_type = rec["edit_type"]
            logger.info(f"\n[{edit_id}] ({edit_type}): {edit_prompt[:70]}")

            case_dir = ef_output / edit_id

            # --- Get edited image: auto-select best view ---
            mv_ori, mv_edt, n_mv = load_multiview_edits(
                edits_2d_dir, edit_id, max_views=args.num_views)

            if n_mv >= 2:
                best_idx, view_scores = select_best_view(mv_ori, mv_edt)
                score_str = " ".join(f"v{i}={s:.1f}" for i, s in enumerate(view_scores))
                logger.info(f"  View selection: best=v{best_idx} ({score_str})")
                edited_img_raw = mv_edt[best_idx]
                ori_img_for_cond = mv_ori[best_idx]
            elif n_mv == 1:
                best_idx = 0
                logger.info(f"  Single view available (v0)")
                edited_img_raw = mv_edt[0]
                ori_img_for_cond = mv_ori[0]
            elif not args.skip_vlm:
                logger.info(f"  No cached 2D edit, generating via VLM...")
                try:
                    edited_img_raw = vlm_edit_image(
                        ori_img_raw, edit_prompt, cfg)
                    edits_2d_dir.mkdir(parents=True, exist_ok=True)
                    edited_img_path = edits_2d_dir / f"{edit_id}_edited_v0.png"
                    edited_img_raw.save(str(edited_img_path))
                    ori_img_raw.save(str(
                        edits_2d_dir / f"{edit_id}_input_v0.png"))
                except Exception as e:
                    logger.error(f"  VLM failed: {e}")
                    results.append({"edit_id": edit_id, "status": "failed",
                                    "reason": f"VLM: {e}"})
                    continue
                best_idx = 0
                ori_img_for_cond = ori_img_raw
            else:
                logger.warning(f"  No cached 2D edit, skipping (--skip-vlm)")
                continue

            # Preprocess: use the best-view ori as condition (not the
            # pre-cached single-view ori_img_pp which is always v0/front)
            edited_img_pp = preprocess_image(edited_img_raw, no_crop=False)
            ori_cond_pp = preprocess_image(ori_img_for_cond, no_crop=False)
            edited_tensor = img_to_tensor(edited_img_pp)
            ori_cond_tensor = img_to_tensor(ori_cond_pp)

            try:
                with torch.no_grad():
                    # --- Stage 1: SS editing (always single-view) ---
                    logger.info("  Stage 1: SS editing...")
                    ss_noise = torch.randn_like(z_s)
                    data_t = {
                        'ori_cond_img': ori_cond_tensor,
                        'edited_cond_img': edited_tensor,
                        'ori_ss_latent': z_s,
                        'edited_ss_latent': z_s,
                    }
                    inf_args = ss_trainer.get_inference_cond(**data_t)
                    inf_args['cond'] = torch.clone(inf_args['edited_cond_img'])

                    res_ss = ss_sampler.sample(
                        ss_trainer.models['denoiser'],
                        noise=ss_noise, **inf_args,
                        steps=total_steps, cfg_strength=0, rescale_t=3.0,
                        start_step=0, end_step=total_steps, verbose=False,
                    ).samples

                    voxel_edit = slat_trainer.dataset.decode_latent(res_ss) > 0
                    coords_edit = torch.argwhere(voxel_edit)[:, [0, 2, 3, 4]].int()
                    logger.info(f"    Edited voxels: {coords_edit.shape[0]}")

                    # --- Stage 2: SLAT editing (always single-view) ---
                    logger.info("  Stage 2: SLAT editing...")
                    noise_slat = sp.SparseTensor(
                        feats=torch.randn(
                            coords_edit.shape[0],
                            slat_trainer.models['denoiser'].in_channels).cuda(),
                        coords=coords_edit,
                    )

                    ori_c = ori_coords.clone()
                    if ori_c.shape[1] == 3:
                        ori_c = torch.cat(
                            [torch.zeros_like(ori_c[:, :1]), ori_c], dim=1)
                    ori_slat_sp = sp.SparseTensor(
                        feats=ori_feats, coords=ori_c.int())

                    data_t['ori_ss_latent'] = ori_slat_sp
                    data_t['edited_ss_latent'] = ori_slat_sp
                    inf_args = slat_trainer.get_inference_cond(**data_t)
                    inf_args['cond'] = torch.clone(inf_args['edited_cond_img'])

                    slat_no_norm = slat_sampler.sample(
                        slat_trainer.models['denoiser'],
                        noise=noise_slat, **inf_args,
                        steps=total_steps, cfg_strength=0, rescale_t=3.0,
                        start_step=0, end_step=total_steps, verbose=False,
                    ).samples

                    # --- Part-aware SLAT merge (if --merge) ---
                    if args.merge:
                        logger.info(f"  Building part mask from GT meshes "
                                    f"(pad={args.pad_voxels})...")
                        part_mask, edit_grid, edit_part_ids = \
                            build_part_mask_on_slat(
                                obj_id=obj_id,
                                edit_spec=rec,
                                ori_coords=ori_c.int(),
                                data_root=args.data_root,
                                vd_mesh_dir=args.vd_img_dir,
                                pad_voxels=args.pad_voxels,
                            )
                        logger.info(
                            f"    Part mask: {int(part_mask.sum())} edit / "
                            f"{int((~part_mask).sum())} preserved "
                            f"(of {ori_c.shape[0]} SLAT voxels)")

                        # Save mask debug visualization
                        debug_path = save_mask_debug(
                            ori_c.int(), part_mask, edit_grid,
                            case_dir, edit_id, edit_part_ids)
                        logger.info(f"    Mask debug: {debug_path}")

                        merged_feats, merged_coords, merge_stats = \
                            merge_slat_partaware(
                                ori_feats=ori_feats,
                                ori_coords=ori_c.int(),
                                edt_feats=slat_no_norm.feats,
                                edt_coords=slat_no_norm.coords,
                                part_mask=part_mask,
                            )
                        logger.info(
                            f"    Merge: kept={merge_stats['ori_kept']}, "
                            f"replaced={merge_stats['ori_replaced']}, "
                            f"dropped={merge_stats['ori_dropped']}, "
                            f"new={merge_stats['edt_new']}, "
                            f"edt_dropped={merge_stats['edt_dropped']} "
                            f"→ {merge_stats['merged_total']} total")

                        slat_edited = sp.SparseTensor(
                            feats=merged_feats,
                            coords=merged_coords,
                        )
                        # Denormalize
                        slat_edited = slat_edited * slat_norm_std + slat_norm_mean
                    else:
                        # Denormalize directly
                        slat_edited = slat_no_norm * slat_norm_std + slat_norm_mean

                    # --- Decode + Export ---
                    logger.info("  Decoding...")
                    decoded = pipeline.decode_slat(
                        slat_edited, ['gaussian', 'mesh'])
                    gs_edit = decoded['gaussian'][0]

                    # Decode original for comparison — use raw (denormalized)
                    # feats directly to avoid normalize/denormalize round-trip
                    ori_slat_denorm = sp.SparseTensor(
                        feats=ori_raw_feats, coords=ori_c.int())
                    gs_ori = pipeline.decode_slat(
                        ori_slat_denorm, ['gaussian'])['gaussian'][0]

                # Render comparison video
                logger.info("  Rendering comparison...")
                case_dir.mkdir(parents=True, exist_ok=True)
                n_frames = 120
                vis_res = 512

                ori_vid = render_utils.render_video(
                    gs_ori, resolution=vis_res, num_frames=n_frames,
                    bg_color=(0, 0, 0), verbose=False)['color']
                edit_vid = render_utils.render_video(
                    gs_edit, resolution=vis_res, num_frames=n_frames,
                    bg_color=(0, 0, 0), verbose=False)['color']

                ori_thumb = np.array(ori_cond_pp.resize(
                    (vis_res, vis_res), Image.Resampling.LANCZOS))
                edit_thumb = np.array(edited_img_pp.resize(
                    (vis_res, vis_res), Image.Resampling.LANCZOS))

                # Save multi-view grid for reference
                if n_mv >= 2:
                    grid_cols = min(n_mv, 4)
                    grid_rows = (n_mv + grid_cols - 1) // grid_cols
                    cell = vis_res // 2
                    grid = Image.new('RGB', (cell * grid_cols * 2, cell * grid_rows))
                    for vi in range(n_mv):
                        r, c = vi // grid_cols, vi % grid_cols
                        grid.paste(mv_ori[vi].resize((cell, cell)), (c * 2 * cell, r * cell))
                        grid.paste(mv_edt[vi].resize((cell, cell)), (c * 2 * cell + cell, r * cell))
                    grid.save(str(case_dir / "multiview_pairs.png"))

                comp = [np.concatenate(
                    [ori_thumb, ori_vid[i], edit_thumb, edit_vid[i]], axis=1)
                    for i in range(len(ori_vid))]

                vid_path = case_dir / "comparison.mp4"
                imageio.mimsave(str(vid_path), comp, fps=30,
                                codec="libx264", quality=8,
                                pixelformat="yuv420p")

                # Export mesh
                mesh_path = case_dir / "edit.glb"
                glb = postprocessing_utils.to_glb(
                    gs_edit, decoded['mesh'][0],
                    simplify=0.95, texture_size=1024, verbose=False)
                glb.export(str(mesh_path))

                # Save SLATs
                for tag, st in [('before', ori_slat_denorm),
                                ('after', slat_edited)]:
                    sd = case_dir / f"{tag}_slat"
                    sd.mkdir(parents=True, exist_ok=True)
                    torch.save(st.feats.cpu(), sd / "feats.pt")
                    torch.save(st.coords.cpu(), sd / "coords.pt")

                # Save 2D images for reference
                ori_cond_pp.save(str(case_dir / "ori_img.png"))
                edited_img_pp.save(str(case_dir / "edited_img.png"))

                logger.info(f"  -> {vid_path}")
                logger.info(f"  -> {mesh_path}")

                result_rec = {
                    "edit_id": edit_id, "edit_type": edit_type,
                    "edit_prompt": edit_prompt, "status": "success",
                    "video": str(vid_path), "mesh": str(mesh_path),
                    "best_view": best_idx, "n_views": n_mv,
                    "merge": args.merge,
                }
                if args.merge:
                    result_rec["pad_voxels"] = args.pad_voxels
                    result_rec["merge_stats"] = merge_stats
                results.append(result_rec)

            except Exception as e:
                logger.error(f"  Failed: {e}")
                traceback.print_exc()
                results.append({"edit_id": edit_id, "status": "failed",
                                "reason": str(e)})

    # Summary
    summary_path = ef_output / "results.jsonl"
    with open(summary_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r["status"] == "success")
    fail = sum(1 for r in results if r["status"] == "failed")
    logger.info(f"\n{'='*60}")
    logger.info(f"3DEditFormer: {ok} success, {fail} failed")
    logger.info(f"Results: {ef_output}")
    logger.info(f"VD results: {output_base / f'vis_compare{tag_suffix}'}")


if __name__ == "__main__":
    main()
