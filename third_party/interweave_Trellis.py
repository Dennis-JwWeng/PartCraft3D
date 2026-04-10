
import torch
import numpy as np
from tqdm import tqdm
from typing import *
from easydict import EasyDict as edict
from sklearn.neighbors import NearestNeighbors
from collections import deque

import trellis.modules.sparse as sp

def get_times(
    steps,
    rescale_t,
    int_len,
    num_iter,
    inverse,
):
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_seq = t_seq[::-1]
    t_seq_new = []
    for i in range(0, steps+1, int_len):
        interval = t_seq[i:min(i+int_len, steps+1)]
        if len(interval) == 1:
            t_seq_new.extend(interval)
            continue
        for cnt in range(num_iter):
            t_seq_new.extend(interval)
            if cnt < num_iter - 1:
                t_seq_new.extend(interval[::-1][1:-1])
    t_seq = np.array(t_seq_new[::-1])
    if inverse:
        t_seq = t_seq[::-1]
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    return t_seq, t_pairs

def inference_model(model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

@torch.no_grad()
def sample_once(
    model,
    x_t,
    t: float,
    t_prev: float,
    cond: Optional[Any] = None,
    neg_cond = None,
    cfg_strength: float = 3.0,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
    **kwargs
):
    if cfg_interval[0] <= t <= cfg_interval[1]:
        pred = inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = inference_model(model, x_t, t, neg_cond, **kwargs)
        pred_v = (1 + cfg_strength) * pred - cfg_strength * neg_pred
    else:
        pred_v = inference_model(model, x_t, t, cond, **kwargs)
    
    sigma_min = 1e-5
    assert x_t.shape == pred_v.shape
    pred_eps = (1 - t) * pred_v + x_t
    pred_x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred_v
    
    pred_x_prev = x_t - (t - t_prev) * pred_v
    return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_v": pred_v})

@torch.no_grad()
def RF_sample_once(
    model,
    x_t,
    t_curr: float,
    t_prev: float,
    inverse: bool = False,
    **kwargs
):

    out = sample_once(model, x_t, t_curr, t_prev, **kwargs)
    pred_vec = out.pred_v

    sample_mid = x_t + (t_prev - t_curr) / 2 * pred_vec

    out = sample_once(model, sample_mid, (t_curr+t_prev)/2, t_prev, **kwargs)
    pred_vec_mid = out.pred_v

    first_order = (pred_vec_mid - pred_vec) / ((t_prev - t_curr) / 2)
    x_t = x_t + (t_prev - t_curr) * pred_vec + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return x_t

def get_s1_mask(slat, mask_input):

    # Reshape the 64x64x64 mask into 16x16x16 blocks of 4x4x4
    mask_reshaped = mask_input.reshape(16, 4, 16, 4, 16, 4)
    mask_reshaped = mask_reshaped.sum(dim=(1,3,5))
    # Sum over the 4x4x4 blocks and threshold at half the block size
    # If more than 32 voxels in a 4x4x4 block (64 total) are True, the downsampled voxel will be True
    mask = (mask_reshaped / 64 >= 0.1).to(mask_input.device)

    mask = (~mask)
    truth_mask = torch.zeros(1, 8, 16, 16, 16, dtype=torch.bool).to(mask_input.device)
    truth_mask[:,:,mask] = True
    mask = truth_mask

    return mask


def compute_contact_boundary(mask, slat_coords, device):
    """Analyze how the edit mask connects to preserved geometry.

    Returns:
        contact_64: (64,64,64) bool — edit voxels adjacent to preserved SLAT voxels
        contact_ratio: float — fraction of edit surface that contacts preserved parts
            High ratio (>0.3): part deeply embedded (e.g. leg in body)
            Low ratio (<0.1): part loosely attached (e.g. hat on head)
        s1_sigma: float — dynamic S1 sigma (16³ space)
        s2_sigma: float — dynamic S2 sigma (64³ space)
    """
    from scipy import ndimage

    # Build preserved occupancy from SLAT coords outside mask
    preserved = torch.zeros(64, 64, 64, dtype=torch.bool, device=device)
    sc = slat_coords[:, 1:]
    in_mask = mask[sc[:, 0], sc[:, 1], sc[:, 2]]
    pres_sc = sc[~in_mask]
    if pres_sc.shape[0] > 0:
        preserved[pres_sc[:, 0], pres_sc[:, 1], pres_sc[:, 2]] = True

    # Contact boundary: mask voxels whose 6-neighbors include preserved voxels
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    pres_dilated = torch.from_numpy(
        ndimage.binary_dilation(
            preserved.cpu().numpy(), structure=struct, iterations=1)
    ).to(device)
    contact_64 = mask & pres_dilated

    # Surface of the edit region (for normalizing contact)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_dilated = ndimage.binary_dilation(mask_np, structure=struct, iterations=1)
    edit_surface = (torch.from_numpy(mask_dilated).to(device) & ~mask)
    edit_surface_count = max(int(edit_surface.sum()), 1)

    contact_count = int(contact_64.sum())
    contact_ratio = contact_count / edit_surface_count
    contact_ratio = min(contact_ratio, 1.0)

    # Dynamic sigma: more contact → wider transition needed
    # S1 is 16³ (each unit = 4 voxels), S2 is 64³
    #   deeply embedded (ratio~0.5+): s1=3.5, s2=7 — wide blend to close hole
    #   loosely attached (ratio~0.05): s1=1.5, s2=2 — narrow, mostly preserve
    s1_sigma = 1.5 + contact_ratio * 4.0   # [1.5, 5.5]
    s2_sigma = 2.0 + contact_ratio * 10.0  # [2.0, 12.0]

    return contact_64, contact_ratio, s1_sigma, s2_sigma


def get_s1_soft_mask(mask_input, sigma=3.0, contact_mask=None):
    """Build a soft float mask for S1 (16³) with distance-based decay.

    If contact_mask is provided, decay is computed from the contact
    boundary (where edit meets preserved) instead of the entire edit
    boundary.  This avoids softening toward empty space where no
    blending is needed.

    Returns a (1, 8, 16, 16, 16) float tensor where:
      - 1.0 = fully preserved (far from contact boundary)
      - 0.0 = fully editable (inside edit region)
      - (0, 1) = smooth transition at boundary
    """
    from scipy import ndimage

    # Downsample 64³ → 16³
    mask_reshaped = mask_input.float().reshape(16, 4, 16, 4, 16, 4)
    edit_frac = mask_reshaped.mean(dim=(1, 3, 5))
    edit_16 = (edit_frac >= 0.1).cpu().numpy().astype(np.uint8)

    if contact_mask is not None:
        # Downsample contact mask to 16³
        contact_reshaped = contact_mask.float().reshape(16, 4, 16, 4, 16, 4)
        contact_frac = contact_reshaped.mean(dim=(1, 3, 5))
        contact_16 = (contact_frac > 0).cpu().numpy().astype(np.uint8)

        # Distance from contact boundary (not entire edit boundary)
        # Preserved voxels far from contact zones stay fully preserved
        non_contact = 1 - contact_16
        dist = ndimage.distance_transform_edt(non_contact).astype(np.float32)
    else:
        # Fallback: distance from entire edit boundary
        preserved_16 = 1 - edit_16
        dist = ndimage.distance_transform_edt(preserved_16).astype(np.float32)

    soft = 1.0 - np.exp(-dist / max(sigma, 0.1))
    soft[edit_16 == 1] = 0.0

    soft_t = torch.from_numpy(soft).to(mask_input.device).float()
    soft_t = soft_t.unsqueeze(0).unsqueeze(0).expand(1, 8, -1, -1, -1)
    return soft_t

def get_coords_mask(coords, mask_input):
        mask = mask_input[coords[:,1], coords[:,2], coords[:,3]]
        mask = ~mask
        return mask

def remove_small_components(coords_edit, feats_edit):

    # Set a threshold for minimum voxels in component
    min_component_size = 50  # You may adjust this value as needed

    # Create a 3D grid to mark the presence of voxels
    vox_grid = torch.zeros((64, 64, 64), dtype=torch.bool, device=coords_edit.device)
    vox_grid[coords_edit[:,1], coords_edit[:,2], coords_edit[:,3]] = True

    visited = torch.zeros_like(vox_grid, dtype=torch.bool)
    keep_mask = torch.zeros(coords_edit.shape[0], dtype=torch.bool, device=coords_edit.device)

    # Mapping from coord to index in coords_edit for marking keep_mask
    # Build a 3D array with -1 for no voxel, otherwise the index of the voxel in coords_edit
    coord_to_idx = -torch.ones((64, 64, 64), dtype=torch.long, device=coords_edit.device)
    coord_to_idx[coords_edit[:,1], coords_edit[:,2], coords_edit[:,3]] = torch.arange(coords_edit.shape[0], device=coords_edit.device)

    # 6-connectivity neighbors
    neighbor_shifts = torch.tensor([
        [ 1, 0, 0],
        [-1, 0, 0],
        [ 0, 1, 0],
        [ 0,-1, 0],
        [ 0, 0, 1],
        [ 0, 0,-1],
    ], device=coords_edit.device)

    for i in range(coords_edit.shape[0]):
        x, y, z = [coords_edit[i,1], coords_edit[i,2], coords_edit[i,3]]
        if not visited[x, y, z]:
            # BFS to find connected voxels
            q = deque()
            q.append((x, y, z))
            component = []
            visited[x, y, z] = True
            idx_in_edit = coord_to_idx[x, y, z].item()
            if idx_in_edit != -1:
                component.append(idx_in_edit)

            while q:
                cx, cy, cz = q.popleft()
                for dx, dy, dz in neighbor_shifts:
                    nx, ny, nz = cx+dx, cy+dy, cz+dz
                    if 0 <= nx < 64 and 0 <= ny < 64 and 0 <= nz < 64:
                        if vox_grid[nx, ny, nz] and not visited[nx, ny, nz]:
                            visited[nx, ny, nz] = True
                            q.append((nx, ny, nz))
                            idx_n = coord_to_idx[nx, ny, nz].item()
                            if idx_n != -1:
                                component.append(idx_n)
            if len(component) >= min_component_size:
                keep_mask[component] = True

    # Remove small components from coords_edit/feats_edit
    coords_edit = coords_edit[keep_mask]
    feats_edit = feats_edit[keep_mask]

    return coords_edit, feats_edit

def get_s2_noise_new(s2_noise, coords_new, mask, in_channels):
    coords_ori = s2_noise.coords
    feats_ori = s2_noise.feats

    mask_ori = get_coords_mask(coords_ori, mask)
    mask_new = get_coords_mask(coords_new, mask)
    coords_new = torch.cat([coords_ori[mask_ori], coords_new[~mask_new]], dim=0)
    coords_ori = torch.cat([coords_ori[mask_ori], coords_ori[~mask_ori]], dim=0)
    feats_ori = torch.cat([feats_ori[mask_ori], feats_ori[~mask_ori]], dim=0)
    mask_ori = get_coords_mask(coords_ori, mask)

    num_known = torch.sum(mask_ori)
    feats_known = feats_ori[mask_ori]
    feats_unknown_new = torch.randn(coords_new.shape[0] - num_known, in_channels).to(s2_noise.device)
    feats_new = torch.cat([feats_known, feats_unknown_new], dim=0)

    mask_new = get_coords_mask(coords_new, mask)
    coords_edit = coords_new[~mask_new]
    feats_edit = feats_new[~mask_new]
    coords_edit, feats_edit = remove_small_components(coords_edit, feats_edit)
    coords_new = torch.cat([coords_new[mask_new], coords_edit], dim=0)
    feats_new = torch.cat([feats_new[mask_new], feats_edit], dim=0)
    
    noise_new = sp.SparseTensor(
        feats=feats_new,
        coords=coords_new,
    )

    return noise_new

def get_soft_weights(preserved_coords, edit_coords, sigma):
    knn = NearestNeighbors(n_neighbors=1, metric='manhattan')
    knn.fit(edit_coords.cpu().numpy())
    min_distances, _ = knn.kneighbors(preserved_coords.cpu().numpy())
    min_distances = torch.from_numpy(min_distances).squeeze(-1).to(preserved_coords.device)

    if sigma is not None:
        # Using exponential decay: exp(-d/sigma) maps distance 0->1, large distances->0
        weights = torch.exp(-min_distances/sigma).to(torch.float32) # Controls how quickly similarity drops off with distance
    else:
        weights = (min_distances <= 2).to(torch.float32)
    return weights

def get_s2_soft_mask(slat, mask, sigma=5.0, contact_mask=None):
    """Compute per-voxel soft weights for S2 preserved voxels.

    If contact_mask is given, distance is measured from the contact
    boundary instead of the entire edit region.  Preserved voxels far
    from any contact zone get weight≈0 (fully preserved), while those
    near contact zones get weight≈1 (allow generation to blend).
    """
    preserved_coords = slat.coords[:,1:][get_coords_mask(slat.coords, mask)]
    if contact_mask is not None:
        # Distance from contact boundary only
        ref_coords = torch.nonzero(contact_mask)
    else:
        # Fallback: distance from entire edit region
        ref_coords = torch.nonzero(mask)
    if ref_coords.shape[0] == 0:
        # No contact / no edit region → fully preserve everything
        return torch.zeros(preserved_coords.shape[0], 1,
                           device=slat.device, dtype=torch.float32)
    soft_mask = get_soft_weights(preserved_coords, ref_coords, sigma).unsqueeze(1)
    return soft_mask

def _use_text_step(cnt: int, mode: str) -> bool:
    """Return True if this forward-repaint step should use the text flow model.

    mode choices:
      'interleaved' - alternate text/image every step (original behaviour)
      'text'        - always use the text model
      'image'       - always use the image model
    """
    if mode == 'text':
        return True
    if mode == 'image':
        return False
    return cnt % 2 == 0  # 'interleaved'

def interweave_Trellis_TI(args, trellis_text, trellis_img,
    slat, mask, 
    prompts, 
    img_new,
    seed,
    mode: str = 'interleaved'):

    device = slat.device
    torch.manual_seed(seed)

    conds = {}
    conds["ori_cpl"] = trellis_text.get_cond([prompts["ori_cpl"]])
    conds["new_cpl"] = trellis_text.get_cond([prompts["new_cpl"]])
    conds["ori_s1_cpl"] = trellis_text.get_cond([prompts["ori_s1_cpl"]])
    conds["ori_s2_cpl"] = trellis_text.get_cond([prompts["ori_s2_cpl"]])
    conds["ori_s1_part"] = trellis_text.get_cond([prompts["ori_s1_part"]])
    conds["ori_s2_part"] = trellis_text.get_cond([prompts["ori_s2_part"]])
    conds["new_s1_cpl"] = trellis_text.get_cond([prompts["new_s1_cpl"]])
    conds["new_s2_cpl"] = trellis_text.get_cond([prompts["new_s2_cpl"]])
    conds["new_s1_part"] = trellis_text.get_cond([prompts["new_s1_part"]])
    conds["new_s2_part"] = trellis_text.get_cond([prompts["new_s2_part"]])
    text_null_cond = conds["ori_s1_cpl"]["neg_cond"]
    conds["null"] = {"cond": text_null_cond}

    if img_new is not None:
        img_new = trellis_img.preprocess_image(img_new)
        ret = trellis_img.get_cond([img_new])
        img_cond_new, img_null_cond = ret["cond"], ret["neg_cond"]
    else:
        img_cond_new, img_null_cond = None, None

    text_s1_flow_model = trellis_text.models['sparse_structure_flow_model']
    s1_text_sampler_params = {**trellis_text.sparse_structure_sampler_params}
    text_s2_flow_model = trellis_text.models['slat_flow_model']
    s2_text_sampler_params = {**trellis_text.slat_sampler_params}
    s1_encoder = trellis_text.models['sparse_structure_encoder']
    s1_decoder = trellis_text.models['sparse_structure_decoder']

    img_s1_flow_model = trellis_img.models['sparse_structure_flow_model']
    s1_img_sampler_params = {**trellis_img.sparse_structure_sampler_params}
    img_s2_flow_model = trellis_img.models['slat_flow_model']
    s2_img_sampler_params = {**trellis_img.slat_sampler_params}

    std = torch.tensor(trellis_text.slat_normalization['std'])[None].to(slat.device)
    mean = torch.tensor(trellis_text.slat_normalization['mean'])[None].to(slat.device)
    slat = (slat - mean) / std
    steps = s1_text_sampler_params['steps']
    rescale_t = s1_text_sampler_params['rescale_t']
    num_iter = args['cnt']
    int_len = 1 if num_iter == 1 else 2

    # --- Dynamic soft mask: analyze contact surface ---
    contact_64, contact_ratio, dyn_s1_sigma, dyn_s2_sigma = \
        compute_contact_boundary(mask, slat.coords, device)
    eff_s1_sigma = args.get('s1_soft_sigma', dyn_s1_sigma)
    eff_s2_sigma = args.get('s2_soft_sigma', dyn_s2_sigma)

    # stage2 inversion
    inverse_dict = {'s2_0.0': slat}
    sample = slat
    t_seq, t_pairs = get_times(steps, rescale_t, 1, 1, True)
    s2_text_sampler_params['cfg_strength'] = 0
    for t_curr, t_prev in tqdm(t_pairs, desc="Sampling", disable=False):
        sample = RF_sample_once(text_s2_flow_model, sample, t_curr, t_prev, inverse=True, cond=conds["ori_cpl"]['cond'], neg_cond=text_null_cond, **s2_text_sampler_params)
        inverse_dict[f's2_{t_prev}'] = sample
    s2_noise = sample

    if args['edit_type'] == "Addition" or args['edit_type'] == "Modification":

        sparse_voxels = torch.zeros(64, 64, 64).to(device)
        sparse_voxels[s2_noise.coords[:, 1], s2_noise.coords[:, 2], s2_noise.coords[:, 3]] = 1
        sparse_voxels = sparse_voxels.unsqueeze(0).unsqueeze(0)
        z_s = s1_encoder(sparse_voxels)

        # stage1 inversion
        s1_mask = get_s1_mask(slat, mask)
        s1_soft = get_s1_soft_mask(mask, sigma=eff_s1_sigma,
                                   contact_mask=contact_64)
        t_seq, t_pairs = get_times(steps, rescale_t, 1, 1, True)
        sample = z_s
        inverse_dict['s1_0.0'] = sample
        s1_text_sampler_params['cfg_strength'] = 0
        for t_curr, t_prev in tqdm(t_pairs, desc="Sampling", disable=False):
            sample = RF_sample_once(text_s1_flow_model, sample, t_curr, t_prev, inverse=True, cond=conds["ori_cpl"]['cond'], neg_cond=text_null_cond, **s1_text_sampler_params)
            inverse_dict[f's1_{t_prev}'] = sample
        s1_noise = sample

        # stage1 repaint (soft blend at boundary)
        t_seq, t_pairs = get_times(steps, rescale_t, int_len, num_iter, False)
        sample = s1_noise
        cnt = 0
        for t_curr, t_prev in tqdm(t_pairs, desc="Sampling", disable=False):
            x_t = sample
            if t_curr > t_prev:
                if _use_text_step(cnt, mode):
                    s1_text_sampler_params['cfg_strength'] = args['cfg_strength']
                    x_t_1 = RF_sample_once(text_s1_flow_model, x_t, t_curr, t_prev, inverse=False, cond=conds[args['s1_pos_cond']]['cond'], neg_cond=conds[args['s1_neg_cond']]['cond'], **s1_text_sampler_params)
                else:
                    s1_img_sampler_params['cfg_strength'] = 5.0
                    x_t_1 = RF_sample_once(img_s1_flow_model, x_t, t_curr, t_prev, inverse=False, cond=img_cond_new, neg_cond=img_null_cond, **s1_img_sampler_params)

                # Soft blend: boundary voxels get partial influence from generation
                inv_feats = inverse_dict[f's1_{t_prev}']
                x_t_1 = x_t_1 * (1.0 - s1_soft) + inv_feats * s1_soft
            else:
                s1_text_sampler_params['cfg_strength'] = 0
                x_t_1 = RF_sample_once(text_s1_flow_model, x_t, t_curr, t_prev, inverse=True, cond=conds["new_s1_cpl"]['cond'], neg_cond=text_null_cond, **s1_text_sampler_params)
            sample = x_t_1
            cnt += 1
        z_s_new = sample
        coords_new = torch.argwhere(s1_decoder(z_s_new)>0)[:, [0, 2, 3, 4]].int()
    elif args['edit_type'] == "TextureOnly":
        # Keep original sparse structure unchanged — only S2 changes texture
        sparse_voxels = torch.zeros(64, 64, 64).to(device)
        sparse_voxels[slat.coords[:, 1], slat.coords[:, 2], slat.coords[:, 3]] = 1
        sparse_voxels = sparse_voxels.unsqueeze(0).unsqueeze(0)
        z_s = s1_encoder(sparse_voxels)
        z_s_new = z_s
        coords_new = slat.coords.clone()
    elif args['edit_type'] in ("Deletion", "HybridDeletion"):
        # Pure voxel removal — no S1 repaint, no new structure generated
        sparse_voxels = torch.zeros(64, 64, 64).to(device)
        sparse_voxels[slat.coords[:, 1], slat.coords[:, 2], slat.coords[:, 3]] = 1
        sparse_voxels = sparse_voxels.unsqueeze(0).unsqueeze(0)
        z_s = s1_encoder(sparse_voxels)
        coords_new = slat.coords[get_coords_mask(slat.coords, mask)]
        sparse_voxels_new = torch.zeros(64, 64, 64).to(device)
        sparse_voxels_new[coords_new[:, 1], coords_new[:, 2], coords_new[:, 3]] = 1
        sparse_voxels_new = sparse_voxels_new.unsqueeze(0).unsqueeze(0)
        z_s_new = s1_encoder(sparse_voxels_new)
    else:
        raise ValueError(f"Invalid edit type: {args['edit_type']}")
        
    # stage2 repaint
    s2_noise_new = get_s2_noise_new(s2_noise, coords_new, mask, text_s2_flow_model.in_channels)
    mask_new = get_coords_mask(s2_noise_new.coords, mask)
    if torch.sum(mask_new) > 0:
        s2_soft_mask = get_s2_soft_mask(s2_noise_new, mask, sigma=eff_s2_sigma,
                                        contact_mask=contact_64)

    t_seq, t_pairs = get_times(steps, rescale_t, int_len, num_iter, False)
    sample = s2_noise_new
    cnt = 0
    for t_curr, t_prev in tqdm(t_pairs, desc="Sampling", disable=False):
        x_t = sample
        if t_curr > t_prev:
            if args['edit_type'] in ("Addition", "Modification", "HybridDeletion", "TextureOnly"):
                if _use_text_step(cnt, mode):
                    s2_text_sampler_params['cfg_strength'] = args['cfg_strength']
                    x_t_1 = RF_sample_once(text_s2_flow_model, x_t, t_curr, t_prev, inverse=False, cond=conds[args['s2_pos_cond']]['cond'], neg_cond=conds[args['s2_neg_cond']]['cond'], **s2_text_sampler_params)
                else:
                    s2_img_sampler_params['cfg_strength'] = 5.0
                    x_t_1 = RF_sample_once(img_s2_flow_model, x_t, t_curr, t_prev, inverse=False, cond=img_cond_new, neg_cond=img_null_cond, **s2_img_sampler_params)
            else:
                s2_text_sampler_params['cfg_strength'] = 0
                x_t_1 = RF_sample_once(text_s2_flow_model, x_t, t_curr, t_prev, inverse=False, cond=text_null_cond, neg_cond=text_null_cond, **s2_text_sampler_params)

            if torch.sum(mask_new) > 0:
                mask_ori = get_coords_mask(inverse_dict[f's2_{t_prev}'].coords, mask)
                x_t_1.feats[mask_new] = x_t_1.feats[mask_new]*s2_soft_mask + inverse_dict[f's2_{t_prev}'].feats[mask_ori]*(1-s2_soft_mask)
        else:
            s2_text_sampler_params['cfg_strength'] = 0
            x_t_1 = RF_sample_once(text_s2_flow_model, x_t, t_curr, t_prev, inverse=True, cond=conds["new_s2_cpl"]['cond'], neg_cond=text_null_cond, **s2_text_sampler_params)
        sample = x_t_1
        cnt += 1
    slat_new = sample * std + mean
    
    return {"slat": slat_new, "z_s_before": z_s, "z_s_after": z_s_new}
