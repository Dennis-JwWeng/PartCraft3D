"""Phase 2.5: TRELLIS 3D editing following Vinedresser3D's pipeline exactly.

Replaces PartField segmentation with HY3D-Part ground-truth parts.
Rendering and SLAT encoding are done in prerender.py (decoupled).

Pipeline per edit spec:
  1. Load pre-encoded SLAT (from prerender.py)
  2. SLAT → Gaussian decoding
  3. Ground-truth part segmentation → voxel mask (replaces PartField)
  4. Render multiview from Gaussian → 2D image editing (Gemini)
  5. TRELLIS Flow Inversion + Repaint (Vinedresser3D)
  6. Gaussian Splatting export (PLY + video)

Prerequisite: Run scripts/prerender.py first to render + encode all objects.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt building (replaces VLM prompt generation)
# ---------------------------------------------------------------------------

# Adjectives that describe appearance (color, texture, material)
_APPEARANCE_WORDS = {
    # colors
    'red', 'blue', 'green', 'black', 'white', 'silver', 'gold', 'golden',
    'pink', 'brown', 'grey', 'gray', 'purple', 'orange', 'yellow', 'beige',
    'bronze', 'copper', 'chrome', 'ivory', 'crimson', 'burgundy', 'teal',
    'navy', 'maroon', 'cream', 'tan', 'dark', 'light', 'bright', 'pale',
    'deep', 'vibrant', 'matte', 'glossy', 'shiny', 'polished', 'brushed',
    'frosted', 'translucent', 'transparent', 'opaque', 'iridescent',
    # textures & materials
    'textured', 'smooth', 'rough', 'ridged', 'ribbed', 'grooved', 'knurled',
    'woven', 'striped', 'checkered', 'patterned', 'plain', 'solid',
    'metallic', 'wooden', 'plastic', 'rubber', 'leather', 'fabric',
    'ceramic', 'glass', 'chrome', 'steel', 'iron', 'aluminum', 'titanium',
    'marble', 'stone', 'concrete', 'brick', 'velvet', 'silk', 'linen',
    'cotton', 'denim', 'suede', 'fur', 'feathered', 'rustic', 'aged',
    'weathered', 'worn', 'faded', 'tarnished', 'corroded', 'oxidized',
}

# Adjectives that describe structure (shape, size, geometry)
_STRUCTURE_WORDS = {
    'round', 'rounded', 'circular', 'spherical', 'cylindrical', 'conical',
    'rectangular', 'square', 'triangular', 'hexagonal', 'octagonal',
    'curved', 'angular', 'pointed', 'blunt', 'flat', 'concave', 'convex',
    'tapered', 'flared', 'bulbous', 'elongated', 'stubby', 'slender',
    'wide', 'narrow', 'thick', 'thin', 'tall', 'short', 'long',
    'large', 'small', 'tiny', 'massive', 'compact', 'slim', 'bulky',
    'streamlined', 'aerodynamic', 'boxy', 'blocky', 'dome-shaped',
    'wedge-shaped', 'bell-shaped', 'teardrop', 'oval', 'oblong',
}


def _strip_words(text: str, words_to_strip: set[str]) -> str:
    """Remove specific adjectives from a description.

    Reproduces VD's decompose_prompt() logic without a VLM call:
    - For s1 (structure): strip appearance words
    - For s2 (appearance): strip structure words
    """
    tokens = text.split()
    filtered = []
    for tok in tokens:
        # Normalize for matching but keep original token
        clean = tok.strip('.,;:!?()[]"\'').lower()
        if clean not in words_to_strip:
            filtered.append(tok)
    result = ' '.join(filtered)
    # Clean up double spaces and dangling commas
    while '  ' in result:
        result = result.replace('  ', ' ')
    result = result.replace(' ,', ',').replace(' .', '.')
    return result.strip()


def _decompose_local(desc: str) -> tuple[str, str]:
    """Split a description into structure-only and appearance-only.

    Mirrors VD's decompose_prompt() without an LLM call.
    Returns (s1_structure, s2_appearance).
    """
    s1 = _strip_words(desc, _APPEARANCE_WORDS)  # remove appearance → structure
    s2 = _strip_words(desc, _STRUCTURE_WORDS)    # remove structure → appearance
    return s1, s2


def build_prompts_from_spec(spec) -> dict:
    """Build the prompt dict expected by interweave_Trellis_TI.

    Maps Phase 0/1 pre-computed descriptions to the format Vinedresser3D uses
    after its 7 VLM/LLM calls (obtain_overall_prompts, decompose_prompt,
    identify_ori_part, identify_new_part).

    Key difference from the original: decomposes descriptions into
    structure (s1) and appearance (s2) channels locally, matching VD's
    decompose_prompt() which uses an LLM to strip shape-adjectives
    for s2 and color/material-adjectives for s1.

    Supports all edit types from partcraft.edit_types:
      - modification/scale → Modification (S1+S2 repaint)
      - material           → TextureOnly (S2 only, part mask)
      - global             → TextureOnly (S2 only, full mask)
      - deletion           → Deletion (voxel removal)
      - addition           → Addition (S1+S2 generation)
      - identity           → not routed here (handled by pipeline)
    """
    from partcraft.edit_types import trellis_effective_type

    raw_type = spec.edit_type
    edit_type = trellis_effective_type(raw_type)

    obj_desc = spec.object_desc or ""
    after_desc = spec.after_desc or obj_desc
    edit_prompt = spec.edit_prompt or ""

    # Deletion now uses the "Deletion" path in interweave_Trellis_TI
    # which directly removes voxels (no S1 repaint), so the "smooth
    # closed surface" prompt enrichment is no longer needed.

    old_label = getattr(spec, 'old_label', '') or ''
    before_part = getattr(spec, 'before_part_desc', '') or old_label
    after_part = getattr(spec, 'after_part_desc', '') or old_label

    # Decompose complete descriptions into structure / appearance
    ori_s1_cpl, ori_s2_cpl = _decompose_local(obj_desc)
    new_s1_cpl, new_s2_cpl = _decompose_local(after_desc)

    # Decompose part-level descriptions
    ori_s1_part, ori_s2_part = _decompose_local(before_part)
    new_s1_part, new_s2_part = _decompose_local(after_part)

    return {
        "edit_prompt": edit_prompt,
        "edit_type": edit_type,         # TRELLIS-level type
        "raw_edit_type": raw_type,      # PartCraft-level type
        "editing_part": old_label,
        "target_part": after_part,
        # Complete descriptions (ori = before, new = after)
        "ori_cpl": obj_desc,
        "new_cpl": after_desc,
        # Structure descriptions (shape adjectives only)
        "ori_s1_cpl": ori_s1_cpl,
        "new_s1_cpl": new_s1_cpl,
        # Appearance descriptions (color/material adjectives only)
        "ori_s2_cpl": ori_s2_cpl,
        "new_s2_cpl": new_s2_cpl,
        # Part-level descriptions
        "ori_s1_part": ori_s1_part,
        "ori_s2_part": ori_s2_part,
        "new_part": after_part,
        "new_s1_part": new_s1_part,
        "new_s2_part": new_s2_part,
    }


# ---------------------------------------------------------------------------
# TrellisRefiner: full pipeline manager
# ---------------------------------------------------------------------------

class TrellisRefiner:
    """TRELLIS editing pipeline following Vinedresser3D exactly.

    Replaces only:
      - PartField clustering → HY3D-Part ground-truth part segmentation
      - VLM prompt generation → Phase 0/1 pre-computed prompts
    """

    def __init__(
        self,
        cache_dir: str | Path,
        device: str = "cuda",
        image_edit_model: str = "gemini-2.5-flash-image",
        ckpt_dir: str | None = None,
        image_edit_backend: str = "api",
        image_edit_base_url: str | None = None,
        debug: bool = False,
        slat_dir: str | Path | None = None,
        img_enc_dir: str | Path | None = None,
        # Legacy: ignored, kept for config compat
        vinedresser_path: str | None = None,
    ):
        self._project_root = Path(__file__).parents[2]
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self.image_edit_model = image_edit_model

        # Local image edit server (HTTP)
        self.image_edit_backend = image_edit_backend
        self.image_edit_base_url = image_edit_base_url

        # Checkpoint directory: default to PartCraft3D/checkpoints
        if ckpt_dir is None:
            ckpt_dir = str(self._project_root / "checkpoints")
        self.ckpt_dir = Path(ckpt_dir)

        # Data directories (SLAT, img_Enc) — config-driven, fallback to partobjaverse_tiny
        _default_data = self._project_root / "data" / "partobjaverse_tiny"
        self.slat_dir = Path(slat_dir) if slat_dir else _default_data / "slat"
        self.img_enc_dir = Path(img_enc_dir) if img_enc_dir else _default_data / "img_Enc"

        self.debug = debug or os.environ.get("PARTCRAFT_DEBUG", "").lower() in ("1", "true")

        self.trellis_text = None
        self.trellis_img = None

        # Ensure third_party/ is on sys.path for trellis, interweave, encode_asset
        third_party = str(self._project_root / "third_party")
        if third_party not in sys.path:
            sys.path.insert(0, third_party)

    # ---- Model loading ----

    def load_models(self):
        """Load TRELLIS pipelines from local checkpoints."""
        from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
        import trellis.models as trellis_models

        text_ckpt = str(self.ckpt_dir / "TRELLIS-text-xlarge")
        image_ckpt = str(self.ckpt_dir / "TRELLIS-image-large")

        logger.info(f"Loading TRELLIS text pipeline from {text_ckpt}...")
        self.trellis_text = TrellisTextTo3DPipeline.from_pretrained(text_ckpt)

        logger.info(f"Loading TRELLIS image pipeline from {image_ckpt}...")
        self.trellis_img = TrellisImageTo3DPipeline.from_pretrained(image_ckpt)

        # sparse_structure_encoder needed by interweave
        if 'sparse_structure_encoder' not in self.trellis_text.models:
            ss_enc_path = str(self.ckpt_dir / "TRELLIS-text-xlarge"
                              / "ckpts" / "ss_enc_conv3d_16l8_fp16")
            ss_encoder = trellis_models.from_pretrained(ss_enc_path)
            self.trellis_text.models['sparse_structure_encoder'] = ss_encoder

        self.trellis_text.cuda()
        self.trellis_img.cuda()
        logger.info("All models loaded")

    # ---- Step 1: Load pre-encoded SLAT ----

    def _find_slat_file(self, obj_id: str, suffix: str) -> str:
        """Locate a SLAT file, supporting both flat and sharded layouts.

        Flat:    slat_dir/{obj_id}_{suffix}.pt
        Sharded: slat_dir/{shard}/{obj_id}_{suffix}.pt
        """
        fname = f"{obj_id}_{suffix}.pt"
        flat = self.slat_dir / fname
        if flat.exists():
            return str(flat)
        # Search shard subdirectories (00, 01, ...)
        for entry in sorted(self.slat_dir.iterdir()):
            if entry.is_dir():
                candidate = entry / fname
                if candidate.exists():
                    return str(candidate)
        return str(flat)  # return flat path for error message

    def encode_object(self, glb_path: str, obj_id: str) -> Any:
        """Load pre-encoded SLAT for an object.

        Supports both flat (slat/{oid}_feats.pt) and sharded
        (slat/{shard}/{oid}_feats.pt) directory layouts.
        """
        from trellis.modules import sparse as sp

        feats_path = self._find_slat_file(obj_id, "feats")
        coords_path = self._find_slat_file(obj_id, "coords")

        if not os.path.exists(feats_path) or not os.path.exists(coords_path):
            raise FileNotFoundError(
                f"Pre-encoded SLAT not found for {obj_id}. "
                f"Run prerender.py first.\n"
                f"  Expected: {feats_path}")

        logger.info(f"Loading pre-encoded SLAT for {obj_id}")
        slat = sp.SparseTensor(
            feats=torch.load(feats_path, weights_only=True),
            coords=torch.load(coords_path, weights_only=True),
        )
        return slat

    # ---- Step 2: Decode SLAT → Gaussian ----

    def decode_to_gaussian(self, slat):
        """SLAT → Gaussian (same as main.py line 291)."""
        outputs = self.trellis_text.decode_slat(slat, ['gaussian'])
        return outputs['gaussian'][0]

    # ---- Step 3: Part mask (replaces PartField + VLM grounding) ----

    def build_part_mask(
        self,
        obj_id: str,
        obj_record,
        edit_part_ids: list[int],
        slat,
        edit_type: str = "Modification",
        large_part_threshold: float = 0.35,
    ) -> tuple[torch.Tensor, str]:
        """Build 64x64x64 voxel mask from HY3D-Part ground-truth parts.

        Replaces: PartField segmentation + VLM grounding + compute_editing_region
        (main.py lines 86-195).

        Maps HY3D-Part meshes from [-1, 1] space to VD's [-0.5, 0.5] space
        via bounding box alignment, then voxelizes using Open3D.

        Returns:
            (mask, effective_edit_type): The 64³ bool mask and the effective
            edit type — may differ from input if a large part (>threshold)
            is automatically promoted to Global.
        """
        device = self.device

        # Global: full mask, no part voxelization needed
        if edit_type == "Global":
            mask = torch.ones(64, 64, 64, device=device, dtype=torch.bool)
            logger.info("Global edit: full mask (64³ = 262144 voxels)")
            return mask, "Global"

        import open3d as o3d
        import tempfile

        # Load VD's normalized mesh for reference frame.
        # Prefer img_Enc/mesh.ply; fall back to full.ply packed in mesh NPZ
        # (mesh.ply is deleted after packing, but full.ply is equivalent).
        vd_mesh_path = str(self.img_enc_dir / obj_id / "mesh.ply")
        if os.path.exists(vd_mesh_path):
            vd_mesh = o3d.io.read_triangle_mesh(vd_mesh_path)
        else:
            npz = np.load(obj_record.mesh_npz_path, allow_pickle=False)
            if "full.ply" not in npz:
                raise FileNotFoundError(
                    f"VD mesh.ply not found at {vd_mesh_path} and "
                    f"full.ply not found in {obj_record.mesh_npz_path}.")
            _tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
            _tmp.write(npz["full.ply"].tobytes())
            _tmp.close()
            vd_mesh = o3d.io.read_triangle_mesh(_tmp.name)
            os.unlink(_tmp.name)
        vd_verts = np.asarray(vd_mesh.vertices)
        vd_center = (vd_verts.max(0) + vd_verts.min(0)) / 2
        vd_extent = (vd_verts.max(0) - vd_verts.min(0)).max()

        # Load HY3D-Part full mesh for bounding box
        hy3d_full = obj_record.get_full_mesh(colored=False)
        hy3d_verts = np.array(hy3d_full.vertices)
        hy3d_center = (hy3d_verts.max(0) + hy3d_verts.min(0)) / 2
        hy3d_extent = (hy3d_verts.max(0) - hy3d_verts.min(0)).max()

        # Transform: HY3D → VD space (uniform scaling + translation)
        scale_factor = vd_extent / hy3d_extent if hy3d_extent > 0 else 1.0
        logger.info(f"Part mask transform: scale={scale_factor:.4f}, "
                    f"hy3d_extent={hy3d_extent:.4f}, vd_extent={vd_extent:.4f}")

        # Merge group parts into combined meshes, then voxelize once each.
        # This treats multi-part groups as a single contiguous region,
        # producing accurate contact boundaries and consistent voxelization.
        import trimesh as _trimesh

        all_part_ids = [p.part_id for p in obj_record.parts]
        edit_set = set(edit_part_ids)
        edit_meshes = []
        preserved_meshes = []

        for pid in all_part_ids:
            try:
                part_mesh = obj_record.get_part_mesh(pid, colored=False)
            except KeyError:
                continue
            if pid in edit_set:
                edit_meshes.append(part_mesh)
            else:
                preserved_meshes.append(part_mesh)

        def _voxelize_combined(meshes: list) -> torch.Tensor:
            """Merge trimesh list → transform to VD space → voxelize once.

            SLAT coordinates are voxelized directly from VD space (no axis
            reorder).  VD's voxelize() clips mesh.ply to [-0.5, 0.5] and
            creates a 64³ grid — SLAT coords are those grid indices.
            """
            grid = torch.zeros(64, 64, 64, device=device, dtype=torch.bool)
            if not meshes:
                return grid
            combined = _trimesh.util.concatenate(meshes)
            verts = np.array(combined.vertices)
            verts_vd = (verts - hy3d_center) * scale_factor + vd_center
            # No axis reorder — SLAT coords are in VD space directly
            verts_vd = np.clip(verts_vd, -0.5 + 1e-6, 0.5 - 1e-6)

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_vd)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(
                np.array(combined.faces))
            try:
                vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                    o3d_mesh, voxel_size=1/64,
                    min_bound=(-0.5, -0.5, -0.5),
                    max_bound=(0.5, 0.5, 0.5))
                voxels = np.array([v.grid_index for v in vg.get_voxels()])
            except Exception:
                vg_grid = ((verts_vd + 0.5) * 64).astype(np.int32)
                voxels = np.unique(np.clip(vg_grid, 0, 63), axis=0)
            if len(voxels) > 0:
                vt = torch.from_numpy(voxels).long().to(device)
                grid[vt[:, 0], vt[:, 1], vt[:, 2]] = True
            return grid

        edit_parts = _voxelize_combined(edit_meshes)
        preserved_parts = _voxelize_combined(preserved_meshes)

        logger.info(f"Part mask (combined-voxelized): "
                    f"edit={int(edit_parts.sum())} voxels ({len(edit_meshes)} parts), "
                    f"preserved={int(preserved_parts.sum())} voxels ({len(preserved_meshes)} parts)")

        # ---- Align masks to SLAT coordinates ----
        # VD builds masks from SLAT coords directly (via PartField).
        # Mesh voxelization may not align with SLAT coords, causing
        # wrong voxels to be edited/preserved.  Re-label each SLAT
        # voxel based on the mesh-voxelized grid, using KNN for any
        # SLAT voxels that fall in neither edit_parts nor preserved_parts.
        edit_parts, preserved_parts = self._align_masks_to_slat(
            slat, edit_parts, preserved_parts)

        n_edit_slat = int(edit_parts.sum())
        n_pres_slat = int(preserved_parts.sum())
        logger.info(f"Part mask (SLAT-aligned): edit={n_edit_slat} voxels, "
                    f"preserved={n_pres_slat} voxels")

        # Warn if the edit part is too small to be reliably edited
        MIN_EDIT_VOXELS = 10
        if n_edit_slat < MIN_EDIT_VOXELS and edit_type != "Addition":
            logger.warning(
                f"Edit part has only {n_edit_slat} SLAT voxels "
                f"(< {MIN_EDIT_VOXELS}). Part may be too small for "
                f"reliable editing at 64³ resolution.")

        # ---- Large part auto-promotion to Global ----
        # If the edit part covers >threshold of SLAT voxels, part-level
        # del/mod will cause severe artifacts (too much geometry removed).
        # Promote to Global: full mask + Inverse Flow preserves structure,
        # TRELLIS regenerates the whole object guided by the edit prompt.
        n_slat_total = slat.coords.shape[0]
        if n_slat_total > 0 and edit_type in ("Modification", "Scale"):
            part_ratio = n_edit_slat / n_slat_total
            if part_ratio > large_part_threshold:
                logger.warning(
                    f"Large part detected: edit covers {part_ratio:.1%} of "
                    f"SLAT voxels (>{large_part_threshold:.0%} threshold). "
                    f"Promoting {edit_type} → Global to avoid severe artifacts.")
                edit_type = "Global"

        # ---- Build final mask per edit type ----
        # Strategy: keep the hard mask tight (just the edit part itself),
        # and rely on soft S1/S2 blending for smooth transitions.
        # TRELLIS generation fills the edit region; soft mask handles
        # boundary continuity without cutting into preserved geometry.
        if edit_type in ("Modification", "Scale"):
            # Expanded mask: edit_parts + surrounding empty space (pad=3).
            # S1 repaint needs room to generate a replacement part that
            # may differ in size/shape from the original.  Using only
            # edit_parts + 1-voxel dilation constrains S1 to regenerate
            # in the exact same footprint, producing near-identical output.
            mask = self._compute_editing_region(
                slat, edit_parts, preserved_parts, pad=3)
        elif edit_type == "Material":
            # Material: S2 only within part mask. Tight mask (no expansion)
            # since geometry is unchanged — only texture gets repainted.
            mask = edit_parts.clone()
            logger.info(f"Material mask: {int(mask.sum())} voxels "
                        f"(S2 texture-only, no geometry change)")
        elif edit_type == "Deletion":
            # Direct deletion: mask = exact edit part, no dilation needed.
            # Voxels in mask are directly removed from SLAT; remaining
            # voxels keep original features (including texture) unchanged.
            mask = edit_parts.clone()
            logger.info(f"Deletion mask: {int(mask.sum())} voxels "
                        f"(direct removal, no generation)")
        elif edit_type == "Addition":
            # Addition needs room for new geometry in empty space.
            mask = self._compute_editing_region(
                slat, edit_parts, preserved_parts, pad=3)
        elif edit_type == "Global":
            # Full mask — everything is editable. Flow Inversion
            # preserves the original structure through the inverted
            # noise trajectory; cfg_strength controls divergence.
            mask = torch.ones(64, 64, 64, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Invalid edit type: {edit_type}")

        # ---- Diagnostics ----
        slat_coords = slat.coords[:, 1:]
        coords_in_mask = mask[slat_coords[:, 0], slat_coords[:, 1], slat_coords[:, 2]]
        slat_in_edit = int(coords_in_mask.sum())
        slat_total = slat_coords.shape[0]
        total_64 = 64 ** 3
        logger.info(f"Final mask: {int(mask.sum())} voxels "
                    f"({int(mask.sum())/total_64*100:.1f}% of 64³), "
                    f"SLAT overlap: {slat_in_edit}/{slat_total} "
                    f"({slat_in_edit/slat_total*100:.1f}%), "
                    f"edit_type={edit_type}")
        if slat_in_edit == 0 and edit_type != "Addition":
            logger.warning("WARNING: No SLAT voxels overlap with mask! "
                           "Coordinate space may be misaligned.")

        # Debug: save mask projections + SLAT overlay
        if self.debug:
            self._save_mask_debug(mask, edit_parts, preserved_parts,
                                  obj_id, edit_part_ids, slat=slat)

        return mask, edit_type

    def _align_masks_to_slat(
        self, slat, edit_parts, preserved_parts,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-project mesh-voxelized masks onto SLAT coordinates.

        In Vinedresser3D, PartField clusters SLAT voxels directly, so
        masks are inherently aligned.  Here, mesh voxelization may not
        overlap with SLAT coords.  This method labels each SLAT voxel
        as edit/preserved using the mesh-voxelized grid, then uses KNN
        to assign any SLAT voxel that falls in neither.

        Returns new edit_parts and preserved_parts tensors built from
        SLAT coordinates only (like VD does from PartField).
        """
        from sklearn.neighbors import NearestNeighbors
        device = self.device
        sc = slat.coords[:, 1:]  # [N, 3] in [0, 63]
        n_slat = sc.shape[0]

        slat_is_edit = edit_parts[sc[:, 0], sc[:, 1], sc[:, 2]]
        slat_is_preserved = preserved_parts[sc[:, 0], sc[:, 1], sc[:, 2]]
        unassigned = ~(slat_is_edit | slat_is_preserved)

        n_edit = int(slat_is_edit.sum())
        n_pres = int(slat_is_preserved.sum())
        n_unas = int(unassigned.sum())
        logger.info(f"SLAT label check: edit={n_edit}, preserved={n_pres}, "
                    f"unassigned={n_unas} (of {n_slat} SLAT voxels)")

        # Assign unassigned SLAT voxels via KNN to nearest assigned voxel
        if n_unas > 0 and (n_edit + n_pres) > 0:
            assigned_mask = slat_is_edit | slat_is_preserved
            assigned_coords = sc[assigned_mask].cpu().numpy().astype(np.float32)
            assigned_is_edit = slat_is_edit[assigned_mask].cpu().numpy()
            unassigned_coords = sc[unassigned].cpu().numpy().astype(np.float32)

            nbrs = NearestNeighbors(
                n_neighbors=1, algorithm='ball_tree'
            ).fit(assigned_coords)
            _, indices = nbrs.kneighbors(unassigned_coords)
            nn_labels = assigned_is_edit[indices.flatten()]
            slat_is_edit[unassigned] = torch.from_numpy(nn_labels).to(device)
            slat_is_preserved[unassigned] = ~slat_is_edit[unassigned]

            logger.info(f"SLAT after KNN assignment: edit={int(slat_is_edit.sum())}, "
                        f"preserved={int(slat_is_preserved.sum())}")

        # Rebuild 64³ masks from SLAT coords (aligned like VD's PartField)
        edit_aligned = torch.zeros(64, 64, 64, device=device, dtype=torch.bool)
        preserved_aligned = torch.zeros(64, 64, 64, device=device, dtype=torch.bool)
        edit_sc = sc[slat_is_edit]
        pres_sc = sc[slat_is_preserved]
        if edit_sc.shape[0] > 0:
            edit_aligned[edit_sc[:, 0], edit_sc[:, 1], edit_sc[:, 2]] = True
        if pres_sc.shape[0] > 0:
            preserved_aligned[pres_sc[:, 0], pres_sc[:, 1], pres_sc[:, 2]] = True

        return edit_aligned, preserved_aligned

    def _dilate_mask(
        self, mask: torch.Tensor, preserved: torch.Tensor,
        slat, radius: int = 2,
        exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dilate mask by `radius` voxels along the boundary with preserved parts.

        Only dilates INTO preserved regions that are adjacent to the edit mask,
        creating a smooth transition zone.

        Args:
            mask: current edit mask (64³ bool tensor)
            preserved: preserved parts mask (64³ bool tensor)
            slat: SLAT object (used to limit dilation to occupied voxels)
            radius: number of voxels to dilate (default 2)
            exclude: voxels to never include in dilation result (64³ bool).
                     For deletion, this should be the non-contact edit voxels
                     to prevent dilation from expanding back into the part
                     being deleted.

        Returns:
            Dilated mask (64³ bool tensor)
        """
        from scipy import ndimage

        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Create a spherical structuring element for smooth dilation
        struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
        dilated = ndimage.binary_dilation(
            mask_np, structure=struct, iterations=radius)
        dilated_t = torch.from_numpy(dilated).to(mask.device)

        # The transition zone = dilated region that overlaps preserved parts
        # or SLAT-occupied voxels (don't expand into pure empty space far away)
        slat_occupied = torch.zeros(64, 64, 64, device=mask.device, dtype=torch.bool)
        sc = slat.coords[:, 1:]
        slat_occupied[sc[:, 0], sc[:, 1], sc[:, 2]] = True

        # For deletion: exclude non-contact edit voxels from slat_occupied
        # so dilation only expands into preserved geometry, not back into
        # the part being deleted.
        if exclude is not None:
            slat_occupied = slat_occupied & ~exclude

        # Final mask: original edit region + boundary transition zone
        # Transition = newly dilated voxels that are on occupied geometry
        transition = dilated_t & ~mask & (preserved | slat_occupied)
        result = mask | transition

        n_transition = int(transition.sum())
        if n_transition > 0:
            logger.info(f"Mask dilation: +{n_transition} transition voxels "
                        f"(radius={radius})")

        return result

    def _compute_editing_region(
        self, slat, edit_parts, preserved_parts, pad: int = 3,
    ) -> torch.Tensor:
        """Compute editing mask for Modification / Addition.

        Adapted from Vinedresser3D compute_editing_region with a key
        fix: removed ``mask | ~bbox_preserved`` which inflated the mask
        to 84-94% of 64³.  Instead, KNN expansion is limited to the
        object's SLAT bounding box + ``pad`` voxels.

        Args:
            pad: voxel margin around the SLAT bbox for KNN expansion.
                 Use 3 for Modification, 5 for Addition.
        """
        from sklearn.neighbors import NearestNeighbors
        device = self.device

        # Limit KNN expansion to the object's vicinity (SLAT bbox + margin)
        sc = slat.coords[:, 1:]
        slat_min = sc.min(dim=0)[0]
        slat_max = sc.max(dim=0)[0]
        obj_vicinity = torch.zeros(64, 64, 64, device=device, dtype=torch.bool)
        obj_vicinity[
            max(slat_min[0] - pad, 0):min(slat_max[0] + pad + 1, 64),
            max(slat_min[1] - pad, 0):min(slat_max[1] + pad + 1, 64),
            max(slat_min[2] - pad, 0):min(slat_max[2] + pad + 1, 64),
        ] = True

        mask = torch.zeros(64, 64, 64, device=device, dtype=torch.bool)
        # Only consider empty voxels within object vicinity
        empty_coords = torch.argwhere(
            ~(preserved_parts | edit_parts) & obj_vicinity)

        if len(empty_coords) > 0 and slat.coords.shape[0] > 0:
            k = min(100, slat.coords.shape[0])
            nbrs = NearestNeighbors(
                n_neighbors=k, algorithm='ball_tree'
            ).fit(slat.coords[:, 1:].cpu().numpy())
            distances, indices = nbrs.kneighbors(empty_coords.cpu().numpy())
            indices = torch.from_numpy(indices).to(device)
            neighbor_masks = edit_parts[
                slat.coords[indices, 1],
                slat.coords[indices, 2],
                slat.coords[indices, 3]]
            mask_proportions = neighbor_masks.float().mean(dim=1)
            mask[empty_coords[:, 0], empty_coords[:, 1],
                 empty_coords[:, 2]] = (mask_proportions > 0.5)

        mask = mask | edit_parts
        mask = mask & ~preserved_parts

        return mask

    def _save_mask_debug(self, mask, edit_parts, preserved_parts,
                         obj_id, edit_part_ids, slat=None):
        """Save mask projection images for debugging, with SLAT overlay."""
        debug_dir = self.cache_dir / "debug_masks"
        debug_dir.mkdir(parents=True, exist_ok=True)

        tag = f"{obj_id}_parts{'_'.join(map(str, edit_part_ids))}"
        scale = 4

        for name, tensor in [("mask", mask), ("edit", edit_parts),
                              ("preserved", preserved_parts)]:
            arr = tensor.cpu().numpy()
            proj_xy = arr.any(axis=2).astype(np.uint8) * 255
            proj_xz = arr.any(axis=1).astype(np.uint8) * 255
            proj_yz = arr.any(axis=0).astype(np.uint8) * 255

            w = proj_xy.shape[1] * scale
            canvas = Image.new("L", (w * 3 + 20, w), 0)
            for j, proj in enumerate([proj_xy, proj_xz, proj_yz]):
                p = Image.fromarray(proj).resize(
                    (proj.shape[1]*scale, proj.shape[0]*scale), Image.NEAREST)
                canvas.paste(p, (j * (w + 10), 0))
            canvas.save(str(debug_dir / f"{tag}_{name}.png"))

        # Save RGB overlay: red=mask, green=SLAT coords, yellow=overlap
        if slat is not None:
            slat_vol = torch.zeros(64, 64, 64, device=mask.device, dtype=torch.bool)
            sc = slat.coords[:, 1:]
            slat_vol[sc[:, 0], sc[:, 1], sc[:, 2]] = True

            mask_np = mask.cpu().numpy()
            slat_np = slat_vol.cpu().numpy()

            for axis, label in [(2, "xy"), (1, "xz"), (0, "yz")]:
                m_proj = mask_np.any(axis=axis)
                s_proj = slat_np.any(axis=axis)

                h, w_ = m_proj.shape
                rgb = np.zeros((h, w_, 3), dtype=np.uint8)
                rgb[m_proj & ~s_proj] = [255, 0, 0]      # red: mask only
                rgb[s_proj & ~m_proj] = [0, 255, 0]       # green: SLAT only
                rgb[m_proj & s_proj] = [255, 255, 0]      # yellow: overlap
                rgb[~m_proj & ~s_proj] = [30, 30, 30]     # dark: neither

                img = Image.fromarray(rgb).resize(
                    (w_ * scale, h * scale), Image.NEAREST)
                img.save(str(debug_dir / f"{tag}_overlay_{label}.png"))

    # ---- Step 4: Multi-view 2D image editing ----

    def obtain_edited_images(
        self,
        gaussian,
        prompts: dict,
        vlm_client,
        obj_id: str,
        edit_id: str = "",
        num_views: int = 4,
        edit_dir: str | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Get multiple edited images from different viewpoints.

        Lookup order:
          1. Pre-generated single image in {edit_dir}/{edit_id}_edited.png
          2. Cached multi-view images in {cache_dir}/2d_edits/
          3. Call VLM API to generate multi-view edits

        Args:
            num_views: Number of viewpoints to edit (default 4).
            edit_dir: Subdir name for pre-generated 2D edits (e.g. '2d_edits_action').

        Returns:
            (original_images, edited_images) — each a list of PIL images (518x518).
        """
        from trellis.utils import render_utils

        # ---- 1. Check pre-generated single image (from run_2d_edit.py) ----
        if edit_dir:
            pre_dir = self.cache_dir / edit_dir
            pre_edited = pre_dir / f"{edit_id}_edited.png"
            pre_input = pre_dir / f"{edit_id}_input.png"
            if pre_edited.exists():
                edited_img = Image.open(str(pre_edited)).resize((518, 518))
                if pre_input.exists():
                    input_img = Image.open(str(pre_input)).resize((518, 518))
                else:
                    # No input image cached — render front view from Gaussian
                    imgs = render_utils.Trellis_render_multiview_images(
                        gaussian, [0.0], [0.0])['color']
                    input_img = Image.fromarray(imgs[0]).resize((518, 518))
                logger.info(f"Using pre-generated 2D edit for {edit_id} "
                            f"from {pre_dir}")
                return [input_img], [edited_img]

        views_dir = self.cache_dir / "2d_edits"
        if self.debug:
            views_dir.mkdir(parents=True, exist_ok=True)

        # ---- 2. Check cached multi-view images (only if debug saved them) ----
        if self.debug:
            cached_edited = []
            cached_original = []
            for v in range(num_views):
                p_edited = views_dir / f"{edit_id}_edited_v{v}.png"
                p_input = views_dir / f"{edit_id}_input_v{v}.png"
                if p_edited.exists() and p_input.exists():
                    cached_edited.append(Image.open(str(p_edited)).resize((518, 518)))
                    cached_original.append(Image.open(str(p_input)).resize((518, 518)))
            if len(cached_edited) == num_views:
                logger.info(f"Loading {num_views} cached edited images for {edit_id}")
                return cached_original, cached_edited

        # ---- 3. Call VLM API to generate multi-view edits ----
        if vlm_client is None:
            logger.warning(f"No pre-generated edit and no VLM client for {edit_id}")
            return [], []

        # Render N views from Gaussian with pitch variation
        # Even views: equator (pitch=0), odd views: elevated (pitch=0.4 ~23°)
        yaws_deg = np.linspace(0, 360, num_views, endpoint=False)
        yaws = (yaws_deg / 360 * 2 * np.pi).tolist()
        pitches = [0.0 if i % 2 == 0 else 0.4 for i in range(num_views)]

        imgs = render_utils.Trellis_render_multiview_images(
            gaussian, yaws, pitches)['color']

        original_images = []
        edited_images = []
        for v, img_arr in enumerate(imgs):
            img_pil = Image.fromarray(img_arr).resize(
                (518, 518), Image.Resampling.LANCZOS)
            original_images.append(img_pil)

            img_edited = self._call_vlm_edit(
                vlm_client, img_pil,
                prompts['edit_prompt'],
                prompts.get('new_part', ''),
                prompts.get('editing_part', ''))

            if img_edited is not None:
                img_edited = img_edited.resize(
                    (518, 518), Image.Resampling.LANCZOS)
                # Composite: keep original background, blend edit on foreground
                img_edited = self._composite_edit(img_pil, img_edited)
                edited_images.append(img_edited)
            else:
                logger.warning(f"2D edit failed for view {v}, using original")
                edited_images.append(img_pil)

            # Save debug views
            if self.debug:
                img_pil.save(str(views_dir / f"{edit_id}_input_v{v}.png"))
                if img_edited is not None:
                    edited_images[-1].save(
                        str(views_dir / f"{edit_id}_edited_v{v}.png"))

        logger.info(f"Generated {len(edited_images)} edited views for {edit_id}")
        return original_images, edited_images

    def encode_multiview_cond(
        self,
        edited_images: list[Image.Image],
        original_images: list[Image.Image] | None = None,
        edit_strength: float = 1.0,
    ) -> torch.Tensor:
        """Encode edited images with DINOv2.

        Default (edit_strength=1.0): uses edited features directly,
        same as original Vinedresser3D.  Background protection is
        already handled by _composite_edit() at the pixel level.

        When edit_strength < 1.0 and original_images is provided:
            feat = feat_ori + edit_strength * (feat_edited - feat_ori)
        This blends toward the original, weakening the edit signal.

        Returns:
            Averaged conditioning tensor [1, 1369, 1024].
        """
        pp_edited = [self.trellis_img.preprocess_image(img)
                     for img in edited_images]
        feat_edited = self.trellis_img.encode_image(pp_edited)
        feat_edited = feat_edited.mean(dim=0, keepdim=True)

        if original_images is not None:
            pp_orig = [self.trellis_img.preprocess_image(img)
                       for img in original_images]
            feat_orig = self.trellis_img.encode_image(pp_orig)
            feat_orig = feat_orig.mean(dim=0, keepdim=True)

            # Feature residual: original + alpha * delta
            feat_cond = feat_orig + edit_strength * (feat_edited - feat_orig)
            logger.info(
                f"DINOv2 feature residual: {len(edited_images)} views, "
                f"edit_strength={edit_strength}, "
                f"delta_norm={float((feat_edited - feat_orig).norm()):.2f}")
        else:
            feat_cond = feat_edited
            logger.info(
                f"DINOv2 features from {len(edited_images)} views "
                f"(no residual)")

        return feat_cond

    @staticmethod
    def _composite_edit(
        original: Image.Image, edited: Image.Image,
        bg_threshold: int = 245, blur_radius: int = 5,
    ) -> Image.Image:
        """Composite edited image back onto the original.

        Prevents VLM-introduced background changes and non-target-part
        modifications from leaking into DINOv2 conditioning.

        Uses foreground mask from the original render (Gaussian renders
        have near-white backgrounds).  The edited foreground is blended
        onto the original, keeping the background identical.
        """
        ori = np.array(original.convert('RGB')).astype(np.float32)
        edt = np.array(edited.convert('RGB')).astype(np.float32)

        # Build foreground mask from original (non-white pixels)
        fg_mask = ori.mean(axis=2) < bg_threshold  # True = foreground
        # Dilate slightly to catch edges
        from PIL import ImageFilter
        mask_img = Image.fromarray((fg_mask * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(blur_radius))
        # Smooth the mask edges for clean blending
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(blur_radius))
        blend_mask = np.array(mask_img).astype(np.float32) / 255.0

        # Composite: edited foreground + original background
        blend_3ch = blend_mask[:, :, np.newaxis]
        composited = edt * blend_3ch + ori * (1 - blend_3ch)

        return Image.fromarray(composited.astype(np.uint8))

    def _get_edit_server_url(self) -> str:
        """Return the base URL of the image edit HTTP server."""
        return self.image_edit_base_url or "http://localhost:8001"

    def _call_vlm_edit(
        self, client, img_pil: Image.Image,
        edit_prompt: str, new_part: str = "",
        editing_part: str = "",
    ) -> Image.Image | None:
        """Call VLM for 2D image editing.

        Dispatches to local diffusers pipeline or remote API based on
        self.image_edit_backend.
        """
        # Constrained prompt: explicitly restrict edits to target part
        target = f"the '{editing_part}' part" if editing_part else "the specified part"
        text_input = (
            f"This is a 3D rendered object on a white background. "
            f"Edit ONLY {target} of this object. "
            f"Editing instruction: {edit_prompt}"
        )
        if new_part:
            text_input += f"\nAfter editing, it should look like: {new_part}"
        text_input += (
            "\nIMPORTANT constraints:"
            "\n- Keep the exact same camera viewpoint and angle."
            "\n- Keep the white background completely unchanged."
            "\n- Keep ALL other parts of the object exactly as they are."
            "\n- Do NOT change the overall shape, pose, or position of the object."
            "\n- Only modify the appearance of the target part."
        )

        if self.image_edit_backend == "local_diffusers":
            return self._call_local_edit(img_pil, text_input)

        return self._call_api_edit(client, img_pil, text_input)

    def _call_local_edit(
        self, img_pil: Image.Image, prompt: str,
    ) -> Image.Image | None:
        """Edit image via local HTTP image edit server."""
        import json as _json
        import urllib.request

        url = self._get_edit_server_url()

        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = _json.dumps(
            {"image_b64": image_b64, "prompt": prompt}).encode()

        try:
            req = urllib.request.Request(
                f"{url}/edit", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = _json.loads(resp.read())

            if data.get("status") == "ok":
                img_data = base64.b64decode(data["image_b64"])
                return Image.open(io.BytesIO(img_data))
            else:
                logger.warning(
                    f"Edit server error: {data.get('msg', 'unknown')}")
                return None
        except Exception as e:
            logger.warning(f"Local image editing failed: {e}")
            return None

    def _call_api_edit(
        self, client, img_pil: Image.Image, text_input: str,
    ) -> Image.Image | None:
        """Edit image via OpenAI-compatible API (Gemini, etc.)."""
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        try:
            response = client.chat.completions.create(
                model=self.image_edit_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_input},
                        {"type": "image_url",
                         "image_url": {
                             "url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }],
            )
            msg = response.choices[0].message

            # Gemini style: message.images
            images = getattr(msg, 'images', None)
            if images:
                img0 = images[0]
                url = (img0['image_url']['url'] if isinstance(img0, dict)
                       else img0.image_url.url)
                img_data = base64.b64decode(url.split(",", 1)[1])
                return Image.open(io.BytesIO(img_data))

            # Fallback: content list
            for part in (msg.content if isinstance(msg.content, list) else []):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image"):
                        img_data = base64.b64decode(url.split(",", 1)[1])
                        return Image.open(io.BytesIO(img_data))

            # Fallback: string data URL
            content = msg.content
            if isinstance(content, str) and content.startswith("data:image"):
                img_data = base64.b64decode(content.split(",", 1)[1])
                return Image.open(io.BytesIO(img_data))

            return None

        except Exception as e:
            logger.warning(f"2D image editing failed: {e}")
            return None

    # ---- Step 5: TRELLIS editing ----

    @staticmethod
    def direct_delete_mesh(
        obj_record,
        remove_part_ids: list[int],
        output_dir: str | Path,
    ) -> dict:
        """Delete parts by assembling remaining GT meshes directly.

        No SLAT encoding, no generation — just removes the target parts
        from the ground-truth mesh and exports before/after PLY with
        original vertex colors preserved.

        Returns dict of exported file paths.
        """
        import trimesh as _trimesh

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        all_part_ids = [p.part_id for p in obj_record.parts]
        keep_ids = [pid for pid in all_part_ids
                    if pid not in set(remove_part_ids)]

        n_all = len(all_part_ids)
        n_remove = len(remove_part_ids)
        n_keep = len(keep_ids)
        logger.info(f"DirectDeletion (GT mesh): removing parts "
                    f"{remove_part_ids} ({n_remove}/{n_all}), "
                    f"keeping {n_keep} parts")

        # Before: full mesh with all parts
        before_mesh = obj_record.get_assembled_mesh(
            all_part_ids, colored=True)
        before_path = output_dir / "before.ply"
        before_mesh.export(str(before_path))
        paths['before_ply'] = str(before_path)

        # After: remaining parts only
        if keep_ids:
            after_mesh = obj_record.get_assembled_mesh(
                keep_ids, colored=True)
        else:
            logger.warning("DirectDeletion: all parts removed!")
            after_mesh = _trimesh.Trimesh()
        after_path = output_dir / "after.ply"
        after_mesh.export(str(after_path))
        paths['after_ply'] = str(after_path)

        logger.info(f"Exported GT mesh pair: before={before_path}, "
                    f"after={after_path}")
        return paths

    def edit(
        self,
        slat,
        mask: torch.Tensor,
        prompts: dict,
        img_cond: torch.Tensor | None = None,
        img_new: Image.Image | None = None,
        seed: int = 1,
        combinations: list[dict] | None = None,
    ) -> list:
        """Run TRELLIS Flow Inversion + Repaint.

        Same as Vinedresser3D main.py lines 320-348.
        Uses interweave_Trellis_TI directly.

        Args:
            img_cond: Pre-computed averaged DINOv2 conditioning [1, 1369, 1024]
                      from encode_multiview_cond(). Takes priority over img_new.
            img_new: Single PIL image (legacy fallback, used if img_cond is None).

        For Modification/Addition: alternates text/image conditioning.
        For Global: routed through TextureOnly — S1 is skipped entirely
            (original shape preserved), only S2 repaint changes texture.
        Note: Deletion is handled by direct_delete_mesh() using GT meshes,
            not by this method.
        """
        from interweave_Trellis import interweave_Trellis_TI
        from trellis.modules import sparse as sp
        from partcraft.edit_types import (
            MODIFICATION, SCALE, MATERIAL, GLOBAL,
            S1_S2_TYPES, S2_ONLY_TYPES, trellis_effective_type,
        )

        trellis_type = prompts.get('edit_type', 'Modification')
        raw_type = prompts.get('raw_edit_type', '')

        # --- Route edit types ---
        if combinations is None:
            if raw_type in S2_ONLY_TYPES or trellis_type == "TextureOnly":
                # Material (part-level) / Global (full mask): S2 only
                combinations = [
                    {"s1_pos_cond": "ori_s1_cpl", "s1_neg_cond": "ori_s1_cpl",
                     "s2_pos_cond": "new_s2_cpl", "s2_neg_cond": "ori_s2_cpl",
                     "cnt": 1, "cfg_strength": 5.0},
                ]
            else:
                # Modification / Scale / Addition: S1+S2 repaint
                combinations = [
                    {"s1_pos_cond": "new_s1_cpl", "s1_neg_cond": "ori_s1_cpl",
                     "s2_pos_cond": "new_s2_cpl", "s2_neg_cond": "ori_s2_cpl",
                     "cnt": 1, "cfg_strength": 7.5},
                ]

        # Map to interweave_Trellis_TI's understood types
        effective_edit_type = trellis_effective_type(raw_type) if raw_type else trellis_type

        # Setup image conditioning
        # Priority: img_cond (multi-view averaged) > img_new (single) > blank
        _patched = False
        if img_cond is not None:
            _orig_preprocess = self.trellis_img.preprocess_image
            _orig_get_cond = self.trellis_img.get_cond
            null_cond = torch.zeros_like(img_cond)
            self.trellis_img.preprocess_image = lambda x: x
            self.trellis_img.get_cond = lambda x: {
                "cond": img_cond, "neg_cond": null_cond}
            effective_img = Image.new("RGB", (518, 518), (255, 255, 255))
            _patched = True
            logger.info("Using multi-view averaged DINOv2 conditioning")
        elif img_new is not None:
            effective_img = img_new
        else:
            # S2 image model always needs a conditioning image;
            # blank white image produces neutral features.
            logger.info("No reference image — using blank white image")
            effective_img = Image.new("RGB", (518, 518), (255, 255, 255))

        try:
            slats_edited = []
            for i, combo in enumerate(combinations):
                args = {
                    'edit_type': effective_edit_type,
                    **combo,
                }
                logger.info(f"Running combination {i}/{len(combinations)}: "
                            f"edit_type={effective_edit_type}, "
                            f"cfg={combo['cfg_strength']}")

                slat_new = interweave_Trellis_TI(
                    args, self.trellis_text, self.trellis_img,
                    slat, mask, prompts, effective_img, seed=seed)
                slats_edited.append(slat_new)
        finally:
            if _patched:
                self.trellis_img.preprocess_image = _orig_preprocess
                self.trellis_img.get_cond = _orig_get_cond

        return slats_edited

    # ---- Step 6: Export ----

    def export_pair(
        self,
        slat_before,
        slat_edited,
        output_dir: str | Path,
    ) -> dict:
        """Export before/after pair as Gaussian PLY only.

        Multiview rendering and video generation are decoupled into
        scripts/vis/render_gs_pairs.py for faster editing throughput.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        for tag, slat in [('before', slat_before), ('after', slat_edited)]:
            gaussian = self.decode_to_gaussian(slat)

            # Save Gaussian PLY
            ply_path = output_dir / f"{tag}.ply"
            gaussian.save_ply(str(ply_path))
            paths[f'{tag}_ply'] = str(ply_path)

            # Save SLAT for visualization tool
            slat_path = output_dir / f"{tag}_slat"
            slat_path.mkdir(parents=True, exist_ok=True)
            torch.save(slat.feats, slat_path / "feats.pt")
            torch.save(slat.coords, slat_path / "coords.pt")
            paths[f'{tag}_slat'] = str(slat_path)

            logger.info(f"Exported {tag}: {ply_path}")

        return paths

    def export_pair_shared_before(
        self,
        slat_before,
        slat_edited,
        output_dir: str | Path,
        shared_before_dir: str | Path | None = None,
    ) -> dict:
        """Export before/after pair, reusing a shared 'before' directory.

        When multiple edits share the same original object, the 'before'
        PLY + SLAT only needs to be saved once. Pass ``shared_before_dir``
        pointing to the already-exported before data to create symlinks
        instead of re-exporting.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # ---- Before: reuse or export ----
        if shared_before_dir and Path(shared_before_dir).exists():
            src = Path(shared_before_dir)
            src_has_slat = (src / "before_slat").exists()

            if src_has_slat:
                # Source has GS before — symlink both PLY and SLAT
                before_ply = output_dir / "before.ply"
                if not before_ply.exists():
                    src_ply = src / "before.ply"
                    if src_ply.exists():
                        rel = os.path.relpath(src_ply, output_dir)
                        before_ply.symlink_to(rel)
                paths['before_ply'] = str(before_ply)

                before_slat = output_dir / "before_slat"
                if not before_slat.exists():
                    rel = os.path.relpath(src / "before_slat", output_dir)
                    before_slat.symlink_to(rel)
                paths['before_slat'] = str(before_slat)
            else:
                # Source has no SLAT (e.g. deletion used direct mesh
                # removal).  Export GS before PLY + SLAT from scratch.
                gaussian = self.decode_to_gaussian(slat_before)
                ply_path = output_dir / "before.ply"
                gaussian.save_ply(str(ply_path))
                paths['before_ply'] = str(ply_path)

                before_slat = output_dir / "before_slat"
                before_slat.mkdir(parents=True, exist_ok=True)
                torch.save(slat_before.feats, before_slat / "feats.pt")
                torch.save(slat_before.coords, before_slat / "coords.pt")
                paths['before_slat'] = str(before_slat)
        else:
            # First edit for this object: export before
            gaussian = self.decode_to_gaussian(slat_before)
            ply_path = output_dir / "before.ply"
            gaussian.save_ply(str(ply_path))
            paths['before_ply'] = str(ply_path)
            slat_path = output_dir / "before_slat"
            slat_path.mkdir(parents=True, exist_ok=True)
            torch.save(slat_before.feats, slat_path / "feats.pt")
            torch.save(slat_before.coords, slat_path / "coords.pt")
            paths['before_slat'] = str(slat_path)

        # ---- After: always export ----
        gaussian = self.decode_to_gaussian(slat_edited)
        ply_path = output_dir / "after.ply"
        gaussian.save_ply(str(ply_path))
        paths['after_ply'] = str(ply_path)
        slat_path = output_dir / "after_slat"
        slat_path.mkdir(parents=True, exist_ok=True)
        torch.save(slat_edited.feats, slat_path / "feats.pt")
        torch.save(slat_edited.coords, slat_path / "coords.pt")
        paths['after_slat'] = str(slat_path)

        logger.info(f"Exported: after={ply_path}, "
                    f"before={'shared' if shared_before_dir else 'new'}")
        return paths

    # ---- Utility ----

    @staticmethod
    def extract_glb(mesh_zip_path: str, obj_id: str, out_dir: str) -> str:
        """Extract a single GLB from source/mesh.zip."""
        with zipfile.ZipFile(mesh_zip_path) as zf:
            matches = [n for n in zf.namelist()
                       if obj_id in n and n.endswith('.glb')]
            if not matches:
                raise FileNotFoundError(
                    f"GLB for {obj_id} not found in {mesh_zip_path}")
            out_path = os.path.join(out_dir, f"{obj_id}.glb")
            with zf.open(matches[0]) as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
            return out_path
