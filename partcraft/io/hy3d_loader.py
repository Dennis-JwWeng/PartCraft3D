"""Loader for HY3D-Part dataset (images NPZ + mesh NPZ).

HY3D-Part stores watertight meshes as geometry-only PLY (no vertex color).
Colors exist only in the pre-rendered multi-view RGBA images (42 views).

This loader provides `bake_vertex_colors()` to project colors from rendered
views back onto mesh vertices using the stored camera transforms.
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class PartInfo:
    """Metadata for a single part cluster."""
    part_id: int              # cluster index (0-based)
    cluster_name: str         # e.g. "part_0"
    mesh_node_names: list[str]  # raw mesh node names from part_id_to_name
    cluster_size: int         # face count


@dataclass
class ObjectRecord:
    """All data associated with one object in HY3D-Part."""
    obj_id: str               # filename stem, e.g. "0000c32fde7f45efb8d14e8ba737d50c"
    shard: str                # e.g. "00"
    image_npz_path: str
    mesh_npz_path: str
    num_views: int = 0
    parts: list[PartInfo] = field(default_factory=list)
    part_id_to_name: list[str] = field(default_factory=list)

    # Lazily loaded
    _image_npz: np.lib.npyio.NpzFile | None = field(default=None, repr=False)
    _mesh_npz: np.lib.npyio.NpzFile | None = field(default=None, repr=False)

    def _ensure_image_npz(self):
        if self._image_npz is None:
            self._image_npz = np.load(self.image_npz_path, allow_pickle=True)

    def _ensure_mesh_npz(self):
        if self._mesh_npz is None:
            self._mesh_npz = np.load(self.mesh_npz_path, allow_pickle=True)

    def close(self):
        if self._image_npz is not None:
            self._image_npz.close()
            self._image_npz = None
        if self._mesh_npz is not None:
            self._mesh_npz.close()
            self._mesh_npz = None

    # ---- Image access ----

    def get_image_bytes(self, view_idx: int) -> bytes:
        """Return raw WebP bytes for a given view."""
        self._ensure_image_npz()
        key = f"{view_idx:03d}.webp"
        return self._image_npz[key].tobytes()

    def get_image_pil(self, view_idx: int) -> "Image.Image":
        """Return PIL Image for a given view."""
        assert Image is not None, "Pillow is required"
        return Image.open(io.BytesIO(self.get_image_bytes(view_idx)))

    def get_mask(self, view_idx: int) -> np.ndarray:
        """Return segmentation mask (H, W) int16 array. -1 = background."""
        self._ensure_image_npz()
        key = f"{view_idx:03d}_mask.npy"
        return self._image_npz[key]

    def get_transforms(self) -> dict:
        """Return camera transforms dict."""
        self._ensure_image_npz()
        return json.loads(self._image_npz["transforms.json"].tobytes().decode("utf-8"))

    # ---- Vertex color baking ----

    _baked_vertex_colors: dict[str, np.ndarray] | None = field(
        default=None, repr=False, init=False)

    def bake_vertex_colors(self, mesh: "trimesh.Trimesh",
                           view_step: int = 1) -> "trimesh.Trimesh":
        """Project colors from rendered views onto mesh vertices.

        HY3D-Part stores watertight meshes without vertex color. This method
        reprojects colors from the 42 pre-rendered RGBA views using the stored
        camera transforms.

        Args:
            mesh: Trimesh to colorize (vertices must be in the same coordinate
                  frame as the NPZ meshes).
            view_step: use every Nth view (1 = all 42, 2 = every other, etc.).

        Returns:
            The same mesh, with vertex_colors set (modified in-place).
        """
        assert Image is not None, "Pillow is required for vertex color baking"
        self._ensure_image_npz()

        transforms = self.get_transforms()
        frames = transforms["frames"]
        vertices = np.array(mesh.vertices, dtype=np.float64)
        n_verts = len(vertices)

        color_sum = np.zeros((n_verts, 3), dtype=np.float64)
        color_count = np.zeros(n_verts, dtype=np.float64)

        H = W = 518  # HY3D-Part render resolution
        verts_h = np.hstack([vertices, np.ones((n_verts, 1))])  # (N, 4)

        for i in range(0, len(frames), view_step):
            frame = frames[i]
            # Load RGBA image
            key = f"{i:03d}.webp"
            if key not in self._image_npz:
                continue
            img = Image.open(
                io.BytesIO(self._image_npz[key].tobytes())
            ).convert("RGBA").resize((W, H))
            img_arr = np.array(img)
            alpha = img_arr[:, :, 3].astype(np.float32) / 255.0
            rgb = img_arr[:, :, :3].astype(np.float32)

            # Camera: world-to-camera transform
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            fov = frame["camera_angle_x"]
            focal = (W / 2.0) / math.tan(fov / 2.0)

            # Project vertices → image pixels
            verts_cam = (w2c @ verts_h.T).T  # (N, 4)
            z = verts_cam[:, 2]
            in_front = z < 0  # OpenGL: camera looks down -Z

            z_safe = np.where(in_front, z, -1e-8)
            u = focal * verts_cam[:, 0] / (-z_safe) + W / 2.0
            v = focal * (-verts_cam[:, 1]) / (-z_safe) + H / 2.0

            u_safe = np.nan_to_num(u, nan=0.0, posinf=float(W), neginf=-1.0)
            v_safe = np.nan_to_num(v, nan=0.0, posinf=float(H), neginf=-1.0)
            u_int = np.clip(np.round(u_safe).astype(np.int32), 0, W - 1)
            v_int = np.clip(np.round(v_safe).astype(np.int32), 0, H - 1)

            in_bounds = in_front & (u_safe >= 0) & (u_safe < W) & (v_safe >= 0) & (v_safe < H)
            visible = in_bounds & (alpha[v_int, u_int] > 0.5)

            color_sum[visible] += rgb[v_int[visible], u_int[visible]]
            color_count[visible] += 1

        # Average colors
        has_color = color_count > 0
        result = np.full((n_verts, 4), 128, dtype=np.uint8)  # default gray
        result[:, 3] = 255
        if has_color.any():
            result[has_color, :3] = np.clip(
                color_sum[has_color] / color_count[has_color, None],
                0, 255).astype(np.uint8)

        mesh.visual.vertex_colors = result
        return mesh

    # ---- Mesh access ----

    def get_full_mesh(self, colored: bool = True) -> "trimesh.Trimesh":
        """Load the full (all parts assembled) watertight mesh.

        Args:
            colored: if True, bake vertex colors from rendered views.
        """
        assert trimesh is not None, "trimesh is required"
        self._ensure_mesh_npz()
        ply_bytes = self._mesh_npz["full.ply"].tobytes()
        mesh = trimesh.load(io.BytesIO(ply_bytes), file_type="ply")
        if colored:
            self.bake_vertex_colors(mesh)
        return mesh

    def get_part_mesh(self, part_id: int, colored: bool = True) -> "trimesh.Trimesh":
        """Load watertight mesh for a single part.

        Args:
            colored: if True, bake vertex colors from rendered views.
        """
        assert trimesh is not None, "trimesh is required"
        self._ensure_mesh_npz()
        key = f"part_{part_id}.ply"
        if key not in self._mesh_npz:
            raise KeyError(f"Part mesh '{key}' not found in {self.mesh_npz_path}")
        ply_bytes = self._mesh_npz[key].tobytes()
        mesh = trimesh.load(io.BytesIO(ply_bytes), file_type="ply")
        if colored:
            self.bake_vertex_colors(mesh)
        return mesh

    def get_assembled_mesh(self, part_ids: list[int],
                           colored: bool = True) -> "trimesh.Trimesh":
        """Load and concatenate meshes for specified parts.

        Args:
            colored: if True, bake vertex colors from rendered views.
        """
        assert trimesh is not None, "trimesh is required"
        meshes = []
        for pid in part_ids:
            try:
                meshes.append(self.get_part_mesh(pid, colored=False))
            except KeyError:
                continue
        if not meshes:
            raise ValueError(f"No valid meshes for parts {part_ids}")
        result = trimesh.util.concatenate(meshes)
        if colored:
            self.bake_vertex_colors(result)
        return result

    def get_mask_pixel_counts(self, view_idx: int) -> dict[int, int]:
        """Return {part_id: pixel_count} for a view's mask."""
        mask = self.get_mask(view_idx)
        unique, counts = np.unique(mask, return_counts=True)
        return {int(v): int(c) for v, c in zip(unique, counts) if v >= 0}

    def get_best_view_for_part(self, part_id: int) -> int:
        """Find the view where the given part has the most visible pixels."""
        best_view, best_count = 0, 0
        for v in range(self.num_views):
            mask = self.get_mask(v)
            count = int(np.sum(mask == part_id))
            if count > best_count:
                best_count = count
                best_view = v
        return best_view


class HY3DPartDataset:
    """Iterator over HY3D-Part dataset objects."""

    def __init__(self, image_npz_dir: str, mesh_npz_dir: str, shards: list[str] | None = None):
        self.image_npz_dir = Path(image_npz_dir)
        self.mesh_npz_dir = Path(mesh_npz_dir)

        if shards is None:
            # Auto-discover shards (directories that look like "00", "01", ...)
            shards = sorted(
                d.name for d in self.image_npz_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            )
        self.shards = shards
        self._index: list[tuple[str, str]] | None = None  # (shard, obj_id)

    def _build_index(self):
        """Build flat index of (shard, obj_id) pairs."""
        self._index = []
        for shard in self.shards:
            img_dir = self.image_npz_dir / shard
            mesh_dir = self.mesh_npz_dir / shard
            if not img_dir.exists():
                continue
            for f in sorted(img_dir.iterdir()):
                if f.suffix != ".npz":
                    continue
                obj_id = f.stem
                mesh_path = mesh_dir / f.name
                if mesh_path.exists():
                    self._index.append((shard, obj_id))

    def __len__(self) -> int:
        if self._index is None:
            self._build_index()
        return len(self._index)

    def __iter__(self) -> Iterator[ObjectRecord]:
        if self._index is None:
            self._build_index()
        for shard, obj_id in self._index:
            yield self.load_object(shard, obj_id)

    def load_object(self, shard: str, obj_id: str) -> ObjectRecord:
        """Load a single object record with metadata (does not load heavy data)."""
        img_path = str(self.image_npz_dir / shard / f"{obj_id}.npz")
        mesh_path = str(self.mesh_npz_dir / shard / f"{obj_id}.npz")

        # Parse split_mesh.json for part metadata
        img_npz = np.load(img_path, allow_pickle=True)
        sm = json.loads(img_npz["split_mesh.json"].tobytes().decode("utf-8"))
        part_id_to_name = sm["part_id_to_name"]
        valid_clusters = sm["valid_clusters"]

        parts = []
        for cluster_name, cluster_info in valid_clusters.items():
            pid = int(cluster_name.split("_")[-1])
            mesh_node_names = [
                part_id_to_name[i] for i in cluster_info["part_ids"]
                if i < len(part_id_to_name)
            ]
            parts.append(PartInfo(
                part_id=pid,
                cluster_name=cluster_name,
                mesh_node_names=mesh_node_names,
                cluster_size=cluster_info["cluster_size"],
            ))

        # Count views
        num_views = sum(1 for k in img_npz.keys() if k.endswith(".webp"))
        img_npz.close()

        return ObjectRecord(
            obj_id=obj_id,
            shard=shard,
            image_npz_path=img_path,
            mesh_npz_path=mesh_path,
            num_views=num_views,
            parts=parts,
            part_id_to_name=part_id_to_name,
        )
