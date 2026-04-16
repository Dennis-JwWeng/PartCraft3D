"""Lightweight dataset for pipeline_v3 PartVerse input data.

Replaces the heavy ``PartCraftDataset`` / ``HY3DPartDataset`` shim in
pipeline_v3.  Owns the authoritative schema for all three input data roots
and exposes a single loading / validation surface.


Complete Input Schema (three roots, five file types)
----------------------------------------------------

``images_root/<shard>/<obj_id>.npz``  -- images NPZ (rendered views + part meta)
    Used by: s1 (VLM overview), sq1 (gate_a), s4 (FLUX input), sq3 (gate_e), s6p (preview)

    split_mesh.json   -- part cluster metadata::

                            { "part_id_to_name": [str, ...],
                              "valid_clusters": {
                                  "part_N": {"part_ids": [int,...], "cluster_size": int},
                                  ...
                              } }

    transforms.json   -- Blender camera transforms for all 150 rendered frames::

                            { "frames": [
                                { "file_path": "000.png",
                                  "camera_angle_x": float,
                                  "transform_matrix": [[...4x4...]] }, ...
                            ] }

    {frame:03d}.png   -- RGBA rendered view (512x512), sparse subset of the
                         150 frames, e.g. 008/009/.../035/089/090/091/100.
    {frame:03d}.webp  -- WebP alternative (checked if .png absent for a key).


``mesh_root/<shard>/<obj_id>.npz``    -- mesh NPZ (geometry + annotation)
    Used by: s4 (part masking), s5b (deletion geometry)

    full.glb           -- Whole-object mesh (Y-up GLB, normalised via vd_scale/offset).
    part_{i}.glb       -- Per-part meshes (GLB), one per valid cluster.
    anno_info.json     -- Face-level annotation::

                             { "bboxes": [...], "ordered_faceid": [...],
                               "weights": [...], "ordered_face_label": [...],
                               "ordered_part_level": [...] }

    part_captions.json -- Part text captions for VLM (optional)::

                             { "<part_id_str>": ["caption1", "caption2"], ... }

    vd_scale           -- float64 (1,): normalisation scale (source -> VD space).
    vd_offset          -- float64 (3,): normalisation offset.


``slat_dir/<shard>/<obj_id>_coords.pt``  -- SLAT coords (Trellis structured latent)
``slat_dir/<shard>/<obj_id>_feats.pt``   -- SLAT feats
    Used by: s5_trellis_3d (TrellisRefiner.encode_object), s5b_deletion

    coords: int32 Tensor (N, 4)  -- col 0 = batch index (always 0),
                                    cols 1-3 = xyz voxel coords in [0, 63]
    feats:  float32 Tensor (N, F), F=8 typically -- per-voxel latent features.

    Note: SLAT is loaded directly by TrellisRefiner.encode_object via its own
    slat_dir path.  This dataset exposes path helpers and validates the files
    but does not load tensors itself (avoids hard torch dependency).


Public API
----------
:class:`PartVerseRecord`   -- per-object record (images NPZ + mesh NPZ, lazy)
:class:`PartVerseDataset`  -- factory with validate + SLAT path helpers
:class:`ObjReport`         -- validation result (errors / warnings lists)

Key methods by consumer
-----------------------
* s1 / sq1 / sq3  -- get_image_bytes(view_idx), get_image_pil(), get_transforms()
* s4 (FLUX)       -- get_image_bytes(spec.npz_view)  (npz_view always >= 0 in v3)
* s5b (deletion)  -- get_mesh_bytes('full.glb'), get_part_ids()
* validate        -- record.validate()  or  dataset.validate_object(..., slat_dir=...)
* SLAT paths      -- dataset.slat_pt_paths(shard, obj_id)

Thread safety
-------------
Each PartVerseRecord opens its NPZ lazily; call close() when done.
For concurrent use in ThreadPoolExecutor, create one record per thread
(load_object is cheap -- only parses JSON headers, no image decompression).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


import numpy as np

import re as _re


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass, field as _field


@_dataclass
class ObjReport:
    """Per-object validation result (errors are blocking, warnings are informational).

    Attributes
    ----------
    obj_id:
        Object UUID.
    errors:
        Blocking issues — the object **cannot** be processed.
    warnings:
        Non-blocking issues — the pipeline may still run but results may degrade.
    ok:
        ``True`` if no errors.
    """
    obj_id: str
    errors: list[str] = _field(default_factory=list)
    warnings: list[str] = _field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _check_mesh_npz(npz_path: str, rep: ObjReport) -> None:
    """Check mesh NPZ keys, dtypes, and JSON sub-fields."""
    path = Path(npz_path)
    if not path.is_file():
        rep.error(f"mesh NPZ missing: {path}")
        return
    try:
        npz = np.load(str(path), allow_pickle=True)
    except Exception as exc:
        rep.error(f"mesh NPZ unreadable: {exc}")
        return

    keys = set(npz.files)

    # full mesh
    if "full.glb" not in keys and "full.ply" not in keys:
        rep.error("mesh NPZ: missing full.glb and full.ply")
    else:
        fkey = "full.glb" if "full.glb" in keys else "full.ply"
        if npz[fkey].nbytes == 0:
            rep.error(f"mesh NPZ: {fkey} is empty")

    # per-part meshes
    part_keys = [k for k in keys if _re.match(r"^part_\d+\.glb$", k)]
    if not part_keys:
        rep.error("mesh NPZ: no part_{i}.glb keys found")
    else:
        empty = [k for k in part_keys if npz[k].nbytes == 0]
        if empty:
            rep.warn(f"mesh NPZ: empty part meshes: {empty}")

    # anno_info.json
    if "anno_info.json" not in keys:
        rep.error("mesh NPZ: missing anno_info.json")
    else:
        try:
            anno = json.loads(bytes(npz["anno_info.json"]).decode())
            for req in ("bboxes", "ordered_faceid", "ordered_face_label"):
                if req not in anno:
                    rep.error(f"mesh NPZ: anno_info.json missing key '{req}'")
        except Exception as exc:
            rep.error(f"mesh NPZ: anno_info.json parse error: {exc}")

    # part_captions.json (soft)
    if "part_captions.json" not in keys:
        rep.warn("mesh NPZ: missing part_captions.json (VLM prompts will lack captions)")
    else:
        try:
            caps = json.loads(bytes(npz["part_captions.json"]).decode())
            if not isinstance(caps, dict):
                rep.error("mesh NPZ: part_captions.json must be a dict")
            elif not caps:
                rep.warn("mesh NPZ: part_captions.json is empty")
        except Exception as exc:
            rep.error(f"mesh NPZ: part_captions.json parse error: {exc}")

    # vd_scale / vd_offset
    if "vd_scale" not in keys:
        rep.error("mesh NPZ: missing vd_scale")
    else:
        vs = npz["vd_scale"]
        if vs.shape != (1,):
            rep.error(f"mesh NPZ: vd_scale shape {vs.shape}, expected (1,)")
        elif float(vs[0]) <= 0:
            rep.error(f"mesh NPZ: vd_scale={vs[0]} must be positive")

    if "vd_offset" not in keys:
        rep.error("mesh NPZ: missing vd_offset")
    else:
        vo = npz["vd_offset"]
        if vo.shape != (3,):
            rep.error(f"mesh NPZ: vd_offset shape {vo.shape}, expected (3,)")

    npz.close()


def _check_images_npz(npz_path: str, rep: ObjReport) -> None:
    """Check images NPZ: split_mesh.json, transforms.json, view images."""
    path = Path(npz_path)
    if not path.is_file():
        rep.error(f"images NPZ missing: {path}")
        return
    try:
        npz = np.load(str(path), allow_pickle=True)
    except Exception as exc:
        rep.error(f"images NPZ unreadable: {exc}")
        return

    keys = set(npz.files)

    # split_mesh.json
    if "split_mesh.json" not in keys:
        rep.error("images NPZ: missing split_mesh.json")
    else:
        try:
            sm = json.loads(bytes(npz["split_mesh.json"]).decode())
            for req in ("part_id_to_name", "valid_clusters"):
                if req not in sm:
                    rep.error(f"images NPZ: split_mesh.json missing key '{req}'")
            if "valid_clusters" in sm and not sm["valid_clusters"]:
                rep.error("images NPZ: split_mesh.json has empty valid_clusters")
        except Exception as exc:
            rep.error(f"images NPZ: split_mesh.json parse error: {exc}")

    # transforms.json (soft)
    if "transforms.json" not in keys:
        rep.warn("images NPZ: missing transforms.json (overview render may fail)")

    # view images
    view_pat = _re.compile(r"^\d{3}\.(png|webp)$")
    view_keys = [k for k in keys if view_pat.match(k)]
    if not view_keys:
        rep.error("images NPZ: no view images (e.g. 008.png) found")
    else:
        empty = [k for k in view_keys if npz[k].nbytes == 0]
        if empty:
            rep.warn(f"images NPZ: empty view images: {empty[:5]}")

    npz.close()


def _check_slat(
    slat_dir: "Path | None",
    shard: str,
    obj_id: str,
    rep: ObjReport,
) -> None:
    """Check pre-encoded SLAT .pt files (coords + feats).

    Skipped silently if ``slat_dir`` is *None* or torch is unavailable.
    """
    if slat_dir is None:
        return
    try:
        import torch
    except ImportError:
        rep.warn("torch not available — SLAT check skipped")
        return

    slat_dir = Path(slat_dir)
    c_path = slat_dir / shard / f"{obj_id}_coords.pt"
    f_path = slat_dir / shard / f"{obj_id}_feats.pt"

    if not c_path.is_file():
        rep.error(f"SLAT missing: {c_path}")
        return
    if not f_path.is_file():
        rep.error(f"SLAT missing: {f_path}")
        return

    try:
        coords = torch.load(str(c_path), map_location="cpu", weights_only=True)
        feats  = torch.load(str(f_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        rep.error(f"SLAT unreadable: {exc}")
        return

    if coords.dtype.is_floating_point:
        rep.warn(f"SLAT coords dtype={coords.dtype}, expected int32")
    if not feats.is_floating_point():
        rep.warn(f"SLAT feats dtype={feats.dtype}, expected float32")

    if coords.ndim != 2 or coords.shape[1] != 4:
        rep.error(f"SLAT coords shape={tuple(coords.shape)}, expected (N, 4)")
        return
    if feats.ndim != 2:
        rep.error(f"SLAT feats shape={tuple(feats.shape)}, expected (N, F)")
        return

    N_c, N_f = coords.shape[0], feats.shape[0]
    if N_c != N_f:
        rep.error(f"SLAT N mismatch: coords={N_c}, feats={N_f}")
        return
    if N_c == 0:
        rep.error("SLAT is empty (N=0)")
        return

    vmin = int(coords[:, 1:].min())
    vmax = int(coords[:, 1:].max())
    if vmin < 0 or vmax > 63:
        rep.error(f"SLAT coords xyz out of [0,63]: min={vmin} max={vmax}")

    batch_vals = coords[:, 0].unique().tolist()
    if batch_vals != [0]:
        rep.warn(f"SLAT coords batch col has unexpected values: {batch_vals}")


# ---------------------------------------------------------------------------
# PartInfo — mirrors partcraft_loader.PartInfo for drop-in compatibility
# ---------------------------------------------------------------------------

@dataclass
class PartInfo:
    """Metadata for one part cluster."""
    part_id: int
    cluster_name: str
    mesh_node_names: list[str]
    cluster_size: int


# ---------------------------------------------------------------------------
# PartVerseRecord — lazy, NPZ-backed object record
# ---------------------------------------------------------------------------

@dataclass
class PartVerseRecord:
    """Lazy-loaded record for one PartVerse object.

    Opened from two NPZ files:

    * ``images_npz_path`` — rendered views + camera metadata
    * ``mesh_npz_path``   — part meshes + annotation + captions

    Both files are opened on first access and closed by :meth:`close`.

    Attributes
    ----------
    obj_id, shard:
        Object identity.
    num_views:
        Number of rendered views present in the images NPZ.
    view_indices:
        Sorted list of absolute frame indices available as images
        (e.g. ``[8, 9, 10, 11, 23, ..., 89, 90, 91, 100]``).
    parts:
        Part cluster list, compatible with ``partcraft_loader.PartInfo``.
    part_id_to_name:
        Raw list[str] from split_mesh.json (index = part_id).
    part_captions:
        Mapping part_id (int) -> list[str] captions from part_captions.json.
        Empty dict if the key is absent in the mesh NPZ.
    vd_scale, vd_offset:
        VD-space normalisation coefficients from the mesh NPZ.
        Used by callers that load mesh geometry.
    """
    obj_id: str
    shard: str
    images_npz_path: str
    mesh_npz_path: str

    # Populated by __post_init__ (from NPZ headers — cheap, no decompression)
    num_views: int = 0
    view_indices: list[int] = field(default_factory=list)
    parts: list[PartInfo] = field(default_factory=list)
    part_id_to_name: list[str] = field(default_factory=list)
    part_captions: dict[int, list[str]] = field(default_factory=dict)
    vd_scale: float = 1.0
    vd_offset: "np.ndarray | None" = None

    # Lazy file handles
    _images_npz: "np.lib.npyio.NpzFile | None" = field(default=None, repr=False)
    _mesh_npz:   "np.lib.npyio.NpzFile | None" = field(default=None, repr=False)

    # ------------------------------------------------------------------ init

    def __post_init__(self) -> None:
        """Read NPZ headers to populate metadata (no image decompression)."""
        self._load_headers()

    def _load_headers(self) -> None:
        """Parse split_mesh.json + transforms.json + mesh header (fast path)."""
        # --- images NPZ ------------------------------------------------
        inpz = np.load(self.images_npz_path, allow_pickle=True)
        try:
            sm: dict = json.loads(bytes(inpz["split_mesh.json"]).decode())
            self.part_id_to_name = sm.get("part_id_to_name", [])
            valid_clusters: dict = sm.get("valid_clusters", {})
            self.parts = [
                PartInfo(
                    part_id=int(name.split("_")[-1]),
                    cluster_name=name,
                    mesh_node_names=[
                        self.part_id_to_name[i]
                        for i in info.get("part_ids", [])
                        if i < len(self.part_id_to_name)
                    ],
                    cluster_size=info.get("cluster_size", 0),
                )
                for name, info in valid_clusters.items()
            ]

            # Sparse view indices: keys like "089.png", "008.webp"
            self.view_indices = sorted(
                int(k.split(".")[0])
                for k in inpz.keys()
                if k.split(".")[0].isdigit()
                and k.endswith((".png", ".webp"))
            )
            self.num_views = len(self.view_indices)
        finally:
            inpz.close()

        # --- mesh NPZ --------------------------------------------------
        mnpz = np.load(self.mesh_npz_path, allow_pickle=True)
        try:
            if "vd_scale" in mnpz.files:
                self.vd_scale = float(mnpz["vd_scale"][0])
            if "vd_offset" in mnpz.files:
                self.vd_offset = np.array(mnpz["vd_offset"], dtype=np.float64)
            if "part_captions.json" in mnpz.files:
                raw: dict = json.loads(bytes(mnpz["part_captions.json"]).decode())
                self.part_captions = {int(k): v for k, v in raw.items()}
        finally:
            mnpz.close()

    # ------------------------------------------------------------------ lazy

    def _ensure_images(self) -> "np.lib.npyio.NpzFile":
        if self._images_npz is None:
            self._images_npz = np.load(self.images_npz_path, allow_pickle=True)
        return self._images_npz

    def _ensure_mesh(self) -> "np.lib.npyio.NpzFile":
        if self._mesh_npz is None:
            self._mesh_npz = np.load(self.mesh_npz_path, allow_pickle=True)
        return self._mesh_npz

    def validate(self) -> "ObjReport":
        """Validate both input NPZs for this object.

        Checks structural integrity of the images NPZ and mesh NPZ (file
        presence, required JSON sub-fields, vd_scale/vd_offset, part meshes).
        SLAT is not checked here — use :meth:`PartVerseDataset.validate_object`
        if you also need SLAT validation.

        Returns
        -------
        ObjReport
            ``.ok`` is ``True`` if no blocking errors were found.
        """
        rep = ObjReport(obj_id=self.obj_id)
        _check_images_npz(self.images_npz_path, rep)
        _check_mesh_npz(self.mesh_npz_path, rep)
        return rep

    def close(self) -> None:
        """Close open NPZ file handles and clear lazy caches."""
        if self._images_npz is not None:
            self._images_npz.close()
            self._images_npz = None
        if self._mesh_npz is not None:
            self._mesh_npz.close()
            self._mesh_npz = None

    # ------------------------------------------------------------------ image

    def get_image_bytes(self, view_idx: int) -> bytes:
        """Return raw PNG/WebP bytes for the given absolute frame index.

        Parameters
        ----------
        view_idx:
            Absolute frame index, e.g. ``89`` for the first standard view.
            Must be present in :attr:`view_indices`.

        Raises
        ------
        KeyError
            If ``view_idx`` is not in the images NPZ.
        """
        npz = self._ensure_images()
        for ext in (".png", ".webp"):
            key = f"{view_idx:03d}{ext}"
            if key in npz:
                return bytes(npz[key])
        raise KeyError(
            f"No image for view {view_idx} in {self.images_npz_path}. "
            f"Available: {self.view_indices}"
        )

    def get_image_pil(self, view_idx: int) -> "Image.Image":
        """Return PIL Image for the given view index."""
        from PIL import Image
        return Image.open(io.BytesIO(self.get_image_bytes(view_idx)))

    # ------------------------------------------------------------------ camera

    def get_transforms(self) -> dict:
        """Return the full transforms.json dict (all 150 frames).

        Structure::

            {
              "frames": [
                {"file_path": "000.png", "camera_angle_x": ...,
                 "transform_matrix": [[...], ...]},
                ...
              ]
            }
        """
        npz = self._ensure_images()
        return json.loads(bytes(npz["transforms.json"]).decode())

    # ------------------------------------------------------------------ mesh

    def get_mesh_bytes(self, key: str) -> bytes:
        """Return raw GLB bytes for a mesh entry (e.g. ``'full.glb'``, ``'part_0.glb'``).

        Parameters
        ----------
        key:
            NPZ key of the mesh to load.

        Raises
        ------
        KeyError
            If the key is absent from the mesh NPZ.
        """
        npz = self._ensure_mesh()
        if key not in npz.files:
            raise KeyError(f"{key!r} not in mesh NPZ {self.mesh_npz_path}")
        return bytes(npz[key])

    def get_part_ids(self) -> list[int]:
        """Return sorted list of available part IDs from the mesh NPZ."""
        import re
        npz = self._ensure_mesh()
        pat = re.compile(r"^part_(\d+)\.glb$")
        ids = [int(m.group(1)) for k in npz.files if (m := pat.match(k))]
        return sorted(ids)

    # ------------------------------------------------------------------ anno

    def get_anno_info(self) -> dict:
        """Return anno_info.json dict (face-level annotation).

        Keys: ``bboxes``, ``ordered_faceid``, ``weights``,
              ``ordered_face_label``, ``ordered_part_level``.
        """
        npz = self._ensure_mesh()
        return json.loads(bytes(npz["anno_info.json"]).decode())

    def get_best_view_for_parts(self, part_ids: list[int]) -> int:
        """Select the best view for a set of parts from the available frames.

        Strategy (no pyrender required):

        1. If the images NPZ contains a pre-computed ``{idx:03d}_mask.npy``,
           count pixels per part and return the view with most pixels.
        2. Otherwise fall back to the VIEW_INDICES mapping:
           pick the first of ``VIEW_INDICES`` that appears in
           :attr:`view_indices`; this matches the VLM's natural choice.

        This method is called only when ``spec.npz_view < 0`` (rare in
        pipeline_v3 because ``EditSpec.from_parsed_edit`` always resolves
        ``npz_view`` via ``VIEW_INDICES``).

        Parameters
        ----------
        part_ids:
            Target part IDs to maximise visibility for.
        """
        npz = self._ensure_images()

        # --- Path 1: pre-rendered masks in NPZ ---
        best_view, best_count = self.view_indices[0] if self.view_indices else 89, 0
        found_mask = False
        for vi in self.view_indices:
            mkey = f"{vi:03d}_mask.npy"
            if mkey in npz:
                found_mask = True
                mask = np.array(npz[mkey])
                count = int(sum(np.sum(mask == pid) for pid in part_ids))
                if count > best_count:
                    best_count = count
                    best_view = vi

        if found_mask:
            return best_view

        # --- Path 2: VIEW_INDICES preference order ---
        from partcraft.pipeline_v3.s1_vlm_core import VIEW_INDICES as _VI
        view_set = set(self.view_indices)
        for vi in _VI:
            if vi in view_set:
                return vi
        return self.view_indices[0] if self.view_indices else 89


# ---------------------------------------------------------------------------
# PartVerseDataset — factory
# ---------------------------------------------------------------------------

_SENTINEL = object()  # sentinel for validate_object slat_dir default


class PartVerseDataset:
    """Factory for :class:`PartVerseRecord` objects.

    Parameters
    ----------
    images_root:
        Root of the images NPZs: ``<images_root>/<shard>/<obj_id>.npz``.
    mesh_root:
        Root of the mesh NPZs: ``<mesh_root>/<shard>/<obj_id>.npz``.
    shards:
        Shard subdirectory names to include (e.g. ``["08"]``).
        If *None*, all numeric subdirectories under ``images_root`` are used.

    Example
    -------
    ::

        dataset = PartVerseDataset(
            images_root="/mnt/zsn/data/partverse/bench/inputs/images",
            mesh_root="/mnt/zsn/data/partverse/bench/inputs/mesh",
            shards=["08"],
        )
        rec = dataset.load_object("08", "c3d88711e2f34164b1eb8803a3e2448a")
        img = rec.get_image_pil(89)   # view 89 = first standard pipeline view
        rec.close()
    """

    def __init__(
        self,
        images_root: str | Path,
        mesh_root: str | Path,
        shards: list[str] | None = None,
        *,
        slat_dir: "str | Path | None" = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.mesh_root = Path(mesh_root)
        self.slat_dir: Path | None = Path(slat_dir) if slat_dir else None
        if shards is None:
            shards = sorted(
                d.name for d in self.images_root.iterdir()
                if d.is_dir() and d.name.isdigit()
            )
        self.shards = shards
        self._index: list[tuple[str, str]] | None = None

    # ------------------------------------------------------------------ index

    def _build_index(self) -> None:
        self._index = []
        for shard in self.shards:
            img_dir  = self.images_root / shard
            mesh_dir = self.mesh_root   / shard
            if not img_dir.exists():
                continue
            for f in sorted(img_dir.iterdir()):
                if f.suffix != ".npz":
                    continue
                obj_id = f.stem
                if (mesh_dir / f.name).exists():
                    self._index.append((shard, obj_id))

    def __len__(self) -> int:
        if self._index is None:
            self._build_index()
        return len(self._index)

    def __iter__(self) -> Iterator[PartVerseRecord]:
        if self._index is None:
            self._build_index()
        for shard, obj_id in self._index:
            rec = self.load_object(shard, obj_id)
            try:
                yield rec
            finally:
                rec.close()

    # ------------------------------------------------------------------ load

    def load_object(self, shard: str, obj_id: str) -> PartVerseRecord:
        """Load and return a :class:`PartVerseRecord` for one object.

        Both NPZ files must exist.  Metadata (split_mesh.json, view keys,
        vd_scale, part_captions) are parsed immediately; raw image/mesh
        bytes are loaded lazily on first access.

        Parameters
        ----------
        shard:
            Two-digit shard string, e.g. ``"08"``.
        obj_id:
            Object UUID.

        Returns
        -------
        PartVerseRecord
            Caller is responsible for calling ``.close()`` when done.
        """
        img_path  = str(self.images_root / shard / f"{obj_id}.npz")
        mesh_path = str(self.mesh_root   / shard / f"{obj_id}.npz")
        return PartVerseRecord(
            obj_id=obj_id,
            shard=shard,
            images_npz_path=img_path,
            mesh_npz_path=mesh_path,
        )


    def slat_pt_paths(self, shard: str, obj_id: str) -> tuple[Path, Path]:
        """Return (coords_pt, feats_pt) paths for one object.

        Layout: ``<slat_dir>/<shard>/<obj_id>_{coords,feats}.pt``

        Consumed by TrellisRefiner.encode_object (s5_trellis_3d) and by
        s5b_deletion._write_before_npz.  Provides path resolution without
        loading tensors (no torch dependency in this module).

        Raises
        ------
        RuntimeError
            If the dataset was constructed without a ``slat_dir``.
        """
        if self.slat_dir is None:
            raise RuntimeError(
                "PartVerseDataset was created without slat_dir -- "
                "pass slat_dir= to the constructor to use slat_pt_paths()"
            )
        base = self.slat_dir / shard / obj_id
        return (
            Path(f"{base}_coords.pt"),
            Path(f"{base}_feats.pt"),
        )

    def validate_object(
        self,
        shard: str,
        obj_id: str,
        slat_dir: "Path | str | None" = _SENTINEL,
    ) -> "ObjReport":
        """Validate all input artefacts for a single object.

        Checks the images NPZ and mesh NPZ via :meth:`PartVerseRecord.validate`,
        and optionally the SLAT ``.pt`` files if *slat_dir* is provided.

        Parameters
        ----------
        shard:
            Two-digit shard string, e.g. ``"08"``.
        obj_id:
            Object UUID.
        slat_dir:
            Root directory for SLAT tensors: ``<slat_dir>/<shard>/<obj_id>_coords.pt``.
            Pass *None* to skip SLAT validation.

        Returns
        -------
        ObjReport
        """
        if slat_dir is _SENTINEL:
            slat_dir = self.slat_dir
        img_path  = str(self.images_root / shard / f"{obj_id}.npz")
        mesh_path = str(self.mesh_root   / shard / f"{obj_id}.npz")
        rep = ObjReport(obj_id=obj_id)
        _check_images_npz(img_path, rep)
        _check_mesh_npz(mesh_path, rep)
        _check_slat(Path(slat_dir) if slat_dir else None, shard, obj_id, rep)
        return rep

    def validate_batch(
        self,
        obj_ids: "list[str]",
        shard: str,
        slat_dir: "Path | str | None" = None,
        *,
        verbose: bool = False,
        print_fn: "callable[[str], None] | None" = print,
    ) -> "list[ObjReport]":
        """Validate a list of objects and print a progress table.

        Parameters
        ----------
        obj_ids:
            Object UUIDs to validate.
        shard:
            Two-digit shard string, e.g. ``"08"``.
        slat_dir:
            If provided, also validates SLAT tensors.
        verbose:
            Show warnings even for passing objects.
        print_fn:
            Function used to emit progress lines.  Pass ``None`` to suppress
            all output.

        Returns
        -------
        list[ObjReport]
            One entry per object in *obj_ids* order.
        """
        _pr = print_fn or (lambda _: None)
        n = len(obj_ids)
        reports: list[ObjReport] = []
        for i, obj_id in enumerate(obj_ids, 1):
            rep = self.validate_object(shard, obj_id, slat_dir=slat_dir)
            reports.append(rep)
            status = "OK " if rep.ok else "ERR"
            w_note = f"  ({len(rep.warnings)} warn)" if rep.warnings else ""
            _pr(f"[{i:3d}/{n}] {status} {obj_id}{w_note}")
            if not rep.ok or (verbose and rep.warnings):
                for e in rep.errors:
                    _pr(f"         ERROR: {e}")
                for w in rep.warnings:
                    _pr(f"         WARN : {w}")
        return reports


# ---------------------------------------------------------------------------
# Drop-in alias for legacy callers
# ---------------------------------------------------------------------------

#: Alias kept for callers that still import the old name.
HY3DPartDataset = PartVerseDataset


__all__ = [
    "ObjReport",
    "PartInfo",
    "PartVerseRecord",
    "PartVerseDataset",
    "HY3DPartDataset",
]
