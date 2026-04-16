"""Paths and per-object context for pipeline v3 (object-centric layout).

Layout::

    {root}/
      _global/
        manifest.jsonl              # one line per object
        run_config.yaml             # frozen run config (optional)
        report_full.html
      objects/<shard>/<obj_id>/
        meta.json
        status.json
        phase1/{overview.png, parsed.json, raw.txt}
        highlights/e{idx:02d}.png
        edits_2d/{edit_id}_{input,edited}.png
        edits_3d/<edit_id>/{before,after}.{npz,png}

Every step runner takes an :class:`ObjectContext` and only writes inside
``ctx.dir``. No global cache directories, no shard-wide files.

This module replaces the shard-centric :mod:`scripts.pipeline_paths`.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from partcraft.edit_types import EDIT_TYPE_PREFIX, FLUX_TYPES  # noqa: F401 — single source of truth


def normalize_shard(shard: str | int | None) -> str:
    """Pad shard id to 2 chars (``'1' -> '01'``)."""
    if shard is None or str(shard).strip() == "":
        raise ValueError("shard is required")
    return str(shard).strip().zfill(2)


DEFAULT_IMAGES_ROOT = "data/partverse/images"
DEFAULT_MESH_ROOT = "data/partverse/mesh"


@dataclass(frozen=True)
class DatasetRoots:
    """Canonical input dataset roots for pipeline_v3 (packed NPZ/PT trees).

    All four roots are resolved from the YAML ``data`` block and stored here
    as the single source of truth.  No step should call
    ``cfg["data"]["slat_dir"]`` directly — use :meth:`slat_pt_paths` instead.
    """

    images_root: Path
    mesh_root: Path
    slat_dir: Path | None = None       # bench/inputs/slat  (sharded: /08/{obj_id}_*.pt)
    normalized_glb_dir: Path | None = None
    anno_dir: Path | None = None

    @classmethod
    def from_pipeline_cfg(cls, cfg: dict) -> "DatasetRoots":
        data = cfg.get("data") or {}
        raw_slat = data.get("slat_dir")
        raw_glb  = data.get("normalized_glb_dir")
        raw_anno = data.get("anno_dir")
        return cls(
            images_root=Path(data.get("images_root", DEFAULT_IMAGES_ROOT)),
            mesh_root=Path(data.get("mesh_root", DEFAULT_MESH_ROOT)),
            slat_dir=Path(raw_slat) if raw_slat else None,
            normalized_glb_dir=Path(raw_glb) if raw_glb else None,
            anno_dir=Path(raw_anno) if raw_anno else None,
        )

    def input_npz_paths(self, shard: str | int, obj_id: str) -> tuple[Path, Path]:
        """Return ``(mesh_npz, image_npz)`` under ``<root>/<shard>/<obj_id>.npz``."""
        s = normalize_shard(shard)
        return (
            self.mesh_root / s / f"{obj_id}.npz",
            self.images_root / s / f"{obj_id}.npz",
        )

    def slat_pt_paths(self, shard: str | int, obj_id: str) -> tuple[Path, Path]:
        """Return ``(coords_pt, feats_pt)`` under ``<slat_dir>/<shard>/<obj_id>_*.pt``.

        Raises ``RuntimeError`` if ``slat_dir`` is not configured.
        """
        if self.slat_dir is None:
            raise RuntimeError(
                "data.slat_dir is not configured — cannot locate SLAT files"
            )
        s = normalize_shard(shard)
        base = self.slat_dir / s
        return (
            base / f"{obj_id}_coords.pt",
            base / f"{obj_id}_feats.pt",
        )

    def normalized_glb_path(self, obj_id: str) -> Path | None:
        """Return path to {obj_id}.glb in normalized_glb_dir, or None if not configured."""
        if self.normalized_glb_dir is None:
            return None
        return self.normalized_glb_dir / f"{obj_id}.glb"

    def anno_object_dir(self, obj_id: str) -> Path | None:
        """Return per-object annotation dir (<anno_dir>/{obj_id}/), or None if not configured."""
        if self.anno_dir is None:
            return None
        return self.anno_dir / obj_id



def resolve_blender_executable(cfg: dict) -> str:
    """Resolve Blender executable for pipeline_v3 steps.

    Precedence:
      1. ``cfg["tools"]["blender_path"]`` (if set)
      2. Legacy top-level ``cfg["blender"]`` (common in pipeline YAML)
      3. ``BLENDER_PATH`` environment variable (deprecated; still honored)
      4. ``"blender"`` on ``PATH`` (no machine-specific hardcoded default)
    """
    tools = cfg.get("tools") or {}
    bp = tools.get("blender_path")
    if isinstance(bp, str) and bp.strip():
        return bp.strip()
    legacy = cfg.get("blender")
    if isinstance(legacy, str) and legacy.strip():
        return legacy.strip()
    env = os.environ.get("BLENDER_PATH", "").strip()
    if env:
        return env
    return "blender"



@dataclass(frozen=True)
class PipelineRoot:
    """Root of a pipeline_v3 run output tree."""
    root: Path

    @property
    def objects_root(self) -> Path:
        return self.root / "objects"

    @property
    def global_dir(self) -> Path:
        return self.root / "_global"

    @property
    def manifest_path(self) -> Path:
        return self.global_dir / "manifest.jsonl"

    @property
    def report_path(self) -> Path:
        return self.global_dir / "report_full.html"

    def shard_dir(self, shard: str | int) -> Path:
        return self.objects_root / normalize_shard(shard)

    def object_dir(self, shard: str | int, obj_id: str) -> Path:
        return self.shard_dir(shard) / obj_id

    def context(
        self,
        shard: str | int,
        obj_id: str,
        *,
        mesh_npz: Path | None = None,
        image_npz: Path | None = None,
    ) -> "ObjectContext":
        return ObjectContext(
            root=self,
            shard=normalize_shard(shard),
            obj_id=obj_id,
            mesh_npz=mesh_npz,
            image_npz=image_npz,
        )

    def iter_objects(self) -> list["ObjectContext"]:
        """Discover all object dirs already on disk."""
        out: list[ObjectContext] = []
        if not self.objects_root.is_dir():
            return out
        for shard_dir in sorted(self.objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            for od in sorted(shard_dir.iterdir()):
                if (od / "meta.json").is_file() or od.is_dir():
                    out.append(self.context(shard_dir.name, od.name))
        return out

    def ensure(self) -> None:
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.objects_root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ObjectContext:
    """All paths for one object. Step runners only write inside ``dir``."""
    root: PipelineRoot
    shard: str
    obj_id: str
    mesh_npz: Path | None = None     # source partverse mesh npz (input)
    image_npz: Path | None = None    # source partverse image npz (input)

    # ─── directory layout ────────────────────────────────────────────

    @property
    def dir(self) -> Path:
        return self.root.object_dir(self.shard, self.obj_id)

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def status_path(self) -> Path:
        return self.dir / "status.json"

    @property
    def qc_path(self) -> Path:
        return self.dir / "qc.json"

    @property
    def phase1_dir(self) -> Path:
        return self.dir / "phase1"

    @property
    def parsed_path(self) -> Path:
        return self.phase1_dir / "parsed.json"

    @property
    def overview_path(self) -> Path:
        return self.phase1_dir / "overview.png"

    @property
    def raw_response_path(self) -> Path:
        return self.phase1_dir / "raw.txt"

    @property
    def highlights_dir(self) -> Path:
        return self.dir / "highlights"

    def highlight_path(self, edit_idx: int) -> Path:
        return self.highlights_dir / f"e{edit_idx:02d}.png"

    @property
    def edits_2d_dir(self) -> Path:
        return self.dir / "edits_2d"

    def edit_2d_input(self, edit_id: str) -> Path:
        return self.edits_2d_dir / f"{edit_id}_input.png"

    def edit_2d_output(self, edit_id: str) -> Path:
        return self.edits_2d_dir / f"{edit_id}_edited.png"

    @property
    def edits_3d_dir(self) -> Path:
        return self.dir / "edits_3d"

    def edit_3d_dir(self, edit_id: str) -> Path:
        return self.edits_3d_dir / edit_id

    def edit_3d_npz(self, edit_id: str, which: str) -> Path:
        assert which in ("before", "after"), which
        return self.edit_3d_dir(edit_id) / f"{which}.npz"

    def edit_3d_png(self, edit_id: str, which: str) -> Path:
        assert which in ("before", "after"), which
        return self.edit_3d_dir(edit_id) / f"{which}.png"

    # ─── helpers ─────────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all subdirectories. Idempotent."""
        for d in (self.phase1_dir, self.highlights_dir,
                  self.edits_2d_dir, self.edits_3d_dir):
            d.mkdir(parents=True, exist_ok=True)

    def edit_id(self, edit_type: str, seq: int) -> str:
        """Standard edit_id: ``{prefix}_{obj_id}_{seq:03d}``."""
        try:
            prefix = EDIT_TYPE_PREFIX[edit_type]
        except KeyError as e:
            raise ValueError(f"unknown edit_type: {edit_type}") from e
        return f"{prefix}_{self.obj_id}_{seq:03d}"

    def __repr__(self) -> str:
        return f"ObjectContext(shard={self.shard}, obj_id={self.obj_id})"
