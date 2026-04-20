"""Path layout for the ``data/H3D_v1/`` dataset.

The single source of truth for every dataset path. All other modules
build on top of this — keeps the layout pinned so a future move
(e.g. renaming a directory) is a one-file change.

Layout summary (full doc in
``docs/superpowers/specs/2026-04-19-h3d-v1-design.md`` §3):

::

    <root>/
    ├── _assets/<NN>/<obj_id>/
    │   ├── object.npz
    │   └── orig_views/view{0..4}.png
    ├── deletion/<NN>/<obj_id>/<edit_id>/
    │   ├── meta.json
    │   ├── before.npz       # hardlink to _assets/.../object.npz
    │   ├── after.npz        # physical (s6b output)
    │   ├── before.png       # hardlink to _assets/.../orig_views/view{K}.png
    │   └── after.png        # hardlink to pipeline preview_{K}.png
    │                        # K = meta.json["views"]["best_view_index"]
    ├── addition/<NN>/<obj_id>/<edit_id>/
    │   └── ... (mirror of paired deletion, see spec §3)
    ├── modification|scale|material|color|global/<NN>/<obj_id>/<edit_id>/
    │   └── ...
    └── manifests/
        ├── <type>/<NN>.jsonl
        └── all.jsonl
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EDIT_TYPES_FLUX: tuple[str, ...] = (
    "modification", "scale", "material", "color", "global",
)
EDIT_TYPES_ALL: tuple[str, ...] = ("deletion", "addition") + EDIT_TYPES_FLUX

N_VIEWS: int = 5

# Pipeline edit-id prefix → dataset edit_type.
EDIT_PREFIX_TO_TYPE: dict[str, str] = {
    "del": "deletion",
    "add": "addition",
    "mod": "modification",
    "scl": "scale",
    "mat": "material",
    "clr": "color",
    "glb": "global",
}
TYPE_TO_PREFIX: dict[str, str] = {v: k for k, v in EDIT_PREFIX_TO_TYPE.items()}


@dataclass(frozen=True)
class H3DLayout:
    """Resolves every absolute path in the H3D_v1 dataset.

    Construct once per CLI invocation with the dataset root and pass
    around. All accessors are pure path joins — no IO.
    """

    root: Path

    # ── _assets/ pool ───────────────────────────────────────────────
    def assets_obj_dir(self, shard: str, obj_id: str) -> Path:
        return self.root / "_assets" / shard / obj_id

    def object_npz(self, shard: str, obj_id: str) -> Path:
        return self.assets_obj_dir(shard, obj_id) / "object.npz"

    def orig_views_dir(self, shard: str, obj_id: str) -> Path:
        return self.assets_obj_dir(shard, obj_id) / "orig_views"

    def orig_view(self, shard: str, obj_id: str, k: int) -> Path:
        if not 0 <= k < N_VIEWS:
            raise ValueError(f"view index out of range [0,{N_VIEWS}): {k}")
        return self.orig_views_dir(shard, obj_id) / f"view{k}.png"

    def asset_lock(self, shard: str, obj_id: str) -> Path:
        return self.assets_obj_dir(shard, obj_id) / ".lock"

    # ── per-edit dirs ───────────────────────────────────────────────
    def edit_dir(self, edit_type: str, shard: str, obj_id: str, edit_id: str) -> Path:
        if edit_type not in EDIT_TYPES_ALL:
            raise ValueError(f"unknown edit_type: {edit_type}")
        return self.root / edit_type / shard / obj_id / edit_id

    def meta_json(self, edit_type: str, shard: str, obj_id: str, edit_id: str) -> Path:
        return self.edit_dir(edit_type, shard, obj_id, edit_id) / "meta.json"

    def before_npz(self, edit_type: str, shard: str, obj_id: str, edit_id: str) -> Path:
        return self.edit_dir(edit_type, shard, obj_id, edit_id) / "before.npz"

    def after_npz(self, edit_type: str, shard: str, obj_id: str, edit_id: str) -> Path:
        return self.edit_dir(edit_type, shard, obj_id, edit_id) / "after.npz"

    def before_image(self, edit_type: str, shard: str, obj_id: str, edit_id: str) -> Path:
        """Per-edit single before.png (flat layout, schema v3).

        Resolves to ``<edit_dir>/before.png`` — a hardlink to the original
        object's ``orig_views/view{K}.png`` (or, for additions, to the paired
        deletion's ``after.png``) where ``K = meta.views.best_view_index``.
        """
        return self.edit_dir(edit_type, shard, obj_id, edit_id) / "before.png"

    def after_image(self, edit_type: str, shard: str, obj_id: str, edit_id: str) -> Path:
        """Per-edit single after.png (flat layout, schema v3).

        Resolves to ``<edit_dir>/after.png`` — a hardlink to the pipeline's
        ``preview_{K}.png`` (deletion/flux) or to ``orig_views/view{K}.png``
        (addition) where ``K = meta.views.best_view_index``.
        """
        return self.edit_dir(edit_type, shard, obj_id, edit_id) / "after.png"

    # ── manifests ───────────────────────────────────────────────────
    def manifest_dir(self, edit_type: str) -> Path:
        if edit_type not in EDIT_TYPES_ALL:
            raise ValueError(f"unknown edit_type: {edit_type}")
        return self.root / "manifests" / edit_type

    def manifest_path(self, edit_type: str, shard: str) -> Path:
        return self.manifest_dir(edit_type) / f"{shard}.jsonl"

    def aggregated_manifest(self) -> Path:
        return self.root / "manifests" / "all.jsonl"

    # ── internal / non-distributed ──────────────────────────────────
    # Repro metadata dropped from per-edit ``meta.json`` (pipeline_config,
    # pipeline_git_sha, promoted_at) is appended here for local audit.
    # ``pack_shard`` deliberately does NOT include this directory so the
    # released tarball is free of internal tooling fingerprints.
    def internal_manifest_dir(self) -> Path:
        return self.root / "manifests" / "_internal"

    def promote_log(self) -> Path:
        return self.internal_manifest_dir() / "promote_log.jsonl"


def edit_type_from_id(edit_id: str) -> str:
    """Return the dataset edit_type (e.g. ``"modification"``) for a pipeline edit id.

    Raises ``ValueError`` if the prefix is not recognised.
    """
    prefix = edit_id.split("_", 1)[0]
    try:
        return EDIT_PREFIX_TO_TYPE[prefix]
    except KeyError as exc:
        raise ValueError(f"unrecognised edit_id prefix: {edit_id!r}") from exc


def paired_edit_id(edit_id: str) -> str | None:
    """Return the paired edit_id for a deletion↔addition pair.

    Pairing convention: ``del_<obj>_NNN`` ↔ ``add_<obj>_NNN``. For any
    other edit type returns ``None``.
    """
    if edit_id.startswith("del_"):
        return "add_" + edit_id[4:]
    if edit_id.startswith("add_"):
        return "del_" + edit_id[4:]
    return None


__all__ = [
    "EDIT_TYPES_ALL",
    "EDIT_TYPES_FLUX",
    "EDIT_PREFIX_TO_TYPE",
    "TYPE_TO_PREFIX",
    "N_VIEWS",
    "H3DLayout",
    "edit_type_from_id",
    "paired_edit_id",
]
