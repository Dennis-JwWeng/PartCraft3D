"""Path and naming conventions for the edit dataset v1 layout.

All v1-side path construction goes through ``V1Layout``.  This is the only
module that knows the v1 directory shape; downstream code stays decoupled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EDIT_TYPE_BY_PREFIX: dict[str, str] = {
    "del": "deletion", "add": "addition", "mod": "modification",
    "scl": "scale", "mat": "material", "clr": "color", "glb": "glb",
}

N_VIEWS: int = 5


def parse_edit_id(edit_id: str) -> tuple[str, str, int]:
    parts = edit_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"malformed edit_id: {edit_id!r}")
    prefix, *middle, idx_str = parts
    if prefix not in EDIT_TYPE_BY_PREFIX:
        raise ValueError(f"unknown edit type prefix {prefix!r} in {edit_id!r}")
    obj_id = "_".join(middle)
    try:
        idx = int(idx_str)
    except ValueError as e:
        raise ValueError(f"non-integer suffix in {edit_id!r}") from e
    return EDIT_TYPE_BY_PREFIX[prefix], obj_id, idx


@dataclass(frozen=True)
class V1Layout:
    root: Path

    def object_dir(self, shard: str, obj_id: str) -> Path:
        return self.root / "objects" / shard / obj_id

    def meta_json(self, shard: str, obj_id: str) -> Path:
        return self.object_dir(shard, obj_id) / "meta.json"

    def before_dir(self, shard: str, obj_id: str) -> Path:
        return self.object_dir(shard, obj_id) / "before"

    def before_ss_npz(self, shard: str, obj_id: str) -> Path:
        return self.before_dir(shard, obj_id) / "ss.npz"

    def before_slat_npz(self, shard: str, obj_id: str) -> Path:
        return self.before_dir(shard, obj_id) / "slat.npz"

    def before_view_paths(self, shard: str, obj_id: str) -> list[Path]:
        d = self.before_dir(shard, obj_id) / "views"
        return [d / f"view_{k}.png" for k in range(N_VIEWS)]

    def edit_dir(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.object_dir(shard, obj_id) / "edits" / f"{edit_id}{suffix}"

    def spec_json(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "spec.json"

    def qc_json(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "qc.json"

    def after_npz_path(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "after.npz"

    def after_pending_marker(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> Path:
        return self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "_after_pending.json"

    def after_view_paths(self, shard: str, obj_id: str, edit_id: str, *, suffix: str = "") -> list[Path]:
        d = self.edit_dir(shard, obj_id, edit_id, suffix=suffix) / "views"
        return [d / f"view_{k}.png" for k in range(N_VIEWS)]

    def pending_del_latent_file(self) -> Path:
        return self.root / "_pending" / "del_latent.txt"

    def objects_jsonl(self) -> Path:
        return self.root / "index" / "objects.jsonl"

    def edits_jsonl(self) -> Path:
        return self.root / "index" / "edits.jsonl"

    def last_rebuild_json(self) -> Path:
        return self.root / "index" / "_last_rebuild.json"

    def iter_object_dirs(self) -> Iterable[Path]:
        objects_root = self.root / "objects"
        if not objects_root.is_dir():
            return
        for shard_dir in sorted(objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            for obj_dir in sorted(shard_dir.iterdir()):
                if obj_dir.is_dir():
                    yield obj_dir
