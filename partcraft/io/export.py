"""Export utilities for writing edit pairs to disk."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class EditPairRecord:
    """A single editing data pair ready for export."""
    edit_id: str
    edit_type: str                    # "deletion" | "addition" | "modification"
    instruction: str
    instruction_variants: list[str] = field(default_factory=list)
    source_obj_id: str = ""
    source_shard: str = ""
    donor_obj_id: str | None = None   # for swap edits
    removed_part_ids: list[int] = field(default_factory=list)
    added_part_ids: list[int] = field(default_factory=list)
    old_part_label: str = ""
    new_part_label: str = ""
    object_desc: str = ""
    edit_prompt: str = ""
    after_desc: str = ""
    quality_tier: str = ""
    quality_score: float = 0.0
    quality_checks: dict[str, bool] = field(default_factory=dict)


class EditPairWriter:
    """Writes edit pairs (meshes + renders + manifest) to disk."""

    def __init__(self, output_dir: str | Path, filename: str = "edit_pairs.jsonl"):
        self.output_dir = Path(output_dir)
        self.mesh_dir = self.output_dir / "meshes"
        self.render_dir = self.output_dir / "renders"
        self.manifest_path = self.output_dir / filename

        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_fp = None
        self._count = 0

    def __enter__(self):
        self._manifest_fp = open(self.manifest_path, "a")
        return self

    def __exit__(self, *exc):
        if self._manifest_fp:
            self._manifest_fp.close()
            self._manifest_fp = None

    def write_pair(
        self,
        record: EditPairRecord,
        before_mesh: Any = None,
        after_mesh: Any = None,
        before_images: dict[int, bytes] | None = None,
        after_images: dict[int, bytes] | None = None,
        before_masks: dict[int, np.ndarray] | None = None,
    ):
        """Write a complete edit pair to disk."""
        eid = record.edit_id

        # Export meshes
        if before_mesh is not None and trimesh is not None:
            bp = self.mesh_dir / f"{eid}_before.ply"
            before_mesh.export(str(bp))
            record_dict = asdict(record)
            record_dict["before_mesh"] = str(bp.relative_to(self.output_dir))
        else:
            record_dict = asdict(record)
            record_dict["before_mesh"] = None

        if after_mesh is not None and trimesh is not None:
            ap = self.mesh_dir / f"{eid}_after.ply"
            after_mesh.export(str(ap))
            record_dict["after_mesh"] = str(ap.relative_to(self.output_dir))
        else:
            record_dict["after_mesh"] = None

        # Export before images (from original dataset, zero cost)
        if before_images:
            bdir = self.render_dir / f"{eid}_before"
            bdir.mkdir(exist_ok=True)
            for vid, img_bytes in before_images.items():
                with open(bdir / f"{vid:03d}.webp", "wb") as f:
                    f.write(img_bytes)
            record_dict["before_renders"] = str(bdir.relative_to(self.output_dir))

        if before_masks:
            bdir = self.render_dir / f"{eid}_before"
            bdir.mkdir(exist_ok=True)
            for vid, mask in before_masks.items():
                np.save(str(bdir / f"{vid:03d}_mask.npy"), mask)

        # Export after images (rendered externally)
        if after_images:
            adir = self.render_dir / f"{eid}_after"
            adir.mkdir(exist_ok=True)
            for vid, img_bytes in after_images.items():
                with open(adir / f"{vid:03d}.webp", "wb") as f:
                    f.write(img_bytes)
            record_dict["after_renders"] = str(adir.relative_to(self.output_dir))

        # Write manifest line
        if self._manifest_fp:
            self._manifest_fp.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
            self._manifest_fp.flush()

        self._count += 1

    @property
    def count(self) -> int:
        return self._count
