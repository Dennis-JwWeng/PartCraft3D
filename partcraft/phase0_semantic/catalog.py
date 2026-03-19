"""Part Catalog: build a global semantic index from Phase 0 labels."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CatalogEntry:
    """A single part in the global catalog."""
    obj_id: str
    shard: str
    part_id: int
    label: str               # semantic label from VLM, e.g. "chair_leg"
    category: str             # normalized category, e.g. "leg"
    core: bool                # core part (cannot be removed)
    desc: str = ""            # appearance description
    desc_without: str = ""    # object description with this part removed (from VLM)
    cluster_size: int = 0     # face count
    # VLM-generated editing instructions from Phase 0
    edits: list[dict] = field(default_factory=list)
    # Geometric metadata (populated lazily during Phase 2)
    bbox_extents: tuple[float, float, float] | None = None  # (dx, dy, dz)
    aspect_ratio: float = 0.0  # max_extent / min_extent


# Positional/ordinal tokens to strip when normalizing
_STRIP_TOKENS = frozenset({
    "left", "right", "front", "rear", "back", "top", "bottom",
    "upper", "lower", "inner", "outer", "first", "second",
    "main", "small", "large", "big", "little",
})


def normalize_category(label: str) -> str:
    """Normalize a semantic label to a canonical category.

    'left_front_leg' → 'leg'
    'rear_wheel_01' → 'wheel'
    'decorative_flower_petal' → 'flower_petal'
    """
    tokens = re.split(r"[_\-\s]+", label.lower().strip())
    tokens = [t for t in tokens if t and t not in _STRIP_TOKENS and not t.isdigit()]
    return "_".join(tokens) if tokens else label.lower()


def extract_object_type(desc: str) -> str:
    """Extract a coarse object type from the VLM description.

    'a modern office chair with armrests' → 'chair'
    'a red sports car with spoiler' → 'car'
    """
    if not desc:
        return ""
    # Take the last noun-like word from the first noun phrase
    # Simple heuristic: split by common prepositions, take first chunk
    chunk = re.split(r"\s+(?:with|having|featuring|including|and)\s+", desc.lower())[0]
    tokens = chunk.split()
    # Walk backwards to find the head noun (skip adjectives)
    _ADJECTIVES = {"a", "an", "the", "modern", "old", "new", "red", "blue", "green",
                   "black", "white", "small", "large", "big", "little", "wooden",
                   "metal", "plastic", "simple", "ornate", "decorative", "tall",
                   "short", "round", "square", "stylish", "vintage", "classic"}
    for t in reversed(tokens):
        t_clean = re.sub(r"[^a-z]", "", t)
        if t_clean and t_clean not in _ADJECTIVES:
            return t_clean
    return tokens[-1] if tokens else ""


class PartCatalog:
    """Global index of all parts across all objects, keyed by semantic category."""

    def __init__(self):
        self.entries: list[CatalogEntry] = []
        self.by_category: dict[str, list[int]] = defaultdict(list)  # category → entry indices
        self.by_object: dict[str, list[int]] = defaultdict(list)    # obj_id → entry indices
        self.object_descs: dict[str, str] = {}                      # obj_id → description
        self.object_global_edits: dict[str, list[dict]] = {}        # obj_id → global edit prompts
        self.object_group_edits: dict[str, list[dict]] = {}         # obj_id → group-level edits
        self.object_ortho_views: dict[str, list[int]] = {}          # obj_id → [front,right,back,left] NPZ view indices
        self._object_types: dict[str, str] = {}                     # obj_id → coarse type

    def add(self, entry: CatalogEntry):
        idx = len(self.entries)
        self.entries.append(entry)
        self.by_category[entry.category].append(idx)
        self.by_object[entry.obj_id].append(idx)

    def get_object_type(self, obj_id: str) -> str:
        """Get coarse object type, with caching."""
        if obj_id not in self._object_types:
            self._object_types[obj_id] = extract_object_type(
                self.object_descs.get(obj_id, ""))
        return self._object_types[obj_id]

    def get_entries_for_object(self, obj_id: str) -> list[CatalogEntry]:
        return [self.entries[i] for i in self.by_object.get(obj_id, [])]

    def get_entries_for_category(self, category: str) -> list[CatalogEntry]:
        return [self.entries[i] for i in self.by_category.get(category, [])]

    def get_swap_candidates(self, entry: CatalogEntry, max_candidates: int = 10,
                            require_same_object_type: bool = True,
                            max_size_ratio: float = 5.0) -> list[CatalogEntry]:
        """Find compatible swap candidates with semantic + geometric filtering.

        Filters applied:
          1. Same normalized category (always)
          2. Different object (always)
          3. Same coarse object type if require_same_object_type (e.g. both chairs)
          4. Similar face count (within max_size_ratio)
          5. Similar aspect ratio if available (within 2x)
        """
        src_type = self.get_object_type(entry.obj_id) if require_same_object_type else ""
        candidates = []

        for idx in self.by_category.get(entry.category, []):
            cand = self.entries[idx]

            # Must be from a different object
            if cand.obj_id == entry.obj_id:
                continue

            # Same object type filter (chair↔chair, car↔car)
            if require_same_object_type and src_type:
                cand_type = self.get_object_type(cand.obj_id)
                if cand_type and cand_type != src_type:
                    continue

            # Face count similarity (proxy for geometric complexity)
            if entry.cluster_size > 0 and cand.cluster_size > 0:
                ratio = max(entry.cluster_size, cand.cluster_size) / \
                        min(entry.cluster_size, cand.cluster_size)
                if ratio > max_size_ratio:
                    continue

            # Aspect ratio similarity (if both have geometric metadata)
            if entry.aspect_ratio > 0 and cand.aspect_ratio > 0:
                ar_ratio = max(entry.aspect_ratio, cand.aspect_ratio) / \
                           min(entry.aspect_ratio, cand.aspect_ratio)
                if ar_ratio > 2.0:
                    continue

            candidates.append(cand)
            if len(candidates) >= max_candidates:
                break

        return candidates

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    @property
    def num_objects(self) -> int:
        return len(self.by_object)

    @property
    def categories(self) -> list[str]:
        return sorted(self.by_category.keys())

    def summary(self) -> str:
        lines = [
            f"PartCatalog: {self.num_entries} parts, {self.num_objects} objects, "
            f"{len(self.by_category)} categories",
            "Top categories:"
        ]
        sorted_cats = sorted(self.by_category.items(), key=lambda x: -len(x[1]))
        for cat, indices in sorted_cats[:20]:
            n_objs = len({self.entries[i].obj_id for i in indices})
            lines.append(f"  {cat}: {len(indices)} parts across {n_objs} objects")
        return "\n".join(lines)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [
                {
                    "obj_id": e.obj_id, "shard": e.shard, "part_id": e.part_id,
                    "label": e.label, "category": e.category, "core": e.core,
                    "desc": e.desc, "desc_without": e.desc_without,
                    "cluster_size": e.cluster_size,
                    "edits": e.edits,
                }
                for e in self.entries
            ],
            "object_descs": self.object_descs,
            "object_global_edits": self.object_global_edits,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PartCatalog":
        with open(path) as f:
            data = json.load(f)
        catalog = cls()
        catalog.object_descs = data.get("object_descs", {})
        catalog.object_global_edits = data.get("object_global_edits", {})
        for e in data["entries"]:
            catalog.add(CatalogEntry(**e))
        return catalog

    @classmethod
    def from_phase0_output(cls, labels_jsonl: str | Path,
                           cluster_sizes: dict[str, dict[int, int]] | None = None,
                           dataset=None) -> "PartCatalog":
        """Build catalog from Phase 0 semantic_labels.jsonl output.

        Args:
            labels_jsonl: path to semantic_labels.jsonl
            cluster_sizes: optional pre-computed {obj_id: {part_id: face_count}}
            dataset: optional HY3DPartDataset — if provided AND cluster_sizes is
                     None, reads cluster_size from the dataset's split_mesh metadata.
        """
        # Pre-load cluster sizes from dataset if available
        _ds_sizes: dict[str, dict[int, int]] = {}
        if cluster_sizes is None and dataset is not None:
            try:
                for obj in dataset:
                    _ds_sizes[obj.obj_id] = {
                        p.part_id: p.cluster_size for p in obj.parts
                    }
                    obj.close()
            except Exception:
                pass

        effective_sizes = cluster_sizes or _ds_sizes

        catalog = cls()
        with open(labels_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                obj_id = rec["obj_id"]
                shard = rec.get("shard", "00")
                catalog.object_descs[obj_id] = rec.get("object_desc", "")
                # Global edits (whole-object style/theme changes)
                if rec.get("global_edits"):
                    catalog.object_global_edits[obj_id] = rec["global_edits"]
                # Group edits (from orthogonal 4-view enrichment)
                if rec.get("group_edits"):
                    catalog.object_group_edits[obj_id] = rec["group_edits"]
                if rec.get("orthogonal_views"):
                    catalog.object_ortho_views[obj_id] = rec["orthogonal_views"]

                for p in rec.get("parts", []):
                    label = p.get("label", f"part_{p['part_id']}")
                    category = normalize_category(label)
                    csize = 0
                    if effective_sizes and obj_id in effective_sizes:
                        csize = effective_sizes[obj_id].get(p["part_id"], 0)
                    catalog.add(CatalogEntry(
                        obj_id=obj_id,
                        shard=shard,
                        part_id=p["part_id"],
                        label=label,
                        category=category,
                        core=p.get("core", False),
                        desc=p.get("desc", ""),
                        desc_without=p.get("desc_without", ""),
                        cluster_size=csize,
                        edits=p.get("edits", []),
                    ))
        return catalog
