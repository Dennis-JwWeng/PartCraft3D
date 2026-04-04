"""PyTorch Dataset for object-centric edit pairs.

Loads ``(before, after, prompt, metadata)`` tuples from the repacked
object-centric directory layout::

    {root}/
      shard_XX/
        {obj_id}/
          original.npz       # shared before state
          mod_000.npz         # after states
          ...
          metadata.json
      manifest.jsonl          # flat index

Each NPZ contains ``slat_feats [N,C]``, ``slat_coords [N,4]``, ``ss [C,R,R,R]``,
and optionally ``dino_voxel_mean [N,1024]`` (float16, multi-view averaged DINOv2
features projected onto voxels).

Follows the Trellis ``SLat`` dataset conventions for ``collate_fn`` —
``SparseTensor`` batching with batch-index prepending and layout slices.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class EditPairDataset(Dataset):
    """Dataset yielding ``(before_slat, after_slat, prompt, metadata)`` pairs.

    Parameters
    ----------
    root : str | Path
        Root directory of repacked data (contains ``shard_XX/`` dirs and
        ``manifest.jsonl``).
    shards : list[str] | None
        Restrict to these shard IDs.  ``None`` = all discovered shards.
    edit_types : set[str] | None
        Restrict to these edit types (e.g. ``{"modification", "scale"}``).
        ``None`` = all types.
    max_voxels : int
        Skip edits whose before *or* after voxel count exceeds this.
    normalization : dict | None
        ``{"mean": [...], "std": [...]}`` applied to ``slat_feats``.
    original_cache_size : int
        LRU cache size for ``original.npz`` (number of objects).
    quality_dir : str | Path | None
        If provided, path to cleaning output (same layout as ``root``,
        with ``quality.json`` per object).  Entries below ``min_tier``
        are excluded.  Can also be the same as ``root`` if
        ``quality.json`` files sit alongside data.
    min_tier : str
        Minimum quality tier to include (``"high"``, ``"medium"``,
        ``"low"``).  Only used when ``quality_dir`` is set.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        shards: list[str] | None = None,
        edit_types: set[str] | None = None,
        max_voxels: int = 32768,
        normalization: Optional[dict] = None,
        original_cache_size: int = 2048,
        quality_dir: Optional[str | Path] = None,
        min_tier: str = "medium",
    ):
        self.root = Path(root)
        self.max_voxels = max_voxels
        self.normalization = normalization

        if normalization is not None:
            self._mean = torch.tensor(normalization["mean"]).reshape(1, -1)
            self._std = torch.tensor(normalization["std"]).reshape(1, -1)

        # Pre-load all object metadata keyed by (shard, obj_id)
        self._meta: dict[tuple[str, str], dict] = {}
        # Flat index: list of (shard, obj_id, edit_idx)
        self._entries: list[tuple[str, str, int]] = []

        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"manifest.jsonl not found in {self.root}. "
                f"Run repack_to_object_dirs.py first."
            )

        with open(manifest) as f:
            for line in f:
                rec = json.loads(line)
                shard = rec["shard"]
                if shards is not None and shard not in shards:
                    continue
                etype = rec["type"]
                if edit_types is not None and etype not in edit_types:
                    continue
                obj_id = rec["obj_id"]
                edit_idx = rec["edit_idx"]

                key = (shard, obj_id)
                if key not in self._meta:
                    meta_path = (self.root / f"shard_{shard}" / obj_id
                                 / "metadata.json")
                    if not meta_path.exists():
                        continue
                    with open(meta_path) as mf:
                        self._meta[key] = json.load(mf)

                self._entries.append((shard, obj_id, edit_idx))

        # Quality-based filtering
        if quality_dir is not None:
            allowed = self._load_quality_tiers(Path(quality_dir), min_tier)
            if allowed is not None:
                before_n = len(self._entries)
                self._entries = [
                    e for e in self._entries
                    if self._meta[(e[0], e[1])]["edits"][e[2]]["edit_id"]
                    in allowed
                ]
                filtered = before_n - len(self._entries)
                if filtered > 0:
                    import logging as _log
                    _log.getLogger(__name__).info(
                        "Quality filter: %d/%d edits removed (min_tier=%s)",
                        filtered, before_n, min_tier,
                    )

        # Build the LRU-cached original loader
        self._load_original = functools.lru_cache(maxsize=original_cache_size)(
            self._load_original_impl
        )

        # Pre-compute per-object deletion seq → edit_idx mapping
        # so addition edits can resolve their source file efficiently
        self._del_seq_to_file: dict[tuple[str, str, int], str] = {}
        for key, meta in self._meta.items():
            for edit in meta["edits"]:
                if edit["type"] == "deletion" and edit.get("file"):
                    self._del_seq_to_file[(*key, edit["seq"])] = edit["file"]

    # ─────────────────── public API ───────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, obj_id, edit_idx = self._entries[index]
        meta = self._meta[(shard, obj_id)]
        edit = meta["edits"][edit_idx]
        obj_dir = self.root / f"shard_{shard}" / obj_id

        etype = edit["type"]

        try:
            if etype == "identity":
                before = self._load_original(shard, obj_id)
                after = before  # same object
            elif etype == "addition":
                # before = source deletion's after, after = original
                del_seq = edit.get("source_del_seq", -1)
                del_file = self._del_seq_to_file.get(
                    (shard, obj_id, del_seq)
                )
                if del_file is None:
                    raise FileNotFoundError(
                        f"Source deletion seq {del_seq} not found for "
                        f"addition {edit['edit_id']}"
                    )
                before = self._load_npz(obj_dir / del_file)
                after = self._load_original(shard, obj_id)
            else:
                before = self._load_original(shard, obj_id)
                fname = edit.get("file")
                if fname is None:
                    raise FileNotFoundError(
                        f"No file for edit {edit['edit_id']}"
                    )
                after = self._load_npz(obj_dir / fname)

            if (before["coords"].shape[0] > self.max_voxels
                    or after["coords"].shape[0] > self.max_voxels):
                return self._random_fallback()

            result = {
                "before_coords": before["coords"],
                "before_feats": before["feats"],
                "before_ss": before["ss"],
                "after_coords": after["coords"],
                "after_feats": after["feats"],
                "after_ss": after["ss"],
                "prompt": edit.get("prompt", ""),
                "edit_type": etype,
            }
            if before.get("dino_voxel_mean") is not None:
                result["before_dino"] = before["dino_voxel_mean"]
            if after.get("dino_voxel_mean") is not None:
                result["after_dino"] = after["dino_voxel_mean"]
            return result
        except Exception:
            return self._random_fallback()

    # ─────────────────── collation ────────────────────────────────────

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Batch edit pairs into dual SparseTensors.

        Follows the Trellis ``SLat.collate_fn`` convention: prepend a
        batch-index column to coords and register a ``layout`` spatial cache.
        """
        from trellis.modules.sparse.basic import SparseTensor

        pack: dict[str, Any] = {}

        for prefix in ("before", "after"):
            coords_list, feats_list, layouts = [], [], []
            start = 0
            for i, item in enumerate(batch):
                c = item[f"{prefix}_coords"]
                f = item[f"{prefix}_feats"]
                n = c.shape[0]
                batch_col = torch.full((n, 1), i, dtype=torch.int32)
                coords_list.append(torch.cat([batch_col, c], dim=-1))
                feats_list.append(f)
                layouts.append(slice(start, start + n))
                start += n

            coords = torch.cat(coords_list)
            feats = torch.cat(feats_list)
            st = SparseTensor(coords=coords, feats=feats)
            st._shape = torch.Size([len(batch), *batch[0][f"{prefix}_feats"].shape[1:]])
            st.register_spatial_cache("layout", layouts)
            pack[f"{prefix}_slat"] = st

        pack["before_ss"] = torch.stack([b["before_ss"] for b in batch])
        pack["after_ss"] = torch.stack([b["after_ss"] for b in batch])
        pack["prompt"] = [b["prompt"] for b in batch]
        pack["edit_type"] = [b["edit_type"] for b in batch]

        # Optional DINOv2 voxel features (same sparse layout as SLAT)
        for prefix in ("before", "after"):
            dino_key = f"{prefix}_dino"
            if dino_key in batch[0]:
                dino_list = [item[dino_key] for item in batch]
                pack[dino_key] = torch.cat(dino_list)
            # else: key absent — dataset has no dino_voxel_mean

        return pack

    # ─────────────────── internal helpers ─────────────────────────────

    def _load_original_impl(self, shard: str, obj_id: str) -> dict[str, torch.Tensor]:
        """Load and cache ``original.npz`` for an object."""
        path = self.root / f"shard_{shard}" / obj_id / "original.npz"
        return self._load_npz(path)

    def _load_npz(self, path: Path) -> dict[str, torch.Tensor | None]:
        data = np.load(path)
        coords = torch.tensor(data["slat_coords"]).int()
        feats = torch.tensor(data["slat_feats"]).float()
        ss = torch.tensor(data["ss"]).float()

        if self.normalization is not None:
            feats = (feats - self._mean) / self._std

        result: dict[str, torch.Tensor | None] = {
            "coords": coords, "feats": feats, "ss": ss,
        }
        if "dino_voxel_mean" in data.files:
            result["dino_voxel_mean"] = torch.tensor(data["dino_voxel_mean"]).float()
        else:
            result["dino_voxel_mean"] = None
        return result

    def _random_fallback(self) -> dict[str, Any]:
        """Return a random valid item (error recovery, matches Trellis convention)."""
        idx = np.random.randint(0, len(self))
        return self[idx]

    # ─────────────────── quality filtering ─────────────────────────────

    @staticmethod
    def _load_quality_tiers(
        quality_dir: Path,
        min_tier: str = "medium",
    ) -> set[str] | None:
        """Load passing edit IDs from quality.json files.

        Returns a set of edit_ids that meet the minimum tier, or None if
        no quality data is found.
        """
        tier_order = {"high": 0, "medium": 1, "low": 2, "negative": 3, "rejected": 4}
        min_val = tier_order.get(min_tier, 1)

        allowed: set[str] = set()
        found_any = False

        for shard_dir in quality_dir.iterdir():
            if not shard_dir.is_dir() or not shard_dir.name.startswith("shard_"):
                continue
            for obj_dir in shard_dir.iterdir():
                qpath = obj_dir / "quality.json"
                if not qpath.exists():
                    continue
                found_any = True
                with open(qpath) as f:
                    q = json.load(f)
                for edit in q.get("edits", []):
                    tier = edit.get("tier", "rejected")
                    if tier_order.get(tier, 4) <= min_val:
                        allowed.add(edit["edit_id"])

        return allowed if found_any else None

    # ─────────────────── diagnostics ──────────────────────────────────

    def __str__(self) -> str:
        from collections import Counter
        type_counts = Counter(
            self._meta[self._entries[i][:2]]["edits"][self._entries[i][2]]["type"]
            for i in range(len(self._entries))
        )
        lines = [
            f"EditPairDataset",
            f"  root: {self.root}",
            f"  objects: {len(self._meta)}",
            f"  edits:   {len(self._entries)}",
            f"  types:   {dict(type_counts)}",
        ]
        return "\n".join(lines)
