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

Each NPZ contains ``slat_feats [N,C]``, ``slat_coords [N,4]``, ``ss [C,R,R,R]``.

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

        return pack

    # ─────────────────── internal helpers ─────────────────────────────

    def _load_original_impl(self, shard: str, obj_id: str) -> dict[str, torch.Tensor]:
        """Load and cache ``original.npz`` for an object."""
        path = self.root / f"shard_{shard}" / obj_id / "original.npz"
        return self._load_npz(path)

    def _load_npz(self, path: Path) -> dict[str, torch.Tensor]:
        data = np.load(path)
        coords = torch.tensor(data["slat_coords"]).int()
        feats = torch.tensor(data["slat_feats"]).float()
        ss = torch.tensor(data["ss"]).float()

        if self.normalization is not None:
            feats = (feats - self._mean) / self._std

        return {"coords": coords, "feats": feats, "ss": ss}

    def _random_fallback(self) -> dict[str, Any]:
        """Return a random valid item (error recovery, matches Trellis convention)."""
        idx = np.random.randint(0, len(self))
        return self[idx]

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
