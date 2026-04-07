"""PyTorch Dataset for object-centric edit pairs (pipeline_v2 layout).

Loads ``(before, after, prompt, metadata)`` tuples from the new
object-centric directory layout produced by the v2 pipeline::

    {root}/
      _global/manifest.jsonl              # one line per object (optional)
      objects/<shard>/<obj_id>/
          meta.json                       # object + edits list
          phase1/{overview.png, parsed.json, raw.txt}
          highlights/e{idx:02d}.png
          edits_2d/{edit_id}_{input,edited}.png
          edits_3d/<edit_id>/{before,after}.npz   # SLAT pair
          status.json

Each NPZ contains ``slat_feats [N,C]``, ``slat_coords [N,4]``, ``ss [C,R,R,R]``.

Differences vs the old flat layout:
  * No shared ``original.npz`` — every edit's own ``before.npz`` is loaded.
  * ``meta.json`` (not ``metadata.json``); edits use ``edit_type`` not ``type``.
  * Iteration scans ``objects/<shard>/<obj_id>/meta.json`` directly; the
    ``manifest.jsonl`` is per-object and only used as an optional shard filter.
  * Edits whose ``edits_3d/<edit_id>/{before,after}.npz`` do not (yet) exist
    are silently skipped at index time — covers deletion placeholders and
    not-yet-backfilled additions.

Follows the Trellis ``SLat`` dataset conventions for ``collate_fn`` —
``SparseTensor`` batching with batch-index prepending and layout slices.
"""
from __future__ import annotations

import json
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
        Root directory of pipeline_v2 output (contains ``objects/<shard>/``).
    shards : list[str] | None
        Restrict to these shard IDs.  ``None`` = all discovered shards.
    edit_types : set[str] | None
        Restrict to these edit types
        (``modification`` / ``scale`` / ``material`` / ``global`` /
        ``deletion`` / ``addition``).  ``None`` = all available.
    max_voxels : int
        Skip edits whose before *or* after voxel count exceeds this.
    normalization : dict | None
        ``{"mean": [...], "std": [...]}`` applied to ``slat_feats``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        shards: list[str] | None = None,
        edit_types: set[str] | None = None,
        max_voxels: int = 32768,
        normalization: Optional[dict] = None,
    ):
        self.root = Path(root)
        self.max_voxels = max_voxels
        self.normalization = normalization

        if normalization is not None:
            self._mean = torch.tensor(normalization["mean"]).reshape(1, -1)
            self._std = torch.tensor(normalization["std"]).reshape(1, -1)

        objects_root = self.root / "objects"
        if not objects_root.is_dir():
            raise FileNotFoundError(
                f"{objects_root} not found — expected pipeline_v2 layout "
                f"with objects/<shard>/<obj_id>/meta.json"
            )

        self._meta: dict[tuple[str, str], dict] = {}
        # entries: (shard, obj_id, edit_idx, edit_id)
        self._entries: list[tuple[str, str, int, str]] = []

        for shard_dir in sorted(objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            shard = shard_dir.name
            if shards is not None and shard not in shards:
                continue
            for obj_dir in sorted(shard_dir.iterdir()):
                meta_path = obj_dir / "meta.json"
                if not meta_path.is_file():
                    continue
                meta = json.loads(meta_path.read_text())
                obj_id = meta.get("obj_id", obj_dir.name)
                self._meta[(shard, obj_id)] = meta

                for edit in meta.get("edits", []):
                    et = edit.get("edit_type", "?")
                    if edit_types is not None and et not in edit_types:
                        continue
                    edit_id = edit.get("edit_id")
                    if not edit_id:
                        continue
                    pair_dir = obj_dir / "edits_3d" / edit_id
                    # Require both before and after npz to exist now;
                    # placeholders (deletion / not-yet-backfilled add) skipped.
                    if not ((pair_dir / "before.npz").is_file()
                            and (pair_dir / "after.npz").is_file()):
                        continue
                    self._entries.append(
                        (shard, obj_id, edit["idx"], edit_id)
                    )

    # ─────────────────── public API ───────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, obj_id, edit_idx, edit_id = self._entries[index]
        meta = self._meta[(shard, obj_id)]
        edit = meta["edits"][edit_idx]
        pair_dir = (self.root / "objects" / shard / obj_id
                    / "edits_3d" / edit_id)

        try:
            before = self._load_npz(pair_dir / "before.npz")
            after = self._load_npz(pair_dir / "after.npz")

            if (before["coords"].shape[0] > self.max_voxels
                    or after["coords"].shape[0] > self.max_voxels):
                return self._random_fallback()

            return {
                "before_coords": before["coords"],
                "before_feats": before["feats"],
                "before_ss": before["ss"],
                "after_coords": after["coords"],
                "after_feats": after["feats"],
                "after_ss": after["ss"],
                "prompt": edit.get("prompt", "") or "",
                "edit_type": edit.get("edit_type", "?"),
                "edit_id": edit_id,
                "obj_id": obj_id,
            }
        except Exception:
            return self._random_fallback()

    # ─────────────────── collation ────────────────────────────────────

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
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
            st._shape = torch.Size(
                [len(batch), *batch[0][f"{prefix}_feats"].shape[1:]]
            )
            st.register_spatial_cache("layout", layouts)
            pack[f"{prefix}_slat"] = st

        pack["before_ss"] = torch.stack([b["before_ss"] for b in batch])
        pack["after_ss"] = torch.stack([b["after_ss"] for b in batch])
        pack["prompt"] = [b["prompt"] for b in batch]
        pack["edit_type"] = [b["edit_type"] for b in batch]
        pack["edit_id"] = [b["edit_id"] for b in batch]
        pack["obj_id"] = [b["obj_id"] for b in batch]
        return pack

    # ─────────────────── internal helpers ─────────────────────────────

    def _load_npz(self, path: Path) -> dict[str, torch.Tensor]:
        data = np.load(path)
        coords = torch.tensor(data["slat_coords"]).int()
        feats = torch.tensor(data["slat_feats"]).float()
        ss = torch.tensor(data["ss"]).float()

        if self.normalization is not None:
            feats = (feats - self._mean) / self._std

        return {"coords": coords, "feats": feats, "ss": ss}

    def _random_fallback(self) -> dict[str, Any]:
        idx = np.random.randint(0, len(self))
        return self[idx]

    # ─────────────────── diagnostics ──────────────────────────────────

    def __str__(self) -> str:
        from collections import Counter
        type_counts = Counter(
            self._meta[(s, o)]["edits"][i]["edit_type"]
            for s, o, i, _ in self._entries
        )
        return (
            f"EditPairDataset\n"
            f"  root: {self.root}\n"
            f"  objects: {len(self._meta)}\n"
            f"  edits:   {len(self._entries)}\n"
            f"  types:   {dict(type_counts)}"
        )
