"""Object-grouped sampler for :class:`EditPairDataset`.

Groups edits by object so that all edits of one object are yielded
consecutively.  This maximises the LRU cache hit rate for
``original.npz`` loading — each object's original is loaded once and
reused for all its edits before being evicted.

Objects are shuffled each epoch; edits within each object are also
shuffled.  Compatible with ``torch.utils.data.DataLoader(sampler=...)``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class ObjectGroupedSampler(Sampler[int]):
    """Yields indices grouped by object for cache-friendly loading.

    Parameters
    ----------
    dataset : EditPairDataset
        Must expose ``_entries`` as ``list[(shard, obj_id, edit_idx)]``.
    shuffle : bool
        Shuffle object order and intra-object edit order each epoch.
    seed : int
        Random seed for reproducibility (only used when ``shuffle=True``).
    drop_last_object : bool
        If True, drop the last (incomplete) object group.  Rarely needed.
    """

    def __init__(
        self,
        dataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last_object: bool = False,
    ):
        self._entries = dataset._entries
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last_object = drop_last_object
        self._epoch = 0

        # Build object → list of global indices
        self._obj_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, (shard, obj_id, _edit_idx) in enumerate(self._entries):
            self._obj_to_indices[(shard, obj_id)].append(idx)
        self._obj_keys = list(self._obj_to_indices.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self._seed + self._epoch)

        obj_order = list(range(len(self._obj_keys)))
        if self._shuffle:
            rng.shuffle(obj_order)

        for obj_idx in obj_order:
            key = self._obj_keys[obj_idx]
            indices = self._obj_to_indices[key].copy()
            if self._shuffle:
                rng.shuffle(indices)
            yield from indices

        self._epoch += 1

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling (DDP compatible)."""
        self._epoch = epoch
