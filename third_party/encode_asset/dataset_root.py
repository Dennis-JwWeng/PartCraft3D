"""Resolve dataset layout without third_party/outputs symlink.

Prerender sets PARTCRAFT_DATASET_ROOT to the dataset root (parent of img_Enc/).

Legacy: if unset, uses <cwd>/outputs so a symlink third_party/outputs → dataset
still works when cwd is third_party/.
"""

from __future__ import annotations

import os


def get_dataset_root() -> str:
    r = os.environ.get("PARTCRAFT_DATASET_ROOT", "").strip()
    if r:
        return os.path.abspath(r)
    return os.path.abspath(os.path.join(os.getcwd(), "outputs"))


def img_enc_root() -> str:
    return os.path.join(get_dataset_root(), "img_Enc")


def slat_flat_root() -> str:
    return os.path.join(get_dataset_root(), "slat")
