"""Backward-compatibility shim — **deprecated**.

Use ``from partcraft.io.partcraft_loader import PartCraftDataset`` instead.
This module will be removed in a future version.
"""
import warnings as _w
_w.warn(
    "partcraft.io.hy3d_loader is deprecated, use partcraft.io.partcraft_loader",
    DeprecationWarning,
    stacklevel=2,
)

from partcraft.io.partcraft_loader import (  # noqa: F401
    ObjectRecord,
    PartCraftDataset,
    PartInfo,
)

# Legacy alias
HY3DPartDataset = PartCraftDataset
