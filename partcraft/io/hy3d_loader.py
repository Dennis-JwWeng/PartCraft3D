"""Backward-compatibility shim — re-exports from partcraft_loader.

All new code should use:
    from partcraft.io.partcraft_loader import PartCraftDataset, ObjectRecord
"""

from partcraft.io.partcraft_loader import (  # noqa: F401
    ObjectRecord,
    PartCraftDataset,
    PartInfo,
)

# Legacy alias
HY3DPartDataset = PartCraftDataset
