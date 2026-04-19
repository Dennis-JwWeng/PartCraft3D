"""H3D_v1 — edit-type-sharded promote workflow for the unified edit dataset.

See ``docs/superpowers/specs/2026-04-19-h3d-v1-design.md`` for the full design.

Public surface (each submodule has its own ``__all__``):

* :mod:`partcraft.cleaning.h3d_v1.layout` — path constants + ``H3DLayout``.
* :mod:`partcraft.cleaning.h3d_v1.filter` — gate-status acceptance rules.
* :mod:`partcraft.cleaning.h3d_v1.pipeline_io` — iterate pipeline_v3 outputs.
* :mod:`partcraft.cleaning.h3d_v1.asset_pool` — ``_assets/`` materialisation.
* :mod:`partcraft.cleaning.h3d_v1.promoter` — per-edit promote routines.
* :mod:`partcraft.cleaning.h3d_v1.manifest` — jsonl manifest io.
"""

__version__ = "0.1.0-alpha"

__all__: list[str] = []
