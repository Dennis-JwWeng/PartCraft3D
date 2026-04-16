"""Object-centric pipeline v3 (bench-scoped).

Pipeline v3; contains the steps used in the shard08
test benchmark config (s1, sq1, s4, s5, s5b, s6p_flux, s6p_del, sq3, s6b, s6).

This module is independently editable.

All step runners take an :class:`ObjectContext` and write under that
single object's directory. Step orchestration lives in :mod:`run`.
"""
from .paths import ObjectContext, PipelineRoot, EDIT_TYPE_PREFIX  # noqa: F401
from .specs import (  # noqa: F401
    VIEW_INDICES, EditSpec,
    iter_all_specs, iter_flux_specs, iter_deletion_specs, iter_specs_for_objects,
)
from .status import (  # noqa: F401
    STATUS_OK, STATUS_FAIL, STATUS_SKIP, STATUS_PENDING,
    load_status, save_status, update_step,
    step_done, needs_step,
    rebuild_manifest, manifest_summary,
)
