"""Canonical access to ``services.*``, ``step_params.*``, and ``pipeline.stages``.

PR-C: pipeline v2 and shared loaders use these helpers; legacy ``phase0`` /
``phase2_5`` / ``phase5`` / ``pipeline.phases`` keys are removed.
"""
from __future__ import annotations

from typing import Any


class PipelineServicesError(ValueError):
    """Raised when a required ``services`` block is missing or invalid."""


def _services(cfg: dict) -> dict:
    s = cfg.get("services")
    if not isinstance(s, dict) or not s:
        raise PipelineServicesError("[CONFIG] services: mapping is required")
    return s


def vlm_service(cfg: dict) -> dict[str, Any]:
    """Return ``services.vlm`` (required for pipeline v2 configs)."""
    v = _services(cfg).get("vlm")
    if not isinstance(v, dict):
        raise PipelineServicesError("[CONFIG] services.vlm: mapping is required")
    return v


def image_edit_service(cfg: dict) -> dict[str, Any]:
    """Return ``services.image_edit`` (required when FLUX/TRELLIS edit settings are used)."""
    ie = _services(cfg).get("image_edit")
    if not isinstance(ie, dict):
        raise PipelineServicesError("[CONFIG] services.image_edit: mapping is required")
    return ie


def trellis_image_edit_flat(cfg: dict) -> dict[str, Any]:
    """Flatten ``services.image_edit`` for TrellisRefiner and s5 helpers.

    Ensures legacy keys ``image_edit_base_url`` / ``image_edit_base_urls`` exist
    when only ``base_urls`` is provided.
    """
    ie = dict(image_edit_service(cfg))
    out: dict[str, Any] = dict(ie)
    if "base_urls" in ie:
        urls = ie["base_urls"]
        if "image_edit_base_urls" not in out:
            out["image_edit_base_urls"] = urls
        if isinstance(urls, list) and urls:
            out.setdefault("image_edit_base_url", str(urls[0]).rstrip("/"))
    return out


def trellis_workers_per_gpu(cfg: dict, *, default: int = 1) -> int:
    """Number of Trellis 3D worker subprocesses per physical GPU.

    Resolution order:
      1. ``TRELLIS_WORKERS_PER_GPU`` env var (ad-hoc override)
      2. ``services.image_edit.trellis_workers_per_gpu`` in YAML
      3. ``default`` (= 1, current behavior)

    Always clamped to >= 1.
    """
    import os
    raw = os.environ.get("TRELLIS_WORKERS_PER_GPU", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    ie = image_edit_service(cfg)
    v = ie.get("trellis_workers_per_gpu", default)
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = default
    return max(1, v)


def vlm_model_name(cfg: dict, *, default: str = "Qwen3.5-27B") -> str:
    v = vlm_service(cfg)
    m = v.get("model", v.get("vlm_model", default))
    return str(m) if m is not None else default


def step_params_for(cfg: dict, step: str) -> dict[str, Any]:
    sp = cfg.get("step_params") or {}
    if not isinstance(sp, dict):
        return {}
    block = sp.get(step)
    return dict(block) if isinstance(block, dict) else {}


def pipeline_stages_raw(cfg: dict) -> list[dict[str, Any]]:
    pl = cfg.get("pipeline") or {}
    if not isinstance(pl, dict):
        raise PipelineServicesError("[CONFIG] pipeline: must be a mapping")
    raw = pl.get("stages")
    if not isinstance(raw, list) or not raw:
        raise PipelineServicesError("[CONFIG] pipeline.stages: non-empty list is required")
    return list(raw)


__all__ = [
    "PipelineServicesError",
    "vlm_service",
    "image_edit_service",
    "trellis_image_edit_flat",
    "trellis_workers_per_gpu",
    "vlm_model_name",
    "step_params_for",
    "pipeline_stages_raw",
]
