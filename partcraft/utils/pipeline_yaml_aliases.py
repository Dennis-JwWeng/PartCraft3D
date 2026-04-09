"""Merge new pipeline config keys into legacy in-memory dicts.

New schema (preferred):
  - ``services.vlm`` / ``services.image_edit`` — model URLs and edit backends
  - ``pipeline.stages`` — ordered stage list (same shape as legacy ``phases``)
  - ``step_params.<step_id>`` — per-step knobs (e.g. ``step_params.s5.num_views``)

Legacy keys (still accepted; merged when new keys are absent):
  - ``phase0``, ``phase2_5``, ``phase5``, ``pipeline.phases``

Call :func:`apply_yaml_aliases` immediately after ``yaml.safe_load`` on any
code path that loads pipeline configs so downstream code can keep reading
``phase0`` / ``phase2_5`` until PR-C removes fallbacks.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _merge_prefer_new(legacy: dict | None, new_block: dict) -> dict:
    base = dict(legacy or {})
    base.update(new_block)
    return base


def _map_vlm_service(vlm: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vlm.items():
        if k == "model":
            out["vlm_model"] = v
        elif k == "base_urls":
            out["vlm_base_urls"] = v
        elif k == "backend":
            out["vlm_backend"] = v
        else:
            out[k] = v
    return out


def _map_image_edit_service(ie: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in ie.items():
        if k == "base_urls":
            out["image_edit_base_urls"] = v
        else:
            out[k] = v
    return out


def apply_yaml_aliases(cfg: dict) -> None:
    """Mutate ``cfg`` in place: map ``services`` / ``step_params`` / ``stages``."""
    services = cfg.get("services")
    if isinstance(services, dict):
        vlm = services.get("vlm")
        if isinstance(vlm, dict) and vlm:
            mapped = _map_vlm_service(vlm)
            cfg["phase0"] = _merge_prefer_new(cfg.get("phase0"), mapped)

        ie = services.get("image_edit")
        if isinstance(ie, dict) and ie:
            mapped = _map_image_edit_service(ie)
            cfg["phase2_5"] = _merge_prefer_new(cfg.get("phase2_5"), mapped)

    sp_all = cfg.get("step_params")
    if isinstance(sp_all, dict):
        s5p = sp_all.get("s5")
        if isinstance(s5p, dict) and s5p:
            cfg["phase5"] = _merge_prefer_new(cfg.get("phase5"), s5p)

    pipe = cfg.get("pipeline")
    if isinstance(pipe, dict) and pipe.get("stages") is not None:
        # Keep legacy readers working; ``stages`` wins when both exist.
        pipe["phases"] = deepcopy(pipe["stages"])


__all__ = ["apply_yaml_aliases"]
