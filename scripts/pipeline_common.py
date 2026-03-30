#!/usr/bin/env python3
"""Shared utilities for PartCraft3D pipeline scripts (batch & streaming)."""

import os
import sys
from pathlib import Path


# =========================================================================
# Project root resolution
# =========================================================================

def _get_project_root() -> Path:
    """Use $PWD to preserve symlinks (os.getcwd() resolves them on Linux)."""
    script = Path(__file__)
    if not script.is_absolute():
        script = Path(os.environ.get('PWD', os.getcwd())) / script
    return script.parents[1]


PROJECT_ROOT = _get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from partcraft.utils.config import load_config  # noqa: E402
from partcraft.utils.logging import setup_logging  # noqa: E402
from partcraft.io.partcraft_loader import PartCraftDataset  # noqa: E402
from partcraft.phase1_planning.planner import EditSpec  # noqa: E402


# =========================================================================
# Cost constants (Gemini 2.5 Flash via OpenAI-compatible API)
# =========================================================================
# Approximate token counts per call type
COST = {
    # Step 1: Semantic — per object
    "phase0_input_tokens": 1532,    # 4 images (4×258) + ~500 text
    "phase0_output_tokens": 2000,   # object desc + part labels
    "enrich_input_tokens": 1258,    # 1 image (258) + ~1000 text
    "enrich_output_tokens": 3000,   # enriched descriptions + prompts
    # Step 3: 2D Edit — per edit spec
    "2d_edit_input_tokens": 458,    # 1 annotated image (258) + ~200 text
    "2d_edit_output_images": 1,     # 1 edited image
    # Step 5: Quality — per edit spec
    "quality_input_tokens": 2564,   # 8 images (8×258) + ~500 text
    "quality_output_tokens": 500,   # scores + rationale
    # Pricing ($/M tokens, Gemini 2.5 Flash)
    "input_price_per_m": 0.15,
    "output_price_per_m": 0.60,
    "image_output_price": 0.02,     # per image generated
}


# =========================================================================
# Utility
# =========================================================================

def resolve_api_key(cfg: dict) -> str:
    """Resolve VLM API key from current config or configured env var."""
    p0 = cfg.get("phase0", {})
    api_key = p0.get("vlm_api_key", "")

    if not api_key:
        env_var = p0.get("vlm_api_key_env", "")
        if env_var:
            api_key = os.environ.get(env_var, "")

    return api_key


def apply_output_shard_layout(cfg: dict) -> None:
    """Nest ``data.output_dir`` under ``shard_<id>/`` when requested.

    If ``data.output_by_shard`` is true and ``data.shards`` has exactly one
    entry, sets ``data.output_dir`` to ``<output_dir>/shard_<shard>``. After
    :func:`normalize_cache_dirs`, phase caches and streaming ``mesh_pairs``
    live under that subtree (convenient to tar/compress per shard).

    If ``shards`` is empty or lists multiple shards, nesting is skipped and a
    warning is issued — use one shard per config run for per-shard outputs.
    """
    data = cfg.get("data", {})
    if not data.get("output_by_shard"):
        return
    shards = data.get("shards")
    if not shards:
        raise ValueError(
            "[CONFIG_ERROR] data.shards <missing> config "
            "output_by_shard=true requires exactly one shard"
        )
    if len(shards) > 1:
        raise ValueError(
            "[CONFIG_ERROR] data.shards multiple config "
            "output_by_shard=true requires exactly one shard"
        )
    shard = str(shards[0]).strip()
    if not shard:
        return
    nested = f"shard_{shard}"
    out = data.get("output_dir", "outputs")
    root = Path(out)
    if root.name == nested:
        return
    data["output_dir"] = str(root / nested)


def normalize_cache_dirs(cfg: dict) -> None:
    """Rewrite relative cache_dir paths to be under ``data.output_dir`` (once).

    ``load_config`` may already prefix ``cache_dir`` with the pre-shard
    ``output_dir`` (e.g. ``outputs/partverse/cache/phase0``). After
    :func:`apply_output_shard_layout`, ``output_dir`` becomes
    ``.../shard_XX``; naively joining again produced paths like
    ``shard_05/outputs/partverse/cache/phase0``.
    """
    apply_output_shard_layout(cfg)
    _out = cfg["data"].get("output_dir", "outputs")
    _out_norm = _out.replace("\\", "/")
    _markers = (
        "cache/phase0", "cache/phase1", "cache/phase2",
        "cache/phase2_5", "cache/phase3", "cache/phase4",
    )
    for _phase_key in ("phase0", "phase1", "phase2", "phase2_5", "phase3", "phase4"):
        _pcfg = cfg.get(_phase_key, {})
        _cd = _pcfg.get("cache_dir", "")
        if not _cd or os.path.isabs(_cd):
            continue
        cd_norm = _cd.replace("\\", "/")
        if cd_norm == _out_norm or cd_norm.startswith(_out_norm + "/"):
            continue
        rel = None
        for m in _markers:
            if m in cd_norm:
                rel = cd_norm[cd_norm.index(m):]
                break
        if rel is not None:
            _pcfg["cache_dir"] = os.path.join(_out, *rel.split("/"))
        else:
            _pcfg["cache_dir"] = os.path.join(_out, _cd)


def set_attn_backend(cfg: dict) -> None:
    """Set ATTN_BACKEND from config (unless already set via env)."""
    attn_backend = cfg.get("pipeline", {}).get("attn_backend", "")
    if attn_backend and "ATTN_BACKEND" not in os.environ:
        os.environ["ATTN_BACKEND"] = attn_backend


def create_dataset(cfg: dict) -> PartCraftDataset:
    """Create PartCraftDataset from config."""
    return PartCraftDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )


def resolve_data_dirs(cfg: dict) -> tuple[str | None, str | None]:
    """Return (slat_dir, img_enc_dir) from config data section.

    Reads fields:
        data.slat_dir     — pre-encoded SLAT ({obj_id}_feats.pt / _coords.pt)  [required]
        data.img_enc_dir  — Blender render outputs ({obj_id}/000.png .. voxels.ply)  [optional]

    slat_dir is required; img_enc_dir may be None (mesh NPZ fallback used).
    """
    data_cfg = cfg.get("data", {})
    slat_dir = data_cfg.get("slat_dir", None)
    img_enc_dir = data_cfg.get("img_enc_dir", None)
    if not slat_dir:
        raise ValueError(
            "[CONFIG_ERROR] data.slat_dir <missing> config "
            "must be set explicitly; no runtime fallback is allowed"
        )
    return slat_dir, img_enc_dir
