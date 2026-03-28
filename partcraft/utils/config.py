"""Configuration loading and validation."""

from __future__ import annotations

import os
import warnings
import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _apply_data_roots_and_layout(cfg: dict) -> None:
    """Apply env overrides and optional standard dataset subpaths under data_dir.

    Convention (PartVerse / same layout on disk):
      {data_dir}/images, {data_dir}/mesh, {data_dir}/slat, {data_dir}/img_Enc

    - ``PARTCRAFT_DATA_ROOT`` — if set, overrides ``data.data_dir`` after YAML load.
    - ``PARTCRAFT_OUTPUT_ROOT`` — if set, overrides ``data.output_dir`` after YAML load.
    - ``data.derive_dataset_subpaths: true`` — fill ``image_npz_dir``, ``mesh_npz_dir``,
      ``slat_dir``, ``img_enc_dir`` from ``data_dir`` only for keys that are missing
      or explicitly null/empty in YAML (so you can still override a single path).

    Offline dataset scripts under ``scripts/datasets/partverse/`` continue to use
    ``PARTVERSE_DATA_ROOT`` / ``--data-root``; set it to the same path as ``data_dir``.
    """
    data = cfg.setdefault("data", {})

    env_data = os.environ.get("PARTCRAFT_DATA_ROOT", "").strip()
    if env_data:
        data["data_dir"] = env_data
    env_out = os.environ.get("PARTCRAFT_OUTPUT_ROOT", "").strip()
    if env_out:
        data["output_dir"] = env_out

    if not data.get("derive_dataset_subpaths"):
        return
    root = data.get("data_dir")
    if not root or not str(root).strip():
        return
    base = Path(str(root).strip())

    mapping = (
        ("image_npz_dir", "images"),
        ("mesh_npz_dir", "mesh"),
        ("slat_dir", "slat"),
        ("img_enc_dir", "img_Enc"),
    )
    for key, sub in mapping:
        v = data.get(key, None)
        if v is None or (isinstance(v, str) and not v.strip()):
            data[key] = str(base / sub)


def _resolve_path(raw: str | Path | None, *, base: Path) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def _resolve_tool_executable(raw: str | None, *, base: Path) -> str | None:
    """Resolve tool executable path while preserving command names like ``blender``."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if "/" not in s and "\\" not in s:
        return s
    return _resolve_path(s, base=base)


def _apply_prerender_paths(cfg: dict) -> None:
    """Normalize prerender path contract under cfg['paths'] and sync cfg['data']."""
    data = cfg.setdefault("data", {})
    paths = cfg.setdefault("paths", {})

    if not paths.get("dataset_root") and data.get("data_dir"):
        paths["dataset_root"] = data.get("data_dir")

    deprecated_env = os.environ.get("PARTVERSE_DATA_ROOT", "").strip()
    if deprecated_env:
        warnings.warn(
            "PARTVERSE_DATA_ROOT is deprecated; prefer config paths.dataset_root.",
            DeprecationWarning,
            stacklevel=2,
        )
        paths["dataset_root"] = deprecated_env
    compat_root = os.environ.get("PARTCRAFT_DATASET_ROOT", "").strip()
    if compat_root:
        warnings.warn(
            "PARTCRAFT_DATASET_ROOT is deprecated for prerender; "
            "prefer config paths.dataset_root.",
            DeprecationWarning,
            stacklevel=2,
        )
        paths["dataset_root"] = compat_root

    dataset_root = _resolve_path(paths.get("dataset_root"), base=_PROJECT_ROOT)
    if dataset_root:
        paths["dataset_root"] = dataset_root
        data["data_dir"] = dataset_root

    base = Path(dataset_root) if dataset_root else _PROJECT_ROOT

    def _default(key: str, rel: str):
        if not paths.get(key):
            paths[key] = str(base / rel)

    _default("source_glb_dir", "source/normalized_glbs")
    _default("source_mesh_zip", "source/mesh.zip")
    _default("captions_json", "source/text_captions.json")
    _default("img_enc_dir", "img_Enc")
    _default("slat_dir", "slat")
    _default("images_npz_dir", "images")
    _default("mesh_npz_dir", "mesh")
    _default("cache_root", "cache")

    for k in (
        "source_glb_dir",
        "source_mesh_zip",
        "captions_json",
        "img_enc_dir",
        "slat_dir",
        "images_npz_dir",
        "mesh_npz_dir",
        "cache_root",
    ):
        paths[k] = _resolve_path(paths.get(k), base=base)

    # Keep existing pipeline/data consumers working with the normalized contract.
    data["img_enc_dir"] = paths["img_enc_dir"]
    data["slat_dir"] = paths["slat_dir"]
    data["image_npz_dir"] = paths["images_npz_dir"]
    data["mesh_npz_dir"] = paths["mesh_npz_dir"]


def _apply_tool_paths(cfg: dict) -> None:
    tools = cfg.setdefault("tools", {})
    if not tools.get("blender_path"):
        tools["blender_path"] = "blender"
    if not tools.get("blender_script"):
        tools["blender_script"] = str(_PROJECT_ROOT / "scripts" / "blender_render.py")

    env_blender = os.environ.get("BLENDER_PATH", "").strip()
    if env_blender:
        warnings.warn(
            "BLENDER_PATH env override is deprecated; prefer config tools.blender_path.",
            DeprecationWarning,
            stacklevel=2,
        )
        tools["blender_path"] = env_blender

    env_blender_script = os.environ.get("BLENDER_SCRIPT", "").strip()
    if env_blender_script:
        warnings.warn(
            "BLENDER_SCRIPT env override is deprecated; prefer config tools.blender_script.",
            DeprecationWarning,
            stacklevel=2,
        )
        tools["blender_script"] = env_blender_script

    tools["blender_path"] = _resolve_tool_executable(
        tools.get("blender_path"),
        base=_PROJECT_ROOT,
    )
    tools["blender_script"] = _resolve_path(
        tools.get("blender_script"),
        base=_PROJECT_ROOT,
    )


def _validate_prerender_config(cfg: dict, *, mode: str | None) -> None:
    paths = cfg.get("paths", {})
    tools = cfg.get("tools", {})
    missing = []
    for key in ("dataset_root", "img_enc_dir", "slat_dir", "images_npz_dir", "mesh_npz_dir"):
        if not paths.get(key):
            missing.append(f"paths.{key}")
    for key in ("blender_path", "blender_script"):
        if not tools.get(key):
            missing.append(f"tools.{key}")
    if mode == "partverse" and not paths.get("source_glb_dir"):
        missing.append("paths.source_glb_dir")
    if mode in {"partobjaverse", "partobjaverse_prepare"} and not paths.get("source_mesh_zip"):
        missing.append("paths.source_mesh_zip")
    if missing:
        msg = ", ".join(missing)
        raise ValueError(
            "Missing required prerender config keys: "
            f"{msg}. See configs/prerender_partverse.yaml "
            "or configs/prerender_partobjaverse.yaml."
        )


def _resolve_trellis_ckpt_path(value: str, ckpt_root: Path) -> str:
    """Turn YAML trellis_*_ckpt into an absolute path under ckpt_root when relative."""
    if not value or not isinstance(value, str):
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    parts = p.parts
    if parts and parts[0] == "checkpoints" and len(parts) > 1:
        rel = Path(*parts[1:])
    else:
        rel = p
    return str((ckpt_root / rel).resolve())


def _apply_ckpt_root(cfg: dict) -> None:
    """Resolve ``ckpt_root`` and expand checkpoint paths (TRELLIS, local VLM on disk).

    Resolution order:
      1. ``PARTCRAFT_CKPT_ROOT`` env (if set)
      2. YAML top-level ``ckpt_root`` (relative paths are under project root)
      3. ``/mnt/zsn/ckpts`` if that directory exists, else ``<project>/checkpoints``

    Writes absolute string to ``cfg["ckpt_root"]``.

    When ``phase0.vlm_backend`` is ``local``, relative ``local_model_path`` and
    ``vlm_model`` values (no ``/`` in the string) are joined to ``ckpt_root``
    so API-style model ids like ``gemini-…`` are unchanged.

    ``phase2_5.trellis_text_ckpt`` / ``trellis_image_ckpt``: relative paths and
    ``checkpoints/...`` prefixes are resolved under ``ckpt_root``; absolute paths kept.
    """
    env = os.environ.get("PARTCRAFT_CKPT_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
    elif cfg.get("ckpt_root"):
        raw = cfg["ckpt_root"]
        r = Path(str(raw).strip())
        root = r.resolve() if r.is_absolute() else (_PROJECT_ROOT / r).resolve()
    else:
        mnt = Path("/mnt/zsn/ckpts")
        local = _PROJECT_ROOT / "checkpoints"
        root = mnt.resolve() if mnt.is_dir() else local.resolve()

    cfg["ckpt_root"] = str(root)

    p25 = cfg.setdefault("phase2_5", {})
    for key in ("trellis_text_ckpt", "trellis_image_ckpt"):
        v = p25.get(key)
        if isinstance(v, str) and v.strip():
            p25[key] = _resolve_trellis_ckpt_path(v.strip(), root)

    if cfg.get("phase0", {}).get("vlm_backend") == "local":
        p0 = cfg.setdefault("phase0", {})
        for key in ("local_model_path", "vlm_model"):
            v = p0.get(key)
            if not isinstance(v, str) or not v.strip():
                continue
            v = v.strip()
            if v.startswith("http://") or v.startswith("https://"):
                continue
            if os.path.isabs(v) or "/" in v:
                continue
            p0[key] = str((root / v).resolve())


def load_config(
    config_path: str | Path = None,
    *,
    for_prerender: bool = False,
    prerender_mode: str | None = None,
) -> dict:
    """Load YAML config, falling back to default.yaml."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "default.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _apply_data_roots_and_layout(cfg)
    _apply_ckpt_root(cfg)
    if for_prerender:
        _apply_prerender_paths(cfg)
        _apply_tool_paths(cfg)
        _validate_prerender_config(cfg, mode=prerender_mode)

    # Resolve environment variables for API keys
    phase0 = cfg.get("phase0", {})
    api_key_env = phase0.get("vlm_api_key_env", "")
    if api_key_env:
        env_val = os.environ.get(api_key_env, "")
        if env_val:
            phase0["vlm_api_key"] = env_val

    # Resolve cache_dir paths relative to output_dir
    output_dir = cfg["data"]["output_dir"]
    for phase_key in ["phase0", "phase1", "phase2", "phase2_5", "phase3", "phase4"]:
        cache = cfg.get(phase_key, {}).get("cache_dir", "")
        if cache and not os.path.isabs(cache):
            cfg[phase_key]["cache_dir"] = os.path.join(output_dir, cache)

    cfg.setdefault("logging", {})
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    if not os.path.isabs(log_dir):
        cfg["logging"]["log_dir"] = os.path.join(output_dir, log_dir)

    return cfg
