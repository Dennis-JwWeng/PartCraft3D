"""Configuration loading and validation."""

from __future__ import annotations

import os
import logging
import warnings
import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CFG_LOGGER = logging.getLogger("partcraft.config")


def _sources(cfg: dict) -> dict:
    return cfg.setdefault("_config_path_sources", {})


def _mark_source(cfg: dict, key: str, source: str) -> None:
    _sources(cfg)[key] = source


def _get_source(cfg: dict, key: str) -> str:
    return _sources(cfg).get(key, "config")


def _config_error(key: str, value: str | None, source: str, reason: str) -> ValueError:
    v = value if value is not None else "<missing>"
    return ValueError(f"[CONFIG_ERROR] {key} {v} {source} {reason}")


def _seed_sources_from_yaml(cfg: dict) -> None:
    data = cfg.get("data", {})
    paths = cfg.get("paths", {})
    tools = cfg.get("tools", {})
    for key, val in (
        ("data.data_dir", data.get("data_dir")),
        ("data.output_dir", data.get("output_dir")),
        ("data.images_root", data.get("images_root")),
        ("data.mesh_root", data.get("mesh_root")),
        ("data.image_npz_dir", data.get("image_npz_dir")),
        ("data.mesh_npz_dir", data.get("mesh_npz_dir")),
        ("data.slat_dir", data.get("slat_dir")),
        ("data.img_enc_dir", data.get("img_enc_dir")),
        ("paths.dataset_root", paths.get("dataset_root")),
        ("paths.source_glb_dir", paths.get("source_glb_dir")),
        ("paths.source_mesh_zip", paths.get("source_mesh_zip")),
        ("paths.captions_json", paths.get("captions_json")),
        ("paths.img_enc_dir", paths.get("img_enc_dir")),
        ("paths.slat_dir", paths.get("slat_dir")),
        ("paths.images_npz_dir", paths.get("images_npz_dir")),
        ("paths.mesh_npz_dir", paths.get("mesh_npz_dir")),
        ("paths.cache_root", paths.get("cache_root")),
        ("tools.blender_path", tools.get("blender_path")),
        ("tools.blender_script", tools.get("blender_script")),
        ("ckpt_root", cfg.get("ckpt_root")),
    ):
        if val is not None and str(val).strip():
            _mark_source(cfg, key, "config")


def _log_resolved_paths(cfg: dict, *, for_prerender: bool) -> None:
    data = cfg.get("data", {})
    paths = cfg.get("paths", {})
    entries = [
        ("data.data_dir", data.get("data_dir")),
        ("data.output_dir", data.get("output_dir")),
        ("data.images_root", data.get("images_root")),
        ("data.mesh_root", data.get("mesh_root")),
        ("data.image_npz_dir", data.get("image_npz_dir")),
        ("data.mesh_npz_dir", data.get("mesh_npz_dir")),
        ("data.slat_dir", data.get("slat_dir")),
        ("data.img_enc_dir", data.get("img_enc_dir")),
        ("ckpt_root", cfg.get("ckpt_root")),
    ]
    if for_prerender:
        entries.extend([
            ("paths.dataset_root", paths.get("dataset_root")),
            ("paths.source_glb_dir", paths.get("source_glb_dir")),
            ("paths.source_mesh_zip", paths.get("source_mesh_zip")),
            ("paths.captions_json", paths.get("captions_json")),
            ("paths.img_enc_dir", paths.get("img_enc_dir")),
            ("paths.slat_dir", paths.get("slat_dir")),
            ("paths.images_npz_dir", paths.get("images_npz_dir")),
            ("paths.mesh_npz_dir", paths.get("mesh_npz_dir")),
            ("paths.cache_root", paths.get("cache_root")),
            ("tools.blender_path", cfg.get("tools", {}).get("blender_path")),
            ("tools.blender_script", cfg.get("tools", {}).get("blender_script")),
        ])
    for key, val in entries:
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        _CFG_LOGGER.info(
            "[CONFIG_PATH] %s=%s source=%s",
            key,
            val,
            _get_source(cfg, key),
        )


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
        _mark_source(cfg, "data.data_dir", "env_override")
    env_out = os.environ.get("PARTCRAFT_OUTPUT_ROOT", "").strip()
    if env_out:
        data["output_dir"] = env_out
        _mark_source(cfg, "data.output_dir", "env_override")

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
    )
    for key, sub in mapping:
        v = data.get(key, None)
        if v is None or (isinstance(v, str) and not v.strip()):
            data[key] = str(base / sub)
            _mark_source(cfg, f"data.{key}", "derived")


def _norm_abs_path(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return str(Path(s).expanduser().resolve())


def _sync_pipeline_v2_data_paths(cfg: dict) -> None:
    """Align ``data.images_root`` / ``data.mesh_root`` with ``image_npz_dir`` / ``mesh_npz_dir``.

    ``partcraft.pipeline_v2`` reads ``images_root`` and ``mesh_root``; older docs and
    ``derive_dataset_subpaths`` populate ``image_npz_dir`` / ``mesh_npz_dir``. When only
    one naming scheme is set, copy to the other. If both disagree, fail fast.
    """
    data = cfg.setdefault("data", {})
    pairs = (
        ("images_root", "image_npz_dir"),
        ("mesh_root", "mesh_npz_dir"),
    )
    for root_key, npz_key in pairs:
        raw_r = data.get(root_key)
        raw_n = data.get(npz_key)
        abs_r = _norm_abs_path(raw_r) if raw_r is not None else None
        abs_n = _norm_abs_path(raw_n) if raw_n is not None else None
        if abs_r and abs_n and abs_r != abs_n:
            raise _config_error(
                f"data.{root_key} vs data.{npz_key}",
                f"{raw_r} vs {raw_n}",
                "config",
                "paths must match when both are set (same dataset roots)",
            )
        if abs_r and not abs_n:
            data[npz_key] = abs_r
            _mark_source(cfg, f"data.{npz_key}", f"derived_from_{root_key}")
        elif abs_n and not abs_r:
            data[root_key] = abs_n
            _mark_source(cfg, f"data.{root_key}", f"derived_from_{npz_key}")


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

    deprecated_env = os.environ.get("PARTVERSE_DATA_ROOT", "").strip()
    if deprecated_env:
        warnings.warn(
            "PARTVERSE_DATA_ROOT is deprecated; prefer config paths.dataset_root.",
            DeprecationWarning,
            stacklevel=2,
        )
        paths["dataset_root"] = deprecated_env
        _mark_source(cfg, "paths.dataset_root", "env_override")
    compat_root = os.environ.get("PARTCRAFT_DATASET_ROOT", "").strip()
    if compat_root:
        warnings.warn(
            "PARTCRAFT_DATASET_ROOT is deprecated for prerender; "
            "prefer config paths.dataset_root.",
            DeprecationWarning,
            stacklevel=2,
        )
        paths["dataset_root"] = compat_root
        _mark_source(cfg, "paths.dataset_root", "env_override")

    dataset_root = _resolve_path(paths.get("dataset_root"), base=_PROJECT_ROOT)
    if not dataset_root:
        raise _config_error(
            "paths.dataset_root",
            None,
            _get_source(cfg, "paths.dataset_root"),
            "must be explicitly set for prerender",
        )
    paths["dataset_root"] = dataset_root
    data["data_dir"] = dataset_root
    _mark_source(cfg, "data.data_dir", "derived")

    base = Path(dataset_root)

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
        raw = paths.get(k)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        paths[k] = _resolve_path(raw, base=base)

    # Keep existing pipeline/data consumers working with the normalized contract.
    if paths.get("img_enc_dir"):
        data["img_enc_dir"] = paths["img_enc_dir"]
        _mark_source(cfg, "data.img_enc_dir", "derived")
    if paths.get("slat_dir"):
        data["slat_dir"] = paths["slat_dir"]
        _mark_source(cfg, "data.slat_dir", "derived")
    if paths.get("images_npz_dir"):
        data["image_npz_dir"] = paths["images_npz_dir"]
        _mark_source(cfg, "data.image_npz_dir", "derived")
    if paths.get("mesh_npz_dir"):
        data["mesh_npz_dir"] = paths["mesh_npz_dir"]
        _mark_source(cfg, "data.mesh_npz_dir", "derived")


def _apply_tool_paths(cfg: dict) -> None:
    tools = cfg.setdefault("tools", {})
    env_blender = os.environ.get("BLENDER_PATH", "").strip()
    if env_blender:
        warnings.warn(
            "BLENDER_PATH env override is deprecated; prefer config tools.blender_path.",
            DeprecationWarning,
            stacklevel=2,
        )
        tools["blender_path"] = env_blender
        _mark_source(cfg, "tools.blender_path", "env_override")

    env_blender_script = os.environ.get("BLENDER_SCRIPT", "").strip()
    if env_blender_script:
        warnings.warn(
            "BLENDER_SCRIPT env override is deprecated; prefer config tools.blender_script.",
            DeprecationWarning,
            stacklevel=2,
        )
        tools["blender_script"] = env_blender_script
        _mark_source(cfg, "tools.blender_script", "env_override")

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
    for key in ("dataset_root", "img_enc_dir", "slat_dir", "images_npz_dir", "mesh_npz_dir", "cache_root"):
        if not paths.get(key):
            missing.append(f"paths.{key}")
    for key in ("blender_path", "blender_script"):
        if not tools.get(key):
            missing.append(f"tools.{key}")
    if mode == "partverse" and not paths.get("source_glb_dir"):
        missing.append("paths.source_glb_dir")
    if mode == "partverse" and not paths.get("captions_json"):
        missing.append("paths.captions_json")
    if mode in {"partobjaverse", "partobjaverse_prepare"} and not paths.get("source_mesh_zip"):
        missing.append("paths.source_mesh_zip")
    if missing:
        msg = ", ".join(missing)
        raise _config_error("prerender.required_keys", msg, "config", "missing required keys")

    dataset_root = Path(paths["dataset_root"])
    if not dataset_root.exists():
        raise _config_error("paths.dataset_root", str(dataset_root), _get_source(cfg, "paths.dataset_root"), "path does not exist")
    if not dataset_root.is_dir():
        raise _config_error("paths.dataset_root", str(dataset_root), _get_source(cfg, "paths.dataset_root"), "must be a directory")

    if mode == "partverse":
        glb_dir = Path(paths["source_glb_dir"])
        if not glb_dir.is_dir():
            raise _config_error("paths.source_glb_dir", str(glb_dir), _get_source(cfg, "paths.source_glb_dir"), "must be an existing directory")
        captions = Path(paths["captions_json"])
        if not captions.is_file():
            raise _config_error("paths.captions_json", str(captions), _get_source(cfg, "paths.captions_json"), "must be an existing file")

    if mode in {"partobjaverse", "partobjaverse_prepare"}:
        mesh_zip = Path(paths["source_mesh_zip"])
        if not mesh_zip.is_file():
            raise _config_error("paths.source_mesh_zip", str(mesh_zip), _get_source(cfg, "paths.source_mesh_zip"), "must be an existing file")

    blender_script = tools.get("blender_script")
    if blender_script and ("/" in blender_script or "\\" in blender_script):
        bsp = Path(blender_script)
        if not bsp.is_file():
            raise _config_error("tools.blender_script", str(bsp), _get_source(cfg, "tools.blender_script"), "must be an existing file")


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

    When ``services.vlm.vlm_backend`` / ``services.vlm.backend`` is ``local``, relative ``local_model_path`` and
    ``vlm_model`` values (no ``/`` in the string) are joined to ``ckpt_root``
    so API-style model ids like ``gemini-…`` are unchanged.

    ``services.image_edit.trellis_text_ckpt`` / ``trellis_image_ckpt``: relative paths and
    ``checkpoints/...`` prefixes are resolved under ``ckpt_root``; absolute paths kept.
    """
    env = os.environ.get("PARTCRAFT_CKPT_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        _mark_source(cfg, "ckpt_root", "env_override")
    elif cfg.get("ckpt_root"):
        raw = cfg["ckpt_root"]
        r = Path(str(raw).strip())
        root = r.resolve() if r.is_absolute() else (_PROJECT_ROOT / r).resolve()
    else:
        raise _config_error("ckpt_root", None, "config", "must be explicitly set (or override PARTCRAFT_CKPT_ROOT)")

    cfg["ckpt_root"] = str(root)
    if not root.is_dir():
        raise _config_error("ckpt_root", str(root), _get_source(cfg, "ckpt_root"), "directory does not exist")

    srv = cfg.get("services")
    if isinstance(srv, dict):
        p25 = srv.get("image_edit")
        if isinstance(p25, dict):
            for key in ("trellis_text_ckpt", "trellis_image_ckpt"):
                v = p25.get(key)
                if isinstance(v, str) and v.strip():
                    p25[key] = _resolve_trellis_ckpt_path(v.strip(), root)

        p0 = srv.get("vlm")
        if isinstance(p0, dict):
            backend = p0.get("vlm_backend") or p0.get("backend")
            if backend == "local":
                for key in ("local_model_path", "vlm_model", "model"):
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
    _seed_sources_from_yaml(cfg)

    _apply_data_roots_and_layout(cfg)
    if not for_prerender:
        _sync_pipeline_v2_data_paths(cfg)
    _apply_ckpt_root(cfg)
    if for_prerender:
        _apply_prerender_paths(cfg)
        _apply_tool_paths(cfg)
        _validate_prerender_config(cfg, mode=prerender_mode)

    # Resolve environment variables for API keys (services.vlm)
    srv = cfg.get("services")
    if isinstance(srv, dict):
        vlm = srv.get("vlm")
        if isinstance(vlm, dict):
            api_key_env = vlm.get("vlm_api_key_env", "")
            if api_key_env:
                env_val = os.environ.get(api_key_env, "")
                if env_val:
                    vlm["vlm_api_key"] = env_val

    # Resolve cache_dir paths relative to output_dir
    output_dir = cfg["data"]["output_dir"]
    if isinstance(srv, dict):
        vlm = srv.get("vlm")
        if isinstance(vlm, dict):
            cache = vlm.get("cache_dir", "")
            if cache and not os.path.isabs(cache):
                vlm["cache_dir"] = os.path.join(output_dir, cache)
        ie = srv.get("image_edit")
        if isinstance(ie, dict):
            cache = ie.get("cache_dir", "")
            if cache and not os.path.isabs(cache):
                ie["cache_dir"] = os.path.join(output_dir, cache)

    for phase_key in ["phase1", "phase2", "phase3", "phase4"]:
        cache = cfg.get(phase_key, {}).get("cache_dir", "")
        if cache and not os.path.isabs(cache):
            cfg[phase_key]["cache_dir"] = os.path.join(output_dir, cache)

    cfg.setdefault("logging", {})
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    if not os.path.isabs(log_dir):
        cfg["logging"]["log_dir"] = os.path.join(output_dir, log_dir)

    _log_resolved_paths(cfg, for_prerender=for_prerender)

    return cfg
