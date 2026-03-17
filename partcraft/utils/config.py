"""Configuration loading and validation."""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field


def load_config(config_path: str | Path = None) -> dict:
    """Load YAML config, falling back to default.yaml."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "default.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

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

    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    if not os.path.isabs(log_dir):
        cfg["logging"]["log_dir"] = os.path.join(output_dir, log_dir)

    return cfg
