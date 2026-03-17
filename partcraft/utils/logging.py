"""Logging setup."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


_CONFIGURED = False


def setup_logging(cfg: dict, phase_name: str = "") -> logging.Logger:
    """Configure logging for a pipeline phase."""
    global _CONFIGURED

    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs")
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    os.makedirs(log_dir, exist_ok=True)

    logger_name = f"partcraft.{phase_name}" if phase_name else "partcraft"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not _CONFIGURED:
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logging.getLogger("partcraft").addHandler(ch)
        logging.getLogger("partcraft").setLevel(level)

        # File handler
        if phase_name:
            fh = logging.FileHandler(os.path.join(log_dir, f"{phase_name}.log"))
            fh.setFormatter(fmt)
            logging.getLogger("partcraft").addHandler(fh)

        _CONFIGURED = True

    return logger
