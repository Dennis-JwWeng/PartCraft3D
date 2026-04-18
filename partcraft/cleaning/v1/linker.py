"""Materialize a source file into the v1 dataset by hardlink/symlink/copy."""
from __future__ import annotations

import enum
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


class LinkMode(str, enum.Enum):
    HARDLINK = "hardlink"
    SYMLINK = "symlink"
    COPY = "copy"


@dataclass(frozen=True)
class LinkResult:
    src: Path
    dst: Path
    mode_used: LinkMode
    skipped: bool = False
    fell_back: bool = False


def link_one(
    src: Path, dst: Path, *, mode: LinkMode, force: bool = False,
) -> LinkResult:
    src = Path(src); dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"link_one: source missing: {src}")
    if dst.exists() and not force:
        return LinkResult(src=src, dst=dst, mode_used=mode, skipped=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    fell_back = False
    actual_mode = mode
    try:
        if mode is LinkMode.HARDLINK:
            os.link(src, dst)
        elif mode is LinkMode.SYMLINK:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
    except OSError:
        if mode is LinkMode.HARDLINK:
            shutil.copy2(src, dst)
            fell_back = True
            actual_mode = LinkMode.COPY
        else:
            raise
    return LinkResult(src=src, dst=dst, mode_used=actual_mode, fell_back=fell_back)
