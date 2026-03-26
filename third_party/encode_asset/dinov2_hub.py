"""DINOv2 ViT-L/14 (registers) for encode: local weights + dinov2 Python package.

- Weights: downloaded once to disk (file-locked), then loaded via ``weights=<path>`` so
  no ``torch.hub.load_state_dict_from_url`` to Meta CDN on every process.
- Model code: shallow ``git clone`` of facebookresearch/dinov2 under the torch hub dir,
  or a path you provide. We intentionally do **not** use ``torch.hub.load(..., 'github/...')``
  because that always hits ``https://github.com/.../tree/main/`` and fails on blocked networks
  (``RemoteDisconnected`` / connection reset).

Env:
  PARTCRAFT_DINOV2_REPO — root of a facebookresearch/dinov2 checkout (contains ``dinov2/hub/``).
    If set, no clone is attempted; must already exist.
  PARTCRAFT_DINOV2_GIT_URL — clone URL when auto-cloning (default: official GitHub repo).
    Use an internal mirror if GitHub is unreachable.
  PARTCRAFT_DINOV2_WEIGHTS — explicit path to ``*_pretrain.pth`` (skip download if present).
  PARTCRAFT_CKPT_ROOT — directory for weights; default ``/mnt/zsn/ckpts`` if it exists,
    else ``<torch.hub.get_dir()>/partcraft_ckpts``. File lives at
    ``{root}/dinov2/dinov2_vitl14_reg4_pretrain.pth``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

DINOV2_VITL14_REG4_PRETRAIN_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
)
DINOV2_WEIGHTS_FILENAME = "dinov2_vitl14_reg4_pretrain.pth"

_dinov2_model = None


def _hub_dir() -> Path:
    return Path(torch.hub.get_dir())


def _dinov2_repo_root() -> Path:
    env = os.environ.get("PARTCRAFT_DINOV2_REPO", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _hub_dir() / "facebookresearch_dinov2_main"


def _is_user_dinov2_repo() -> bool:
    return bool(os.environ.get("PARTCRAFT_DINOV2_REPO", "").strip())


def _dinov2_hub_ready() -> bool:
    return (_dinov2_repo_root() / "dinov2" / "hub" / "backbones.py").is_file()


def _default_ckpt_root() -> Path:
    env = os.environ.get("PARTCRAFT_CKPT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    zsn = Path("/mnt/zsn/ckpts")
    if zsn.is_dir():
        return zsn.resolve()
    return (_hub_dir() / "partcraft_ckpts").resolve()


def dinov2_weights_path() -> Path:
    """Resolved path to the ViT-L/14 reg4 pretrained checkpoint."""
    explicit = os.environ.get("PARTCRAFT_DINOV2_WEIGHTS", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return _default_ckpt_root() / "dinov2" / DINOV2_WEIGHTS_FILENAME


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        os.replace(tmp, dest)
    except BaseException:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def ensure_dinov2_vitl14_reg_weights_file() -> Path:
    """Ensure pretrained .pth exists on disk (download under cross-process lock if needed)."""
    path = dinov2_weights_path()
    if path.is_file():
        return path
    try:
        import fcntl
    except ImportError:
        _download_file(DINOV2_VITL14_REG4_PRETRAIN_URL, path)
        return path

    lock_path = path.parent / ".dinov2_download.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            if not path.is_file():
                _download_file(DINOV2_VITL14_REG4_PRETRAIN_URL, path)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
    return path


def _clone_dinov2_repo_shallow() -> None:
    """Shallow-clone dinov2 next to torch hub cache (no torch.hub GitHub probe)."""
    if _dinov2_hub_ready():
        return
    dest = _dinov2_repo_root()
    hub_default = (_hub_dir() / "facebookresearch_dinov2_main").resolve()

    if dest.exists() and not _dinov2_hub_ready():
        if _is_user_dinov2_repo():
            raise RuntimeError(
                f"PARTCRAFT_DINOV2_REPO={dest} is missing dinov2/hub/backbones.py — "
                "point it at a full facebookresearch/dinov2 checkout."
            )
        if dest.resolve() == hub_default:
            shutil.rmtree(dest)
        else:
            raise RuntimeError(
                f"DINOv2 path {dest} exists but is not a valid checkout; remove it or set "
                "PARTCRAFT_DINOV2_REPO to a good clone."
            )

    url = (
        os.environ.get("PARTCRAFT_DINOV2_GIT_URL", "").strip()
        or "https://github.com/facebookresearch/dinov2.git"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "`git` not found; install git or set PARTCRAFT_DINOV2_REPO to an existing "
            "facebookresearch/dinov2 directory."
        ) from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + (e.stdout or "")
        raise RuntimeError(
            "Could not clone facebookresearch/dinov2 (GitHub or default git URL may be blocked).\n"
            "Options:\n"
            "  • On a machine with access: git clone --depth 1 <url> <dir> then "
            "export PARTCRAFT_DINOV2_REPO=<dir>\n"
            "  • Or set PARTCRAFT_DINOV2_GIT_URL to an internal mirror, then rerun.\n"
            f"Git said:\n{err.strip()}"
        ) from e

    if not _dinov2_hub_ready():
        raise RuntimeError(
            "git clone finished but dinov2/hub/backbones.py is missing — check PARTCRAFT_DINOV2_GIT_URL."
        )


def ensure_facebook_dinov2_hub_clone() -> None:
    """Ensure dinov2 Python package exists (local path or shallow git clone)."""
    if _dinov2_hub_ready():
        return
    if _is_user_dinov2_repo():
        raise RuntimeError(
            f"PARTCRAFT_DINOV2_REPO={_dinov2_repo_root()} is set but dinov2/hub/backbones.py "
            "was not found. Clone facebookresearch/dinov2 there, or unset PARTCRAFT_DINOV2_REPO "
            "to auto-clone under the torch hub directory."
        )
    try:
        import fcntl
    except ImportError:
        _clone_dinov2_repo_shallow()
        return

    hub = _hub_dir()
    hub.mkdir(parents=True, exist_ok=True)
    lock_path = hub / ".partcraft_dinov2_hub.lock"
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            if not _dinov2_hub_ready():
                _clone_dinov2_repo_shallow()
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def load_dinov2_vitl14_reg(pretrained: bool = True):
    """Build ViT-L/14+reg; load weights from local file when pretrained=True."""
    ensure_facebook_dinov2_hub_clone()
    root = str(_dinov2_repo_root().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from dinov2.hub.backbones import dinov2_vitl14_reg

    if not pretrained:
        return dinov2_vitl14_reg(pretrained=False)
    wpath = ensure_dinov2_vitl14_reg_weights_file()
    return dinov2_vitl14_reg(pretrained=True, weights=str(wpath))


def get_dinov2_vitl14_reg():
    """Single process-wide instance (avoids reloading ~1.2GB weights per object)."""
    global _dinov2_model
    if _dinov2_model is None:
        _dinov2_model = load_dinov2_vitl14_reg(pretrained=True)
    return _dinov2_model
