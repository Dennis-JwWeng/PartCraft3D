from __future__ import annotations

import json
import shutil
from pathlib import Path


def normalize_shard(shard: str | None) -> str | None:
    if shard is None:
        return None
    s = str(shard).strip()
    if not s:
        return None
    return s.zfill(2)


def run_token(tag: str | None, shard: str | None) -> str:
    """Build output token. Prefer explicit tag, fallback to shard token."""
    if tag:
        return str(tag)
    if shard:
        return f"shard{shard}"
    return ""


def shard_output_root(cfg: dict, shard: str | None) -> Path:
    out = Path(cfg["data"]["output_dir"])
    shard_leaf = f"shard_{shard}" if shard else "shard_unknown"
    return out if out.name == shard_leaf else (out / shard_leaf)


def pipeline_report_dir(cfg: dict, shard: str | None) -> Path:
    report_dir = shard_output_root(cfg, shard) / "pipeline" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def write_stage_diag(report_dir: Path, stage: str, payload: dict):
    with open(report_dir / f"{stage}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def pipeline_manifest_dir(cfg: dict, shard: str | None, phase: str) -> Path:
    mdir = shard_output_root(cfg, shard) / "pipeline" / "manifests" / phase
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir


def sync_manifest_link(cfg: dict, shard: str | None, phase: str, filename: str, src: Path):
    if not src or not src.exists():
        return
    mdir = pipeline_manifest_dir(cfg, shard, phase)
    dst = mdir / filename
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink(missing_ok=True)
    dst.symlink_to(src)
