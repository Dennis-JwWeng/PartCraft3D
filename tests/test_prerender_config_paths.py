#!/usr/bin/env python3
"""Tests for prerender config-driven path normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config


def _write_cfg(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_prerender_paths_are_normalized_and_synced(tmp_path: Path):
    cfg_path = tmp_path / "prerender.yaml"
    _write_cfg(
        cfg_path,
        {
            "data": {
                "data_dir": "data/partverse",
                "derive_dataset_subpaths": True,
                "output_dir": "outputs/partverse",
                "shards": ["00"],
            },
            "paths": {
                "dataset_root": "data/partverse",
                "source_glb_dir": "data/partverse/source/normalized_glbs",
                "img_enc_dir": "data/partverse/img_Enc",
                "slat_dir": "data/partverse/slat",
                "images_npz_dir": "data/partverse/images",
                "mesh_npz_dir": "data/partverse/mesh",
            },
            "tools": {"blender_path": "blender", "blender_script": "scripts/blender_render.py"},
        },
    )

    cfg = load_config(str(cfg_path), for_prerender=True, prerender_mode="partverse")

    assert Path(cfg["paths"]["dataset_root"]).is_absolute()
    assert Path(cfg["paths"]["source_glb_dir"]).is_absolute()
    assert Path(cfg["paths"]["img_enc_dir"]).is_absolute()
    assert cfg["data"]["image_npz_dir"] == cfg["paths"]["images_npz_dir"]
    assert cfg["data"]["mesh_npz_dir"] == cfg["paths"]["mesh_npz_dir"]
    assert cfg["data"]["slat_dir"] == cfg["paths"]["slat_dir"]
    assert cfg["tools"]["blender_path"] == "blender"
    assert Path(cfg["tools"]["blender_script"]).is_absolute()


def test_prerender_env_compat_override_dataset_root(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "prerender.yaml"
    _write_cfg(
        cfg_path,
        {
            "data": {
                "data_dir": "data/partverse",
                "derive_dataset_subpaths": True,
                "output_dir": "outputs/partverse",
                "shards": ["00"],
            },
            "paths": {"dataset_root": "data/partverse"},
            "tools": {"blender_path": "blender", "blender_script": "scripts/blender_render.py"},
        },
    )
    compat_root = tmp_path / "compat_data"
    monkeypatch.setenv("PARTCRAFT_DATASET_ROOT", str(compat_root))

    with pytest.warns(DeprecationWarning):
        cfg = load_config(str(cfg_path), for_prerender=True, prerender_mode="partverse")

    assert cfg["paths"]["dataset_root"] == str(compat_root.resolve())
    assert cfg["paths"]["img_enc_dir"] == str((compat_root / "img_Enc").resolve())

