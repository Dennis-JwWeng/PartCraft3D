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


def _layout_partverse_tree(root: Path) -> None:
    (root / "source/normalized_glbs").mkdir(parents=True, exist_ok=True)
    (root / "img_Enc").mkdir(parents=True, exist_ok=True)
    (root / "slat").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "mesh").mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    (root / "captions.json").write_text("{}", encoding="utf-8")


def test_prerender_paths_are_normalized_and_synced(tmp_path: Path):
    root = tmp_path / "dataset"
    _layout_partverse_tree(root)
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)

    cfg_path = tmp_path / "prerender.yaml"
    blender_script = tmp_path / "fake_blender.py"
    blender_script.write_text("# stub", encoding="utf-8")

    _write_cfg(
        cfg_path,
        {
            "ckpt_root": str(ckpts),
            "data": {
                "data_dir": str(root),
                "derive_dataset_subpaths": True,
                "output_dir": str(tmp_path / "outputs"),
                "shards": ["00"],
            },
            "paths": {
                "dataset_root": str(root),
                "source_glb_dir": str(root / "source/normalized_glbs"),
                "captions_json": str(root / "captions.json"),
                "img_enc_dir": str(root / "img_Enc"),
                "slat_dir": str(root / "slat"),
                "images_npz_dir": str(root / "images"),
                "mesh_npz_dir": str(root / "mesh"),
                "cache_root": str(root / "cache"),
            },
            "tools": {"blender_path": "blender", "blender_script": str(blender_script)},
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
    """``PARTCRAFT_DATASET_ROOT`` overrides ``paths.dataset_root``; relative subpaths resolve under it."""
    placeholder_root = tmp_path / "placeholder"
    placeholder_root.mkdir(parents=True, exist_ok=True)
    compat_root = tmp_path / "compat_data"
    _layout_partverse_tree(compat_root)
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)

    cfg_path = tmp_path / "prerender.yaml"
    blender_script = tmp_path / "fake_blender.py"
    blender_script.write_text("# stub", encoding="utf-8")

    _write_cfg(
        cfg_path,
        {
            "ckpt_root": str(ckpts),
            "data": {
                "data_dir": str(placeholder_root),
                "derive_dataset_subpaths": True,
                "output_dir": str(tmp_path / "outputs"),
                "shards": ["00"],
            },
            "paths": {
                "dataset_root": str(placeholder_root),
                "source_glb_dir": "source/normalized_glbs",
                "captions_json": "captions.json",
                "img_enc_dir": "img_Enc",
                "slat_dir": "slat",
                "images_npz_dir": "images",
                "mesh_npz_dir": "mesh",
                "cache_root": "cache",
            },
            "tools": {"blender_path": "blender", "blender_script": str(blender_script)},
        },
    )
    monkeypatch.setenv("PARTCRAFT_DATASET_ROOT", str(compat_root))

    with pytest.warns(DeprecationWarning):
        cfg = load_config(str(cfg_path), for_prerender=True, prerender_mode="partverse")

    assert cfg["paths"]["dataset_root"] == str(compat_root.resolve())
    assert cfg["paths"]["img_enc_dir"] == str((compat_root / "img_Enc").resolve())
