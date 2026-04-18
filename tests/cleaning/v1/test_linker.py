from pathlib import Path

import pytest

from partcraft.cleaning.v1.linker import LinkMode, link_one


@pytest.fixture
def src(tmp_path: Path) -> Path:
    p = tmp_path / "src" / "a.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"hello")
    return p


def test_hardlink_same_fs(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    res = link_one(src, dst, mode=LinkMode.HARDLINK)
    assert dst.read_bytes() == b"hello"
    assert dst.stat().st_ino == src.stat().st_ino
    assert res.mode_used == LinkMode.HARDLINK


def test_symlink_mode(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    res = link_one(src, dst, mode=LinkMode.SYMLINK)
    assert dst.is_symlink()
    assert dst.read_bytes() == b"hello"
    assert res.mode_used == LinkMode.SYMLINK


def test_copy_mode(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    res = link_one(src, dst, mode=LinkMode.COPY)
    assert not dst.is_symlink()
    assert dst.stat().st_ino != src.stat().st_ino
    assert res.mode_used == LinkMode.COPY


def test_existing_dst_is_skipped(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"existing")
    res = link_one(src, dst, mode=LinkMode.HARDLINK)
    assert res.skipped is True
    assert dst.read_bytes() == b"existing"


def test_force_overwrites_existing(tmp_path: Path, src: Path):
    dst = tmp_path / "dst" / "a.bin"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"existing")
    res = link_one(src, dst, mode=LinkMode.HARDLINK, force=True)
    assert res.skipped is False
    assert dst.read_bytes() == b"hello"


def test_missing_src_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        link_one(tmp_path / "nope.bin", tmp_path / "dst.bin", mode=LinkMode.HARDLINK)
