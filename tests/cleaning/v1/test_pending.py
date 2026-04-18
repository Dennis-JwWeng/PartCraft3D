from pathlib import Path

from partcraft.cleaning.v1.pending import DelLatentPending, PendingEntry


def test_append_and_iter(tmp_path: Path):
    pending = DelLatentPending(tmp_path / "del_latent.txt")
    pending.append(PendingEntry("05", "objA", "del_objA_000", suffix=""))
    pending.append(PendingEntry("08", "objB", "del_objB_000", suffix="__r2"))
    entries = list(pending.iter_entries())
    assert entries == [
        PendingEntry("05", "objA", "del_objA_000", suffix=""),
        PendingEntry("08", "objB", "del_objB_000", suffix="__r2"),
    ]


def test_remove_keeps_others(tmp_path: Path):
    pending = DelLatentPending(tmp_path / "del_latent.txt")
    e1 = PendingEntry("05", "objA", "del_objA_000", suffix="")
    e2 = PendingEntry("08", "objB", "del_objB_000", suffix="")
    pending.append(e1); pending.append(e2)
    pending.remove(e1)
    assert list(pending.iter_entries()) == [e2]


def test_dedup_on_append(tmp_path: Path):
    pending = DelLatentPending(tmp_path / "del_latent.txt")
    e = PendingEntry("05", "objA", "del_objA_000", suffix="")
    pending.append(e); pending.append(e)
    assert len(list(pending.iter_entries())) == 1
