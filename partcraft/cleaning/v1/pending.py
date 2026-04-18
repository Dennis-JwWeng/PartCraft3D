"""Pending-list manager for deletion edits awaiting latent encoding."""
from __future__ import annotations

import fcntl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PendingEntry:
    shard: str
    obj_id: str
    edit_id: str
    suffix: str = ""

    def to_line(self) -> str:
        return f"{self.shard}\t{self.obj_id}\t{self.edit_id}\t{self.suffix}"

    @classmethod
    def from_line(cls, line: str) -> "PendingEntry":
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 4:
            raise ValueError(f"malformed pending line: {line!r}")
        return cls(parts[0], parts[1], parts[2], parts[3])


class DelLatentPending:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def iter_entries(self) -> Iterator[PendingEntry]:
        if not self.path.is_file():
            return
        for line in self.path.read_text().splitlines():
            if line.strip():
                yield PendingEntry.from_line(line)

    def append(self, entry: PendingEntry) -> None:
        self._ensure()
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                existing = {PendingEntry.from_line(ln) for ln in f.read().splitlines() if ln.strip()}
                if entry in existing:
                    return
                f.seek(0, 2)
                f.write(entry.to_line() + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def remove(self, entry: PendingEntry) -> None:
        if not self.path.is_file():
            return
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                kept = [ln for ln in f.read().splitlines()
                        if ln.strip() and PendingEntry.from_line(ln) != entry]
                f.seek(0); f.truncate()
                if kept:
                    f.write("\n".join(kept) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
