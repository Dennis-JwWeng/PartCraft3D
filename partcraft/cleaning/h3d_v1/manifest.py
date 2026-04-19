"""JSONL manifest io for H3D_v1.

Each ``manifests/<edit_type>/<NN>.jsonl`` is the system of record for
"what's in the dataset" — one JSON record per promoted edit. The
aggregated ``manifests/all.jsonl`` is rebuilt by ``build_h3d_v1_index``.

This module guarantees:

* **Atomic appends** under concurrent writers (``fcntl.flock`` on the
  jsonl file itself, with the standard ``open(..., "a")`` ordering
  semantics — Linux only, which matches our deployment).
* **Fault-tolerant reads** — malformed lines are logged and skipped.

Records are plain dicts; schema enforcement lives in ``promoter.py``
(it builds the records). Keeping this module schema-agnostic means the
same primitives serve manifest io and the future ``--validate`` index
walker.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def _flock_exclusive(fd: int) -> Iterator[None]:
    """Hold an exclusive ``flock`` for the duration of the ``with`` block."""
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append ``record`` as one JSON line, atomically under contention.

    Creates parent directories on demand. Acquires an exclusive
    ``flock`` on the file before writing — concurrent appenders block
    rather than interleaving partial lines.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        with _flock_exclusive(f.fileno()):
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def append_jsonl_many(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append multiple records under a single ``flock``. Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        with _flock_exclusive(f.fileno()):
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from ``path``. Skips and logs malformed lines.

    Returns an empty iterator if ``path`` does not exist (so callers can
    treat first-run as "no manifest yet").
    """
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                LOGGER.warning("skipping malformed line %s:%d (%s)", path, lineno, exc)
                continue
            if isinstance(rec, dict):
                yield rec
            else:
                LOGGER.warning("skipping non-object line %s:%d", path, lineno)


def rewrite_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Rewrite ``path`` atomically (write to ``.tmp`` + rename).

    Used by ``build_h3d_v1_index`` for the aggregated ``all.jsonl``.
    Returns the number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n


__all__ = [
    "append_jsonl",
    "append_jsonl_many",
    "read_jsonl",
    "rewrite_jsonl",
]
