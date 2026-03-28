from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def parse_csv_or_space_list(values) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        for x in str(v).split(","):
            x = x.strip()
            if x:
                out.append(x)
    return out


def load_ids_file(path: str | None) -> list[str]:
    if not path:
        return []
    ids: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def dedupe_ids_preserve_order(ids: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    out: list[str] = []
    dup = 0
    for x in ids:
        if x in seen:
            dup += 1
            continue
        seen.add(x)
        out.append(x)
    return out, dup


def dedupe_specs_by_edit_id(specs: list, logger, stage: str):
    seen: set[str] = set()
    unique_specs = []
    dup = 0
    for s in specs:
        eid = getattr(s, "edit_id", None)
        if not eid:
            continue
        if eid in seen:
            dup += 1
            continue
        seen.add(eid)
        unique_specs.append(s)
    if dup:
        logger.warning("%s: dropped %d duplicate edit_id rows before dispatch", stage, dup)
    return unique_specs


def collect_success_ids(path: Path, id_key: str = "edit_id") -> set[str]:
    done_ids: set[str] = set()
    if not path.exists():
        return done_ids
    for rec in iter_jsonl(path):
        try:
            if rec.get("status") == "success" and rec.get(id_key):
                done_ids.add(rec[id_key])
        except Exception:
            pass
    return done_ids


def collect_existing_ids(path: Path, id_key: str) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for rec in iter_jsonl(path):
        if rec.get(id_key):
            ids.add(rec[id_key])
    return ids


def read_records_by_id(path: Path, id_key: str) -> OrderedDict[str, dict]:
    merged: OrderedDict[str, dict] = OrderedDict()
    if not path.exists():
        return merged
    for rec in iter_jsonl(path):
        rid = rec.get(id_key)
        if rid:
            merged[rid] = rec
    return merged


def write_records(path: Path, records: OrderedDict[str, dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
