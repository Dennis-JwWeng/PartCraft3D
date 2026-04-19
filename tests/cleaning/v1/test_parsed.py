"""Schema tolerance for v2/v3 ``phase1/parsed.json``."""
from __future__ import annotations

from partcraft.cleaning.v1._parsed import extract_edits_and_parts


def test_extract_handles_wrapped_v2_schema():
    """Real v2/v3 layout: edits + object.parts live under ``parsed``."""
    doc = {
        "obj_id": "x", "shard": "05",
        "validation": {"ok": True},
        "parsed": {
            "object": {
                "full_desc": "an object",
                "parts": [
                    {"part_id": 0, "color": "red", "name": "handle"},
                    {"part_id": 1, "color": "orange", "name": "canopy"},
                ],
            },
            "edits": [
                {"edit_type": "deletion", "selected_part_ids": [0],
                 "prompt": "Remove the handle."},
                {"edit_type": "modification", "selected_part_ids": [1],
                 "prompt": "Make the canopy wooden."},
            ],
        },
    }
    edits, parts = extract_edits_and_parts(doc)
    assert len(edits) == 2
    assert edits[0]["prompt"] == "Remove the handle."
    assert parts == {0: "handle", 1: "canopy"}


def test_extract_handles_legacy_unwrapped_schema():
    """Older fixtures had edits/parts at the top level with key ``id``."""
    doc = {
        "object": {"full_desc": "..."},
        "parts": [{"id": 0, "name": "leg"}],
        "edits": [{"edit_type": "deletion", "prompt": "Remove the leg."}],
    }
    edits, parts = extract_edits_and_parts(doc)
    assert len(edits) == 1 and edits[0]["prompt"] == "Remove the leg."
    assert parts == {0: "leg"}


def test_extract_skips_malformed_parts_silently():
    doc = {
        "parsed": {"object": {"parts": [
            {"part_id": "not-int", "name": "bad"},  # bogus id
            {"name": "missing-id"},                  # no part_id
            {"part_id": 7, "name": "good"},
        ]}, "edits": []},
    }
    _, parts = extract_edits_and_parts(doc)
    assert parts == {7: "good"}


def test_extract_returns_empty_on_missing_doc():
    assert extract_edits_and_parts(None) == ([], {})
    assert extract_edits_and_parts({}) == ([], {})
