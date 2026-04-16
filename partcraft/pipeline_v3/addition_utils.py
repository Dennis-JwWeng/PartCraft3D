"""Utilities shared by addition (backfill) edit generation."""
from __future__ import annotations

# Verb-level rule rewrites. Longer matches first to avoid substring collisions.
_INVERT_VERBS: list[tuple[str, str]] = [
    ("get rid of", "add"),
    ("take away", "add"),
    ("delete", "add"),
    ("remove", "add"),
    ("strip", "add"),
    ("erase", "add"),
]


def invert_delete_prompt(prompt: str) -> str:
    """Convert a deletion imperative into an addition imperative.

    Handles 'from' locatives: 'Remove X from Y' becomes 'Add X to Y'.
    Pure rule-based; goal is good-enough training labels.
    """
    if not prompt:
        return prompt
    p = prompt.strip()
    if not p:
        return ""
    low = p.lower()
    result = None
    for old, new in _INVERT_VERBS:
        if low.startswith(old):
            result = new.capitalize() + p[len(old):]
            break
        idx = low.find(" " + old + " ")
        if idx >= 0:
            result = p[:idx + 1] + new + p[idx + 1 + len(old):]
            break
    if result is None:
        result = "Add back " + p
    # Replace first 'from' with 'to': "Add X from Y" -> "Add X to Y"
    result = result.replace(" from ", " to ", 1)
    return result


__all__ = ["invert_delete_prompt"]
