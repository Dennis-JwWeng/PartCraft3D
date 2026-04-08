"""Phase 1.5 VLM judge: prompts + tier mapping for scoring phase1_v2 edits.

The runner sends the VLM a 2-image stitch (ORIGINAL | HIGHLIGHTED) plus a
short text block with the phase1 edit metadata, and asks for a structured
JSON score. This module owns the prompt templates and the deterministic
score → tier mapping.

Tier rules (most important = semantic correspondence):
  - selection_correct == 1     → reject  (highlight is on the wrong part(s))
  - overall >= 4 and sel >= 4  → high
  - overall == 3               → medium
  - else                       → low

NOTE: deleting / modifying the structural body of the object is NOT a reject —
phase1 sometimes intentionally chooses these to produce hard cases. They land
in `low` (low overall, sel still correct).
"""
from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a strict reviewer for a 3D-object part-editing dataset. \
You are given two images side by side and a small JSON describing one proposed edit. \
Your job is to score the edit. Output ONE JSON object — no prose, no markdown. \
Begin with '{' and end with '}'."""


# Single user-prompt template used for all edit types.
# Image layout: LEFT = original photo of the object, RIGHT = same view with the
# selected target parts painted MAGENTA and all other parts in light gray.
# (For `global` edits there are no target parts; the right image is identical
#  to the left and the model is told to ignore highlight semantics.)
USER_PROMPT_TEMPLATE = """[The image is a side-by-side panel.
LEFT  = ORIGINAL photo of the 3D object (one chosen view).
RIGHT = SAME view with the parts targeted by this edit painted in bright \
MAGENTA. All other parts of the object are painted light gray. The right \
panel is purely a *highlight overlay* showing the spatial extent of the \
edit's selected_part_ids. Use it to judge whether the right *region* really \
corresponds to the part(s) the prompt is talking about.
For "global" edits there is no highlight (the whole object is the target); \
ignore the right panel and look at the left.]

## EDIT BEING REVIEWED
```json
{edit_meta_json}
```

## YOUR JOB

Output a single JSON object with these fields:

  selection_correct  Integer 1-5. Does the magenta-highlighted region in the
                     RIGHT panel actually correspond to the part(s) the prompt
                     and target_part_desc are talking about?
                       5 = highlight is exactly the named part(s)
                       4 = mostly correct, minor over/under-coverage
                       3 = partial overlap, ambiguous
                       2 = wrong part but related region
                       1 = clearly the wrong part / unrelated region
                     For "global" edits there is no selection — set to 5.

  edit_plausibility  Integer 1-5. Is the edit ACTION itself a sensible
                     operation on this object?
                       5 = clearly a meaningful, well-formed edit
                       4 = reasonable, some rough edges
                       3 = OK but odd / generic / weakly motivated
                       2 = unlikely to make sense visually
                       1 = nonsensical / contradicts the object
                     IMPORTANT: deleting or replacing a large structural part
                     (the "body" of the object) is allowed — score it 2-3 if
                     the result becomes unrecognizable, but DO NOT mark it 1.

  prompt_clarity     Integer 1-5. Does the natural-language `prompt` field
                     accurately and unambiguously describe what should change?
                       5 = unambiguous, mentions the right part by name
                       3 = understandable but vague
                       1 = empty / contradictory / wrong subject

  overall            Integer 1-5. Your overall judgement.

  issues             List of short string tags. Use only from this set:
                       "wrong_part"          highlight does not match prompt target
                       "ambiguous_prompt"    prompt is vague about which part
                       "destroys_object"     edit removes/replaces the main body
                       "redundant"           edit duplicates another likely edit
                       "tiny_target"         the highlighted region is degenerate
                       "off_target_action"   action does not fit the part
                     Empty list = no issues.

  rationale          ONE short sentence (<= 25 words) explaining the scores.

  improved_prompt    A rewritten version of `prompt` that fixes the issues you
                     listed. If `prompt` is already good, copy it verbatim.

## HARD RULES

H1. Output ONLY the JSON object. No prose, no code fences.
H2. selection_correct is the most important field — be honest and strict.
H3. For "global" edit_type: the right panel has no highlight; set
    selection_correct = 5 and judge edit_plausibility / prompt_clarity from
    the left panel + the prompt.
H4. NEVER mark something reject just because it removes the body. Use a low
    edit_plausibility instead.
"""


def build_user_prompt(edit: dict) -> str:
    """Build the user message for one edit. We strip noise fields before
    embedding the json so the VLM focuses on what matters."""
    keep = {}
    for k in ("edit_type", "prompt", "target_part_desc", "selected_part_ids",
              "edit_params", "rationale"):
        if k in edit:
            keep[k] = edit[k]
    meta = json.dumps(keep, ensure_ascii=False, indent=2)
    return USER_PROMPT_TEMPLATE.format(edit_meta_json=meta)


# ──────────────────────────────────────────────────────────────────────────
#                          parsing & tier mapping
# ──────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = ("selection_correct", "edit_plausibility", "prompt_clarity",
                 "overall", "issues", "rationale", "improved_prompt")


def _coerce_int(x: Any, default: int = 0) -> int:
    try:
        v = int(round(float(x)))
        return max(1, min(5, v))
    except Exception:
        return default


def parse_score(raw_obj: dict | None) -> dict:
    """Validate + normalize a raw VLM score dict. Always returns a dict with
    every required field. Adds 'tier' deterministically.

    If `raw_obj` is None or fundamentally broken, returns a sentinel score
    with tier='low' and an issue tag so the runner can keep going.
    """
    if not isinstance(raw_obj, dict):
        return _bad_score("vlm_parse_error")

    out = {
        "selection_correct": _coerce_int(raw_obj.get("selection_correct"), 0),
        "edit_plausibility": _coerce_int(raw_obj.get("edit_plausibility"), 0),
        "prompt_clarity":    _coerce_int(raw_obj.get("prompt_clarity"), 0),
        "overall":           _coerce_int(raw_obj.get("overall"), 0),
    }
    issues = raw_obj.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    out["issues"] = [str(x)[:40] for x in issues][:8]
    out["rationale"] = str(raw_obj.get("rationale") or "")[:300]
    out["improved_prompt"] = str(raw_obj.get("improved_prompt") or "")[:400]

    # Backstop: if any of the 1-5 fields ended up 0 (parse fail), force low.
    if any(out[k] == 0 for k in ("selection_correct", "edit_plausibility",
                                 "prompt_clarity", "overall")):
        out["tier"] = "low"
        if "vlm_parse_partial" not in out["issues"]:
            out["issues"].append("vlm_parse_partial")
        return out

    out["tier"] = _tier(out)
    return out


def _tier(s: dict) -> str:
    sel = s["selection_correct"]
    ovl = s["overall"]
    if sel <= 1:
        return "reject"
    if ovl >= 4 and sel >= 4:
        return "high"
    if ovl == 3:
        return "medium"
    return "low"


def _bad_score(reason: str) -> dict:
    return {
        "selection_correct": 0,
        "edit_plausibility": 0,
        "prompt_clarity": 0,
        "overall": 0,
        "issues": [reason],
        "rationale": "",
        "improved_prompt": "",
        "tier": "low",
    }
