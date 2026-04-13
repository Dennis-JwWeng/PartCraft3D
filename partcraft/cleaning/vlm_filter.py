"""VLM judge library for edit-pair quality scoring.

This module is a pure **library** (no main, no batch runner).  The previous
``run_vlm_filter`` / ``evaluate_edit`` / mesh-prefilter / PLY-render entries
have been removed; the active entry point for VLM cleaning is
``scripts/tools/run_vlm_cleaning.py`` (object-centric ``partverse_pairs/``
layout, decoupled render + score, multi-GPU launcher).

Public API:
  * ``VLMScore``                    — dataclass for one edit's score
  * ``compute_composite_score``     — weighted scalar from VLMScore fields
  * ``classify_tier``               — high / medium / low / negative / rejected
  * ``compose_comparison``          — top=before / bottom=after PNG grid
  * ``build_judge_prompt``          — VLM prompt (Part 1 edit + Part 2 prompt eval)
  * ``call_vlm_judge``              — OpenAI-compatible chat call with JSON parse
  * ``_VLM_YAWS`` / ``_VLM_PITCHES``— 3-view optimal-coverage angles (single source)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from dataclasses import dataclass, asdict

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "VLMScore",
    "compute_composite_score",
    "classify_tier",
    "compose_comparison",
    "build_judge_prompt",
    "call_vlm_judge",
    "_VLM_YAWS",
    "_VLM_PITCHES",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VLMScore:
    """VLM quality score for a single edit pair."""
    edit_id: str
    edit_type: str
    edit_executed: bool = False
    correct_region: bool = False
    preserve_other: bool = False
    visual_quality: int = 0        # 1-5
    artifact_free: bool = False
    reason: str = ""
    prompt_quality: int = 0        # 1-5
    improved_prompt: str = ""
    improved_after_desc: str = ""
    score: float = 0.0             # composite [0, 1]
    quality_tier: str = "rejected" # high / medium / low / negative / rejected

    def to_dict(self) -> dict:
        return asdict(self)


def compute_composite_score(s: VLMScore) -> float:
    """Weighted composite score from VLM judgments."""
    total = 0.0
    total += 0.3 * (1.0 if s.edit_executed else 0.0)
    total += 0.2 * (1.0 if s.correct_region else 0.0)
    total += 0.2 * (1.0 if s.preserve_other else 0.0)
    total += 0.2 * max(0, (s.visual_quality - 1)) / 4.0
    total += 0.1 * (1.0 if s.artifact_free else 0.0)
    return round(total, 4)


def classify_tier(s: VLMScore) -> str:
    """Classify edit quality into tiers.

    - high:     all criteria met, visual_quality >= 4 — ideal training data
    - medium:   all criteria met, visual_quality = 3 — usable training data
    - low:      minor issues — use with caution
    - negative: edit failed or wrong region — usable as negative sample
    - rejected: evaluation error / no VLM response — discard
    """
    if s.score == 0.0 and not s.edit_executed:
        if s.reason.startswith("Evaluation error") or \
           s.reason == "VLM returned no valid response":
            return "rejected"

    if not s.edit_executed or not s.correct_region:
        return "negative"
    if s.preserve_other and s.artifact_free and s.visual_quality >= 4:
        return "high"
    if s.preserve_other and s.artifact_free and s.visual_quality >= 3:
        return "medium"
    if s.visual_quality >= 2:
        return "low"
    return "negative"


# ---------------------------------------------------------------------------
# 3-view optimal coverage angles (single source of truth)
# ---------------------------------------------------------------------------

# View 1 (0°, 26°): front + sides;  View 2 (120°, 26°): right-back;
# View 3 (240°, 63°): left-back + top surface.
_VLM_YAWS    = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
_VLM_PITCHES = [0.45, 0.45, 1.1]


# ---------------------------------------------------------------------------
# Image composition
# ---------------------------------------------------------------------------

def compose_comparison(before_imgs: list[np.ndarray],
                       after_imgs: list[np.ndarray]) -> bytes:
    """Compose a grid image: top row = before views, bottom row = after views.

    Returns PNG bytes.
    """
    from PIL import Image, ImageDraw, ImageFont

    n = min(len(before_imgs), len(after_imgs))
    h, w = before_imgs[0].shape[:2]

    label_h = 28
    canvas_w = w * n
    canvas_h = (h + label_h) * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i in range(n):
        canvas.paste(Image.fromarray(before_imgs[i]), (i * w, label_h))
        canvas.paste(Image.fromarray(after_imgs[i]), (i * w, h + label_h * 2))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((canvas_w // 2 - 30, 4), "Before", fill=(0, 0, 0), font=font)
    draw.text((canvas_w // 2 - 25, h + label_h + 4), "After",
              fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# VLM judge prompt — type-specific
# ---------------------------------------------------------------------------

_JUDGE_SCHEMA = """\
Your entire reply MUST be one JSON object only: no prose, no markdown fences, \
no text before or after. First character must be "{{" and last must be "}}".

Schema (fill ALL fields):
{{"edit_executed":<bool>,"correct_region":<bool>,"preserve_other":<bool>,\
"visual_quality":<1-5>,"artifact_free":<bool>,"reason":"<one sentence>",\
"prompt_quality":<1-5>,"improved_prompt":"<imperative>",\
"improved_after_desc":"<concise>"}}

## Part 2 — Prompt quality
- prompt_quality: How well does the edit prompt describe the actual visual \
change? (1=wrong/missing, 2=vague, 3=roughly correct, 4=accurate, 5=precise)
- improved_prompt: Rewrite the prompt to precisely describe what you observe. \
Imperative form (e.g. "Remove the ...", "Change ... to ..."). Always fill.
- improved_after_desc: Concise description of the AFTER object. Always fill."""


def _header(object_desc: str, edit_type: str, part_label: str,
            edit_prompt: str) -> str:
    prompt_line = (f'- Edit prompt: "{edit_prompt}"' if edit_prompt.strip()
                   else "- Edit prompt: (none — infer from images)")
    return (
        "You are a quality judge for 3D object editing.\n\n"
        "The image shows two rows of multi-view renders of a 3D object:\n"
        "- TOP row: BEFORE the edit\n"
        "- BOTTOM row: AFTER the edit\n\n"
        "Edit information:\n"
        f"- Object: {object_desc}\n"
        f"- Edit type: {edit_type}\n"
        f"- Target part: {part_label}\n"
        f"{prompt_line}\n"
    )


def _build_deletion_prompt(object_desc, part_label, edit_prompt,
                            target_part_desc) -> str:
    tpd = f'- Target part description: "{target_part_desc}"' if target_part_desc else ""
    return (
        _header(object_desc, "deletion", part_label, edit_prompt)
        + (f"{tpd}\n" if tpd else "")
        + """
## Part 1 — Deletion quality (WHAT TO CHECK)
A deletion edit removes one or more parts from the object. Judge strictly:

- edit_executed: Is the target part GONE from the AFTER model? (true/false)
- correct_region: Was the CORRECT part removed — matching "{part_label}" \
and the edit prompt? (true/false). False if a wrong/different part was removed.
- preserve_other: Are ALL remaining parts intact, with no floating fragments, \
gaps, or broken surfaces left behind where the part was? (true/false)
- visual_quality: Overall quality of the AFTER model (1=terrible, 2=poor, \
3=acceptable, 4=good, 5=excellent). Penalise holes, seam artefacts, \
distorted remaining parts.
- artifact_free: No floating blobs, stray geometry, or jagged seams at the \
removal site? (true/false)
- reason: One sentence on the quality of the deletion.

""".replace("{part_label}", part_label)
        + _JUDGE_SCHEMA
    )


def _build_modification_prompt(object_desc, part_label, edit_prompt,
                                target_part_desc, new_part_desc) -> str:
    tpd = f'- Target part (before): "{target_part_desc}"' if target_part_desc else ""
    npd = f'- Expected shape after: "{new_part_desc}"' if new_part_desc else ""
    extras = "\n".join(x for x in [tpd, npd] if x)
    return (
        _header(object_desc, "modification", part_label, edit_prompt)
        + (f"{extras}\n" if extras else "")
        + """
## Part 1 — Modification quality (WHAT TO CHECK)
A modification edit changes the SHAPE, SILHOUETTE, or FUNCTIONAL ROLE of a \
part — NOT its colour or material. Judge strictly:

- edit_executed: Is the target part's SHAPE or FORM visibly different in AFTER? \
(true/false). Return false if only colour/material changed, or no change visible.
- correct_region: Was the shape change applied to the correct part \
("{part_label}"), not some other part? (true/false)
- preserve_other: Are all OTHER parts unchanged in shape and position? \
(true/false). Minor texture differences are acceptable; geometry must be intact.
- visual_quality: Quality of the modified AFTER model (1–5). Penalise \
implausible geometry, broken meshes, or shape changes that ignore the prompt.
- artifact_free: No floating blobs, intersecting geometry, or broken normals? \
(true/false)
- reason: One sentence on whether the shape change is correct and clean.

""".replace("{part_label}", part_label)
        + _JUDGE_SCHEMA
    )


def _build_scale_prompt(object_desc, part_label, edit_prompt,
                         target_part_desc, factor) -> str:
    tpd = f'- Target part description: "{target_part_desc}"' if target_part_desc else ""
    fac = f"- Scale factor (shrink): {factor}" if factor else ""
    extras = "\n".join(x for x in [tpd, fac] if x)
    return (
        _header(object_desc, "scale", part_label, edit_prompt)
        + (f"{extras}\n" if extras else "")
        + """
## Part 1 — Scale quality (WHAT TO CHECK)
A scale edit SHRINKS the target part. Judge strictly:

- edit_executed: Is the target part SMALLER in AFTER than in BEFORE? \
(true/false). Return false if size is unchanged or part grew.
- correct_region: Was the CORRECT part ("{part_label}") scaled, not a \
neighbouring part? (true/false)
- preserve_other: Are all OTHER parts at their original size and position? \
(true/false)
- visual_quality: Quality of the scaled AFTER model (1–5). Penalise \
disproportionate scaling, floating connectors, or visible seams.
- artifact_free: No floating or disconnected geometry after scaling? (true/false)
- reason: One sentence on the scale change quality.

""".replace("{part_label}", part_label)
        + _JUDGE_SCHEMA
    )


def _build_material_prompt(object_desc, part_label, edit_prompt,
                            target_part_desc, target_material) -> str:
    tpd = f'- Target part description: "{target_part_desc}"' if target_part_desc else ""
    mat = f'- Expected material/finish: "{target_material}"' if target_material else ""
    extras = "\n".join(x for x in [tpd, mat] if x)
    return (
        _header(object_desc, "material", part_label, edit_prompt)
        + (f"{extras}\n" if extras else "")
        + """
## Part 1 — Material quality (WHAT TO CHECK)
A material edit changes the SURFACE MATERIAL or FINISH of a part (e.g. wood, \
steel, glass) while leaving the GEOMETRY unchanged. Judge strictly:

- edit_executed: Is the surface material/texture of the target part visibly \
different in AFTER? (true/false)
- correct_region: Did the material change on the CORRECT part ("{part_label}") \
only, with no spillover to adjacent parts? (true/false)
- preserve_other: Is the geometry of ALL parts (including the target) intact — \
no shape changes, no new holes? (true/false)
- visual_quality: Quality of the material change (1–5). Does it look like the \
expected material? Is the shading/texture plausible?
- artifact_free: No UV seams, tiling artefacts, or broken surfaces? (true/false)
- reason: One sentence on the material change quality.

""".replace("{part_label}", part_label)
        + _JUDGE_SCHEMA
    )


def _build_global_prompt(object_desc, part_label, edit_prompt,
                          target_style) -> str:
    sty = f'- Expected style: "{target_style}"' if target_style else ""
    return (
        _header(object_desc, "global", "entire object", edit_prompt)
        + (f"{sty}\n" if sty else "")
        + """
## Part 1 — Global style quality (WHAT TO CHECK)
A global edit changes the ENTIRE OBJECT's artistic or rendering aesthetic \
(e.g. cel-shading, Art Deco, origami geometry) — NOT a material or colour \
change to individual parts. Judge strictly:

- edit_executed: Is the overall rendering/art style of the object visibly \
changed in AFTER? (true/false). Return false if the object looks identical.
- correct_region: Is the style applied CONSISTENTLY across the ENTIRE object \
with no partial areas left in the original style? (true/false). Return false \
if only some parts changed style.
- preserve_other: Is the object's underlying GEOMETRY and STRUCTURE still \
recognisable — correct number of parts, overall form preserved? (true/false). \
Return false if geometry was destroyed or heavily distorted.
- visual_quality: How well does the style transfer look? (1–5). Penalise \
inconsistent style application, artefacts, or style that does not match the \
target description.
- artifact_free: No floating geometry, broken normals, or severe rendering \
artefacts? (true/false)
- reason: One sentence on the global style transfer quality.

"""
        + _JUDGE_SCHEMA
    )


def _build_addition_prompt(object_desc, part_label, edit_prompt,
                            target_part_desc) -> str:
    tpd = f'- New element description: "{target_part_desc}"' if target_part_desc else ""
    return (
        _header(object_desc, "addition", part_label, edit_prompt)
        + (f"{tpd}\n" if tpd else "")
        + """
## Part 1 — Addition quality (WHAT TO CHECK)
An addition edit ADDS a new element to the object. Judge strictly:

- edit_executed: Is there a NEW element in AFTER that was NOT present in \
BEFORE? (true/false)
- correct_region: Is the new element placed in a REASONABLE position — \
attached or adjacent to the correct existing part, not floating arbitrarily? \
(true/false)
- preserve_other: Are all ORIGINAL parts still present and intact? (true/false)
- visual_quality: Quality of the addition (1–5). Does it blend naturally with \
the original object's style and scale?
- artifact_free: No interpenetrating geometry, floating blobs, or broken \
surfaces at the attachment point? (true/false)
- reason: One sentence on the addition quality.

"""
        + _JUDGE_SCHEMA
    )


def build_judge_prompt(
    edit_prompt: str,
    edit_type: str,
    object_desc: str,
    part_label: str,
    target_part_desc: str = "",
    edit_params: dict | None = None,
    expected_after_desc: str = "",
) -> str:
    """Build a type-specific VLM judge prompt.

    Args:
        edit_prompt:       The original phase-1 prompt string.
        edit_type:         One of deletion/modification/scale/material/global/addition.
        object_desc:       Full object description.
        part_label:        Comma-joined label(s) of the target part(s).
        target_part_desc:  Phase-1 visual description of the target part (before).
        edit_params:       edit_params dict from EditSpec (target_style, target_material,
                           new_part_desc, factor …).
        expected_after_desc: Phase-1 after_desc_full (unused in prompt body but kept
                             for future use / logging).
    """
    ep = edit_params or {}
    et = edit_type.lower()
    if et == "deletion":
        return _build_deletion_prompt(object_desc, part_label, edit_prompt,
                                      target_part_desc)
    if et == "modification":
        return _build_modification_prompt(object_desc, part_label, edit_prompt,
                                          target_part_desc,
                                          ep.get("new_part_desc", ""))
    if et == "scale":
        return _build_scale_prompt(object_desc, part_label, edit_prompt,
                                   target_part_desc, ep.get("factor"))
    if et == "material":
        return _build_material_prompt(object_desc, part_label, edit_prompt,
                                      target_part_desc,
                                      ep.get("target_material", ""))
    if et == "global":
        return _build_global_prompt(object_desc, part_label, edit_prompt,
                                    ep.get("target_style", ""))
    if et == "addition":
        return _build_addition_prompt(object_desc, part_label, edit_prompt,
                                      target_part_desc)
    # fallback — generic (unknown edit type)
    prompt_line = (f'- Edit prompt: "{edit_prompt}"' if edit_prompt.strip()
                   else "- Edit prompt: (none — infer from images)")
    return (
        _header(object_desc, edit_type, part_label, edit_prompt)
        + """
## Part 1 — Edit quality
- edit_executed: Did the described edit visibly happen? (true/false)
- correct_region: Was the change applied to the correct part? (true/false)
- preserve_other: Are all other parts preserved and intact? (true/false)
- visual_quality: Overall quality of the AFTER model (1–5)
- artifact_free: No floating blobs, broken surfaces? (true/false)
- reason: One sentence on quality.

"""
        + _JUDGE_SCHEMA
    )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _balanced_brace_object(s: str, start: int) -> str | None:
    """If s[start] == '{', return the balanced JSON object substring, else None."""
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for j in range(start, len(s)):
        c = s[j]
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : j + 1]
    return None


def _iter_json_object_substrings(content: str) -> list[str]:
    """All top-level `{...}` spans in document order (deduped)."""
    seen: set[str] = set()
    out: list[str] = []
    for i, c in enumerate(content):
        if c != "{":
            continue
        block = _balanced_brace_object(content, i)
        if block and block not in seen:
            seen.add(block)
            out.append(block)
    return out


def _parse_vlm_score_dict(blob: str) -> dict | None:
    """Parse JSON object; require VLM judge schema key."""
    try:
        d = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict) and "edit_executed" in d:
        return d
    return None


def _extract_json_from_vlm(content: str) -> dict | None:
    """Robustly extract JSON from VLM response (CoT, markdown, multiple objects)."""
    if not content:
        return None
    content = content.strip()

    # Strip `think`...`</think>` blocks (Gemini 2.5 Flash thinking mode)
    content = re.sub(
        r"`think`.*?`</think>`", "", content, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        got = _parse_vlm_score_dict(inner)
        if got is not None:
            return got
        content = inner

    got = _parse_vlm_score_dict(content)
    if got is not None:
        return got

    # Prefer the last valid object with schema keys (final answer after CoT)
    for blob in reversed(_iter_json_object_substrings(content)):
        got = _parse_vlm_score_dict(blob)
        if got is not None:
            return got

    return None


# ---------------------------------------------------------------------------
# VLM judge call
# ---------------------------------------------------------------------------

def call_vlm_judge(client, model: str, img_bytes: bytes,
                   edit_prompt: str, edit_type: str,
                   object_desc: str, part_label: str,
                   max_retries: int = 4,
                   max_tokens: int = 1024,
                   json_object_mode: bool = False) -> dict | None:
    """Call VLM to judge edit quality. Returns parsed JSON or None."""
    import time
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    base_text = build_judge_prompt(edit_prompt, edit_type, object_desc, part_label)
    strict_suffix = (
        "\n\nIf you already wrote analysis above, IGNORE it for the parser: "
        "output ONE new line that is ONLY the JSON object, starting with { "
        "and ending with }."
    )

    for attempt in range(max_retries + 1):
        text = base_text + (strict_suffix if attempt > 0 else "")
        sys_msg = (
            "You output only valid JSON for machine parsing. "
            "Never write explanations, headings, or markdown."
        )
        if attempt > 0:
            sys_msg += " Your reply must be a single JSON object; no chain-of-thought."
        try:
            create_kw: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": text},
                        ],
                    },
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "timeout": 120,
            }
            # Disable thinking/CoT for Qwen3.5 via SGLang; silently dropped on
            # backends that don't support extra_body.
            create_kw["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if json_object_mode and attempt == 0:
                create_kw["response_format"] = {"type": "json_object"}
            try:
                resp = client.chat.completions.create(**create_kw)
            except TypeError:
                create_kw.pop("extra_body", None)
                resp = client.chat.completions.create(**create_kw)
            except Exception as e0:
                if json_object_mode and attempt == 0 and create_kw.pop(
                        "response_format", None
                ) is not None:
                    logger.warning(
                        "VLM json_object mode rejected by server (%s); retrying without",
                        e0,
                    )
                    resp = client.chat.completions.create(**create_kw)
                else:
                    raise

            content = resp.choices[0].message.content
            if not content:
                logger.warning(
                    "VLM returned empty content (attempt %d/%d)",
                    attempt + 1, max_retries + 1)
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
                continue

            result = _extract_json_from_vlm(content)
            if result is not None:
                return result

            logger.warning(
                "VLM JSON extraction failed (attempt %d/%d), raw: %s",
                attempt + 1, max_retries + 1, content[:300])

        except Exception as e:
            logger.warning("VLM judge call failed (attempt %d/%d): %s",
                           attempt + 1, max_retries + 1, e)

        if attempt < max_retries:
            time.sleep(2 * (attempt + 1))

    return None
