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
# VLM judge prompt
# ---------------------------------------------------------------------------

def build_judge_prompt(edit_prompt: str, edit_type: str,
                       object_desc: str, part_label: str) -> str:
    """Build the VLM judge prompt.

    Asks the VLM to:
      1. Evaluate edit quality (5 criteria)
      2. Rate the edit_prompt quality and provide an improved version
    """
    prompt_section = f'- Edit prompt: "{edit_prompt}"' if edit_prompt.strip() else \
        "- Edit prompt: (none provided — you MUST infer what happened from the images)"
    return f"""You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
- Top row: BEFORE editing
- Bottom row: AFTER editing

Edit information:
- Object: {object_desc}
- Edit type: {edit_type}
- Target part: {part_label}
{prompt_section}

Your entire reply MUST be one JSON object only: no prose, no markdown fences, no text before or after. First character must be "{{" and the last must be "}}".

Example shape:
{{"edit_executed":true,"correct_region":true,"preserve_other":true,"visual_quality":4,"artifact_free":true,"reason":"brief explanation","prompt_quality":3,"improved_prompt":"Remove the red wheel from the car","improved_after_desc":"A blue car without its front wheel"}}

## Part 1: Edit quality (judge the 3D edit result)
- edit_executed: Did the described edit visibly happen? (true/false). If no prompt was provided, judge whether ANY meaningful edit is visible.
- correct_region: Was the change applied to the correct part ({part_label})? (true/false)
- preserve_other: Are all other parts of the object preserved and intact? (true/false)
- visual_quality: Overall visual quality of the AFTER model (1=terrible, 2=poor, 3=acceptable, 4=good, 5=excellent)
- artifact_free: Is the AFTER model free of obvious artifacts like floating blobs, broken surfaces, or missing geometry? (true/false)
- reason: One sentence explaining your quality assessment

## Part 2: Prompt quality (judge and improve the edit prompt)
- prompt_quality: How well does the edit prompt describe the actual visual change? (1=completely wrong or missing, 2=vague/misleading, 3=roughly correct, 4=accurate, 5=precise and natural)
- improved_prompt: Write a better edit prompt that precisely describes the visual change you observe. Use natural English, imperative form (e.g. "Remove the ...", "Change the ... to ..."). Always fill this even if the original is good.
- improved_after_desc: Write a concise description of the AFTER object as seen in the images. Always fill this.

Be strict but fair. Minor imperfections are acceptable (quality=3-4). Only fail edit_executed if there is NO visible change."""


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
