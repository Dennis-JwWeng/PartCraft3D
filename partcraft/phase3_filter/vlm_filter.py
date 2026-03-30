"""Phase 3: VLM-based quality filter for Phase 2.5 SLAT edit pairs.

Optional **mesh prefilter** (``phase3.vlm_mesh_prefilter``): cheap ``trimesh``
checks on ``before.ply`` / ``after.ply`` *before* TRELLIS decode and VLM.
Failsures are scored as ``negative`` and skip rendering + API calls.

The VLM evaluates:
  1. edit_executed   — did the edit actually happen?
  2. correct_region  — was the right part modified?
  3. preserve_other  — are unedited parts preserved?
  4. visual_quality  — overall visual quality (1-5)
  5. artifact_free   — no broken geometry / floating blobs?

Each edit pair gets a structured JSON score.  Pairs below threshold
are marked as failed.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Mesh prefilter: full ``evaluate_pair`` (volume / edit ratio / …) only for types
# where vertex motion is expected.  Light checks for del/add.  Skip for identity
# and appearance-only edits (material/global) where metrics misfire.
_MESH_PREFILTER_FULL = frozenset({"modification", "scale"})
_MESH_PREFILTER_LIGHT = frozenset({"deletion", "addition"})


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
    score: float = 0.0             # composite [0, 1]
    quality_tier: str = "rejected" # "high" / "medium" / "low" / "negative" / "rejected"

    def to_dict(self) -> dict:
        return asdict(self)


def compute_composite_score(s: VLMScore) -> float:
    """Weighted composite score from VLM judgments."""
    total = 0.0
    # edit_executed: 30%
    total += 0.3 * (1.0 if s.edit_executed else 0.0)
    # correct_region: 20%
    total += 0.2 * (1.0 if s.correct_region else 0.0)
    # preserve_other: 20%
    total += 0.2 * (1.0 if s.preserve_other else 0.0)
    # visual_quality: 20% (normalized 1-5 → 0-1)
    total += 0.2 * max(0, (s.visual_quality - 1)) / 4.0
    # artifact_free: 10%
    total += 0.1 * (1.0 if s.artifact_free else 0.0)
    return round(total, 4)


def classify_tier(s: VLMScore) -> str:
    """Classify edit quality into tiers.

    - high:     All criteria met, visual_quality >= 4 — ideal training data
    - medium:   All criteria met, visual_quality = 3 — usable training data
    - low:      Minor issues (e.g. small artifacts) — use with caution
    - negative: Edit failed or wrong region — negative sample for training
    - rejected: Evaluation error / no VLM response — discard
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
# Rendering helpers (reuse from vis module)
# ---------------------------------------------------------------------------

def load_slat(slat_dir: Path, device: str = "cuda"):
    """Load SLAT from feats.pt + coords.pt.

    Raises RuntimeError if files are corrupted or contain invalid data.
    """
    from trellis.modules import sparse as sp
    feats_path = slat_dir / "feats.pt"
    coords_path = slat_dir / "coords.pt"
    try:
        feats = torch.load(feats_path, weights_only=True)
        coords = torch.load(coords_path, weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Corrupted SLAT in {slat_dir} — delete and re-encode"
        ) from e
    if feats.shape[0] != coords.shape[0]:
        raise RuntimeError(
            f"SLAT shape mismatch in {slat_dir}: "
            f"feats {feats.shape} vs coords {coords.shape}")
    if not torch.isfinite(feats).all():
        raise RuntimeError(f"Non-finite SLAT feats in {slat_dir}")
    return sp.SparseTensor(feats=feats.to(device), coords=coords.to(device))


def render_views(pipeline, slat, num_views: int = 4,
                 pitch: float = 0.45) -> list[np.ndarray]:
    """Render multiview images from SLAT via pipeline.decode_slat → Gaussian."""
    from trellis.utils import render_utils
    outputs = pipeline.decode_slat(slat, ['gaussian'])
    gaussian = outputs['gaussian'][0]
    yaws = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]
    pitches = torch.tensor([pitch] * num_views)
    imgs = render_utils.Trellis_render_multiview_images(
        gaussian, yaws.tolist(), pitches.tolist())['color']
    return imgs


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
        # Before row
        img_b = Image.fromarray(before_imgs[i])
        canvas.paste(img_b, (i * w, label_h))
        # After row
        img_a = Image.fromarray(after_imgs[i])
        canvas.paste(img_a, (i * w, h + label_h * 2))

    # Labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((canvas_w // 2 - 30, 4), "Before", fill=(0, 0, 0), font=font)
    draw.text((canvas_w // 2 - 25, h + label_h + 4), "After",
              fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# VLM judge
# ---------------------------------------------------------------------------

def _build_judge_prompt(edit_prompt: str, edit_type: str,
                        object_desc: str, part_label: str) -> str:
    """Build the VLM judge prompt."""
    return f"""You are a quality judge for 3D object editing.

The image shows two rows of multi-view renders of a 3D object:
- Top row: BEFORE editing
- Bottom row: AFTER editing

Edit information:
- Object: {object_desc}
- Edit type: {edit_type}
- Target part: {part_label}
- Edit prompt: "{edit_prompt}"

Evaluate the edit quality. Your entire reply MUST be one JSON object only: no prose,
no markdown fences, no numbered analysis, no text before or after the object.
First character must be "{{" and the last must be "}}".
Example shape:
{{"edit_executed":true,"correct_region":true,"preserve_other":true,"visual_quality":4,"artifact_free":true,"reason":"brief explanation"}}

Scoring criteria:
- edit_executed: Did the described edit visibly happen? (true/false)
- correct_region: Was the change applied to the correct part ({part_label})? (true/false)
- preserve_other: Are all other parts of the object preserved and intact? (true/false)
- visual_quality: Overall visual quality of the AFTER model (1=terrible, 2=poor, 3=acceptable, 4=good, 5=excellent)
- artifact_free: Is the AFTER model free of obvious artifacts like floating blobs, broken surfaces, or missing geometry? (true/false)
- reason: One sentence explaining your assessment

Be strict but fair. Minor imperfections are acceptable (quality=3-4). Only fail edit_executed if there is NO visible change matching the prompt."""


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

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        got = _parse_vlm_score_dict(inner)
        if got is not None:
            return got
        content = inner

    # Whole buffer
    got = _parse_vlm_score_dict(content)
    if got is not None:
        return got

    # Prefer the last valid object with schema keys (final answer after CoT)
    blocks = _iter_json_object_substrings(content)
    for blob in reversed(blocks):
        got = _parse_vlm_score_dict(blob)
        if got is not None:
            return got

    return None


def call_vlm_judge(client, model: str, img_bytes: bytes,
                   edit_prompt: str, edit_type: str,
                   object_desc: str, part_label: str,
                   max_retries: int = 4,
                   max_tokens: int = 4096,
                   json_object_mode: bool = False) -> dict | None:
    """Call VLM to judge edit quality. Returns parsed JSON or None."""
    import time
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    base_text = _build_judge_prompt(edit_prompt, edit_type, object_desc, part_label)
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
            sys_msg += (
                " Your reply must be a single JSON object; no chain-of-thought."
            )
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
            if json_object_mode and attempt == 0:
                create_kw["response_format"] = {"type": "json_object"}
            try:
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
                    f"VLM returned empty content (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
                continue

            result = _extract_json_from_vlm(content)
            if result is not None:
                return result

            logger.warning(
                f"VLM JSON extraction failed (attempt {attempt+1}/{max_retries+1}), "
                f"raw response: {content[:300]}")

        except Exception as e:
            logger.warning(f"VLM judge call failed (attempt {attempt+1}/{max_retries+1}): {e}")

        if attempt < max_retries:
            time.sleep(2 * (attempt + 1))

    return None


# ---------------------------------------------------------------------------
# Mesh prefilter (before TRELLIS + VLM)
# ---------------------------------------------------------------------------

def mesh_prefilter_before_vlm(
    pair_dir: Path,
    entry: dict,
    cfg: dict,
) -> VLMScore | None:
    """If enabled, run fast PLY checks. Returns a finished score to skip VLM, or None."""
    p3 = cfg.get("phase3", {})
    if not p3.get("vlm_mesh_prefilter", False):
        return None

    before_ply = pair_dir / "before.ply"
    after_ply = pair_dir / "after.ply"
    if not before_ply.is_file() or not after_ply.is_file():
        return None

    eid = entry["edit_id"]
    et = (entry.get("edit_type") or "").lower()

    try:
        import trimesh
        from partcraft.phase3_filter.filter import (
            MetricResult,
            QualityReport,
            evaluate_pair,
            metric_connected_components,
            metric_not_degenerate,
        )

        before = trimesh.load(str(before_ply), process=False)
        after = trimesh.load(str(after_ply), process=False)
    except Exception as e:
        s = VLMScore(edit_id=eid, edit_type=et, reason=f"Mesh prefilter load: {e}")
        s.quality_tier = "rejected"
        return s

    def _negative_from_failed(report, prefix: str) -> VLMScore:
        failed = [m for m in report.metrics if not m.passed]
        reasons = "; ".join(
            f"{m.name}: {m.reason or 'fail'}" for m in failed[:4])
        s = VLMScore(edit_id=eid, edit_type=et)
        s.reason = f"{prefix}{reasons}"
        s.score = round(report.score, 4)
        s.quality_tier = "negative"
        s.edit_executed = False
        return s

    if et in _MESH_PREFILTER_FULL:
        report = evaluate_pair(eid, et, before, after, cfg)
        if not report.passed:
            return _negative_from_failed(report, "Mesh prefilter: ")
        return None

    if et in _MESH_PREFILTER_LIGHT:
        results = []
        for fn in (metric_not_degenerate, metric_connected_components):
            try:
                results.append(fn(before, after, cfg))
            except Exception as e:
                results.append(
                    MetricResult(fn.__name__.replace("metric_", ""), 0.0, False,
                                 reason=str(e)))

        bv, av = len(before.vertices), len(after.vertices)
        min_v = int(p3.get("vlm_mesh_prefilter_min_vertices", 8))
        if bv < min_v or av < min_v:
            results.append(
                MetricResult(
                    "min_vertices",
                    float(min(bv, av)),
                    False,
                    weight=2.0,
                    reason=f"before={bv}, after={av} (min={min_v})",
                ))

        total_weight = sum(r.weight for r in results)
        score = (
            sum(r.weight * (1.0 if r.passed else 0.0) for r in results)
            / max(total_weight, 1e-8)
        )
        passed = all(r.passed for r in results)
        if not passed:
            qr = QualityReport(
                edit_id=eid,
                edit_type=et,
                passed=False,
                score=score,
                metrics=results,
            )
            return _negative_from_failed(qr, "Mesh prefilter (light): ")
        return None

    return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_edit(
    pipeline,
    slat_dir_before: Path,
    slat_dir_after: Path,
    edit_id: str,
    edit_type: str,
    edit_prompt: str,
    object_desc: str,
    part_label: str,
    vlm_client,
    vlm_model: str,
    num_views: int = 4,
    device: str = "cuda",
    vlm_max_tokens: int = 4096,
    vlm_json_object_mode: bool = False,
) -> VLMScore:
    """Evaluate a single edit pair: render → compose → VLM judge."""
    score = VLMScore(edit_id=edit_id, edit_type=edit_type)

    try:
        # Load SLATs
        slat_before = load_slat(slat_dir_before, device)
        slat_after = load_slat(slat_dir_after, device)

        # Render views
        before_imgs = render_views(pipeline, slat_before, num_views)
        after_imgs = render_views(pipeline, slat_after, num_views)

        # Compose comparison image
        comp_bytes = compose_comparison(before_imgs, after_imgs)

        # Call VLM judge
        result = call_vlm_judge(
            vlm_client, vlm_model, comp_bytes,
            edit_prompt, edit_type, object_desc, part_label,
            max_tokens=vlm_max_tokens,
            json_object_mode=vlm_json_object_mode,
        )

        if result is None:
            score.reason = "VLM returned no valid response"
            return score

        score.edit_executed = bool(result.get("edit_executed", False))
        score.correct_region = bool(result.get("correct_region", False))
        score.preserve_other = bool(result.get("preserve_other", False))
        score.visual_quality = int(result.get("visual_quality", 0))
        score.artifact_free = bool(result.get("artifact_free", False))
        score.reason = result.get("reason", "")
        score.score = compute_composite_score(score)
        score.quality_tier = classify_tier(score)

    except Exception as e:
        score.reason = f"Evaluation error: {e}"
        logger.error(f"  {edit_id}: {e}")

    return score


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_vlm_filter(
    cfg: dict,
    results_path: str | Path,
    mesh_pairs_dir: str | Path,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    num_views: int = 4,
    device: str = "cuda",
) -> list[dict]:
    """Run Phase 3 VLM filter on Phase 2.5 edit results.

    All edits are kept and scored with quality tiers:
      high/medium — positive training samples
      low         — usable with caution
      negative    — failed edits, usable as negative samples
      rejected    — evaluation errors, discard

    Args:
        cfg: pipeline config
        results_path: edit_results.jsonl from Phase 2.5
        mesh_pairs_dir: directory with {edit_id}/before_slat, after_slat
        output_dir: where to write filter results (default: cache/phase3)
        limit: max edits to evaluate
        num_views: number of views to render per model

    Returns:
        All scored entries (sorted by score descending)
    """
    from openai import OpenAI
    from tqdm import tqdm

    results_path = Path(results_path)
    mesh_pairs_dir = Path(mesh_pairs_dir)

    p3_cfg = cfg.get("phase3", {})
    num_views = int(p3_cfg.get("num_views", num_views))
    vlm_max_tokens = int(p3_cfg.get("vlm_max_tokens", 4096))
    vlm_json_object_mode = bool(p3_cfg.get("vlm_json_response_format", False))
    if output_dir is None:
        output_dir = Path(p3_cfg.get("cache_dir", "cache/phase3"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load edit results
    entries = []
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == "success":
                entries.append(rec)

    if limit:
        entries = entries[:limit]

    if not entries:
        logger.info("No successful edits to filter")
        return []

    # VLM client
    p0 = cfg.get("phase0", {})
    api_key = p0.get("vlm_api_key", "")
    if not api_key:
        import yaml
        default_cfg = Path(__file__).parents[2] / "configs" / "default.yaml"
        if default_cfg.exists():
            with open(default_cfg) as f:
                dcfg = yaml.safe_load(f)
            api_key = dcfg.get("phase0", {}).get("vlm_api_key", "")

    if not api_key:
        raise ValueError("No API key for VLM judge")

    client = OpenAI(
        base_url=p0.get("vlm_base_url", ""),
        api_key=api_key,
    )
    vlm_model = p0.get("vlm_model", "gemini-3.1-flash-lite-preview")

    # Load TRELLIS decoder
    logger.info("Loading TRELLIS decoder for rendering...")
    p25 = cfg.get("phase2_5", {})

    import sys
    project_root = Path(__file__).parents[2]
    third_party = str(project_root / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    from trellis.pipelines import TrellisImageTo3DPipeline

    # Resolve checkpoint path (load_config makes this absolute when using ckpt_root)
    ckpt_spec = p25.get("trellis_image_ckpt", "checkpoints/TRELLIS-image-large")
    ckpt_path = Path(ckpt_spec)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_spec
    if not (ckpt_path / "pipeline.json").exists():
        raise FileNotFoundError(
            f"TRELLIS checkpoint not found at {ckpt_path}")
    ckpt = str(ckpt_path)

    logger.info(f"TRELLIS checkpoint: {ckpt}")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(ckpt)
    pipeline.to(device)

    pre_on = bool(p3_cfg.get("vlm_mesh_prefilter", False))
    if pre_on:
        logger.info(
            "Mesh prefilter enabled: full=%s light=%s (skips TRELLIS+VLM on fail)",
            sorted(_MESH_PREFILTER_FULL),
            sorted(_MESH_PREFILTER_LIGHT),
        )

    logger.info(
        f"Phase 3 VLM filter: {len(entries)} edits, {num_views} views each, "
        f"model={vlm_model}, max_tokens={vlm_max_tokens}, "
        f"json_object_mode={vlm_json_object_mode}"
    )

    # Evaluate — all results kept, classified by quality tier
    all_scored: list[dict] = []
    scores_path = output_dir / "vlm_scores.jsonl"
    mesh_prefilter_skipped_vlm = 0

    with open(scores_path, "w") as fp:
        for i, entry in enumerate(tqdm(entries, desc="Phase 3: VLM Filter")):
            eid = entry["edit_id"]
            pair_dir = mesh_pairs_dir / eid

            has_before = (pair_dir / "before.npz").exists() or (pair_dir / "before_slat").exists()
            has_after = (pair_dir / "after.npz").exists() or (pair_dir / "after_slat").exists()

            if not has_before or not has_after:
                score = VLMScore(edit_id=eid, edit_type=entry.get("edit_type", ""),
                                 reason="SLAT files missing")
                score_dict = score.to_dict()
                fp.write(json.dumps(score_dict, ensure_ascii=False) + "\n")
                fp.flush()
                all_scored.append({**entry, **score_dict})
                continue

            pre_score = mesh_prefilter_before_vlm(pair_dir, entry, cfg)
            if pre_score is not None:
                mesh_prefilter_skipped_vlm += 1
                score_dict = pre_score.to_dict()
                fp.write(json.dumps(score_dict, ensure_ascii=False) + "\n")
                fp.flush()
                all_scored.append({**entry, **score_dict})
                logger.info(
                    f"  [{i+1}/{len(entries)}] {eid}: "
                    f"{pre_score.quality_tier.upper()} (mesh prefilter, skip VLM) "
                    f"{pre_score.reason[:80]}")
                continue

            # Extract part label from edit_prompt or entry
            part_label = entry.get("old_label", "")
            if not part_label:
                # Try to extract from remove_labels or add_labels
                labels = entry.get("remove_labels", entry.get("add_labels", []))
                part_label = labels[0] if labels else "unknown"

            score = evaluate_edit(
                pipeline=pipeline,
                slat_dir_before=slat_before,
                slat_dir_after=slat_after,
                edit_id=eid,
                edit_type=entry.get("edit_type", ""),
                edit_prompt=entry.get("edit_prompt", ""),
                object_desc=entry.get("object_desc", ""),
                part_label=part_label,
                vlm_client=client,
                vlm_model=vlm_model,
                num_views=num_views,
                device=device,
                vlm_max_tokens=vlm_max_tokens,
                vlm_json_object_mode=vlm_json_object_mode,
            )

            score_dict = score.to_dict()
            fp.write(json.dumps(score_dict, ensure_ascii=False) + "\n")
            fp.flush()
            all_scored.append({**entry, **score_dict})

            tier = score.quality_tier
            logger.info(f"  [{i+1}/{len(entries)}] {eid}: "
                        f"{tier.upper()} (score={score.score:.2f}) "
                        f"{score.reason}")

    if pre_on and mesh_prefilter_skipped_vlm:
        logger.info(
            "Mesh prefilter dropped %d edits before TRELLIS+VLM "
            "(see reasons in vlm_scores.jsonl)",
            mesh_prefilter_skipped_vlm,
        )

    # Write tiered output
    _write_tiered_output(all_scored, output_dir)

    return all_scored


def _write_tiered_output(all_scored: list[dict], output_dir: Path):
    """Write tiered output: all edits kept, grouped by quality."""
    from collections import defaultdict

    summary_path = output_dir / "summary.json"

    # Group by tier
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for e in all_scored:
        tier = e.get("quality_tier", "rejected")
        by_tier[tier].append(e)

    # Write per-tier JSONL files
    for tier, items in by_tier.items():
        tier_path = output_dir / f"tier_{tier}.jsonl"
        with open(tier_path, "w") as f:
            for e in items:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Write combined scored output (all tiers, sorted by score desc)
    all_path = output_dir / "all_scored.jsonl"
    all_sorted = sorted(all_scored, key=lambda x: x.get("score", 0), reverse=True)
    with open(all_path, "w") as f:
        for e in all_sorted:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Stats
    tier_counts = {t: len(items) for t, items in by_tier.items()}
    scores = [e["score"] for e in all_scored if "score" in e]

    by_type = defaultdict(lambda: defaultdict(int))
    for e in all_scored:
        et = e.get("edit_type", "unknown")
        tier = e.get("quality_tier", "rejected")
        by_type[et][tier] += 1
        by_type[et]["total"] += 1

    summary = {
        "total": len(all_scored),
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "tier_counts": tier_counts,
        "by_edit_type": {k: dict(v) for k, v in by_type.items()},
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Phase 3 VLM Quality Assessment")
    print(f"{'='*60}")
    print(f"  Total: {summary['total']}, Avg score: {summary['avg_score']:.3f}")
    print(f"\n  Quality tiers:")
    tier_order = ["high", "medium", "low", "negative", "rejected"]
    tier_desc = {
        "high": "ideal training data",
        "medium": "usable training data",
        "low": "use with caution",
        "negative": "negative samples",
        "rejected": "evaluation failed",
    }
    for tier in tier_order:
        n = tier_counts.get(tier, 0)
        pct = n / max(len(all_scored), 1)
        desc = tier_desc.get(tier, "")
        print(f"    {tier:10s}: {n:4d} ({pct:5.1%})  — {desc}")

    print(f"\n  By edit type:")
    for et, counts in sorted(by_type.items()):
        total = counts["total"]
        parts = ", ".join(f"{t}={counts.get(t, 0)}"
                          for t in tier_order if counts.get(t, 0) > 0)
        print(f"    {et:15s}: {total:4d} [{parts}]")

    print(f"\n  Output: {all_path}")
    print(f"{'='*60}")
