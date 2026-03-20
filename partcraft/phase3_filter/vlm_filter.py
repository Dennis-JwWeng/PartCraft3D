"""Phase 3: VLM-based quality filter for Phase 2.5 SLAT edit pairs.

Renders before/after Gaussian views from SLAT, composes side-by-side
comparison images, and sends them to a VLM for semantic quality scoring.

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
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
    """Load SLAT from feats.pt + coords.pt."""
    from trellis.modules import sparse as sp
    feats = torch.load(slat_dir / "feats.pt", weights_only=True)
    coords = torch.load(slat_dir / "coords.pt", weights_only=True)
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

Evaluate the edit quality and return JSON only, no fences:
{{"edit_executed":true,"correct_region":true,"preserve_other":true,"visual_quality":4,"artifact_free":true,"reason":"brief explanation"}}

Scoring criteria:
- edit_executed: Did the described edit visibly happen? (true/false)
- correct_region: Was the change applied to the correct part ({part_label})? (true/false)
- preserve_other: Are all other parts of the object preserved and intact? (true/false)
- visual_quality: Overall visual quality of the AFTER model (1=terrible, 2=poor, 3=acceptable, 4=good, 5=excellent)
- artifact_free: Is the AFTER model free of obvious artifacts like floating blobs, broken surfaces, or missing geometry? (true/false)
- reason: One sentence explaining your assessment

Be strict but fair. Minor imperfections are acceptable (quality=3-4). Only fail edit_executed if there is NO visible change matching the prompt."""


def _extract_json_from_vlm(content: str) -> dict | None:
    """Robustly extract JSON from VLM response, handling various formats."""
    if not content:
        return None
    content = content.strip()

    # Strip <think>...</think> blocks (Gemini 2.5 Flash thinking mode)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting the first {...} block
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None


def call_vlm_judge(client, model: str, img_bytes: bytes,
                   edit_prompt: str, edit_type: str,
                   object_desc: str, part_label: str,
                   max_retries: int = 4) -> dict | None:
    """Call VLM to judge edit quality. Returns parsed JSON or None."""
    import time
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    text = _build_judge_prompt(edit_prompt, edit_type, object_desc, part_label)

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": text},
                    ]
                }],
                temperature=0.3,
                max_tokens=512,
                timeout=120,
            )
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
            edit_prompt, edit_type, object_desc, part_label)

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
        return [], []

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

    # Resolve checkpoint path
    ckpt_rel = p25.get("trellis_image_ckpt", "checkpoints/TRELLIS-image-large")
    ckpt_path = project_root / ckpt_rel
    if not (ckpt_path / "pipeline.json").exists():
        raise FileNotFoundError(
            f"TRELLIS checkpoint not found at {ckpt_path}")
    ckpt = str(ckpt_path)

    logger.info(f"TRELLIS checkpoint: {ckpt}")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(ckpt)
    pipeline.to(device)

    logger.info(f"Phase 3 VLM filter: {len(entries)} edits, "
                f"{num_views} views each, model={vlm_model}")

    # Evaluate — all results kept, classified by quality tier
    all_scored: list[dict] = []
    scores_path = output_dir / "vlm_scores.jsonl"

    with open(scores_path, "w") as fp:
        for i, entry in enumerate(tqdm(entries, desc="Phase 3: VLM Filter")):
            eid = entry["edit_id"]
            pair_dir = mesh_pairs_dir / eid

            slat_before = pair_dir / "before_slat"
            slat_after = pair_dir / "after_slat"

            if not slat_before.exists() or not slat_after.exists():
                score = VLMScore(edit_id=eid, edit_type=entry.get("edit_type", ""),
                                 reason="SLAT files missing")
                score_dict = score.to_dict()
                fp.write(json.dumps(score_dict, ensure_ascii=False) + "\n")
                fp.flush()
                all_scored.append({**entry, **score_dict})
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
            )

            score_dict = score.to_dict()
            fp.write(json.dumps(score_dict, ensure_ascii=False) + "\n")
            fp.flush()
            all_scored.append({**entry, **score_dict})

            tier = score.quality_tier
            logger.info(f"  [{i+1}/{len(entries)}] {eid}: "
                        f"{tier.upper()} (score={score.score:.2f}) "
                        f"{score.reason}")

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
