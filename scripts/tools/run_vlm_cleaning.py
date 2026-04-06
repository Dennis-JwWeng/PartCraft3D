#!/usr/bin/env python3
"""VLM-based quality cleaning for repacked edit pairs.

Replaces computational Layer 2 pair checks with Qwen VLM visual judgment.
Rendering paths:
  - deletion:        Blender PLY rendering (no TRELLIS needed)
  - mod/scl/mat/glb: TRELLIS SLAT decode → Gaussian → multi-view render

Identity edits are skipped (auto-pass).  Addition edits inherit scores
from their corresponding deletion (they share the same before/after pair,
swapped).

Usage:
    # 1. Start Qwen VLM server (GPU 0):
    conda activate qwen_test
    CUDA_VISIBLE_DEVICES=0 VLM_MODEL=/Node11_nvme/zsn/checkpoints/Qwen3.5-27B \\
        bash scripts/tools/launch_local_vlm.sh

    # 2. Run deletion-only (no TRELLIS GPU needed):
    python scripts/tools/run_vlm_cleaning.py \\
        --root outputs/partverse/partverse_pairs \\
        --output-root outputs/partverse \\
        --vlm-url http://localhost:8002/v1 \\
        --vlm-model Qwen3.5-27B \\
        --shards 01 --only-types deletion

    # 3. Run others (TRELLIS on GPU 3):
    CUDA_VISIBLE_DEVICES=3 python scripts/tools/run_vlm_cleaning.py \\
        --root outputs/partverse/partverse_pairs \\
        --output-root outputs/partverse \\
        --vlm-url http://localhost:8002/v1 \\
        --vlm-model Qwen3.5-27B \\
        --shards 01 --only-types modification scale material global

    # 4. Multi-GPU (see run_vlm_cleaning_multi_gpu.sh)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from partcraft.phase3_filter.vlm_filter import (
    VLMScore,
    call_vlm_judge,
    classify_tier,
    compose_comparison,
    compute_composite_score,
)

logger = logging.getLogger(__name__)

_TYPES_NEEDING_TRELLIS = frozenset({"modification", "scale", "material", "global"})


# ─── Helpers ───────────────────────────────────────────────────────

def _get_part_label(edit: dict) -> str:
    """Extract a human-readable part label from edit metadata."""
    for key in ("remove_labels", "target_part_labels"):
        labels = edit.get(key)
        if labels:
            return labels[0] if isinstance(labels, list) else str(labels)
    return ""


def _load_existing_scores(path: Path) -> dict[str, dict]:
    """Load already-scored edit_ids from JSONL for resume support."""
    scores: dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    scores[rec["edit_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return scores


# ─── Rendering ─────────────────────────────────────────────────────

# Import 3-view angles from the single source of truth.
from partcraft.phase3_filter.vlm_filter import _VLM_YAWS, _VLM_PITCHES


def _render_ply_pair(
    edit_id: str,
    shard: str,
    output_root: Path,
    blender_path: str,
    num_views: int = 3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render deletion edit from PLY via Blender (3 optimal-coverage views)."""
    mesh_dir = (
        output_root / f"shard_{shard}" / f"mesh_pairs_shard{shard}" / edit_id
    )
    before_ply = mesh_dir / "before.ply"
    after_ply = mesh_dir / "after.ply"

    if not before_ply.exists() or not after_ply.exists():
        raise FileNotFoundError(f"PLY not found in {mesh_dir}")

    vis_dir = str(_PROJECT_ROOT / "scripts" / "vis")
    if vis_dir not in sys.path:
        sys.path.insert(0, vis_dir)
    from render_ply_pairs import render_3views  # noqa: E402

    before_imgs = render_3views(str(before_ply), blender_path=blender_path)
    after_imgs = render_3views(str(after_ply), blender_path=blender_path)
    return before_imgs[:num_views], after_imgs[:num_views]


def _render_slat_views(
    pipeline,
    npz_path: Path,
    device: str,
    num_views: int = 3,
) -> list[np.ndarray]:
    """Load NPZ → SparseTensor → TRELLIS decode → Gaussian → render (3 views)."""
    from trellis.modules.sparse.basic import SparseTensor
    from trellis.utils import render_utils

    data = np.load(npz_path)
    feats = torch.tensor(data["slat_feats"]).float().to(device)
    coords = torch.tensor(data["slat_coords"]).int().to(device)
    slat = SparseTensor(feats=feats, coords=coords)

    outputs = pipeline.decode_slat(slat, ["gaussian"])
    gaussian = outputs["gaussian"][0]

    yaws = _VLM_YAWS[:num_views]
    pitches = _VLM_PITCHES[:num_views]
    imgs = render_utils.Trellis_render_multiview_images(
        gaussian, yaws, pitches
    )["color"]

    del slat, outputs, gaussian
    torch.cuda.empty_cache()
    return imgs[:num_views]


# ─── VLM scoring ──────────────────────────────────────────────────

def _score_one(
    client,
    model: str,
    comp_bytes: bytes,
    edit: dict,
    object_desc: str,
    max_tokens: int = 4096,
) -> VLMScore:
    """Send comparison image to VLM and parse structured score."""
    edit_id = edit["edit_id"]
    edit_type = edit["type"]
    prompt = edit.get("prompt", "")
    part_label = _get_part_label(edit)

    score = VLMScore(edit_id=edit_id, edit_type=edit_type)

    result = call_vlm_judge(
        client,
        model,
        comp_bytes,
        prompt,
        edit_type,
        object_desc,
        part_label,
        max_tokens=max_tokens,
    )
    if result is None:
        score.reason = "VLM returned no valid response"
        score.quality_tier = "rejected"
        return score

    score.edit_executed = bool(result.get("edit_executed", False))
    score.correct_region = bool(result.get("correct_region", False))
    score.preserve_other = bool(result.get("preserve_other", False))
    score.visual_quality = int(result.get("visual_quality", 0))
    score.artifact_free = bool(result.get("artifact_free", False))
    score.reason = result.get("reason", "")
    score.prompt_quality = int(result.get("prompt_quality", 0))
    score.improved_prompt = str(result.get("improved_prompt", ""))
    score.improved_after_desc = str(result.get("improved_after_desc", ""))
    score.score = compute_composite_score(score)
    score.quality_tier = classify_tier(score)
    return score


# ─── Propagation & output ─────────────────────────────────────────

def _propagate_del_to_add(
    scores: dict[str, dict],
    root: Path,
    shards: list[str],
) -> dict[str, dict]:
    """For each addition edit, inherit score from corresponding deletion."""
    manifest = root / "manifest.jsonl"
    meta_cache: dict[tuple[str, str], dict] = {}
    added: dict[str, dict] = {}

    with open(manifest) as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] != "addition":
                continue
            shard = rec["shard"]
            if shards and shard not in shards:
                continue

            obj_id = rec["obj_id"]
            key = (shard, obj_id)
            if key not in meta_cache:
                meta_path = root / f"shard_{shard}" / obj_id / "metadata.json"
                if not meta_path.exists():
                    continue
                with open(meta_path) as mf:
                    meta_cache[key] = json.load(mf)

            meta = meta_cache[key]
            edit = meta["edits"][rec["edit_idx"]]
            add_id = edit["edit_id"]
            del_seq = edit.get("source_del_seq", -1)

            # Find corresponding deletion edit_id
            del_id = None
            for e in meta["edits"]:
                if e["type"] == "deletion" and e["seq"] == del_seq:
                    del_id = e["edit_id"]
                    break

            if del_id and del_id in scores:
                src = scores[del_id]
                added[add_id] = {
                    "edit_id": add_id,
                    "edit_type": "addition",
                    "quality_tier": src.get(
                        "quality_tier", src.get("tier", "rejected")
                    ),
                    "score": src.get("score", 0.0),
                    "source": del_id,
                    "edit_executed": src.get("edit_executed", False),
                    "correct_region": src.get("correct_region", False),
                    "preserve_other": src.get("preserve_other", False),
                    "visual_quality": src.get("visual_quality", 0),
                    "artifact_free": src.get("artifact_free", False),
                    "reason": f"Inherited from {del_id}",
                }

    return added


def _write_quality_json(
    all_scores: dict[str, dict],
    root: Path,
    shards: list[str],
):
    """Write quality.json per object, compatible with EditPairDataset."""
    manifest = root / "manifest.jsonl"
    obj_edits: dict[tuple[str, str], list[dict]] = defaultdict(list)
    meta_cache: dict[tuple[str, str], dict] = {}

    with open(manifest) as f:
        for line in f:
            rec = json.loads(line)
            shard = rec["shard"]
            if shards and shard not in shards:
                continue
            obj_id = rec["obj_id"]
            key = (shard, obj_id)
            if key not in meta_cache:
                meta_path = root / f"shard_{shard}" / obj_id / "metadata.json"
                if not meta_path.exists():
                    continue
                with open(meta_path) as mf:
                    meta_cache[key] = json.load(mf)

            meta = meta_cache[key]
            edit = meta["edits"][rec["edit_idx"]]
            edit_id = edit["edit_id"]
            etype = rec["type"]

            if etype == "identity":
                obj_edits[key].append(
                    {"edit_id": edit_id, "tier": "high", "score": 1.0}
                )
            elif edit_id in all_scores:
                s = all_scores[edit_id]
                obj_edits[key].append(
                    {
                        "edit_id": edit_id,
                        "tier": s.get(
                            "quality_tier", s.get("tier", "rejected")
                        ),
                        "score": s.get("score", 0.0),
                    }
                )
            else:
                obj_edits[key].append(
                    {"edit_id": edit_id, "tier": "rejected", "score": 0.0}
                )

    n_written = 0
    for (shard, obj_id), edits in obj_edits.items():
        obj_dir = root / f"shard_{shard}" / obj_id
        num_passed = sum(
            1 for e in edits if e["tier"] in ("high", "medium")
        )
        quality = {
            "obj_id": obj_id,
            "shard": shard,
            "num_edits": len(edits),
            "num_passed": num_passed,
            "edits": edits,
        }
        with open(obj_dir / "quality.json", "w") as f:
            json.dump(quality, f, indent=2, ensure_ascii=False)
        n_written += 1

    logger.info("Wrote quality.json for %d objects", n_written)


def _print_summary(all_scores: dict[str, dict]):
    """Print tier distribution summary."""
    tier_counts: Counter = Counter()
    type_tier: dict[str, Counter] = defaultdict(Counter)

    for s in all_scores.values():
        tier = s.get("quality_tier", s.get("tier", "rejected"))
        etype = s.get("edit_type", "unknown")
        tier_counts[tier] += 1
        type_tier[etype][tier] += 1

    print(f"\n{'=' * 60}")
    print("VLM Quality Assessment Summary")
    print(f"{'=' * 60}")
    print(f"  Total scored: {len(all_scores)}")
    for tier in ("high", "medium", "low", "negative", "rejected"):
        n = tier_counts[tier]
        pct = n / max(len(all_scores), 1)
        print(f"    {tier:10s}: {n:5d} ({pct:5.1%})")
    print(f"\n  By edit type:")
    for et in sorted(type_tier):
        counts = type_tier[et]
        total = sum(counts.values())
        parts = ", ".join(
            f"{t}={counts[t]}"
            for t in ("high", "medium", "low", "negative", "rejected")
            if counts[t] > 0
        )
        print(f"    {et:15s}: {total:5d} [{parts}]")
    print(f"{'=' * 60}")


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLM-based quality cleaning for repacked edit pairs"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Repacked data root (with manifest.jsonl and shard_XX/)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Pipeline output root (parent of shard_XX/mesh_pairs_*)",
    )
    parser.add_argument("--vlm-url", default="http://localhost:8002/v1")
    parser.add_argument("--vlm-model", default="Qwen3.5-27B")
    parser.add_argument("--vlm-max-tokens", type=int, default=4096)
    parser.add_argument("--shards", nargs="+", default=[])
    parser.add_argument(
        "--only-types",
        nargs="+",
        default=None,
        help="Only process these edit types (e.g. deletion modification)",
    )
    parser.add_argument(
        "--trellis-ckpt",
        default="checkpoints/TRELLIS-image-large",
        help="TRELLIS checkpoint (absolute or relative to project root)",
    )
    parser.add_argument(
        "--blender-path",
        default="/Node11_nvme/artgen/lac/.tools/"
        "blender-4.2.0-linux-x64/blender",
    )
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument(
        "--render-cache",
        default=None,
        help="Dir to cache rendered comparison PNGs (for resume/debug)",
    )
    parser.add_argument(
        "--scores-file",
        default=None,
        help="JSONL output for scores (default: {root}/vlm_scores.jsonl)",
    )
    parser.add_argument(
        "--include-objects",
        default=None,
        help="File listing obj_ids to process (for multi-GPU splitting)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.root)
    output_root = Path(args.output_root)

    # Scores file (for incremental resume)
    scores_path = Path(args.scores_file) if args.scores_file else root / "vlm_scores.jsonl"

    # Render cache
    render_cache = (
        Path(args.render_cache)
        if args.render_cache
        else root / "_vlm_render_cache"
    )
    render_cache.mkdir(parents=True, exist_ok=True)

    # Load existing scores
    existing = _load_existing_scores(scores_path)
    logger.info("Loaded %d existing scores from %s", len(existing), scores_path)

    # Object filter (multi-GPU splitting)
    include_objs: set[str] | None = None
    if args.include_objects:
        with open(args.include_objects) as f:
            include_objs = {line.strip() for line in f if line.strip()}
        logger.info("Object filter: %d objects", len(include_objs))

    # ── Build work list ────────────────────────────────────────────
    manifest = root / "manifest.jsonl"
    entries: list[dict] = []
    meta_cache: dict[tuple[str, str], dict] = {}

    with open(manifest) as f:
        for line in f:
            rec = json.loads(line)
            shard = rec["shard"]
            if args.shards and shard not in args.shards:
                continue

            etype = rec["type"]
            # Skip identity (auto-pass) and addition (inherit from del)
            if etype in ("identity", "addition"):
                continue
            if args.only_types and etype not in args.only_types:
                continue

            obj_id = rec["obj_id"]
            if include_objs is not None and obj_id not in include_objs:
                continue

            key = (shard, obj_id)
            if key not in meta_cache:
                meta_path = root / f"shard_{shard}" / obj_id / "metadata.json"
                if not meta_path.exists():
                    continue
                with open(meta_path) as mf:
                    meta_cache[key] = json.load(mf)

            meta = meta_cache[key]
            edit = meta["edits"][rec["edit_idx"]]
            edit_id = edit["edit_id"]

            if edit_id in existing:
                continue

            entries.append(
                {
                    "shard": shard,
                    "obj_id": obj_id,
                    "edit_idx": rec["edit_idx"],
                    "edit_id": edit_id,
                    "type": etype,
                    "edit": edit,
                    "object_desc": meta.get("object_desc", ""),
                }
            )

    logger.info("Work list: %d edits to evaluate", len(entries))

    if not entries:
        logger.info("Nothing to do — generating quality.json from existing scores")
        if include_objs is None:
            add_scores = _propagate_del_to_add(existing, root, args.shards)
            all_scores = {**existing, **add_scores}
            _write_quality_json(all_scores, root, args.shards)
            _print_summary(all_scores)
        return

    # ── Group by object for cache efficiency ───────────────────────
    obj_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in entries:
        obj_groups[(entry["shard"], entry["obj_id"])].append(entry)

    # ── Load TRELLIS if needed ─────────────────────────────────────
    need_trellis = any(e["type"] in _TYPES_NEEDING_TRELLIS for e in entries)
    pipeline = None

    if need_trellis:
        logger.info("Loading TRELLIS pipeline for SLAT rendering...")
        third_party = str(_PROJECT_ROOT / "third_party")
        if third_party not in sys.path:
            sys.path.insert(0, third_party)
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        from trellis.pipelines import TrellisImageTo3DPipeline

        ckpt = args.trellis_ckpt
        if not Path(ckpt).is_absolute():
            ckpt = str(_PROJECT_ROOT / ckpt)
        pipeline = TrellisImageTo3DPipeline.from_pretrained(ckpt)
        pipeline.to(args.device)
        logger.info("TRELLIS loaded on %s", args.device)

    # ── VLM client ─────────────────────────────────────────────────
    from openai import OpenAI

    client = OpenAI(base_url=args.vlm_url, api_key="dummy")
    logger.info("VLM: %s model=%s", args.vlm_url, args.vlm_model)

    # ── Process ────────────────────────────────────────────────────
    scored = 0
    errors = 0

    with open(scores_path, "a") as fp:
        pbar = tqdm(total=len(entries), desc="VLM cleaning")

        for (shard, obj_id), edits in obj_groups.items():
            obj_dir = root / f"shard_{shard}" / obj_id
            orig_trellis_views: list[np.ndarray] | None = None

            for entry in edits:
                edit_id = entry["edit_id"]
                edit = entry["edit"]
                etype = entry["type"]

                try:
                    # ── Render (with disk cache) ───────────────────
                    cache_png = render_cache / f"{edit_id}.png"
                    if cache_png.exists():
                        comp_bytes = cache_png.read_bytes()
                    elif etype == "deletion":
                        before_imgs, after_imgs = _render_ply_pair(
                            edit_id,
                            shard,
                            output_root,
                            args.blender_path,
                            args.num_views,
                        )
                        comp_bytes = compose_comparison(
                            before_imgs, after_imgs
                        )
                        cache_png.write_bytes(comp_bytes)
                    elif etype in _TYPES_NEEDING_TRELLIS:
                        if orig_trellis_views is None:
                            orig_trellis_views = _render_slat_views(
                                pipeline,
                                obj_dir / "original.npz",
                                args.device,
                                args.num_views,
                            )
                        fname = edit.get("file")
                        if not fname:
                            raise FileNotFoundError(
                                f"No file for {edit_id}"
                            )
                        after_views = _render_slat_views(
                            pipeline,
                            obj_dir / fname,
                            args.device,
                            args.num_views,
                        )
                        comp_bytes = compose_comparison(
                            orig_trellis_views, after_views
                        )
                        cache_png.write_bytes(comp_bytes)
                    else:
                        raise ValueError(f"Unsupported type: {etype}")

                    # ── VLM score ──────────────────────────────────
                    score = _score_one(
                        client,
                        args.vlm_model,
                        comp_bytes,
                        edit,
                        entry["object_desc"],
                        max_tokens=args.vlm_max_tokens,
                    )

                    score_dict = score.to_dict()
                    fp.write(
                        json.dumps(score_dict, ensure_ascii=False) + "\n"
                    )
                    fp.flush()
                    existing[edit_id] = score_dict
                    scored += 1

                    pbar.set_postfix(
                        scored=scored,
                        err=errors,
                        tier=score.quality_tier,
                    )

                except Exception as e:
                    logger.error("  %s: %s", edit_id, e)
                    err_score = VLMScore(
                        edit_id=edit_id,
                        edit_type=etype,
                        reason=f"Error: {e}",
                    )
                    err_score.quality_tier = "rejected"
                    score_dict = err_score.to_dict()
                    fp.write(
                        json.dumps(score_dict, ensure_ascii=False) + "\n"
                    )
                    fp.flush()
                    existing[edit_id] = score_dict
                    errors += 1

                pbar.update(1)

            # Clear TRELLIS cache between objects
            orig_trellis_views = None
            if pipeline:
                torch.cuda.empty_cache()

        pbar.close()

    logger.info("Scored %d edits (%d errors)", scored, errors)

    # ── Finalize (skip when running as multi-GPU worker) ───────────
    if include_objs is not None:
        logger.info(
            "Worker mode — skipping propagation / quality.json "
            "(run launcher to finalize)"
        )
        return

    # Propagate del → add
    add_scores = _propagate_del_to_add(existing, root, args.shards)
    logger.info(
        "Propagated %d deletion scores to addition edits", len(add_scores)
    )
    with open(scores_path, "a") as fp:
        for s in add_scores.values():
            fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    all_scores = {**existing, **add_scores}
    _write_quality_json(all_scores, root, args.shards)
    _print_summary(all_scores)


if __name__ == "__main__":
    main()
