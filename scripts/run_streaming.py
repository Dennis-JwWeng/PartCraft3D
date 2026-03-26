#!/usr/bin/env python3
"""PartCraft3D Streaming Pipeline — per-object full pipeline.

Each object goes through the full chain (enrich → plan → 2D+3D edit)
before moving to the next. First results appear fast, lower peak memory,
and immediate feedback on failures.

Per object:
  1. VLM enrichment → semantic record
  2. Plan edits from record → EditSpecs
  3. For each spec: 2D edit → 3D TRELLIS edit (inline)

Note: No cross-object swap modifications (addition from other objects).
For cross-object swaps, use the batch pipeline (run_pipeline.py).

After streaming completes, run quality scoring + export with:
    python scripts/run_pipeline.py --config <same> --steps 5 6 --tag <same>

Usage:
    # Basic streaming
    ATTN_BACKEND=xformers python scripts/run_streaming.py \\
        --config configs/hybrid_streaming.yaml --tag v1

    # Multi-GPU parallel (2 workers)
    ATTN_BACKEND=xformers python scripts/run_streaming.py \\
        --config configs/hybrid_streaming.yaml --tag v1 \\
        --num-workers 2 --worker-id 0   # GPU 0
    ATTN_BACKEND=xformers python scripts/run_streaming.py \\
        --config configs/hybrid_streaming.yaml --tag v1 \\
        --num-workers 2 --worker-id 1   # GPU 1

  # Overlap: pipeline.lookahead_objects or --lookahead-objects N (0=serial).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path before any project imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image

from scripts.pipeline_common import (
    PROJECT_ROOT,
    load_config, setup_logging,
    PartCraftDataset, EditSpec,
    resolve_api_key, normalize_cache_dirs, set_attn_backend, create_dataset,
    resolve_data_dirs,
)


def run_streaming(cfg, dataset, logger, args):
    """Streaming pipeline: each object goes through the full chain."""
    from openai import OpenAI
    from partcraft.phase1_planning.enricher import (
        _enrich_one_object_visual, _call_vlm, _fallback_enrichment,
        _result_to_phase0_record, load_thumbnail_from_npz,
    )
    from partcraft.phase1_planning.planner import plan_edits_for_record

    logger.info("=" * 60)
    logger.info("STREAMING MODE: per-object pipeline")
    logger.info("=" * 60)

    p0 = cfg.get("phase0", {})
    p25 = cfg.get("phase2_5", {})

    # ---- Paths (resolve relative paths against project root) ----
    project_root = PROJECT_ROOT

    def _resolve(p) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else project_root / pp

    tag_suffix = f"_{args.tag}" if args.tag else ""
    num_workers = args.num_workers
    worker_id = args.worker_id
    # Per-worker output files to avoid write conflicts
    worker_suffix = f"_w{worker_id}" if num_workers > 1 else ""
    labels_path = (_resolve(p0["cache_dir"])
                   / f"semantic_labels{tag_suffix}{worker_suffix}.jsonl")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    specs_path = (_resolve(cfg["phase1"]["cache_dir"])
                  / f"edit_specs{tag_suffix}{worker_suffix}.jsonl")
    specs_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = (_resolve(p25["cache_dir"])
                    / f"edit_results{tag_suffix}{worker_suffix}.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # 2D edit cache directory (shared across workers for same tag)
    edit_2d_subdir = f"2d_edits{tag_suffix}"
    edit_2d_dir = _resolve(p25["cache_dir"]) / edit_2d_subdir
    edit_2d_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _resolve(cfg["data"]["output_dir"])
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # ---- Resolve object list from dataset (NPZ files) ----
    # uid_info: obj_id -> (category, labels, actual_pids, shard_id)
    uid_info: dict[str, tuple[str, list[str], list[int], str]] = {}
    if dataset._index is None:
        dataset._build_index()
    for shard_id, obj_id in dataset._index:
        obj_rec = dataset.load_object(shard_id, obj_id)
        labels = []
        actual_pids = []
        for p in obj_rec.parts:
            if p.mesh_node_names:
                raw = p.mesh_node_names[0]
                label = raw.rsplit("_", 1)[0] if "_" in raw else raw
            else:
                label = p.cluster_name
            labels.append(label)
            actual_pids.append(p.part_id)
        uid_info[obj_id] = ("object", labels, actual_pids, shard_id)
        obj_rec.close()
    logger.info(f"Discovered {len(uid_info)} objects from dataset")

    all_uids = sorted(uid_info.keys())
    if args.limit:
        all_uids = all_uids[:args.limit]

    # ---- Worker partitioning for multi-GPU parallel ----
    if num_workers > 1:
        all_uids = [uid for i, uid in enumerate(all_uids)
                     if i % num_workers == worker_id]
        logger.info(f"Worker {worker_id}/{num_workers}: "
                    f"processing {len(all_uids)} objects")

    # ---- VLM client for enrichment ----
    api_key = resolve_api_key(cfg)
    if not api_key:
        logger.error("No API key for VLM enrichment")
        return
    vlm_model = p0.get("vlm_model", "gemini-3.1-flash-lite-preview")
    vlm_base_url = p0.get("vlm_base_url", "")
    vlm_client = OpenAI(base_url=vlm_base_url, api_key=api_key)

    # ---- Image edit setup ----
    image_edit_backend = p25.get("image_edit_backend", "api")
    edit_vlm_client = None
    if image_edit_backend != "local_diffusers":
        image_edit_url = p25.get("image_edit_base_url") or vlm_base_url
        edit_vlm_client = OpenAI(base_url=image_edit_url, api_key=api_key)

    # ---- TRELLIS setup ----
    from partcraft.phase2_assembly.trellis_refine import (
        TrellisRefiner, build_prompts_from_spec)

    slat_dir, img_enc_dir = resolve_data_dirs(cfg)
    refiner = TrellisRefiner(
        cache_dir=str(_resolve(p25["cache_dir"])),
        device="cuda",
        image_edit_model=p25.get("image_edit_model", "gemini-2.5-flash-image"),
        ckpt_dir=cfg.get("ckpt_root"),
        image_edit_backend=image_edit_backend,
        image_edit_base_url=p25.get("image_edit_base_url", "http://localhost:8001"),
        debug=args.debug,
        slat_dir=slat_dir,
        img_enc_dir=img_enc_dir,
    )
    refiner.load_models()

    # ---- Resume tracking ----
    done_labels: set[str] = set()
    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                try:
                    done_labels.add(json.loads(line)["obj_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    done_edits: set[str] = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_edits.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    logger.info(f"Streaming: {len(all_uids)} total objects, "
                f"{len(done_labels)} already enriched (resume)")

    # ---- Image source ----
    npz_dir = _resolve(cfg["data"]["image_npz_dir"])

    # Import 2D edit helpers
    from scripts.run_2d_edit import (
        prepare_input_image, call_local_edit, call_vlm_edit)
    pipeline_cfg = cfg.get("pipeline", {})
    la = getattr(args, "lookahead_objects", None)
    if la is None:
        lookahead = int(pipeline_cfg.get("lookahead_objects", 0) or 0)
    else:
        lookahead = int(la)
    if lookahead < 0:
        lookahead = 0
    logger.info(f"streaming lookahead_objects={lookahead}")


    # ---- Main loop (optional lookahead prefetch) ----
    from partcraft.streaming_lookahead import run_streaming_with_lookahead

    existing_specs_by_obj: dict[str, list] = {}
    if specs_path.exists():
        with open(specs_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                existing_specs_by_obj.setdefault(d["obj_id"], []).append(d)

    run_streaming_with_lookahead(
        all_uids=all_uids,
        uid_info=uid_info,
        lookahead=lookahead,
        p0=p0,
        p25=p25,
        labels_path=labels_path,
        specs_path=specs_path,
        results_path=results_path,
        edit_2d_dir=edit_2d_dir,
        mesh_pairs_dir=mesh_pairs_dir,
        npz_dir=npz_dir,
        dataset=dataset,
        cfg=cfg,
        args=args,
        logger=logger,
        refiner=refiner,
        build_prompts_from_spec=build_prompts_from_spec,
        vlm_client=vlm_client,
        vlm_model=vlm_model,
        image_edit_backend=image_edit_backend,
        edit_vlm_client=edit_vlm_client,
        done_labels=done_labels,
        done_edits=done_edits,
        existing_specs_by_obj=existing_specs_by_obj,
        prepare_input_image=prepare_input_image,
        call_local_edit=call_local_edit,
        call_vlm_edit=call_vlm_edit,
        EditSpec=EditSpec,
        plan_edits_for_record=plan_edits_for_record,
        _enrich_one_object_visual=_enrich_one_object_visual,
        _call_vlm=_call_vlm,
        _fallback_enrichment=_fallback_enrichment,
        _result_to_phase0_record=_result_to_phase0_record,
        load_thumbnail_from_npz=load_thumbnail_from_npz,
    )


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PartCraft3D Streaming Pipeline (per-object)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each object goes through: enrich → plan → 2D+3D edit before moving on.

After streaming completes, run quality scoring + export separately:
    python scripts/run_pipeline.py --config <same> --steps 5 6 --tag <same>

Examples:
  # Basic streaming
  ATTN_BACKEND=xformers python scripts/run_streaming.py \\
      --config configs/hybrid_streaming.yaml --tag v1

  # Multi-GPU parallel
  ATTN_BACKEND=xformers python scripts/run_streaming.py \\
      --config configs/hybrid_streaming.yaml --tag v1 \\
      --num-workers 2 --worker-id 0
""")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag for output isolation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-2d-edit", dest="use_2d", action="store_false",
                        default=True)
    parser.add_argument("--debug", action="store_true",
                        help="Save debug output files")
    parser.add_argument("--num-workers", type=int, default=1,
                        dest="num_workers",
                        help="Number of parallel streaming workers "
                             "(for multi-GPU)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="This worker's ID (0-indexed)")
    parser.add_argument(
        "--lookahead-objects", type=int, default=None,
        dest="lookahead_objects",
        help="Prefetch N upcoming objects (VLM/plan/2D) while GPU runs TRELLIS; "
             "0 disables. Overrides pipeline.lookahead_objects in config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    normalize_cache_dirs(cfg)
    set_attn_backend(cfg)
    if cfg.get("ckpt_root"):
        os.environ.setdefault("PARTCRAFT_CKPT_ROOT", cfg["ckpt_root"])

    logger = setup_logging(cfg, "streaming")

    dataset = create_dataset(cfg)
    logger.info(f"Dataset: {len(dataset)} objects")

    run_streaming(cfg, dataset, logger, args)


if __name__ == "__main__":
    main()
