#!/usr/bin/env python3
"""Run the full PartCraft3D pipeline end-to-end.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --config configs/custom.yaml --limit 100
    python scripts/run_all.py --phases 1 2 4       # skip phase 0, 2.5, and 3
    python scripts/run_all.py --phases 0 1 2 2.5   # include TRELLIS generative editing

Note: Phase 2.5 (TRELLIS) requires ATTN_BACKEND=xformers and GPU.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging
from partcraft.io.hy3d_loader import HY3DPartDataset
from partcraft.io.export import EditPairWriter, EditPairRecord
from partcraft.phase0_semantic.labeler import run_phase0
from partcraft.phase0_semantic.catalog import PartCatalog
from partcraft.phase1_planning.planner import run_phase1, EditSpec
from partcraft.phase2_assembly.assembler import run_phase2
from partcraft.phase4_filter.instruction import generate_instructions


def run_phase2_5(cfg, dataset, logger):
    """Phase 2.5: TRELLIS generative editing for style modification specs."""
    p25_cfg = cfg.get("phase2_5", {})

    if not p25_cfg.get("enabled", False):
        logger.info("Phase 2.5 disabled in config (set phase2_5.enabled: true to enable)")
        return

    vinedresser_path = p25_cfg.get(
        "vinedresser_path", "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main")
    sys.path.insert(0, vinedresser_path)

    from partcraft.phase2_assembly.trellis_refine import TrellisRefiner
    from partcraft.phase1_planning.planner import EditSpec

    # Load modification specs
    specs_path = Path(cfg["phase1"]["cache_dir"]) / "edit_specs.jsonl"
    mod_specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if spec.edit_type == "modification" and spec.edit_id.startswith("mod_"):
                mod_specs.append(spec)

    limit = p25_cfg.get("max_refine", 1000)
    mod_specs = mod_specs[:limit]

    if not mod_specs:
        logger.info("No modification specs found for Phase 2.5")
        return

    output_dir = Path(cfg["data"]["output_dir"])
    cache_dir = Path(p25_cfg.get("cache_dir", str(output_dir / "cache/phase2_5")))
    mesh_pairs_dir = output_dir / "mesh_pairs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / "modification_pairs.jsonl"

    # Resume support
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [s for s in mod_specs if s.edit_id not in done_ids]
    logger.info(f"Phase 2.5: {len(pending)} modifications to process "
                f"({len(done_ids)} already done)")

    if not pending:
        return

    seed = p25_cfg.get("seed", 42)
    num_combs = p25_cfg.get("num_combinations", 1)
    output_fmt = p25_cfg.get("output_format", "ply")

    refiner = TrellisRefiner(
        device="cuda",
        cache_dir=str(cache_dir),
        vinedresser_path=vinedresser_path,
        trellis_text_ckpt=p25_cfg.get("trellis_text_ckpt"),
        trellis_image_ckpt=p25_cfg.get("trellis_image_ckpt"),
    )
    refiner.load_models()

    success, fail = 0, 0
    with open(output_path, "a") as out_fp:
        for i, spec in enumerate(pending):
            logger.info(f"[{i+1}/{len(pending)}] {spec.edit_id}")
            try:
                obj = dataset.load_object(spec.shard, spec.obj_id)
                result = refiner.refine_modification(
                    obj_record=obj,
                    edit_part_ids=[spec.old_part_id],
                    edit_prompt=spec.edit_prompt,
                    after_desc=spec.after_desc,
                    old_part_label=spec.old_label,
                    before_part_desc=spec.before_part_desc,
                    after_part_desc=spec.after_part_desc,
                    obj_desc=spec.object_desc,
                    seed=seed,
                    num_combinations=num_combs,
                )
                if result is not None:
                    pair_dir = mesh_pairs_dir / spec.edit_id
                    mesh_paths = refiner.export_mesh_pair(result, pair_dir, fmt=output_fmt)
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id,
                        "edit_type": "modification",
                        "obj_id": spec.obj_id,
                        "shard": spec.shard,
                        "before_mesh": mesh_paths["before_mesh"],
                        "after_mesh": mesh_paths["after_mesh"],
                        "status": "success",
                    }, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                else:
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                    }) + "\n")
                    out_fp.flush()
                    fail += 1
                obj.close()
            except Exception as e:
                logger.error(f"Failed {spec.edit_id}: {e}")
                fail += 1

    logger.info(f"Phase 2.5: {success} succeeded, {fail} failed -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PartCraft3D Full Pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max objects for Phase 0")
    parser.add_argument("--phases", nargs="+", type=float, default=[0, 1, 2, 2.5, 3, 4],
                        help="Which phases to run (default: all, use 2.5 for TRELLIS)")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip Phase 3 rendering (produce meshes only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "pipeline")
    phases = set(args.phases)
    has_phase = lambda p: p in phases

    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )
    logger.info(f"Dataset: {len(dataset)} objects, shards={cfg['data']['shards']}")

    # ======================== Phase 0: Semantic Labeling ========================
    labels_path = Path(cfg["phase0"]["cache_dir"]) / "semantic_labels.jsonl"
    if has_phase(0) or has_phase(0.0):
        logger.info("=" * 60)
        logger.info("PHASE 0: VLM Semantic Labeling")
        logger.info("=" * 60)
        labels_path = run_phase0(cfg, dataset, limit=args.limit)
    else:
        logger.info("Phase 0 skipped (using cached labels)")

    if not labels_path.exists():
        logger.error(f"Phase 0 output not found: {labels_path}")
        logger.error("Run Phase 0 first, or provide cached semantic_labels.jsonl")
        sys.exit(1)

    # ======================== Phase 1: Edit Planning ========================
    catalog_path = Path(cfg["phase1"]["cache_dir"]) / "part_catalog.json"
    specs_path = Path(cfg["phase1"]["cache_dir"]) / "edit_specs.jsonl"

    if has_phase(1) or has_phase(1.0):
        logger.info("=" * 60)
        logger.info("PHASE 1: Edit Planning")
        logger.info("=" * 60)
        catalog = PartCatalog.from_phase0_output(labels_path)
        catalog.save(catalog_path)
        specs = run_phase1(cfg, catalog)
    else:
        logger.info("Phase 1 skipped (using cached specs)")
        specs = None

    # ======================== Phase 2: Mesh Assembly ========================
    # Handles deletion, addition, and swap edits via direct mesh manipulation
    if has_phase(2) or has_phase(2.0):
        logger.info("=" * 60)
        logger.info("PHASE 2: Mesh Assembly (deletion/addition/swap)")
        logger.info("=" * 60)
        run_phase2(cfg, specs, dataset)
    else:
        logger.info("Phase 2 skipped")

    # ======================== Phase 2.5: TRELLIS Generative Editing ========================
    # Handles style modification edits via TRELLIS Flow Inversion + Repaint
    if has_phase(2.5):
        logger.info("=" * 60)
        logger.info("PHASE 2.5: TRELLIS Generative Editing (style modification)")
        logger.info("=" * 60)
        run_phase2_5(cfg, dataset, logger)
    else:
        logger.info("Phase 2.5 skipped")

    # ======================== Phase 3: Quality Filter ========================
    # Reads from Phase 2 manifest + Phase 2.5 manifest
    manifest_path = Path(cfg["phase2"]["cache_dir"]) / "assembled_pairs.jsonl"

    if has_phase(3) or has_phase(3.0):
        logger.info("=" * 60)
        logger.info("PHASE 3: Quality Scoring & Filtering")
        logger.info("=" * 60)
        if manifest_path.exists():
            from partcraft.phase3_filter.filter import run_phase3 as run_quality_filter
            passed_entries, failed_entries = run_quality_filter(cfg, str(manifest_path))
            logger.info(f"Phase 3: {len(passed_entries)} passed, {len(failed_entries)} failed")
        else:
            logger.warning(f"Phase 2 manifest not found: {manifest_path}")
            passed_entries = []
    else:
        logger.info("Phase 3 skipped")
        passed_entries = []

    # ======================== Phase 4: Instruction Generation + Export ========================
    if has_phase(4) or has_phase(4.0):
        logger.info("=" * 60)
        logger.info("PHASE 4: Instruction Generation + Export")
        logger.info("=" * 60)

        # Read passed pairs from Phase 3 output
        passed_path = Path(cfg.get("phase3", {}).get("cache_dir", "")) / "passed_pairs.jsonl"
        if not passed_entries and passed_path.exists():
            passed_entries = []
            with open(passed_path) as f:
                for line in f:
                    if line.strip():
                        passed_entries.append(json.loads(line))

        if not passed_entries:
            logger.warning("No passed pairs found for Phase 4")
        else:
            output_dir = Path(cfg["data"]["output_dir"])
            n_variants = cfg.get("phase3", {}).get("instructions_per_edit", 3)

            # Load edit specs for instruction generation
            specs_map: dict[str, EditSpec] = {}
            specs_path = Path(cfg["phase1"]["cache_dir"]) / "edit_specs.jsonl"
            if specs_path.exists():
                with open(specs_path) as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            specs_map[d["edit_id"]] = EditSpec(**d)

            exported = 0
            with EditPairWriter(output_dir) as writer:
                for entry in passed_entries:
                    eid = entry["edit_id"]
                    spec = specs_map.get(eid)
                    if spec is None:
                        continue

                    instructions = generate_instructions(spec, n_variants)
                    if not instructions:
                        continue

                    record = EditPairRecord(
                        edit_id=eid,
                        edit_type=spec.edit_type,
                        instruction=instructions[0],
                        instruction_variants=instructions[1:],
                        source_obj_id=spec.obj_id,
                        source_shard=spec.shard,
                        donor_obj_id=spec.new_obj_id or None,
                        removed_part_ids=spec.remove_part_ids,
                        added_part_ids=spec.add_part_ids,
                        old_part_label=spec.old_label,
                        new_part_label=spec.new_label,
                        object_desc=spec.object_desc,
                        quality_score=entry.get("quality_score", 0.0),
                        quality_checks=entry.get("quality_metrics", {}),
                    )
                    before_mesh, after_mesh = None, None
                    try:
                        import trimesh as _tm
                        before_mesh = _tm.load(entry["before_mesh"], process=False)
                        after_mesh = _tm.load(entry["after_mesh"], process=False)
                    except Exception:
                        pass

                    writer.write_pair(record, before_mesh=before_mesh, after_mesh=after_mesh)
                    exported += 1

            logger.info(f"Phase 4: {exported} pairs exported")
            logger.info(f"Output: {output_dir / 'edit_pairs.jsonl'}")
    else:
        logger.info("Phase 4 skipped")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
