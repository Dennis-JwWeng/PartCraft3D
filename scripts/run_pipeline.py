#!/usr/bin/env python3
"""Batch pipeline: phases 1–6 (semantic → export), one phase at a time, resumable.

Steps: 1 semantic 2 plan 3 2D 4 TRELLIS 3D 5 quality 6 export.
Examples: ``python scripts/run_pipeline.py`` | ``--steps 3 4 5`` | ``--tag v1``
| ``--no-2d-edit`` | ``--steps 4 --2d-cache-only --edit-dir 2d_edits_TAG``
| ``--dry-run``. Per-object mode: ``scripts/run_streaming.py``."""

import argparse
import faulthandler
import json
import os
import subprocess
import shutil
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path

faulthandler.enable(all_threads=True)

# Ensure project root is on sys.path before any project imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.pipeline_common import (
    PROJECT_ROOT, COST,
    load_config, setup_logging,
    PartCraftDataset, EditSpec,
    resolve_api_key, normalize_cache_dirs, set_attn_backend, create_dataset,
)
from scripts.pipeline_diagnostics import (
    diagnose_step1 as _diagnose_step1,
    diagnose_step2 as _diagnose_step2,
    diagnose_step3 as _diagnose_step3,
    diagnose_step4 as _diagnose_step4,
    diagnose_step5 as _diagnose_step5,
    diagnose_step6 as _diagnose_step6,
)
from scripts.pipeline_dispatch import (
    assert_unique_dispatch as _assert_unique_dispatch_impl,
    discover_step1_worker_results,
    discover_step4_worker_results,
    merge_jsonl_by_key,
    reconcile_step4_results,
    reconcile_worker_results,
    split_obj_groups as _split_obj_groups_impl,
    validate_worker_jsonl_outputs,
    wait_for_workers,
)
from scripts.pipeline_jsonl import (
    collect_success_ids,
    dedupe_ids_preserve_order as _dedupe_ids_preserve_order_impl,
    dedupe_specs_by_edit_id as _dedupe_specs_by_edit_id_impl,
    iter_jsonl as _iter_jsonl_impl,
    load_ids_file as _load_ids_file_impl,
    parse_csv_or_space_list as _parse_csv_or_space_list_impl,
    write_records,
)
from scripts.pipeline_paths import (
    normalize_shard as _normalize_shard_impl,
    pipeline_report_dir as _pipeline_report_dir_impl,
    run_token as _run_token_impl,
    sync_manifest_link as _sync_manifest_link_impl,
    write_stage_diag as _write_stage_diag_impl,
)
from scripts.pipeline_orchestrator import (
    build_runtime_context,
    finalize_summary,
    run_selected_steps,
)
from partcraft.edit_types import (
    ADDITION,
    DELETION,
    IDENTITY,
    S1_S2_TYPES,
    S2_ONLY_TYPES,
    TYPE_ORDER,
)


STEP4_GPU_TYPES = set(S1_S2_TYPES) | set(S2_ONLY_TYPES)
STEP4_NON_GPU_TYPES = {ADDITION, DELETION, IDENTITY}
STEP3_IMAGE_TYPES = ("modification", "scale", "material", "global")


def _normalize_shard(shard: str | None) -> str | None:
    return _normalize_shard_impl(shard)


def _run_token(tag: str | None, shard: str | None) -> str:
    return _run_token_impl(tag, shard)


def _iter_jsonl(path: Path):
    yield from _iter_jsonl_impl(path)


def _pipeline_report_dir(cfg: dict, shard: str | None) -> Path:
    return _pipeline_report_dir_impl(cfg, shard)


def _write_stage_diag(report_dir: Path, stage: str, payload: dict):
    _write_stage_diag_impl(report_dir, stage, payload)


def _pipeline_manifest_dir(cfg: dict, shard: str | None, phase: str) -> Path:
    # Kept for backward compatibility in this script.
    from scripts.pipeline_paths import pipeline_manifest_dir

    return pipeline_manifest_dir(cfg, shard, phase)


def _sync_manifest_link(cfg: dict, shard: str | None, phase: str, filename: str, src: Path):
    _sync_manifest_link_impl(cfg, shard, phase, filename, src)


def _parse_csv_or_space_list(values) -> list[str]:
    return _parse_csv_or_space_list_impl(values)


def _load_edit_ids_file(path: str | None) -> list[str]:
    return _load_ids_file_impl(path)


def _load_generic_ids_file(path: str | None) -> list[str]:
    return _load_ids_file_impl(path)


def _split_obj_groups(specs, n_buckets: int) -> list[list]:
    return _split_obj_groups_impl(specs, n_buckets)


def _dedupe_ids_preserve_order(ids: list[str]) -> tuple[list[str], int]:
    return _dedupe_ids_preserve_order_impl(ids)


def _dedupe_specs_by_edit_id(specs: list, logger, stage: str):
    return _dedupe_specs_by_edit_id_impl(specs, logger, stage)


def _assert_unique_dispatch(groups: list[tuple[str, list]], stage: str):
    _assert_unique_dispatch_impl(groups, stage)


# =========================================================================
# Step 1: Semantic Labeling + Enrichment
# =========================================================================

def run_step_semantic(cfg, dataset, logger, limit=None, force=False, tag=None,
                      debug=False, object_ids: list[str] | None = None,
                      labels_name: str | None = None):
    """Step 1: VLM semantic labeling via enricher.

    Uses dataset NPZ files to discover objects and part labels.
    No dependency on source/semantic.json.
    """
    from partcraft.phase1_planning.enricher import enrich_semantic_labels

    logger.info("=" * 60)
    logger.info("STEP 1: Semantic Labeling + Enrichment")
    logger.info("=" * 60)

    cache_dir = Path(cfg["phase0"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    labels_path = cache_dir / (labels_name or f"semantic_labels{tag_suffix}.jsonl")

    image_npz_dir = cfg["data"].get("image_npz_dir")
    shards = cfg["data"].get("shards")
    if not shards:
        raise ValueError(
            "[CONFIG_ERROR] data.shards <missing> config must be set explicitly"
        )
    max_workers = cfg["phase0"].get("max_workers", 4)

    if force and labels_path.exists():
        backup = labels_path.with_suffix(".jsonl.bak")
        labels_path.rename(backup)
        logger.info(f"--force: backed up old labels to {backup}")

    enrich_semantic_labels(
        cfg,
        semantic_json_path=None,
        output_path=str(labels_path),
        image_npz_dir=image_npz_dir,
        shard=shards[0],
        limit=limit or 0,
        max_workers=max_workers,
        visual_grounding=cfg.get("phase0", {}).get("visual_grounding", True),
        dataset=dataset,
        debug=debug,
        object_ids=object_ids,
    )

    logger.info(f"Labels: {labels_path}")
    return labels_path


def run_step_semantic_multi_gpu(
    cfg,
    dataset,
    logger,
    *,
    vlm_urls: list[str],
    gpus: list[str] | None = None,
    tag=None,
    limit=None,
    force=False,
    debug=False,
    step1_workers: int | None = None,
    config_path: str | None = None,
):
    """Dispatch Step1 across multiple VLM endpoints with object-level sharding."""
    logger.info("=" * 60)
    logger.info("STEP 1: Semantic Labeling — Multi-Worker Dispatch")
    logger.info("=" * 60)
    if len(vlm_urls) <= 1:
        raise ValueError("run_step_semantic_multi_gpu requires >=2 VLM URLs")
    if gpus and len(gpus) != len(vlm_urls):
        raise ValueError("--step1-gpus must match --step1-vlm-urls in length")

    cache_dir = Path(cfg["phase0"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    labels_path = cache_dir / f"semantic_labels{tag_suffix}.jsonl"
    if force and labels_path.exists():
        backup = labels_path.with_suffix(".jsonl.bak")
        labels_path.rename(backup)
        logger.info(f"--force: backed up old labels to {backup}")

    if dataset._index is None:
        dataset._build_index()
    shards = cfg["data"].get("shards")
    if not shards:
        raise ValueError(
            "[CONFIG_ERROR] data.shards <missing> config must be set explicitly"
        )
    target_shard = shards[0]
    all_uids = sorted([obj_id for shard, obj_id in dataset._index if shard == target_shard])
    if limit:
        all_uids = all_uids[:limit]
    all_uids, dup_uids = _dedupe_ids_preserve_order(all_uids)
    if dup_uids:
        logger.warning("Step1: dropped %d duplicate object IDs before dispatch", dup_uids)

    # --- Resume precheck: reconcile historical worker shards ---
    historical = discover_step1_worker_results(cache_dir, tag_suffix)
    if historical or labels_path.exists():
        preflight_merged, preflight_stats = reconcile_worker_results(
            output_path=labels_path,
            worker_paths=historical,
            expected_ids=set(all_uids),
            id_key="obj_id",
            strict=False,
            stage="Step1",
        )
        if preflight_merged:
            write_records(labels_path, preflight_merged)
        logger.info(
            "Step1 resume precheck: worker_shards=%d, merged=%d",
            preflight_stats["worker_shards_found"],
            preflight_stats["merged_records"],
        )
        if preflight_stats["bad_json_lines"] or preflight_stats["missing_id_rows"]:
            logger.warning(
                "Step1 resume precheck quality: bad_json=%d, missing_obj_id=%d",
                preflight_stats["bad_json_lines"],
                preflight_stats["missing_id_rows"],
            )

    done_ids: set[str] = set()
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("obj_id"):
                        done_ids.add(rec["obj_id"])
                except Exception:
                    pass
    pending = [u for u in all_uids if u not in done_ids]
    if not pending:
        logger.info("All semantic labels already done")
        return labels_path

    buckets = [[] for _ in vlm_urls]
    for i, uid in enumerate(pending):
        buckets[i % len(vlm_urls)].append(uid)
    plan = [(i, u, buckets[i]) for i, u in enumerate(vlm_urls) if buckets[i]]
    logger.info("Step1 plan: %s", ", ".join(
        f"w{i}@{url}={len(b)} objs" for i, url, b in plan))

    dispatch_dir = cache_dir / f"dispatch{tag_suffix}"
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    worker_label_paths: list[Path] = []
    procs = []
    worker_log_fps = []
    pending_set = set(pending)
    for i, url, bucket in plan:
        ids_path = dispatch_dir / f"step1_obj_ids_w{i}.txt"
        with open(ids_path, "w", encoding="utf-8") as f:
            for uid in bucket:
                f.write(f"{uid}\n")
        labels_name = f"semantic_labels{tag_suffix}_w{i}.jsonl"
        worker_labels_path = cache_dir / labels_name
        if worker_labels_path.exists():
            worker_labels_path.unlink()
        worker_label_paths.append(worker_labels_path)

        cmd = [sys.executable, str(Path(__file__).resolve())]
        if config_path:
            cmd.extend(["--config", config_path])
        cmd.extend(["--steps", "1"])
        cmd.extend(["--step1-obj-ids-file", str(ids_path)])
        cmd.extend(["--step1-labels-name", labels_name])
        cmd.extend(["--step1-vlm-base-url", url])
        if tag:
            cmd.extend(["--tag", tag])
        if debug:
            cmd.append("--debug")
        if step1_workers and step1_workers > 0:
            cmd.extend(["--step1-workers", str(step1_workers)])

        env = os.environ.copy()
        worker_log_path = cache_dir / f"worker_w{i}.log"
        log_fp = open(worker_log_path, "w", encoding="utf-8")
        worker_log_fps.append(log_fp)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[i])
            logger.info("Launch Step1 worker %d on GPU %s → %s (%d objs, log=%s)",
                        i, gpus[i], url, len(bucket), worker_log_path)
        else:
            logger.info("Launch Step1 worker %d → %s (%d objs, log=%s)",
                        i, url, len(bucket), worker_log_path)
        procs.append((i, subprocess.Popen(
            cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)))

    failed = wait_for_workers(procs, "Step1", fail_fast=False)
    for fp in worker_log_fps:
        fp.close()

    # Always merge whatever was produced — even on partial failure
    merged = merge_jsonl_by_key(
        output_path=labels_path,
        worker_paths=worker_label_paths,
        id_key="obj_id",
    )
    write_records(labels_path, merged)

    if failed:
        failed_names = [f"w{name}(rc={rc})" for name, rc in failed]
        logger.warning(
            "Step1 partial failure: %s. Merged %d/%d objects. "
            "Re-run to complete remaining.",
            ", ".join(failed_names), len(merged), len(all_uids),
        )
        for name, rc in failed:
            wlog = cache_dir / f"worker_w{name}.log"
            if wlog.exists():
                logger.warning("  Worker %s log: %s", name, wlog)
        raise RuntimeError(
            f"Step1 workers failed: {failed} "
            f"(partial merge saved: {len(merged)}/{len(all_uids)} objects)"
        )

    # Full success — validate completeness
    missing_pending, dup_pending, unexpected_worker_ids = validate_worker_jsonl_outputs(
        worker_label_paths,
        pending_ids=pending_set,
        id_key="obj_id",
        stage="Step1",
    )
    if unexpected_worker_ids:
        sample = ", ".join(sorted(unexpected_worker_ids)[:10])
        raise RuntimeError(
            "Step1 merge found unexpected obj_ids from worker outputs "
            f"(sample: {sample}, total={len(unexpected_worker_ids)})"
        )
    if missing_pending:
        sample = ", ".join(missing_pending[:10])
        raise RuntimeError(
            "Step1 merge missing worker outputs for pending obj_ids "
            f"(sample: {sample}, total={len(missing_pending)})"
        )
    if dup_pending:
        sample = ", ".join(dup_pending[:10])
        raise RuntimeError(
            "Step1 merge found duplicate obj_id rows across worker outputs "
            f"(sample: {sample}, total={len(dup_pending)})"
        )

    logger.info("Merged Step1 labels → %s (%d objects)", labels_path, len(merged))
    return labels_path


# =========================================================================
# Step 2: Edit Planning
# =========================================================================

def run_step_planning(cfg, labels_path, logger, suffix=""):
    """Step 2: Generate edit specs from Part Catalog."""
    from partcraft.phase0_semantic.catalog import PartCatalog
    from partcraft.phase1_planning.planner import run_phase1

    logger.info("=" * 60)
    logger.info("STEP 2: Edit Planning")
    logger.info("=" * 60)

    catalog = PartCatalog.from_phase0_output(labels_path)
    catalog_path = Path(cfg["phase1"]["cache_dir"]) / f"part_catalog{suffix}.json"
    catalog.save(catalog_path)
    logger.info(f"Catalog: {catalog_path}")

    specs = run_phase1(cfg, catalog, output_suffix=suffix)
    specs_path = Path(cfg["phase1"]["cache_dir"]) / f"edit_specs{suffix}.jsonl"
    logger.info(f"Specs: {specs_path}")
    return specs_path


# =========================================================================
# Step 3: 2D Image Editing (API, parallelizable)
# =========================================================================

def run_step_2d_edit(cfg, specs_path, dataset, logger,
                     tag=None, workers=4, limit=None, edit_types=None,
                     edit_ids=None, edit_server_urls=None,
                     auto_max_parallel: bool = False):
    """Step 3: Pre-generate 2D edited images for all specs."""
    logger.info("=" * 60)
    logger.info("STEP 3: 2D Image Editing")
    logger.info("=" * 60)

    p0 = cfg["phase0"]
    p25 = cfg.get("phase2_5", {})

    image_edit_backend = p25.get("image_edit_backend", "api")

    # Local edit server or API client
    client = None
    edit_server_url = None
    edit_server_urls = edit_server_urls or []
    model = p25.get("image_edit_model", "gemini-2.5-flash-image")

    if image_edit_backend == "local_diffusers":
        from scripts.run_2d_edit import check_edit_server
        cfg_urls = p25.get("image_edit_base_urls", [])
        cfg_urls = cfg_urls if isinstance(cfg_urls, list) else []
        if edit_server_urls:
            urls = edit_server_urls
        elif cfg_urls:
            urls = [str(u).strip() for u in cfg_urls if str(u).strip()]
        else:
            single_url = str(p25.get("image_edit_base_url", "")).strip()
            if not single_url:
                raise ValueError(
                    "[CONFIG_ERROR] phase2_5.image_edit_base_url <missing> config "
                    "must be set explicitly for local_diffusers backend"
                )
            urls = [single_url]

        live_urls = []
        for u in urls:
            if check_edit_server(u):
                live_urls.append(u)
            else:
                logger.warning(f"Image edit server not reachable at {u}")
        if not live_urls:
            raise RuntimeError(
                "[CONFIG_ERROR] phase2_5.image_edit_base_url unreachable runtime "
                "all configured image edit servers are unreachable"
            )
        edit_server_urls = live_urls
        edit_server_url = edit_server_urls[0]
        logger.info(f"Image edit servers OK: {', '.join(edit_server_urls)}")

        if auto_max_parallel:
            per_srv = int(p25.get("image_edit_workers_per_server", 1))
            auto_workers = max(1, len(edit_server_urls) * max(1, per_srv))
            if auto_workers > workers:
                workers = auto_workers
                logger.info(f"--max-parallel: Step3 workers={workers}")
        else:
            cfg_workers = p25.get("image_edit_workers", 1)
            if workers != cfg_workers:
                logger.info(f"local_diffusers backend: workers={cfg_workers} "
                            f"(from config)")
                workers = cfg_workers
        if workers < len(edit_server_urls):
            workers = len(edit_server_urls)
            logger.info(f"Raised workers to {workers} (>= number of servers)")
    else:
        from openai import OpenAI
        api_key = resolve_api_key(cfg)
        if not api_key:
            raise ValueError(
                "[CONFIG_ERROR] phase0.vlm_api_key <missing> config "
                "API backend requires an API key"
            )

        image_edit_url = str(p25.get("image_edit_base_url", "")).strip()
        if not image_edit_url:
            raise ValueError(
                "[CONFIG_ERROR] phase2_5.image_edit_base_url <missing> config "
                "API backend requires explicit image_edit_base_url"
            )
        client = OpenAI(
            base_url=image_edit_url,
            api_key=api_key,
        )

    # Load specs
    # Step3 pre-generates 2D conditioning only for TRELLIS-bound edit types.
    # deletion/addition/identity do not require image editing.
    edit_types = edit_types or list(STEP3_IMAGE_TYPES)
    specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if edit_ids and spec.edit_id not in edit_ids:
                continue
            if spec.edit_type in edit_types:
                specs.append(spec)

    if not edit_ids and limit:
        specs = specs[:limit]
    specs = _dedupe_specs_by_edit_id(specs, logger, "Step3")

    # Output directory
    cache_dir_raw = p25.get("cache_dir")
    if not cache_dir_raw:
        raise ValueError(
            "[CONFIG_ERROR] phase2_5.cache_dir <missing> config must be set explicitly"
        )
    if not tag:
        raise ValueError(
            "[CONFIG_ERROR] run_token <missing> runtime "
            "Step3 requires a non-empty run token (tag or shard token)"
        )
    cache_dir = Path(cache_dir_raw)
    edit_subdir = f"2d_edits_{tag}"
    output_dir = cache_dir / edit_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.run_2d_edit import process_one, select_best_view, prepare_input_image

    manifest_path = output_dir / "manifest.jsonl"

    # Resume
    done_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [s for s in specs if s.edit_id not in done_ids]
    backend_label = ", ".join(edit_server_urls) if edit_server_urls else (edit_server_url or model)
    logger.info(f"2D edits: {len(pending)} pending ({len(done_ids)} done), "
                f"backend={backend_label}, workers={workers}")

    if not pending:
        logger.info("All 2D edits already done")
        return output_dir

    from concurrent.futures import ThreadPoolExecutor, as_completed
    success, fail = 0, 0
    with open(manifest_path, "a") as fp:
        if workers <= 1:
            for i, spec in enumerate(pending):
                logger.info(f"[{i+1}/{len(pending)}] {spec.edit_id}")
                result = process_one(spec, dataset, client, output_dir,
                                     model, logger,
                                     edit_server_url=(
                                         edit_server_urls[i % len(edit_server_urls)]
                                         if edit_server_urls else edit_server_url
                                     ))
                fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                fp.flush()
                if result["status"] == "success":
                    success += 1
                else:
                    fail += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                for spec in pending:
                    assigned_url = (
                        edit_server_urls[len(futures) % len(edit_server_urls)]
                        if edit_server_urls else edit_server_url
                    )
                    fut = pool.submit(process_one, spec, dataset, client,
                                      output_dir, model, logger,
                                      edit_server_url=assigned_url)
                    futures[fut] = spec

                for i, fut in enumerate(as_completed(futures)):
                    spec = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {"edit_id": spec.edit_id, "status": "failed",
                                  "reason": str(e)}
                    fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fp.flush()
                    if result.get("status") == "success":
                        success += 1
                    else:
                        fail += 1
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Progress: {i+1}/{len(pending)} "
                                    f"({success} ok, {fail} fail)")

    logger.info(f"2D edits: {success} ok, {fail} fail → {output_dir}")
    return output_dir


# =========================================================================
# Step 4: 3D Editing — TRELLIS (main workload)
# =========================================================================

def run_step_3d_edit(cfg, specs_path, dataset, logger,
                     tag=None, seed=1, limit=None, use_2d=True,
                     edit_types=None, edit_ids=None, combinations=None,
                     edit_dir=None, debug=False, cache_only_2d: bool = False,
                     results_name: str | None = None):
    from scripts.pipeline_step_3d import run_step_3d_edit as _run_step_3d_edit_impl

    return _run_step_3d_edit_impl(
        cfg,
        specs_path,
        dataset,
        logger,
        tag=tag,
        seed=seed,
        limit=limit,
        use_2d=use_2d,
        edit_types=edit_types,
        edit_ids=edit_ids,
        combinations=combinations,
        edit_dir=edit_dir,
        debug=debug,
        cache_only_2d=cache_only_2d,
        results_name=results_name,
    )


def run_step_3d_edit_multi_gpu(
    cfg,
    specs_path,
    logger,
    *,
    gpus: list[str],
    tag=None,
    seed=1,
    limit=None,
    use_2d=True,
    edit_ids=None,
    edit_dir=None,
    debug=False,
    cache_only_2d: bool = False,
    config_path: str | None = None,
    resume_merge_precheck: bool = True,
    strict_resume_check: bool = False,
):
    """Dispatch Step4 with split routing: GPU-bound and non-GPU edit types."""
    logger.info("=" * 60)
    logger.info("STEP 4: 3D Editing — Multi-GPU Dispatch")
    logger.info("=" * 60)
    if len(gpus) <= 1:
        raise ValueError("run_step_3d_edit_multi_gpu requires >=2 GPUs")
    if not edit_dir:
        raise ValueError(
            "[CONFIG_ERROR] pipeline.edit_dir <missing> runtime "
            "Step4 multi-GPU requires explicit 2D edit subdir"
        )

    # Load specs with same filter semantics as run_step_3d_edit
    accepted = {"deletion", "addition", "modification", "scale",
                "material", "global", "identity"}
    all_specs = []
    with open(specs_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if edit_ids and spec.edit_id not in edit_ids:
                continue
            if spec.edit_type in accepted or spec.edit_type == "addition":
                all_specs.append(spec)
    if not edit_ids and limit:
        all_specs = all_specs[:limit]
    all_specs = _dedupe_specs_by_edit_id(all_specs, logger, "Step4(multi)")
    if not all_specs:
        logger.info("No edit specs to process")
        return None

    p25_cfg = cfg.get("phase2_5", {})
    cache_dir = Path(p25_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    merged_path = cache_dir / f"edit_results{tag_suffix}.jsonl"
    all_spec_ids = {s.edit_id for s in all_specs}

    if resume_merge_precheck:
        historical_worker_paths = discover_step4_worker_results(cache_dir, tag_suffix)
        if historical_worker_paths or merged_path.exists():
            preflight_merged, preflight_stats = reconcile_step4_results(
                output_path=merged_path,
                worker_paths=historical_worker_paths,
                expected_ids=all_spec_ids,
                strict=strict_resume_check,
            )
            if preflight_merged:
                write_records(merged_path, preflight_merged)
            logger.info(
                "Step4 resume precheck: worker_shards=%d, merged_records=%d, "
                "done_success=%d",
                preflight_stats["worker_shards_found"],
                preflight_stats["merged_records"],
                preflight_stats["done_success_ids"],
            )
            if preflight_stats["bad_json_lines"] or preflight_stats["missing_id_rows"]:
                logger.warning(
                    "Step4 resume precheck quality: bad_json_lines=%d, "
                    "missing_edit_id_rows=%d",
                    preflight_stats["bad_json_lines"],
                    preflight_stats["missing_id_rows"],
                )
            if preflight_stats["duplicate_rows_detected"]:
                logger.warning(
                    "Step4 resume precheck duplicates: %d rows (sample: %s)",
                    preflight_stats["duplicate_rows_detected"],
                    ", ".join(preflight_stats["duplicate_ids_sample"]) or "n/a",
                )
            if preflight_stats["unexpected_ids_count"]:
                logger.warning(
                    "Step4 resume precheck unexpected edit_ids: %d (sample: %s)",
                    preflight_stats["unexpected_ids_count"],
                    ", ".join(preflight_stats["unexpected_ids_sample"]) or "n/a",
                )
        else:
            logger.info("Step4 resume precheck: no historical result shards found")

    done_ids = collect_success_ids(merged_path, id_key="edit_id")
    pending = [s for s in all_specs if s.edit_id not in done_ids]
    if not pending:
        logger.info("All 3D edits already done")
        return merged_path

    nongpu_pending = [s for s in pending if s.edit_type in STEP4_NON_GPU_TYPES]
    gpu_pending = [s for s in pending if s.edit_type not in STEP4_NON_GPU_TYPES]
    _assert_unique_dispatch(
        [("gpu_pending", gpu_pending), ("nongpu_pending", nongpu_pending)],
        "Step4 routing",
    )

    logger.info(
        "Step4 routing: GPU=%d, non-GPU=%d",
        len(gpu_pending), len(nongpu_pending),
    )

    gpu_plan = []
    if gpu_pending:
        buckets = _split_obj_groups(gpu_pending, len(gpus))
        gpu_plan = [(gpu, bucket) for gpu, bucket in zip(gpus, buckets) if bucket]
        _assert_unique_dispatch(
            [(f"gpu_{gpu}", bucket) for gpu, bucket in gpu_plan],
            "Step4 GPU plan",
        )
        logger.info("Step4 GPU plan: %s", ", ".join(
            f"GPU {gpu}: {len(bucket)} edits" for gpu, bucket in gpu_plan))
    else:
        logger.info("Step4 GPU plan: no GPU-bound edits")

    dispatch_dir = cache_dir / f"dispatch{tag_suffix}"
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    worker_result_paths: list[Path] = []
    procs = []
    pending_set = {s.edit_id for s in pending}
    full_run = (not edit_ids and not limit)
    for wi, (gpu, bucket) in enumerate(gpu_plan):
        ids_path = dispatch_dir / f"edit_ids_gpu{gpu}.txt"
        with open(ids_path, "w", encoding="utf-8") as f:
            for s in bucket:
                f.write(f"{s.edit_id}\n")
        res_name = f"edit_results{tag_suffix}_gpu{gpu}.jsonl"
        worker_res_path = cache_dir / res_name
        if worker_res_path.exists():
            worker_res_path.unlink()
        worker_result_paths.append(worker_res_path)

        cmd = [sys.executable, str(Path(__file__).resolve())]
        if config_path:
            cmd.extend(["--config", config_path])
        cmd.extend(["--steps", "4"])
        cmd.extend(["--seed", str(seed)])
        cmd.extend(["--edit-dir", edit_dir])
        cmd.extend(["--results-name", res_name])
        cmd.extend(["--edit-ids-file", str(ids_path)])
        if tag:
            cmd.extend(["--tag", tag])
        if not use_2d:
            cmd.append("--no-2d-edit")
        if cache_only_2d:
            cmd.append("--2d-cache-only")
        if debug:
            cmd.append("--debug")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONFAULTHANDLER"] = "1"
        worker_log = cache_dir / f"worker_gpu{gpu}.log"
        worker_log_fp = open(worker_log, "w")
        logger.info("Launch worker %d on GPU %s with %d edits (log: %s)",
                    wi, gpu, len(bucket), worker_log)
        procs.append((gpu, subprocess.Popen(
            cmd, env=env, stdout=worker_log_fp, stderr=subprocess.STDOUT)))

    if nongpu_pending:
        cpu_ids_path = dispatch_dir / "edit_ids_nongpu.txt"
        with open(cpu_ids_path, "w", encoding="utf-8") as f:
            for s in nongpu_pending:
                f.write(f"{s.edit_id}\n")
        cpu_res_name = f"edit_results{tag_suffix}_nongpu.jsonl"
        cpu_res_path = cache_dir / cpu_res_name
        if cpu_res_path.exists():
            cpu_res_path.unlink()
        worker_result_paths.append(cpu_res_path)

        cpu_cmd = [sys.executable, str(Path(__file__).resolve())]
        if config_path:
            cpu_cmd.extend(["--config", config_path])
        cpu_cmd.extend(["--steps", "4"])
        cpu_cmd.extend(["--seed", str(seed)])
        cpu_cmd.extend(["--edit-dir", edit_dir])
        cpu_cmd.extend(["--results-name", cpu_res_name])
        cpu_cmd.extend(["--edit-ids-file", str(cpu_ids_path)])
        cpu_cmd.append("--no-2d-edit")
        if tag:
            cpu_cmd.extend(["--tag", tag])
        if cache_only_2d:
            cpu_cmd.append("--2d-cache-only")
        if debug:
            cpu_cmd.append("--debug")

        cpu_env = os.environ.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Launch non-GPU worker with %d edits", len(nongpu_pending))
        procs.append(("nongpu", subprocess.Popen(cpu_cmd, env=cpu_env)))

    failed = wait_for_workers(procs, "Step4", fail_fast=False)

    # Always merge whatever was produced — even on partial failure
    merged = merge_jsonl_by_key(
        output_path=merged_path,
        worker_paths=worker_result_paths,
        id_key="edit_id",
    )
    write_records(merged_path, merged)

    if failed:
        failed_names = [f"gpu{name}(rc={rc})" for name, rc in failed]
        n_success = sum(1 for r in merged.values() if r.get("status") == "success")
        logger.warning(
            "Step4 partial failure: %s. Merged %d records (%d success). "
            "Re-run to complete remaining.",
            ", ".join(failed_names), len(merged), n_success,
        )
        raise RuntimeError(
            f"Step4 workers failed: {failed} "
            f"(partial merge saved: {len(merged)} records)"
        )

    # Full success — validate completeness
    missing_pending, dup_pending, unexpected_ids = validate_worker_jsonl_outputs(
        worker_result_paths,
        pending_ids=pending_set,
        id_key="edit_id",
        stage="Step4",
    )
    if missing_pending:
        sample = ", ".join(missing_pending[:10])
        raise RuntimeError(
            "Step4 merge missing pending edit_ids from worker outputs "
            f"(sample: {sample}, total={len(missing_pending)})"
        )
    if dup_pending:
        sample = ", ".join(dup_pending[:10])
        raise RuntimeError(
            "Step4 merge found duplicate edit_id rows across worker outputs "
            f"(sample: {sample}, total={len(dup_pending)})"
        )
    if unexpected_ids:
        sample = ", ".join(sorted(unexpected_ids)[:10])
        raise RuntimeError(
            "Step4 merge found unexpected edit_ids in worker outputs "
            f"(sample: {sample}, total={len(unexpected_ids)})"
        )

    merged_ids = set(merged.keys())
    missing_after_merge = sorted([eid for eid in all_spec_ids if eid not in merged_ids])
    if missing_after_merge:
        sample = ", ".join(missing_after_merge[:10])
        raise RuntimeError(
            "Step4 canonical merge missing edit_ids required for downstream steps "
            f"(sample: {sample}, total={len(missing_after_merge)})"
        )
    if full_run:
        unexpected_after_merge = sorted([eid for eid in merged_ids if eid not in all_spec_ids])
        if unexpected_after_merge:
            sample = ", ".join(unexpected_after_merge[:10])
            raise RuntimeError(
                "Step4 canonical merge contains edit_ids not in current specs "
                f"(sample: {sample}, total={len(unexpected_after_merge)})"
            )

    logger.info("Merged multi-GPU results → %s (%d records)",
                merged_path, len(merged))
    return merged_path


# =========================================================================
# Step 5: Quality Scoring
# =========================================================================

_STEP56_DEPRECATED_MSG = (
    "Step 5/6 of run_pipeline.py have been removed. "
    "The post-Step4 flow now lives outside the main pipeline:\n"
    "  1. scripts/tools/repack_to_object_dirs.py     (flat mesh_pairs/ → object-centric partverse_pairs/)\n"
    "  2. scripts/tools/migrate_slat_to_npz.py --phase 1   (mod/scl/mat/glb *_slat/ → NPZ)\n"
    "  3. scripts/tools/run_vlm_cleaning_multi_gpu.sh      (decoupled render + score, --render-only supported)\n"
    "  4. scripts/tools/extract_passed_deletion_ids.py     (vlm_scores.jsonl → passed_deletion_ids.txt)\n"
    "  5. scripts/tools/migrate_slat_to_npz.py --phase 5 --include-list ...   (deletion PLY → 40-view re-encode)\n"
    "  6. scripts/tools/migrate_slat_to_npz.py --phase 3,4                    (addition/identity backfill)\n"
    "See docs/ARCH.md §389-470 for the full flow."
)


def run_step_quality(cfg, results_path, logger, tag=None, limit=None):
    """Removed. Use scripts/tools/run_vlm_cleaning.py — see deprecation message."""
    raise RuntimeError(_STEP56_DEPRECATED_MSG)


# =========================================================================
# Step 6: Export
# =========================================================================

def run_step_export(cfg, specs_path, scores_path, logger, tag=None):
    """Removed. Use repack_to_object_dirs.py + migrate_slat_to_npz.py — see deprecation message."""
    raise RuntimeError(_STEP56_DEPRECATED_MSG)


# =========================================================================
# Step 7: Data Cleaning (object-centric)
# =========================================================================

def run_step_cleaning(cfg, input_dir, logger, tag=None, workers=4):
    """Step 7: NPZ-based data cleaning on repacked object-centric data."""
    logger.info("=" * 60)
    logger.info("STEP 7: Data Cleaning")
    logger.info("=" * 60)

    from partcraft.cleaning.cleaner import run_cleaning

    # Build cleaning config with defaults
    default_cleaning = {
        "min_voxels": 100, "max_voxels": 40000,
        "max_feat_abs": 50.0, "min_feat_std": 0.01,
        "max_ss_abs": 100.0, "min_ss_std": 0.001,
        "deletion": {"min_voxel_ratio": 0.05, "max_voxel_ratio": 0.95,
                      "min_delete_ratio": 0.02, "max_delete_ratio": 0.80,
                      "min_bbox_iou": 0.15, "max_components": 3},
        "addition": {"min_voxel_ratio": 1.05, "max_voxel_ratio": 20.0,
                      "min_add_ratio": 0.02, "max_add_ratio": 0.80},
        "modification": {"min_voxel_ratio": 0.3, "max_voxel_ratio": 3.0,
                          "min_ss_cosine": 0.3, "min_edit_locality": 0.02,
                          "max_edit_locality": 0.70, "max_components": 5},
        "scale": {"min_voxel_ratio": 0.5, "max_voxel_ratio": 2.0,
                   "min_ss_cosine": 0.5, "min_edit_locality": 0.01,
                   "max_edit_locality": 0.50},
        "material": {"require_coords_match": True, "require_ss_match": True,
                      "ss_match_tol": 1e-4, "min_feat_change": 0.01},
        "global": {"require_coords_match": True, "require_ss_match": True,
                    "ss_match_tol": 1e-4, "min_change_coverage": 0.3},
        "tier_thresholds": {"high": 0.8, "medium": 0.6, "low": 0.4},
    }
    yaml_cleaning = cfg.get("cleaning", {})
    merged = dict(default_cleaning)
    for k, v in yaml_cleaning.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    cfg["cleaning"] = merged

    summary_path = run_cleaning(
        input_dir=input_dir,
        cfg=cfg,
        workers=workers,
    )
    logger.info(f"Cleaning summary: {summary_path}")
    return str(summary_path)


def diagnose_step1(labels_path: Path, shard: str | None) -> dict:
    return _diagnose_step1(labels_path, shard)


def diagnose_step2(specs_path: Path) -> dict:
    return _diagnose_step2(specs_path)


def diagnose_step3(manifest_path: Path) -> dict:
    return _diagnose_step3(manifest_path)


def diagnose_step4(results_path: Path, expected_edit_ids: set[str] | None = None) -> dict:
    return _diagnose_step4(results_path, expected_edit_ids=expected_edit_ids)


def diagnose_step5(scores_path: Path) -> dict:
    return _diagnose_step5(scores_path)


def diagnose_step6(export_path: Path) -> dict:
    return _diagnose_step6(export_path)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PartCraft3D Batch Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps 1–7: semantic, plan, 2D, TRELLIS, quality, export, cleaning. "
               "Examples: run_pipeline.py | --steps 4 5 | --dry-run")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="Steps to run (default: all). E.g. --steps 3 4 5")
    parser.add_argument("--shard", type=str, default=None,
                        help="Target shard ID (e.g. 05). Overrides config data.shards.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional legacy run token. Prefer --shard-only naming.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for 2D editing API calls")
    parser.add_argument("--step1-workers", type=int, default=None,
                        help="Override Step1 VLM parallel workers")
    parser.add_argument("--step1-gpus", nargs="*", default=None,
                        help="GPU IDs for parallel Step1 workers")
    parser.add_argument("--step1-vlm-urls", nargs="*", default=None,
                        help="VLM URLs for Step1 sharding (one per worker)")
    parser.add_argument("--step1-vlm-base-url", type=str, default=None,
                        help="Override Step1 VLM URL for current process")
    parser.add_argument("--step1-obj-ids-file", type=str, default=None,
                        help="Internal: object IDs file for Step1 worker")
    parser.add_argument("--step1-labels-name", type=str, default=None,
                        help="Internal: output labels filename for Step1 worker")
    parser.add_argument("--no-2d-edit", dest="use_2d", action="store_false",
                        default=True)
    parser.add_argument("--2d-cache-only", dest="cache_only_2d",
                        action="store_true", default=False,
                        help="Step 4: 2D from cache only (no image API)")
    parser.add_argument("--edit-dir", type=str, default=None,
                        help="Pre-generated 2D edits subdir")
    parser.add_argument("--edit-ids", nargs="*", default=None,
                        help="Specific edit IDs to process")
    parser.add_argument("--edit-ids-file", type=str, default=None,
                        help="File with edit IDs (one per line)")
    parser.add_argument("--image-edit-urls", nargs="*", default=None,
                        help="Local image edit server URLs (supports comma-separated)")
    parser.add_argument("--gpus", nargs="*", default=None,
                        help="GPU IDs for parallel Step4, e.g. --gpus 0 1 2 3")
    parser.add_argument("--max-parallel", action="store_true",
                        help="Auto maximize per-step parallelism")
    parser.add_argument("--results-name", type=str, default=None,
                        help="Internal: override Step4 result filename")
    parser.add_argument("--resume-merge-precheck", dest="resume_merge_precheck",
                        action="store_true", default=True,
                        help="Step4 multi-GPU: reconcile historical worker shards before dispatch")
    parser.add_argument("--no-resume-merge-precheck", dest="resume_merge_precheck",
                        action="store_false",
                        help="Disable Step4 historical shard reconcile before dispatch")
    parser.add_argument("--strict-resume-check", action="store_true",
                        help="Step4 precheck: fail on malformed or unexpected historical records")
    parser.add_argument("--dry-run", action="store_true",
                        help="Cost estimation only, no actual processing")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for spec files (e.g. '_action')")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug output files (masks, views, enricher "
                             "ortho images). Off by default.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run (delete and regenerate cached results)")
    parser.add_argument("--cleaning-input-dir", type=str, default=None,
                        help="Step 7: root of repacked object-centric data "
                             "(partverse_pairs dir with shard_XX subdirs)")
    parser.add_argument("--cleaning-workers", type=int, default=4,
                        help="Step 7: parallel workers per shard")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_shard = _normalize_shard(
        args.shard or ((cfg.get("data", {}).get("shards") or [None])[0])
    )
    if not target_shard:
        raise ValueError(
            "[CONFIG_ERROR] data.shards <missing> config "
            "batch pipeline requires explicit shard (config or --shard)"
        )
    cfg.setdefault("data", {})["shards"] = [target_shard]

    normalize_cache_dirs(cfg)
    set_attn_backend(cfg)
    if cfg.get("ckpt_root"):
        os.environ.setdefault("PARTCRAFT_CKPT_ROOT", cfg["ckpt_root"])

    if args.step1_workers is not None and args.step1_workers > 0:
        cfg.setdefault("phase0", {})["max_workers"] = int(args.step1_workers)
    if args.step1_vlm_base_url:
        cfg.setdefault("phase0", {})["vlm_base_url"] = args.step1_vlm_base_url
        cfg.setdefault("phase0", {})["local_base_url"] = args.step1_vlm_base_url

    logger = setup_logging(cfg, "pipeline")
    run_token = _run_token(args.tag, target_shard)
    report_dir = _pipeline_report_dir(cfg, target_shard)
    logger.info("Run token: %s (shard=%s)", run_token, target_shard or "N/A")
    logger.info("Diagnostics: %s", report_dir)

    if args.max_parallel and args.workers == 4:
        args.workers = max(8, (os.cpu_count() or 8))
    if args.max_parallel and (args.step1_workers is None):
        cpu_n = os.cpu_count() or 8
        cfg.setdefault("phase0", {})["max_workers"] = max(
            int(cfg.get("phase0", {}).get("max_workers", 4)),
            min(64, cpu_n * 2)
        )

    dataset = create_dataset(cfg)
    logger.info("Dataset: %d objects", len(dataset))

    ctx = build_runtime_context(
        args,
        cfg,
        target_shard=target_shard,
        run_token=run_token,
        report_dir=report_dir,
    )

    if args.dry_run:
        estimate = {
            "dry_run": True,
            "run_token": run_token,
            "shard": target_shard,
            "steps_requested": sorted(ctx["steps"]),
            "cost_table": COST,
            "paths": {
                "labels": str(ctx["labels_path"]),
                "specs": str(ctx["specs_path"]),
            },
        }
        _write_stage_diag(report_dir, "pipeline_summary", estimate)
        logger.info("Dry run only; no processing steps executed.")
        return

    callbacks = {
        "run_token": run_token,
        "parse_list": _parse_csv_or_space_list,
        "load_generic_ids_file": _load_generic_ids_file,
        "load_edit_ids_file": _load_edit_ids_file,
        "iter_jsonl": _iter_jsonl,
        "write_stage_diag": _write_stage_diag,
        "sync_manifest_link": _sync_manifest_link,
        "diagnose_step1": diagnose_step1,
        "diagnose_step2": diagnose_step2,
        "diagnose_step3": diagnose_step3,
        "diagnose_step4": diagnose_step4,
        "diagnose_step5": diagnose_step5,
        "diagnose_step6": diagnose_step6,
        "run_step_semantic": run_step_semantic,
        "run_step_semantic_multi_gpu": run_step_semantic_multi_gpu,
        "run_step_planning": run_step_planning,
        "run_step_2d_edit": run_step_2d_edit,
        "run_step_3d_edit": run_step_3d_edit,
        "run_step_3d_edit_multi_gpu": run_step_3d_edit_multi_gpu,
        "run_step_quality": run_step_quality,
        "run_step_export": run_step_export,
        "run_step_cleaning": run_step_cleaning,
    }
    try:
        ctx = run_selected_steps(
            ctx,
            args=args,
            cfg=cfg,
            logger=logger,
            dataset=dataset,
            callbacks=callbacks,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error("Run prerequisite steps first, or provide cached artifacts")
        sys.exit(1)

    summary = finalize_summary(ctx, run_token=run_token)
    _write_stage_diag(report_dir, "pipeline_summary", summary)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Shard-centric diagnostics -> %s", report_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
