#!/usr/bin/env python3
"""Batch pipeline: phases 1–6 (semantic → export), one phase at a time, resumable.

Steps: 1 semantic 2 plan 3 2D 4 TRELLIS 3D 5 quality 6 export.
Examples: ``python scripts/run_pipeline.py`` | ``--steps 3 4 5`` | ``--tag v1``
| ``--no-2d-edit`` | ``--steps 4 --2d-cache-only --edit-dir 2d_edits_TAG``
| ``--dry-run``. Per-object mode: ``scripts/run_streaming.py``."""

import argparse
import json
import os
import subprocess
import shutil
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path

# Ensure project root is on sys.path before any project imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image

from scripts.pipeline_common import (
    PROJECT_ROOT, COST,
    load_config, setup_logging,
    PartCraftDataset, EditSpec,
    resolve_api_key, normalize_cache_dirs, set_attn_backend, create_dataset,
    resolve_data_dirs,
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


def _normalize_shard(shard: str | None) -> str | None:
    if shard is None:
        return None
    s = str(shard).strip()
    if not s:
        return None
    return s.zfill(2)


def _run_token(tag: str | None, shard: str | None) -> str:
    """Build output token. Prefer explicit tag, fallback to shard token."""
    if tag:
        return str(tag)
    if shard:
        return f"shard{shard}"
    return ""


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _pipeline_report_dir(cfg: dict, shard: str | None) -> Path:
    out = Path(cfg["data"]["output_dir"])
    shard_leaf = f"shard_{shard}" if shard else "shard_unknown"
    shard_dir = out if out.name == shard_leaf else (out / shard_leaf)
    report_dir = shard_dir / "pipeline" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _write_stage_diag(report_dir: Path, stage: str, payload: dict):
    with open(report_dir / f"{stage}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pipeline_manifest_dir(cfg: dict, shard: str | None, phase: str) -> Path:
    out = Path(cfg["data"]["output_dir"])
    shard_leaf = f"shard_{shard}" if shard else "shard_unknown"
    shard_dir = out if out.name == shard_leaf else (out / shard_leaf)
    mdir = shard_dir / "pipeline" / "manifests" / phase
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir


def _sync_manifest_link(cfg: dict, shard: str | None, phase: str, filename: str, src: Path):
    if not src or not src.exists():
        return
    mdir = _pipeline_manifest_dir(cfg, shard, phase)
    dst = mdir / filename
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink(missing_ok=True)
    dst.symlink_to(src)


def _parse_csv_or_space_list(values) -> list[str]:
    """Parse repeated args like ['0,1', '2'] into ['0', '1', '2']."""
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        for x in str(v).split(","):
            x = x.strip()
            if x:
                out.append(x)
    return out


def _load_edit_ids_file(path: str | None) -> list[str]:
    """Load edit IDs from text file (one id per line)."""
    if not path:
        return []
    ids: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def _load_generic_ids_file(path: str | None) -> list[str]:
    """Load generic IDs from file (one per line)."""
    return _load_edit_ids_file(path)


def _split_obj_groups(specs, n_buckets: int) -> list[list]:
    """Greedy split by object groups to balance multi-GPU workloads."""
    buckets = [[] for _ in range(n_buckets)]
    bucket_sizes = [0] * n_buckets
    obj_groups: OrderedDict[str, list] = OrderedDict()
    for s in specs:
        obj_groups.setdefault(s.obj_id, []).append(s)
    groups = sorted(obj_groups.values(), key=lambda g: len(g), reverse=True)
    for g in groups:
        idx = min(range(n_buckets), key=lambda i: bucket_sizes[i])
        buckets[idx].extend(g)
        bucket_sizes[idx] += len(g)
    return buckets


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
    shards = cfg["data"].get("shards", ["00"])
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
        shard=shards[0] if shards else "00",
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
    shards = cfg["data"].get("shards", ["00"])
    target_shard = shards[0] if shards else "00"
    all_uids = sorted([obj_id for shard, obj_id in dataset._index if shard == target_shard])
    if limit:
        all_uids = all_uids[:limit]

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
    for i, url, bucket in plan:
        ids_path = dispatch_dir / f"step1_obj_ids_w{i}.txt"
        with open(ids_path, "w", encoding="utf-8") as f:
            for uid in bucket:
                f.write(f"{uid}\n")
        labels_name = f"semantic_labels{tag_suffix}_w{i}.jsonl"
        worker_label_paths.append(cache_dir / labels_name)

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
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[i])
            logger.info("Launch Step1 worker %d on GPU %s → %s (%d objs)",
                        i, gpus[i], url, len(bucket))
        else:
            logger.info("Launch Step1 worker %d → %s (%d objs)",
                        i, url, len(bucket))
        procs.append((i, subprocess.Popen(cmd, env=env)))

    failed = []
    for i, p in procs:
        ret = p.wait()
        if ret != 0:
            failed.append((i, ret))
    if failed:
        raise RuntimeError(f"Step1 workers failed: {failed}")

    merged: OrderedDict[str, dict] = OrderedDict()
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    oid = rec.get("obj_id")
                    if oid:
                        merged[oid] = rec
                except Exception:
                    pass
    for lp in worker_label_paths:
        if not lp.exists():
            continue
        with open(lp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    oid = rec.get("obj_id")
                    if oid:
                        merged[oid] = rec
                except Exception:
                    pass
    with open(labels_path, "w", encoding="utf-8") as f:
        for rec in merged.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
            urls = [p25.get("image_edit_base_url", "http://localhost:8001")]

        live_urls = []
        for u in urls:
            if check_edit_server(u):
                live_urls.append(u)
            else:
                logger.warning(f"Image edit server not reachable at {u}")
        if not live_urls:
            logger.warning("No image edit server reachable, skipping 2D editing")
            return None
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
            logger.warning("No API key — skipping 2D editing")
            return None

        image_edit_url = p25.get("image_edit_base_url") or p0.get("vlm_base_url", "")
        client = OpenAI(
            base_url=image_edit_url,
            api_key=api_key,
        )

    # Load specs
    edit_types = edit_types or ["modification", "deletion", "global"]
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

    # Output directory
    cache_dir = Path(p25.get("cache_dir", "outputs/cache/phase2_5"))
    edit_subdir = f"2d_edits_{tag}" if tag else "2d_edits"
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
    """TRELLIS Step 4. With ``use_2d``: load ``{edit_dir}/{edit_id}_edited.png``
    first; if ``cache_only_2d``, skip image HTTP/VLM (missing PNG → no cond)."""
    logger.info("=" * 60)
    logger.info("STEP 4: 3D Editing — TRELLIS")
    logger.info("=" * 60)

    p25_cfg = cfg.get("phase2_5", {})

    from partcraft.phase2_assembly.trellis_refine import (
        TrellisRefiner, build_prompts_from_spec)

    # ---- Paths ----
    output_dir = Path(cfg["data"]["output_dir"])
    cache_dir = Path(p25_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    # ---- Load specs (match streaming / PartVerse) ----
    edit_types = edit_types or [
        "deletion", "addition", "modification", "scale", "material",
        "global", "identity",
    ]
    all_specs = []
    with open(specs_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            spec = EditSpec(**d)
            if edit_ids and spec.edit_id not in edit_ids:
                continue
            if spec.edit_type in edit_types or spec.edit_type == "addition":
                all_specs.append(spec)

    if not edit_ids and limit:
        all_specs = all_specs[:limit]

    if not all_specs:
        logger.info("No edit specs to process")
        return None

    # ---- Resume ----
    output_path = (cache_dir / results_name) if results_name else (
        cache_dir / f"edit_results{tag_suffix}.jsonl")
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

    pending = [s for s in all_specs if s.edit_id not in done_ids]
    logger.info(f"3D edits: {len(pending)} pending ({len(done_ids)} done)")
    if not pending:
        logger.info("All 3D edits already done")
        return output_path

    # ---- VLM client for inline 2D editing ----
    image_edit_backend = p25_cfg.get("image_edit_backend", "api")
    vlm_client = None
    if use_2d and image_edit_backend != "local_diffusers":
        from openai import OpenAI
        api_key = resolve_api_key(cfg)
        if api_key:
            p0 = cfg["phase0"]
            image_edit_url = (p25_cfg.get("image_edit_base_url")
                              or p0.get("vlm_base_url", ""))
            vlm_client = OpenAI(
                base_url=image_edit_url,
                api_key=api_key,
            )

    # ---- Lazy init refiner ----
    refiner = None

    def ensure_refiner():
        nonlocal refiner
        if refiner is not None:
            return refiner
        slat_dir, img_enc_dir = resolve_data_dirs(cfg)
        refiner = TrellisRefiner(
            cache_dir=str(cache_dir),
            device="cuda",
            image_edit_model=p25_cfg.get("image_edit_model", "gemini-2.5-flash-image"),
            ckpt_dir=cfg.get("ckpt_root"),
            image_edit_backend=image_edit_backend,
            image_edit_base_url=p25_cfg.get("image_edit_base_url", "http://localhost:8001"),
            debug=debug,
            slat_dir=slat_dir,
            img_enc_dir=img_enc_dir,
        )
        refiner.load_models()
        logger.info("TRELLIS models initialized for GPU edit types")
        return refiner

    # ---- Group by object ----
    obj_groups: OrderedDict[str, list] = OrderedDict()
    for spec in pending:
        obj_groups.setdefault(spec.obj_id, []).append(spec)

    success, fail = 0, 0
    glb_tmp_dir = tempfile.mkdtemp(prefix="partcraft_glb_")

    with open(output_path, "a") as out_fp:
        for obj_id, specs in obj_groups.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Object: {obj_id} ({len(specs)} edits)")

            def _spec_order(s: EditSpec):
                t = s.edit_type
                if t == "deletion":
                    return (0, 0)
                if t == "addition":
                    return (1, 0)
                return (2, TYPE_ORDER.get(t, 99))

            specs.sort(key=_spec_order)

            # ---- Handle additions (swap from deletion) ----
            add_specs = [s for s in specs if s.edit_type == "addition"]
            run_specs = [s for s in specs if s.edit_type != "addition"]

            for spec in add_specs:
                del_pair_dir = mesh_pairs_dir / spec.source_del_id
                add_pair_dir = mesh_pairs_dir / spec.edit_id
                if (del_pair_dir / "before_slat").exists():
                    add_pair_dir.mkdir(parents=True, exist_ok=True)
                    for src, dst in [
                        (del_pair_dir / "before_slat", add_pair_dir / "after_slat"),
                        (del_pair_dir / "after_slat", add_pair_dir / "before_slat"),
                        (del_pair_dir / "before.ply", add_pair_dir / "after.ply"),
                        (del_pair_dir / "after.ply", add_pair_dir / "before.ply"),
                    ]:
                        if src.exists() and not dst.exists():
                            if src.is_dir():
                                shutil.copytree(str(src), str(dst))
                            else:
                                shutil.copy2(str(src), str(dst))
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "edit_type": "addition",
                        "obj_id": obj_id, "status": "success",
                        "source_del_id": spec.source_del_id,
                    }, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                else:
                    run_specs.append(spec)

            idt_specs = [s for s in run_specs if s.edit_type == IDENTITY]
            run_specs = [s for s in run_specs if s.edit_type != IDENTITY]
            first_pair_dir = None
            cpu_specs = [s for s in run_specs if s.edit_type == DELETION]
            gpu_specs = [s for s in run_specs if s.edit_type in STEP4_GPU_TYPES]
            unknown_specs = [
                s for s in run_specs
                if s.edit_type not in (STEP4_NON_GPU_TYPES | STEP4_GPU_TYPES)
            ]
            for spec in unknown_specs:
                logger.warning(f"Unknown edit type: {spec.edit_type}")
                out_fp.write(json.dumps({
                    "edit_id": spec.edit_id, "status": "failed",
                    "reason": f"Unknown edit type: {spec.edit_type}",
                }, ensure_ascii=False) + "\n")
                fail += 1
            if unknown_specs:
                out_fp.flush()

            if not cpu_specs and not gpu_specs:
                # Identity-only object: still need first_pair_dir from prior
                # edits on same object (not supported here).
                for spec in idt_specs:
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": "identity needs prior edits on object",
                    }, ensure_ascii=False) + "\n")
                    fail += 1
                out_fp.flush()
                continue

            obj_record = None
            ori_slat = None
            ori_gaussian = None
            if cpu_specs or gpu_specs:
                try:
                    shard = (cpu_specs[0].shard if cpu_specs
                             else gpu_specs[0].shard)
                    obj_record = dataset.load_object(shard, obj_id)
                except Exception as e:
                    logger.error(f"Failed to load object {obj_id}: {e}")
                    for spec in cpu_specs + gpu_specs:
                        out_fp.write(json.dumps({
                            "edit_id": spec.edit_id, "status": "failed",
                            "reason": f"Object load failed: {e}",
                        }, ensure_ascii=False) + "\n")
                    fail += len(cpu_specs) + len(gpu_specs)
                    out_fp.flush()
                    continue

            for spec in cpu_specs:
                logger.info(f"\n[{success+fail+1}/{len(pending)}] "
                            f"{spec.edit_id} ({spec.edit_type}): "
                            f"\"{spec.edit_prompt[:80]}\"")
                try:
                    pair_dir = mesh_pairs_dir / spec.edit_id
                    export_paths = TrellisRefiner.direct_delete_mesh(
                        obj_record, spec.remove_part_ids, pair_dir)
                    record = {
                        "edit_id": spec.edit_id,
                        "edit_type": spec.edit_type,
                        "effective_edit_type": "DirectDeletion",
                        "obj_id": obj_id,
                        "shard": spec.shard,
                        "object_desc": spec.object_desc,
                        "edit_prompt": spec.edit_prompt,
                        "after_desc": spec.after_desc,
                        **export_paths,
                        "status": "success",
                    }
                    out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                    if first_pair_dir is None:
                        first_pair_dir = pair_dir
                    logger.info(f"  → direct deletion (GT mesh) saved {pair_dir}")
                except Exception as e:
                    import traceback
                    logger.error(f"Failed {spec.edit_id}: {e}")
                    traceback.print_exc()
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": str(e),
                    }, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    fail += 1

            if gpu_specs:
                try:
                    refiner = ensure_refiner()
                    ori_slat = refiner.encode_object(None, obj_id)
                    ori_gaussian = refiner.decode_to_gaussian(ori_slat)
                except Exception as e:
                    logger.error(f"Failed to prepare TRELLIS state for {obj_id}: {e}")
                    for spec in gpu_specs:
                        out_fp.write(json.dumps({
                            "edit_id": spec.edit_id, "status": "failed",
                            "reason": f"Preparation failed: {e}",
                        }, ensure_ascii=False) + "\n")
                    fail += len(gpu_specs)
                    out_fp.flush()
                    gpu_specs = []

            for spec in gpu_specs:
                logger.info(f"\n[{success+fail+1}/{len(pending)}] "
                            f"{spec.edit_id} ({spec.edit_type}): "
                            f"\"{spec.edit_prompt[:80]}\"")

                try:
                    edit_type = spec.edit_type.capitalize()
                    if edit_type in ("Modification", "Scale"):
                        if spec.remove_part_ids:
                            edit_part_ids = spec.remove_part_ids
                        else:
                            edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Material":
                        edit_part_ids = [spec.old_part_id]
                    elif edit_type == "Global":
                        edit_part_ids = []
                    else:
                        logger.warning(f"Skipped non-GPU type in GPU queue: {spec.edit_type}")
                        continue

                    # Build mask (may auto-promote to Global)
                    mask, effective_type = refiner.build_part_mask(
                        obj_id, obj_record, edit_part_ids,
                        ori_slat, edit_type)

                    if effective_type != edit_type:
                        logger.info(f"  Auto-promoted {edit_type} → "
                                    f"{effective_type} (large part)")
                        edit_type = effective_type

                    if mask.sum() == 0:
                        logger.warning(f"Empty mask for {spec.edit_id}")
                        out_fp.write(json.dumps({
                            "edit_id": spec.edit_id, "status": "failed",
                            "reason": "Empty mask",
                        }) + "\n")
                        out_fp.flush()
                        fail += 1
                        continue

                    # Build prompts
                    prompts = build_prompts_from_spec(spec)
                    if prompts["edit_type"] != edit_type:
                        prompts["edit_type"] = edit_type

                    # 2D image conditioning (disk first — same paths as Step 3 / streaming)
                    img_cond = None
                    if edit_type in ("Modification", "Scale", "Material",
                                     "Global") and use_2d:
                        num_edit_views = p25_cfg.get("num_edit_views", 4)
                        edit_strength = p25_cfg.get("edit_strength", 1.0)

                        prerender_img = None
                        _2d_base = cache_dir / (edit_dir or "2d_edits")
                        _ced = _2d_base / f"{spec.edit_id}_edited.png"
                        if _ced.exists():
                            try:
                                from scripts.run_2d_edit import (
                                    prepare_input_image)
                                edited = Image.open(_ced).convert("RGB").resize(
                                    (518, 518))
                                _cin = _2d_base / f"{spec.edit_id}_input.png"
                                if _cin.exists():
                                    pil_in = Image.open(_cin).convert(
                                        "RGB").resize((518, 518))
                                elif (hasattr(spec, "best_view")
                                      and spec.best_view >= 0):
                                    _, pil_img = prepare_input_image(
                                        obj_record, spec.best_view)
                                    pil_in = pil_img.resize((518, 518))
                                else:
                                    pil_in = edited
                                prerender_img = (pil_in, edited)
                                logger.info(
                                    f"  2D from disk ({_2d_base.name}/"
                                    f"{spec.edit_id}_edited.png)")
                            except Exception as e:
                                logger.warning(f"  Cached 2D load failed: {e}")

                        if (prerender_img is None and not cache_only_2d
                                and hasattr(spec, 'best_view')
                                and spec.best_view >= 0):
                            try:
                                from scripts.run_2d_edit import (
                                    prepare_input_image, call_local_edit,
                                    call_vlm_edit)
                                img_bytes, pil_img = prepare_input_image(
                                    obj_record, spec.best_view)
                                after_desc = (spec.after_desc
                                              or spec.after_part_desc or "")
                                before_desc = (getattr(spec, 'before_part_desc', '')
                                               or '')
                                remove_labels = getattr(spec, 'remove_labels', [])
                                old_label = getattr(spec, 'old_label', '') or ''
                                if remove_labels and len(remove_labels) > 1:
                                    part_label = ", ".join(remove_labels)
                                elif remove_labels:
                                    part_label = remove_labels[0]
                                else:
                                    part_label = old_label

                                if image_edit_backend == "local_diffusers":
                                    edit_url = p25_cfg.get(
                                        "image_edit_base_url",
                                        "http://localhost:8001")
                                    edited = call_local_edit(
                                        edit_url, img_bytes, spec.edit_prompt,
                                        after_desc, old_part_label=part_label,
                                        before_part_desc=before_desc,
                                        edit_type=spec.edit_type)
                                elif vlm_client is not None:
                                    edited = call_vlm_edit(
                                        vlm_client, img_bytes, spec.edit_prompt,
                                        after_desc,
                                        p25_cfg.get("image_edit_model", ""),
                                        old_part_label=part_label,
                                        before_part_desc=before_desc,
                                        edit_type=spec.edit_type)
                                else:
                                    edited = None
                                if edited is not None:
                                    edited = edited.resize((518, 518))
                                    prerender_img = (pil_img.resize((518, 518)),
                                                     edited)
                                    logger.info(f"  2D edit from prerender "
                                                f"view {spec.best_view}")
                            except Exception as e:
                                logger.warning(
                                    f"  Prerender 2D edit failed: {e}")

                        if prerender_img is not None:
                            original_images = [prerender_img[0]]
                            edited_images = [prerender_img[1]]
                        elif not cache_only_2d:
                            original_images, edited_images = \
                                refiner.obtain_edited_images(
                                    ori_gaussian, prompts, vlm_client,
                                    obj_id, spec.edit_id,
                                    num_views=num_edit_views,
                                    edit_dir=edit_dir)
                        else:
                            original_images, edited_images = [], []
                            logger.warning(
                                "  --2d-cache-only: missing %s_edited.png → no img cond",
                                spec.edit_id)

                        if edited_images:
                            img_cond = refiner.encode_multiview_cond(
                                edited_images, original_images,
                                edit_strength=edit_strength)

                    # Run TRELLIS editing
                    slats_edited = refiner.edit(
                        ori_slat, mask, prompts,
                        img_cond=img_cond,
                        seed=seed,
                        combinations=combinations,
                    )

                    if not slats_edited:
                        raise RuntimeError("No edited SLATs produced")

                    best_slat = slats_edited[0]

                    # Export before/after pair
                    pair_dir = mesh_pairs_dir / spec.edit_id
                    export_paths = refiner.export_pair(
                        ori_slat, best_slat, pair_dir)

                    record = {
                        "edit_id": spec.edit_id,
                        "edit_type": spec.edit_type,
                        "effective_edit_type": edit_type,
                        "obj_id": obj_id,
                        "shard": spec.shard,
                        "object_desc": spec.object_desc,
                        "edit_prompt": spec.edit_prompt,
                        "after_desc": spec.after_desc,
                        **export_paths,
                        "status": "success",
                    }
                    out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                    if first_pair_dir is None:
                        first_pair_dir = pair_dir
                    logger.info(f"  → saved {pair_dir}")

                except Exception as e:
                    import traceback
                    logger.error(f"Failed {spec.edit_id}: {e}")
                    traceback.print_exc()
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": str(e),
                    }) + "\n")
                    out_fp.flush()
                    fail += 1

            for spec in idt_specs:
                idt_pair = mesh_pairs_dir / spec.edit_id
                if first_pair_dir and (first_pair_dir / "before.ply").exists():
                    idt_pair.mkdir(parents=True, exist_ok=True)
                    before_ply = first_pair_dir / "before.ply"
                    shutil.copy2(str(before_ply),
                                 str(idt_pair / "before.ply"))
                    shutil.copy2(str(before_ply),
                                 str(idt_pair / "after.ply"))
                    before_slat = first_pair_dir / "before_slat"
                    if before_slat.exists():
                        for _slat_name in ("before_slat", "after_slat"):
                            _dst = idt_pair / _slat_name
                            if _dst.exists():
                                shutil.rmtree(_dst)
                            shutil.copytree(
                                str(before_slat), str(_dst))
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id,
                        "edit_type": IDENTITY,
                        "effective_edit_type": "Identity",
                        "obj_id": obj_id, "status": "success",
                        "edit_prompt": spec.edit_prompt,
                    }, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    success += 1
                    logger.info(f"  [{spec.edit_id}] identity: before=after (no-op)")
                else:
                    out_fp.write(json.dumps({
                        "edit_id": spec.edit_id, "status": "failed",
                        "reason": "No before mesh available for identity",
                    }) + "\n")
                    out_fp.flush()
                    fail += 1

            if obj_record is not None:
                obj_record.close()

    shutil.rmtree(glb_tmp_dir, ignore_errors=True)
    logger.info(f"\n3D edits: {success} ok, {fail} fail → {output_path}")
    return output_path


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
):
    """Dispatch Step4 with split routing: GPU-bound and non-GPU edit types."""
    logger.info("=" * 60)
    logger.info("STEP 4: 3D Editing — Multi-GPU Dispatch")
    logger.info("=" * 60)
    if len(gpus) <= 1:
        raise ValueError("run_step_3d_edit_multi_gpu requires >=2 GPUs")

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
    if not all_specs:
        logger.info("No edit specs to process")
        return None

    p25_cfg = cfg.get("phase2_5", {})
    cache_dir = Path(p25_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    merged_path = cache_dir / f"edit_results{tag_suffix}.jsonl"

    done_ids: set[str] = set()
    if merged_path.exists():
        with open(merged_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        done_ids.add(rec["edit_id"])
                except Exception:
                    pass
    pending = [s for s in all_specs if s.edit_id not in done_ids]
    if not pending:
        logger.info("All 3D edits already done")
        return merged_path

    nongpu_pending = [s for s in pending if s.edit_type in STEP4_NON_GPU_TYPES]
    gpu_pending = [s for s in pending if s.edit_type not in STEP4_NON_GPU_TYPES]

    logger.info(
        "Step4 routing: GPU=%d, non-GPU=%d",
        len(gpu_pending), len(nongpu_pending),
    )

    gpu_plan = []
    if gpu_pending:
        buckets = _split_obj_groups(gpu_pending, len(gpus))
        gpu_plan = [(gpu, bucket) for gpu, bucket in zip(gpus, buckets) if bucket]
        logger.info("Step4 GPU plan: %s", ", ".join(
            f"GPU {gpu}: {len(bucket)} edits" for gpu, bucket in gpu_plan))
    else:
        logger.info("Step4 GPU plan: no GPU-bound edits")

    dispatch_dir = cache_dir / f"dispatch{tag_suffix}"
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    worker_result_paths: list[Path] = []
    procs = []
    for wi, (gpu, bucket) in enumerate(gpu_plan):
        ids_path = dispatch_dir / f"edit_ids_gpu{gpu}.txt"
        with open(ids_path, "w", encoding="utf-8") as f:
            for s in bucket:
                f.write(f"{s.edit_id}\n")
        res_name = f"edit_results{tag_suffix}_gpu{gpu}.jsonl"
        worker_result_paths.append(cache_dir / res_name)

        cmd = [sys.executable, str(Path(__file__).resolve())]
        if config_path:
            cmd.extend(["--config", config_path])
        cmd.extend(["--steps", "4"])
        cmd.extend(["--seed", str(seed)])
        cmd.extend(["--edit-dir", edit_dir or "2d_edits"])
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
        logger.info("Launch worker %d on GPU %s with %d edits",
                    wi, gpu, len(bucket))
        procs.append((gpu, subprocess.Popen(cmd, env=env)))

    if nongpu_pending:
        cpu_ids_path = dispatch_dir / "edit_ids_nongpu.txt"
        with open(cpu_ids_path, "w", encoding="utf-8") as f:
            for s in nongpu_pending:
                f.write(f"{s.edit_id}\n")
        cpu_res_name = f"edit_results{tag_suffix}_nongpu.jsonl"
        worker_result_paths.append(cache_dir / cpu_res_name)

        cpu_cmd = [sys.executable, str(Path(__file__).resolve())]
        if config_path:
            cpu_cmd.extend(["--config", config_path])
        cpu_cmd.extend(["--steps", "4"])
        cpu_cmd.extend(["--seed", str(seed)])
        cpu_cmd.extend(["--edit-dir", edit_dir or "2d_edits"])
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

    failed_gpus = []
    for gpu, p in procs:
        ret = p.wait()
        if ret != 0:
            failed_gpus.append((gpu, ret))
    if failed_gpus:
        raise RuntimeError(f"Multi-GPU worker failed: {failed_gpus}")

    # Merge worker result files into canonical edit_results{tag}.jsonl
    merged: OrderedDict[str, dict] = OrderedDict()
    if merged_path.exists():
        with open(merged_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    eid = rec.get("edit_id")
                    if eid:
                        merged[eid] = rec
                except Exception:
                    pass
    for rp in worker_result_paths:
        if not rp.exists():
            continue
        with open(rp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    eid = rec.get("edit_id")
                    if eid:
                        merged[eid] = rec
                except Exception:
                    pass
    with open(merged_path, "w", encoding="utf-8") as f:
        for rec in merged.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Merged multi-GPU results → %s (%d records)",
                merged_path, len(merged))
    return merged_path


# =========================================================================
# Step 5: Quality Scoring
# =========================================================================

def run_step_quality(cfg, results_path, logger, tag=None, limit=None):
    """Step 5: VLM quality scoring + tier classification."""
    logger.info("=" * 60)
    logger.info("STEP 5: Quality Scoring (VLM)")
    logger.info("=" * 60)

    from partcraft.phase3_filter.vlm_filter import run_vlm_filter
    p25_cfg = cfg.get("phase2_5", {})
    cache_dir = Path(p25_cfg["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    tag_suffix = f"_{tag}" if tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"

    phase3_dir = cache_dir / f"phase3{tag_suffix}"
    run_vlm_filter(
        cfg, str(results_path), str(mesh_pairs_dir),
        str(phase3_dir),
        limit=limit)

    scores_file = phase3_dir / "vlm_scores.jsonl"
    out = str(scores_file) if scores_file.is_file() else None
    logger.info(f"Quality scores: {out}")
    return out


# =========================================================================
# Step 6: Export
# =========================================================================

def run_step_export(cfg, specs_path, scores_path, logger, tag=None):
    """Step 6: Instruction generation + final dataset export."""
    logger.info("=" * 60)
    logger.info("STEP 6: Export")
    logger.info("=" * 60)

    from partcraft.io.export import EditPairWriter, EditPairRecord
    from partcraft.phase4_filter.instruction import generate_instructions

    output_dir = Path(cfg["data"]["output_dir"])
    tag_suffix = f"_{tag}" if tag else ""
    n_variants = cfg.get("phase3", {}).get("instructions_per_edit", 3)

    # Load passed entries from quality scoring
    passed_entries = []
    if scores_path and Path(scores_path).exists():
        with open(scores_path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip().replace("\x00", "")
                if not line or not line.startswith("{"):
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid JSON in scores file line %s", lineno
                    )
                    continue
                if entry.get("quality_tier") in ("high", "medium"):
                    passed_entries.append(entry)

    if not passed_entries:
        logger.warning("No passed entries for export")
        return None

    # Load edit specs
    specs_map: dict[str, EditSpec] = {}
    with open(specs_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                specs_map[d["edit_id"]] = EditSpec(**d)

    exported = 0
    export_path = output_dir / f"edit_pairs{tag_suffix}.jsonl"
    with EditPairWriter(output_dir,
                        filename=f"edit_pairs{tag_suffix}.jsonl") as writer:
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
                edit_type=entry.get("effective_edit_type", spec.edit_type),
                instruction=instructions[0],
                instruction_variants=instructions[1:],
                source_obj_id=spec.obj_id,
                source_shard=spec.shard,
                object_desc=spec.object_desc,
                edit_prompt=spec.edit_prompt,
                after_desc=spec.after_desc,
                quality_tier=entry.get("quality_tier", ""),
                quality_score=float(
                    entry.get("quality_score", entry.get("score", 0.0))
                ),
            )
            writer.write_pair(record)
            exported += 1

    logger.info(f"Exported {exported} pairs → {export_path}")
    return export_path


def diagnose_step1(labels_path: Path, shard: str | None) -> dict:
    rows = list(_iter_jsonl(labels_path))
    ids = [r.get("obj_id") for r in rows if r.get("obj_id")]
    shard_counter = Counter(str(r.get("shard", "")).zfill(2) for r in rows)
    dup_count = len(ids) - len(set(ids))
    return {
        "stage": "step1_semantic",
        "file": str(labels_path),
        "count": len(rows),
        "unique_obj_ids": len(set(ids)),
        "duplicate_obj_id_rows": dup_count,
        "target_shard": shard,
        "by_shard": dict(shard_counter),
    }


def diagnose_step2(specs_path: Path) -> dict:
    rows = list(_iter_jsonl(specs_path))
    edit_ids = [r.get("edit_id") for r in rows if r.get("edit_id")]
    type_counter = Counter(r.get("edit_type", "unknown") for r in rows)
    return {
        "stage": "step2_planning",
        "file": str(specs_path),
        "count": len(rows),
        "unique_edit_ids": len(set(edit_ids)),
        "duplicate_edit_id_rows": len(edit_ids) - len(set(edit_ids)),
        "by_edit_type": dict(type_counter),
    }


def diagnose_step3(manifest_path: Path) -> dict:
    rows = list(_iter_jsonl(manifest_path))
    status_counter = Counter(r.get("status", "unknown") for r in rows)
    return {
        "stage": "step3_2d_edit",
        "file": str(manifest_path),
        "count": len(rows),
        "status": dict(status_counter),
    }


def diagnose_step4(results_path: Path, expected_edit_ids: set[str] | None = None) -> dict:
    rows = list(_iter_jsonl(results_path))
    status_counter = Counter(r.get("status", "unknown") for r in rows)
    type_counter = Counter(r.get("edit_type", "unknown") for r in rows)
    fail_reason_counter = Counter(
        (r.get("reason") or "unknown")[:120]
        for r in rows
        if r.get("status") != "success"
    )
    result_ids = {r.get("edit_id") for r in rows if r.get("edit_id")}
    missing = []
    if expected_edit_ids:
        missing = sorted([eid for eid in expected_edit_ids if eid not in result_ids])
    return {
        "stage": "step4_3d_edit",
        "file": str(results_path),
        "count": len(rows),
        "status": dict(status_counter),
        "by_edit_type": dict(type_counter),
        "top_fail_reasons": fail_reason_counter.most_common(20),
        "expected_edit_ids": len(expected_edit_ids or []),
        "missing_edit_ids_count": len(missing),
        "missing_edit_ids_examples": missing[:50],
    }


def diagnose_step5(scores_path: Path) -> dict:
    rows = list(_iter_jsonl(scores_path))
    tier_counter = Counter(r.get("quality_tier", "unknown") for r in rows)
    return {
        "stage": "step5_quality",
        "file": str(scores_path),
        "count": len(rows),
        "by_tier": dict(tier_counter),
    }


def diagnose_step6(export_path: Path) -> dict:
    rows = list(_iter_jsonl(export_path))
    type_counter = Counter(r.get("edit_type", "unknown") for r in rows)
    return {
        "stage": "step6_export",
        "file": str(export_path),
        "count": len(rows),
        "by_edit_type": dict(type_counter),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PartCraft3D Batch Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps 1–6: semantic, plan, 2D, TRELLIS, quality, export. "
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Cost estimation only, no actual processing")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for spec files (e.g. '_action')")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug output files (masks, views, enricher "
                             "ortho images). Off by default.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run (delete and regenerate cached results)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_shard = _normalize_shard(
        args.shard or ((cfg.get("data", {}).get("shards") or [None])[0])
    )
    if target_shard:
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
    token_suffix = f"_{run_token}" if run_token else ""
    report_dir = _pipeline_report_dir(cfg, target_shard)
    logger.info("Run token: %s (shard=%s)", run_token, target_shard or "N/A")
    logger.info("Diagnostics: %s", report_dir)
    stage_diagnostics = {}

    steps = set(args.steps) if args.steps else {1, 2, 3, 4, 5, 6}
    if args.max_parallel and args.workers == 4:
        args.workers = max(8, (os.cpu_count() or 8))
    if args.max_parallel and (args.step1_workers is None):
        cpu_n = os.cpu_count() or 8
        cfg.setdefault("phase0", {})["max_workers"] = max(
            int(cfg.get("phase0", {}).get("max_workers", 4)),
            min(64, cpu_n * 2)
        )

    dataset = create_dataset(cfg)
    logger.info(f"Dataset: {len(dataset)} objects")

    # ---- Resolve paths ----
    suffix = args.suffix
    labels_path = (Path(cfg["phase0"]["cache_dir"])
                   / f"semantic_labels{token_suffix}.jsonl")
    specs_path = (Path(cfg["phase1"]["cache_dir"])
                  / f"edit_specs{suffix}{token_suffix}.jsonl")

    # ---- Step 1: Semantic ----
    if 1 in steps:
        step1_obj_ids = _load_generic_ids_file(args.step1_obj_ids_file)
        step1_urls = _parse_csv_or_space_list(args.step1_vlm_urls)
        if not step1_urls:
            cfg_urls = cfg.get("phase0", {}).get("vlm_base_urls", [])
            if isinstance(cfg_urls, list):
                step1_urls = [str(u).strip() for u in cfg_urls if str(u).strip()]
        step1_gpus = _parse_csv_or_space_list(args.step1_gpus)

        if len(step1_urls) > 1 and not step1_obj_ids:
            labels_path = run_step_semantic_multi_gpu(
                cfg, dataset, logger,
                vlm_urls=step1_urls,
                gpus=step1_gpus if step1_gpus else None,
                tag=run_token, limit=args.limit,
                force=args.force, debug=args.debug,
                step1_workers=args.step1_workers,
                config_path=args.config,
            )
        else:
            labels_path = run_step_semantic(
                cfg, dataset, logger, limit=args.limit,
                force=args.force, tag=run_token, debug=args.debug,
                object_ids=step1_obj_ids or None,
                labels_name=args.step1_labels_name,
            )
        stage_diagnostics["step1"] = diagnose_step1(labels_path, target_shard)
        _write_stage_diag(report_dir, "step1_semantic", stage_diagnostics["step1"])
        _sync_manifest_link(
            cfg, target_shard, "phase0", "semantic_labels.jsonl", labels_path
        )

    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path}")
        logger.error("Run step 1 first, or provide cached labels")
        sys.exit(1)

    # ---- Step 2: Planning ----
    if 2 in steps:
        specs_path = run_step_planning(cfg, labels_path, logger,
                                       suffix=f"{suffix}{token_suffix}")
        stage_diagnostics["step2"] = diagnose_step2(specs_path)
        _write_stage_diag(report_dir, "step2_planning", stage_diagnostics["step2"])
        _sync_manifest_link(
            cfg, target_shard, "phase1", "edit_specs.jsonl", specs_path
        )

    needs_specs = bool({2, 3, 4, 6} & steps)
    if needs_specs and not specs_path.exists():
        logger.error(f"Specs not found: {specs_path}")
        logger.error("Run step 2 first, or provide cached specs")
        sys.exit(1)

    # ---- Step 3: 2D Edit ----
    edit_2d_dir = None
    if 3 in steps:
        image_edit_urls = _parse_csv_or_space_list(args.image_edit_urls)
        edit_2d_dir = run_step_2d_edit(
            cfg, specs_path, dataset, logger,
            tag=run_token, workers=args.workers, limit=args.limit,
            edit_ids=set(args.edit_ids) if args.edit_ids else None,
            edit_server_urls=image_edit_urls,
            auto_max_parallel=args.max_parallel)
        if edit_2d_dir:
            manifest = Path(edit_2d_dir) / "manifest.jsonl"
            stage_diagnostics["step3"] = diagnose_step3(manifest)
            _write_stage_diag(report_dir, "step3_2d_edit", stage_diagnostics["step3"])
            _sync_manifest_link(
                cfg, target_shard, "phase2_5", "2d_manifest.jsonl", manifest
            )

    # ---- Step 4: 3D Edit ----
    edit_subdir = args.edit_dir
    if not edit_subdir and run_token:
        edit_subdir = f"2d_edits_{run_token}"
    elif not edit_subdir:
        edit_subdir = "2d_edits"

    results_path = None
    if 4 in steps:
        file_edit_ids = _load_edit_ids_file(args.edit_ids_file)
        cli_edit_ids = list(args.edit_ids) if args.edit_ids else []
        merged_edit_ids = cli_edit_ids + file_edit_ids
        final_edit_ids = merged_edit_ids if merged_edit_ids else None

        gpu_list = _parse_csv_or_space_list(args.gpus)
        if len(gpu_list) > 1:
            results_path = run_step_3d_edit_multi_gpu(
                cfg, specs_path, logger,
                gpus=gpu_list,
                tag=run_token, seed=args.seed, limit=args.limit,
                use_2d=args.use_2d, edit_ids=final_edit_ids,
                edit_dir=edit_subdir, debug=args.debug,
                cache_only_2d=args.cache_only_2d,
                config_path=args.config,
            )
        else:
            results_path = run_step_3d_edit(
                cfg, specs_path, dataset, logger,
                tag=run_token, seed=args.seed, limit=args.limit,
                use_2d=args.use_2d, edit_ids=final_edit_ids,
                edit_dir=edit_subdir, debug=args.debug,
                cache_only_2d=args.cache_only_2d,
                results_name=args.results_name)
        if results_path and Path(results_path).exists():
            expected_ids = {
                r.get("edit_id") for r in _iter_jsonl(specs_path)
                if r.get("edit_id")
            }
            stage_diagnostics["step4"] = diagnose_step4(
                Path(results_path), expected_edit_ids=expected_ids
            )
            _write_stage_diag(report_dir, "step4_3d_edit", stage_diagnostics["step4"])
            _sync_manifest_link(
                cfg, target_shard, "phase2_5", "edit_results.jsonl", Path(results_path)
            )

    if results_path is None:
        p25_cfg = cfg.get("phase2_5", {})
        results_path = (Path(p25_cfg["cache_dir"])
                        / f"edit_results{token_suffix}.jsonl")

    # ---- Step 5: Quality ----
    scores_path = None
    if 5 in steps and results_path and Path(results_path).exists():
        scores_path = run_step_quality(
            cfg, results_path, logger, tag=run_token, limit=args.limit)
        if scores_path and Path(scores_path).exists():
            stage_diagnostics["step5"] = diagnose_step5(Path(scores_path))
            _write_stage_diag(report_dir, "step5_quality", stage_diagnostics["step5"])
            _sync_manifest_link(
                cfg, target_shard, "phase3", "vlm_scores.jsonl", Path(scores_path)
            )

    if scores_path is None and run_token:
        p25_cfg = cfg.get("phase2_5", {})
        cand = Path(p25_cfg["cache_dir"]) / f"phase3_{run_token}" / "vlm_scores.jsonl"
        if cand.is_file():
            scores_path = str(cand)
            logger.info(f"Using existing quality scores: {scores_path}")
            stage_diagnostics["step5"] = diagnose_step5(cand)
            _write_stage_diag(report_dir, "step5_quality", stage_diagnostics["step5"])
            _sync_manifest_link(
                cfg, target_shard, "phase3", "vlm_scores.jsonl", cand
            )

    # ---- Step 6: Export ----
    export_path = None
    if 6 in steps:
        export_path = run_step_export(cfg, specs_path, scores_path, logger, tag=run_token)
        if export_path and Path(export_path).exists():
            stage_diagnostics["step6"] = diagnose_step6(Path(export_path))
            _write_stage_diag(report_dir, "step6_export", stage_diagnostics["step6"])
            _sync_manifest_link(
                cfg, target_shard, "phase4", "edit_pairs.jsonl", Path(export_path)
            )

    summary = {
        "run_token": run_token,
        "shard": target_shard,
        "steps_requested": sorted(steps),
        "paths": {
            "labels": str(labels_path),
            "specs": str(specs_path),
            "results": str(results_path) if results_path else None,
            "scores": scores_path,
            "export": str(export_path) if export_path else None,
        },
        "stages_with_diagnostics": sorted(stage_diagnostics.keys()),
    }
    _write_stage_diag(report_dir, "pipeline_summary", summary)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Shard-centric diagnostics → %s", report_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
