from __future__ import annotations

from pathlib import Path


def build_runtime_context(args, cfg: dict, *, target_shard: str | None, run_token: str, report_dir: Path) -> dict:
    steps = set(args.steps) if args.steps else {1, 2, 3, 4, 5, 6}
    token_suffix = f"_{run_token}" if run_token else ""
    suffix = args.suffix
    labels_path = Path(cfg["phase0"]["cache_dir"]) / f"semantic_labels{token_suffix}.jsonl"
    specs_path = Path(cfg["phase1"]["cache_dir"]) / f"edit_specs{suffix}{token_suffix}.jsonl"
    return {
        "steps": steps,
        "token_suffix": token_suffix,
        "suffix": suffix,
        "labels_path": labels_path,
        "specs_path": specs_path,
        "target_shard": target_shard,
        "report_dir": report_dir,
        "stage_diagnostics": {},
    }


def run_selected_steps(ctx: dict, *, args, cfg, logger, dataset, callbacks: dict) -> dict:
    steps = ctx["steps"]
    labels_path = ctx["labels_path"]
    specs_path = ctx["specs_path"]
    report_dir = ctx["report_dir"]
    stage_diagnostics = ctx["stage_diagnostics"]
    target_shard = ctx["target_shard"]
    run_token = callbacks["run_token"]

    if 1 in steps:
        step1_obj_ids = callbacks["load_generic_ids_file"](args.step1_obj_ids_file)
        step1_urls = callbacks["parse_list"](args.step1_vlm_urls)
        if not step1_urls:
            cfg_urls = cfg.get("phase0", {}).get("vlm_base_urls", [])
            if isinstance(cfg_urls, list):
                step1_urls = [str(u).strip() for u in cfg_urls if str(u).strip()]
        step1_gpus = callbacks["parse_list"](args.step1_gpus)
        if len(step1_urls) > 1 and not step1_obj_ids:
            labels_path = callbacks["run_step_semantic_multi_gpu"](
                cfg,
                dataset,
                logger,
                vlm_urls=step1_urls,
                gpus=step1_gpus if step1_gpus else None,
                tag=run_token,
                limit=args.limit,
                force=args.force,
                debug=args.debug,
                step1_workers=args.step1_workers,
                config_path=args.config,
            )
        else:
            labels_path = callbacks["run_step_semantic"](
                cfg,
                dataset,
                logger,
                limit=args.limit,
                force=args.force,
                tag=run_token,
                debug=args.debug,
                object_ids=step1_obj_ids or None,
                labels_name=args.step1_labels_name,
            )
        stage_diagnostics["step1"] = callbacks["diagnose_step1"](labels_path, target_shard)
        callbacks["write_stage_diag"](report_dir, "step1_semantic", stage_diagnostics["step1"])
        callbacks["sync_manifest_link"](cfg, target_shard, "phase0", "semantic_labels.jsonl", labels_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    if 2 in steps:
        specs_path = callbacks["run_step_planning"](
            cfg,
            labels_path,
            logger,
            suffix=f"{ctx['suffix']}{ctx['token_suffix']}",
        )
        stage_diagnostics["step2"] = callbacks["diagnose_step2"](specs_path)
        callbacks["write_stage_diag"](report_dir, "step2_planning", stage_diagnostics["step2"])
        callbacks["sync_manifest_link"](cfg, target_shard, "phase1", "edit_specs.jsonl", specs_path)

    needs_specs = bool({2, 3, 4, 6} & steps)
    if needs_specs and not specs_path.exists():
        raise FileNotFoundError(f"Specs not found: {specs_path}")

    edit_2d_dir = None
    if 3 in steps:
        image_edit_urls = callbacks["parse_list"](args.image_edit_urls)
        edit_2d_dir = callbacks["run_step_2d_edit"](
            cfg,
            specs_path,
            dataset,
            logger,
            tag=run_token,
            workers=args.workers,
            limit=args.limit,
            edit_ids=set(args.edit_ids) if args.edit_ids else None,
            edit_server_urls=image_edit_urls,
            auto_max_parallel=args.max_parallel,
        )
        if edit_2d_dir:
            manifest = Path(edit_2d_dir) / "manifest.jsonl"
            stage_diagnostics["step3"] = callbacks["diagnose_step3"](manifest)
            callbacks["write_stage_diag"](report_dir, "step3_2d_edit", stage_diagnostics["step3"])
            callbacks["sync_manifest_link"](cfg, target_shard, "phase2_5", "2d_manifest.jsonl", manifest)

    edit_subdir = args.edit_dir
    if not edit_subdir and run_token:
        edit_subdir = f"2d_edits_{run_token}"
    elif not edit_subdir:
        edit_subdir = "2d_edits"

    results_path = None
    if 4 in steps:
        file_edit_ids = callbacks["load_edit_ids_file"](args.edit_ids_file)
        cli_edit_ids = list(args.edit_ids) if args.edit_ids else []
        merged_edit_ids = cli_edit_ids + file_edit_ids
        final_edit_ids = merged_edit_ids if merged_edit_ids else None
        gpu_list = callbacks["parse_list"](args.gpus)
        if len(gpu_list) > 1:
            results_path = callbacks["run_step_3d_edit_multi_gpu"](
                cfg,
                specs_path,
                logger,
                gpus=gpu_list,
                tag=run_token,
                seed=args.seed,
                limit=args.limit,
                use_2d=args.use_2d,
                edit_ids=final_edit_ids,
                edit_dir=edit_subdir,
                debug=args.debug,
                cache_only_2d=args.cache_only_2d,
                config_path=args.config,
            )
        else:
            results_path = callbacks["run_step_3d_edit"](
                cfg,
                specs_path,
                dataset,
                logger,
                tag=run_token,
                seed=args.seed,
                limit=args.limit,
                use_2d=args.use_2d,
                edit_ids=final_edit_ids,
                edit_dir=edit_subdir,
                debug=args.debug,
                cache_only_2d=args.cache_only_2d,
                results_name=args.results_name,
            )
        if results_path and Path(results_path).exists():
            expected_ids = {
                r.get("edit_id") for r in callbacks["iter_jsonl"](specs_path) if r.get("edit_id")
            }
            stage_diagnostics["step4"] = callbacks["diagnose_step4"](
                Path(results_path), expected_edit_ids=expected_ids
            )
            callbacks["write_stage_diag"](report_dir, "step4_3d_edit", stage_diagnostics["step4"])
            callbacks["sync_manifest_link"](cfg, target_shard, "phase2_5", "edit_results.jsonl", Path(results_path))

    if results_path is None:
        p25_cfg = cfg.get("phase2_5", {})
        results_path = Path(p25_cfg["cache_dir"]) / f"edit_results{ctx['token_suffix']}.jsonl"

    scores_path = None
    if 5 in steps and results_path and Path(results_path).exists():
        scores_path = callbacks["run_step_quality"](cfg, results_path, logger, tag=run_token, limit=args.limit)
        if scores_path and Path(scores_path).exists():
            stage_diagnostics["step5"] = callbacks["diagnose_step5"](Path(scores_path))
            callbacks["write_stage_diag"](report_dir, "step5_quality", stage_diagnostics["step5"])
            callbacks["sync_manifest_link"](cfg, target_shard, "phase3", "vlm_scores.jsonl", Path(scores_path))

    if scores_path is None and run_token:
        p25_cfg = cfg.get("phase2_5", {})
        cand = Path(p25_cfg["cache_dir"]) / f"phase3_{run_token}" / "vlm_scores.jsonl"
        if cand.is_file():
            scores_path = str(cand)
            logger.info("Using existing quality scores: %s", scores_path)
            stage_diagnostics["step5"] = callbacks["diagnose_step5"](cand)
            callbacks["write_stage_diag"](report_dir, "step5_quality", stage_diagnostics["step5"])
            callbacks["sync_manifest_link"](cfg, target_shard, "phase3", "vlm_scores.jsonl", cand)

    export_path = None
    if 6 in steps:
        export_path = callbacks["run_step_export"](cfg, specs_path, scores_path, logger, tag=run_token)
        if export_path and Path(export_path).exists():
            stage_diagnostics["step6"] = callbacks["diagnose_step6"](Path(export_path))
            callbacks["write_stage_diag"](report_dir, "step6_export", stage_diagnostics["step6"])
            callbacks["sync_manifest_link"](cfg, target_shard, "phase4", "edit_pairs.jsonl", Path(export_path))

    ctx["labels_path"] = labels_path
    ctx["specs_path"] = specs_path
    ctx["results_path"] = results_path
    ctx["scores_path"] = scores_path
    ctx["export_path"] = export_path
    return ctx


def finalize_summary(ctx: dict, *, run_token: str) -> dict:
    return {
        "run_token": run_token,
        "shard": ctx["target_shard"],
        "steps_requested": sorted(ctx["steps"]),
        "paths": {
            "labels": str(ctx["labels_path"]),
            "specs": str(ctx["specs_path"]),
            "results": str(ctx.get("results_path")) if ctx.get("results_path") else None,
            "scores": ctx.get("scores_path"),
            "export": str(ctx.get("export_path")) if ctx.get("export_path") else None,
        },
        "stages_with_diagnostics": sorted(ctx["stage_diagnostics"].keys()),
    }
