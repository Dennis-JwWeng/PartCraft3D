#!/usr/bin/env python3
"""Compare orthogonal 4-view VLM enrichment: local OpenAI-compatible vs API.

Uses the same images, prompt, and temperature as streaming enrichment
(:func:`_enrich_one_object_visual`). Writes raw JSON + a short summary.

Example:
  PARTCRAFT_DATA_ROOT=/path/to/partverse \\
    python scripts/tools/compare_vlm_enrichment.py \\
    --obj-id 0008dc75fb3648f2af4ca8c4d711e53e \\
    --out-dir outputs/vlm_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

from partcraft.phase1_planning.enricher import _enrich_one_object_visual
from partcraft.utils.config import load_config
from scripts.pipeline_common import create_dataset, resolve_api_key


def _labels_for_object(obj) -> tuple[str, list[str], list[int]]:
    """Match ``run_streaming.py`` label extraction."""
    category = "object"
    labels: list[str] = []
    actual_pids: list[int] = []
    for p in sorted(obj.parts, key=lambda x: x.part_id):
        if p.mesh_node_names:
            raw = p.mesh_node_names[0]
            label = raw.rsplit("_", 1)[0] if "_" in raw else raw
        else:
            label = p.cluster_name
        labels.append(label)
        actual_pids.append(p.part_id)
    return category, labels, actual_pids


def _summarize(result: dict | None) -> dict:
    if not result or "part_groups" not in result:
        return {"ok": False, "error": "missing part_groups"}
    groups = result.get("part_groups") or []
    globals_ = result.get("global_edits") or []
    return {
        "ok": True,
        "object_desc": result.get("object_desc", ""),
        "num_groups": len(groups),
        "group_names": [g.get("group_name", "") for g in groups],
        "num_global_edits": len(globals_),
        "global_prompts": [g.get("prompt", "") for g in globals_],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-local", default="configs/partverse_local.yaml")
    ap.add_argument("--config-api", default="configs/partverse.yaml")
    ap.add_argument("--obj-id", required=True, help="Object uid (NPZ stem)")
    ap.add_argument("--shard", default=None, help="Shard dir (default: first in config)")
    ap.add_argument("--out-dir", default="outputs/vlm_compare")
    ap.add_argument("--skip-local", action="store_true")
    ap.add_argument("--skip-api", action="store_true")
    args = ap.parse_args()

    cfg_loc = load_config(_PROJECT_ROOT / args.config_local)
    cfg_api = load_config(_PROJECT_ROOT / args.config_api)

    # Same dataset layout as local config (paths resolved via PARTCRAFT_DATA_ROOT)
    shards = cfg_loc["data"]["shards"]
    shard = args.shard or (shards[0] if shards else "00")

    ds = create_dataset(cfg_loc)
    obj = ds.load_object(shard, args.obj_id)
    category, labels, _pids = _labels_for_object(obj)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "obj_id": args.obj_id,
        "shard": shard,
        "category": category,
        "labels": labels,
        "n_parts": len(labels),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    results: dict[str, dict | None] = {}

    if not args.skip_local:
        p0 = cfg_loc["phase0"]
        key = resolve_api_key(cfg_loc) or p0.get("vlm_api_key") or "dummy"
        client = OpenAI(base_url=p0["vlm_base_url"], api_key=key)
        model = p0["vlm_model"]
        try:
            results["local"] = _enrich_one_object_visual(
                client, model, obj, category, labels)
        except Exception as e:
            print(f"[local] FAILED: {e}", file=sys.stderr)
            results["local"] = None
    else:
        results["local"] = None

    if not args.skip_api:
        p0a = cfg_api["phase0"]
        key_a = resolve_api_key(cfg_api)
        if not key_a:
            print("[api] No vlm_api_key in config / env; skip or fix configs/partverse.yaml",
                  file=sys.stderr)
            results["api"] = None
        else:
            client_a = OpenAI(base_url=p0a["vlm_base_url"], api_key=key_a)
            model_a = p0a["vlm_model"]
            try:
                results["api"] = _enrich_one_object_visual(
                    client_a, model_a, obj, category, labels)
            except Exception as e:
                print(f"[api] FAILED: {e}", file=sys.stderr)
                results["api"] = None
    else:
        results["api"] = None

    obj.close()

    for name, res in results.items():
        path = out / f"enrich_{name}_raw.json"
        if res is None:
            path.write_text("null\n", encoding="utf-8")
        else:
            # Drop internal keys starting with _ for readability (optional)
            to_save = {k: v for k, v in res.items() if not k.startswith("_")}
            to_save["_labels"] = res.get("_labels")
            to_save["orthogonal_views"] = res.get("orthogonal_views")
            path.write_text(
                json.dumps(to_save, ensure_ascii=False, indent=2),
                encoding="utf-8")

    summary = {
        "meta": meta,
        "local": _summarize(results.get("local")),
        "api": _summarize(results.get("api")),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
