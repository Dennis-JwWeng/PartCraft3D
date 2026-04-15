"""Pipeline v2 scheduling helpers (control plane).

Pure functions that read the ``pipeline:`` block and ``services`` of a config.

* :func:`stages_for(cfg)` — list[Phase] ordered stage definitions (``pipeline.stages``)
* :func:`select_stages(cfg, names, with_optional)` — stage subset
* :func:`gpus_for` / :func:`vlm_urls_for` / :func:`flux_urls_for` — hardware + URL lists

Imported by :mod:`run` and ``run_pipeline_v2_shard.sh`` (``dump_shell_env``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from . import services_cfg as sc


@dataclass
class Phase:
    """One pipeline stage row from ``pipeline.stages`` (historical class name)."""

    name: str
    desc: str = ""
    servers: str = "none"          # "vlm" | "flux" | "none"
    steps: list[str] = field(default_factory=list)
    server_steps: list[str] = field(default_factory=list)  # subset of steps that need the server; empty → all steps
    use_gpus: bool = False
    optional: bool = False
    parallel_group: str = ""       # non-empty → run concurrently with same-group stages


def _pipeline(cfg: dict) -> dict:
    p = cfg.get("pipeline") or {}
    if not isinstance(p, dict):
        raise ValueError("[CONFIG] pipeline: must be a mapping")
    return p


def gpus_for(cfg: dict) -> list[int]:
    p = _pipeline(cfg)
    raw = p.get("gpus")
    if not raw:
        raise ValueError("[CONFIG] pipeline.gpus is required (e.g. [4,5,6,7])")
    return [int(x) for x in raw]


def n_gpus(cfg: dict) -> int:
    return len(gpus_for(cfg))


def vlm_port(cfg: dict, idx: int) -> int:
    p = _pipeline(cfg)
    base = int(p.get("vlm_port_base", 8002))
    stride = int(p.get("vlm_port_stride", 10))
    return base + idx * stride


def flux_port(cfg: dict, idx: int) -> int:
    p = _pipeline(cfg)
    base = int(p.get("flux_port_base", 8004))
    stride = int(p.get("flux_port_stride", 1))
    return base + idx * stride


def n_vlm_servers(cfg: dict) -> int:
    p = _pipeline(cfg)
    explicit = p.get("n_vlm_servers")
    if explicit is not None:
        return int(explicit)
    return n_gpus(cfg)


def vlm_urls_for(cfg: dict) -> list[str]:
    """One VLM ``/v1`` URL per server instance.

    Override via ``services.vlm.base_urls`` (or legacy ``vlm_base_urls`` inside that block).
    """
    s = cfg.get("services")
    if isinstance(s, dict):
        v = s.get("vlm")
        if isinstance(v, dict):
            override = v.get("base_urls") or v.get("vlm_base_urls")
            if override:
                return list(override)
    return [f"http://localhost:{vlm_port(cfg, i)}/v1"
            for i in range(n_vlm_servers(cfg))]


def flux_urls_for(cfg: dict) -> list[str]:
    """Override via ``services.image_edit.base_urls``."""
    s = cfg.get("services")
    if isinstance(s, dict):
        ie = s.get("image_edit")
        if isinstance(ie, dict):
            override = ie.get("base_urls") or ie.get("image_edit_base_urls")
            if override:
                return list(override)
    return [f"http://localhost:{flux_port(cfg, i)}"
            for i in range(n_gpus(cfg))]


def stages_for(cfg: dict) -> list[Phase]:
    raw_list = sc.pipeline_stages_raw(cfg)
    out: list[Phase] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError(f"[CONFIG] pipeline.stages entry not a dict: {entry}")
        out.append(Phase(
            name=str(entry["name"]),
            desc=str(entry.get("desc", "")),
            servers=str(entry.get("servers", "none")),
            steps=list(entry.get("steps") or []),
            server_steps=list(entry.get("server_steps") or []),
            use_gpus=bool(entry.get("use_gpus", False)),
            optional=bool(entry.get("optional", False)),
            parallel_group=str(entry.get("parallel_group", "")),
        ))
    return out


def select_stages(
    cfg: dict,
    *,
    names: list[str] | None = None,
    with_optional: bool = False,
) -> list[Phase]:
    stages = stages_for(cfg)
    if names:
        wanted = set(names)
        return [p for p in stages if p.name in wanted]
    return [p for p in stages if with_optional or not p.optional]


def get_stage(cfg: dict, name: str) -> Phase:
    for st in stages_for(cfg):
        if st.name == name:
            return st
    raise KeyError(f"stage {name!r} not in config")


def dump_stage_batches(
    cfg: dict,
    stage_names: list[str],
) -> list[list[str]]:
    """Group stage_names into ordered execution batches by parallel_group.

    Stages sharing the same non-empty ``parallel_group`` are placed in the
    same batch and will be run concurrently by the shell orchestrator.
    Stages without a group (or with ``servers != "none"``) form
    single-element batches and run serially.

    The output preserves the original relative order of stage_names.

    Example with D and D2 both having ``parallel_group: "D+D2"``::

        dump_stage_batches(cfg, ["A", "C", "D", "D2", "E"])
        → [["A"], ["C"], ["D", "D2"], ["E"]]
    """
    by_name = {ph.name: ph for ph in stages_for(cfg)}
    batches: list[list[str]] = []
    group_to_idx: dict[str, int] = {}   # parallel_group → index in batches
    group_server_count: dict[str, int] = {}  # parallel_group → stages needing servers

    for name in stage_names:
        ph = by_name.get(name)
        group = (ph.parallel_group if ph else "") or ""
        needs_servers = bool(ph and ph.servers != "none")
        # Safety guard: allow at most one server-backed stage per parallel group.
        # This enables "server stage + CPU/GPU-only stage" concurrency while
        # avoiding duplicate external server startups in the same group.
        if group and needs_servers and group_server_count.get(group, 0) >= 1:
            logging.getLogger("scheduler").warning(
                "[scheduler] stage %s is in parallel_group %r with servers=%r "
                "but that group already has a server-backed stage — running serially",
                name, group, ph.servers,
            )
            group = ""
        if group and group in group_to_idx:
            batches[group_to_idx[group]].append(name)
            if needs_servers:
                group_server_count[group] = group_server_count.get(group, 0) + 1
        else:
            idx = len(batches)
            batches.append([name])
            if group:
                group_to_idx[group] = idx
                group_server_count[group] = 1 if needs_servers else 0

    return batches


def dump_shell_env(
    cfg: dict,
    stage_name: str | None = None,
    *,
    phase_name: str | None = None,
) -> str:
    """Emit shell variables that bash can ``eval``.

    ``phase_name`` is accepted as a deprecated alias for ``stage_name``.

    Exposes ``DEFAULT_STAGES`` / ``ALL_STAGES``. When ``stage_name`` is set, also
    ``STAGE_NAME``, ``STAGE_DESC``, ``STAGE_STEPS``, ``STAGE_SERVERS``.
    """
    gpus = gpus_for(cfg)
    lines = [
        f"GPUS=({' '.join(str(g) for g in gpus)})",
        f"N_VLM_SERVERS={n_vlm_servers(cfg)}",
        f"VLM_PORTS=({' '.join(str(vlm_port(cfg, i)) for i in range(n_vlm_servers(cfg)))})",
        f"FLUX_PORTS=({' '.join(str(flux_port(cfg, i)) for i in range(len(gpus)))})",
        f"DEFAULT_STAGES=({' '.join(p.name for p in select_stages(cfg))})",
        f"ALL_STAGES=({' '.join(p.name for p in stages_for(cfg))})",
    ]
    name = stage_name or phase_name
    if name:
        ph = get_stage(cfg, name)
        lines += [
            f"STAGE_NAME={ph.name}",
            f"STAGE_DESC={ph.desc!r}",
            f"STAGE_SERVERS={ph.servers}",
            f"STAGE_STEPS=({' '.join(ph.steps)})",
            f"STAGE_SERVER_STEPS=({' '.join(ph.server_steps)})",
            f"STAGE_USE_GPUS={1 if ph.use_gpus else 0}",
            f"STAGE_OPTIONAL={1 if ph.optional else 0}",
        ]
    return "\n".join(lines)


# Back-compat aliases (older imports; prefer stages_for / select_stages / get_stage).
def phases_for(cfg: dict) -> list[Phase]:
    return stages_for(cfg)


def select_phases(
    cfg: dict,
    *,
    names: list[str] | None = None,
    with_optional: bool = False,
) -> list[Phase]:
    return select_stages(cfg, names=names, with_optional=with_optional)


def get_phase(cfg: dict, name: str) -> Phase:
    return get_stage(cfg, name)


__all__ = [
    "Phase",
    "gpus_for", "n_gpus",
    "vlm_port", "flux_port",
    "vlm_urls_for", "flux_urls_for",
    "stages_for", "select_stages", "get_stage",
    "phases_for", "select_phases", "get_phase",
    "dump_shell_env",
    "dump_stage_batches",
]
