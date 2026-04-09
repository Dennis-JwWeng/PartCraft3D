"""Pipeline v2 scheduling helpers (control plane).

Pure functions that read the ``pipeline:`` block of a config and
expose:

* :func:`gpus_for(cfg)`         — list[int]    GPU ids in the run pool
* :func:`vlm_urls_for(cfg)`     — list[str]    one /v1 url per GPU
* :func:`flux_urls_for(cfg)`    — list[str]    one /<host:port> per GPU
* :func:`phases_for(cfg)`       — list[Phase]  ordered phase definitions
* :func:`select_phases(cfg, names, with_optional)` — phase subset

These are imported by both the orchestrator (`run.py --phase`) and the
shell scheduler (which calls a tiny `python -c` to dump server
specs in shell-eval format).

The functional layer (`s1_phase1_vlm`, `s4_flux_2d`, …) never imports
this module — they only see ``ObjectContext`` and explicit url lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Phase:
    name: str
    desc: str = ""
    servers: str = "none"          # "vlm" | "flux" | "none"
    steps: list[str] = field(default_factory=list)
    use_gpus: bool = False         # multi-GPU dispatch via subprocess
    optional: bool = False


# ─────────────────── readers ──────────────────────────────────────────

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
    """Number of VLM server instances to start.

    Defaults to n_gpus(cfg) but can be capped via
    ``pipeline.n_vlm_servers`` when thread/memory limits prevent
    running one VLM per GPU.
    """
    p = _pipeline(cfg)
    explicit = p.get("n_vlm_servers")
    if explicit is not None:
        return int(explicit)
    return n_gpus(cfg)


def vlm_urls_for(cfg: dict) -> list[str]:
    """One VLM /v1 URL per server instance.

    Override via ``phase0.vlm_base_urls`` for remote or pre-started servers.
    """
    override = (cfg.get("phase0") or {}).get("vlm_base_urls")
    if override:
        return list(override)
    return [f"http://localhost:{vlm_port(cfg, i)}/v1"
            for i in range(n_vlm_servers(cfg))]


def flux_urls_for(cfg: dict) -> list[str]:
    override = (cfg.get("phase2_5") or {}).get("image_edit_base_urls")
    if override:
        return list(override)
    return [f"http://localhost:{flux_port(cfg, i)}"
            for i in range(n_gpus(cfg))]


# ─────────────────── phase plan ───────────────────────────────────────

def phases_for(cfg: dict) -> list[Phase]:
    raw_list = _pipeline(cfg).get("phases") or []
    if not raw_list:
        raise ValueError("[CONFIG] pipeline.phases is empty")
    out: list[Phase] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError(f"[CONFIG] phase entry not a dict: {entry}")
        out.append(Phase(
            name=str(entry["name"]),
            desc=str(entry.get("desc", "")),
            servers=str(entry.get("servers", "none")),
            steps=list(entry.get("steps") or []),
            use_gpus=bool(entry.get("use_gpus", False)),
            optional=bool(entry.get("optional", False)),
        ))
    return out


def select_phases(
    cfg: dict,
    *,
    names: list[str] | None = None,
    with_optional: bool = False,
) -> list[Phase]:
    """Pick which phases to run.

    * If ``names`` is given, return exactly those phases in the order
      they appear in the config (optional flag is ignored — explicit
      selection always wins).
    * Otherwise return every non-optional phase, plus optional ones if
      ``with_optional=True``.
    """
    phases = phases_for(cfg)
    if names:
        wanted = set(names)
        return [p for p in phases if p.name in wanted]
    return [p for p in phases if with_optional or not p.optional]


def get_phase(cfg: dict, name: str) -> Phase:
    for p in phases_for(cfg):
        if p.name == name:
            return p
    raise KeyError(f"phase {name!r} not in config")


# ─────────────────── shell-eval dump ──────────────────────────────────

def dump_shell_env(cfg: dict, phase_name: str | None = None) -> str:
    """Emit shell variables that the bash scheduler can ``eval``.

    Always exposes:
        GPUS=4 5 6 7
        VLM_PORTS=8002 8012 8022 8032
        FLUX_PORTS=8004 8005 8006 8007
        PHASES=A B C D D2 E F                      (default selection)

    If ``phase_name`` is set, also dumps:
        PHASE_NAME=...
        PHASE_SERVERS=vlm|flux|none
        PHASE_STEPS=s1 s2 ...
        PHASE_USE_GPUS=0|1
        PHASE_OPTIONAL=0|1
    """
    gpus = gpus_for(cfg)
    lines = [
        f"GPUS=({' '.join(str(g) for g in gpus)})",
        f"N_VLM_SERVERS={n_vlm_servers(cfg)}",
        f"VLM_PORTS=({' '.join(str(vlm_port(cfg, i)) for i in range(n_vlm_servers(cfg)))})",
        f"FLUX_PORTS=({' '.join(str(flux_port(cfg, i)) for i in range(len(gpus)))})",
        f"DEFAULT_PHASES=({' '.join(p.name for p in select_phases(cfg))})",
        f"ALL_PHASES=({' '.join(p.name for p in phases_for(cfg))})",
    ]
    if phase_name:
        ph = get_phase(cfg, phase_name)
        lines += [
            f"PHASE_NAME={ph.name}",
            f"PHASE_DESC={ph.desc!r}",
            f"PHASE_SERVERS={ph.servers}",
            f"PHASE_STEPS=({' '.join(ph.steps)})",
            f"PHASE_USE_GPUS={1 if ph.use_gpus else 0}",
            f"PHASE_OPTIONAL={1 if ph.optional else 0}",
        ]
    return "\n".join(lines)


__all__ = [
    "Phase",
    "gpus_for", "n_gpus",
    "vlm_port", "flux_port",
    "vlm_urls_for", "flux_urls_for",
    "phases_for", "select_phases", "get_phase",
    "dump_shell_env",
]
