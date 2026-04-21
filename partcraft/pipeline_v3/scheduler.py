"""Pipeline v2 scheduling helpers (control plane).

Pure functions that read the ``pipeline:`` block and ``services`` of a config.

* :func:`stages_for(cfg)` — list[Phase] ordered stage definitions (``pipeline.stages``)
* :func:`select_stages(cfg, names, with_optional)` — stage subset
* :func:`gpus_for` / :func:`vlm_urls_for` / :func:`flux_urls_for` — hardware + URL lists
* :func:`hooks_for(cfg)` — list[:class:`Hook`] post-stage hooks (``pipeline.hooks``)

Imported by :mod:`run` (``dump_shell_env``).
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
    # Sub-chain support inside a parallel_group: stages sharing the same
    # non-empty ``chain_id`` form a sequential sub-chain within the group's
    # batch (sorted by ``chain_order``). Different chains in the same group
    # run in parallel; stages without ``chain_id`` are treated as singleton
    # chains. ``chain_id`` is meaningful only together with ``parallel_group``.
    chain_id: str = ""
    chain_order: int = 0


_ALLOWED_HOOK_USES = frozenset({"cpu", "none"})
_ALLOWED_HOOK_FIELDS = frozenset({
    "name", "after_stage", "uses", "command", "env_passthrough",
})


@dataclass
class Hook:
    """One post-stage hook row from ``pipeline.hooks`` (spec 2026-04-21)."""

    name: str
    after_stage: str
    uses: str  # "cpu" | "none" in v1
    command: list[str] = field(default_factory=list)
    env_passthrough: list[str] = field(default_factory=list)


def hooks_for(cfg: dict) -> list[Hook]:
    """Parse ``pipeline.hooks`` into a list of :class:`Hook`.

    Returns ``[]`` when the block is absent. Validates:
      * every required field is present;
      * ``after_stage`` names an existing stage in ``pipeline.stages``;
      * ``uses`` is one of ``cpu`` / ``none`` (v1; ``gpu`` is reserved);
      * no unknown keys (guard against typos like ``timeout``);
      * hook names do not collide with stage names.
    """
    p = _pipeline(cfg)
    raw = p.get("hooks")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("[CONFIG] pipeline.hooks: must be a list")

    stage_names = {ph.name for ph in stages_for(cfg)}
    out: list[Hook] = []
    seen_names: set[str] = set()

    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"[CONFIG] pipeline.hooks[{idx}] not a mapping: {entry!r}")

        unknown = set(entry) - _ALLOWED_HOOK_FIELDS
        if unknown:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] unknown fields: {sorted(unknown)}; "
                f"allowed: {sorted(_ALLOWED_HOOK_FIELDS)}"
            )
        for req in ("name", "after_stage", "uses", "command"):
            if req not in entry:
                raise ValueError(
                    f"[CONFIG] pipeline.hooks[{idx}] missing required field {req!r}"
                )

        name = str(entry["name"])
        after_stage = str(entry["after_stage"])
        uses = str(entry["uses"])
        cmd_raw = entry["command"]
        if (
            not isinstance(cmd_raw, list)
            or not cmd_raw
            or not all(isinstance(c, str) for c in cmd_raw)
        ):
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] command must be a non-empty list of strings"
            )
        command = list(cmd_raw)

        env_raw = entry.get("env_passthrough")
        if env_raw is None:
            env_passthrough: list[str] = []
        elif not isinstance(env_raw, list) or not all(isinstance(v, str) for v in env_raw):
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] env_passthrough must be a list of strings"
            )
        else:
            env_passthrough = list(env_raw)
        if uses not in _ALLOWED_HOOK_USES:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] uses={uses!r}; allowed v1: "
                f"{sorted(_ALLOWED_HOOK_USES)} (gpu reserved for follow-up spec)"
            )
        if after_stage not in stage_names:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] after_stage={after_stage!r} is not a "
                f"declared stage; known stages: {sorted(stage_names)}"
            )
        if name in stage_names:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] name={name!r} collides with an "
                "existing stage name"
            )
        if name in seen_names:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] duplicate hook name {name!r}"
            )
        seen_names.add(name)

        out.append(Hook(
            name=name,
            after_stage=after_stage,
            uses=uses,
            command=command,
            env_passthrough=env_passthrough,
        ))
    return out


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
            chain_id=str(entry.get("chain_id", "")),
            chain_order=int(entry.get("chain_order", 0)),
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


def dump_stage_chains(
    cfg: dict,
    stage_names: list[str],
) -> list[list[list[str]]]:
    """Group stage_names into ordered batches → parallel chains → serial stages.

    Topology:
    - **Batch** (top): runs serially after the previous batch finishes.
      A batch is created from a ``parallel_group`` (or one solo stage).
    - **Chain** (middle): chains within the same batch run concurrently.
      A chain is created per non-empty ``chain_id`` inside a group, plus one
      singleton chain per stage without ``chain_id``.
    - **Stage** (inner): stages within a chain run sequentially, sorted by
      ``chain_order`` (then by config order).

    The output preserves the original relative order of ``stage_names`` for
    chain placement (then ``chain_order`` re-sorts inside a chain).

    Examples::

        # Flat sequence (no groups, no chains)
        dump_stage_chains(cfg, ["A", "B", "C"])
        → [[["A"]], [["B"]], [["C"]]]

        # Two stages in same parallel_group, no chains
        dump_stage_chains(cfg, ["A", "D", "D2", "E"])  # D, D2 share group "x"
        → [[["A"]], [["D"], ["D2"]], [["E"]]]

        # Sub-chain inside a parallel_group
        # group=g  chains: del → [del_mesh]; flux → [flux_2d, trellis_preview]
        dump_stage_chains(cfg, ["A", "del_mesh", "flux_2d", "trellis_preview"])
        → [[["A"]], [["del_mesh"], ["flux_2d", "trellis_preview"]]]
    """
    by_name = {ph.name: ph for ph in stages_for(cfg)}
    batches: list[list[list[str]]] = []
    group_to_idx: dict[str, int] = {}                  # parallel_group → batch index
    chain_to_pos: dict[tuple[int, str], int] = {}      # (batch_idx, chain_id) → chain idx
    server_count_per_chain: dict[tuple[int, int], int] = {}  # (batch, chain) → #server stages

    log = logging.getLogger("scheduler")

    for name in stage_names:
        ph = by_name.get(name)
        if ph is None:
            log.warning("[scheduler] stage %r not in config — skipping", name)
            continue

        group = ph.parallel_group or ""
        chain_id = ph.chain_id or ""
        needs_servers = ph.servers != "none"

        # Resolve target batch.
        if group and group in group_to_idx:
            batch_idx = group_to_idx[group]
        else:
            batch_idx = len(batches)
            batches.append([])
            if group:
                group_to_idx[group] = batch_idx

        # Resolve target chain inside the batch.
        if chain_id:
            key = (batch_idx, chain_id)
            if key in chain_to_pos:
                chain_idx = chain_to_pos[key]
            else:
                chain_idx = len(batches[batch_idx])
                batches[batch_idx].append([])
                chain_to_pos[key] = chain_idx
        else:
            chain_idx = len(batches[batch_idx])
            batches[batch_idx].append([])

        # Safety: at most one server-backed stage per chain — chains that need
        # the same external server type would collide on ports if overlapped.
        if needs_servers:
            count = server_count_per_chain.get((batch_idx, chain_idx), 0)
            if count >= 1:
                log.warning(
                    "[scheduler] stage %s adds a 2nd server-backed step to "
                    "chain %r — server lifecycle is per-stage so this is OK, "
                    "but verify the two stages use distinct server types.",
                    name, chain_id or f"<solo:{name}>",
                )
            server_count_per_chain[(batch_idx, chain_idx)] = count + 1

        batches[batch_idx][chain_idx].append(name)

    # Sort stages inside each chain by chain_order (stable for ties).
    for batch in batches:
        for chain in batch:
            chain.sort(key=lambda n: by_name[n].chain_order)

    # Post-stage hooks (spec 2026-04-21): append each hook to the chain
    # ending with its after_stage. Hooks whose after_stage is not the
    # tail of any chain in the selected run are silently dropped — this
    # inherits upstream stage selection (§4.2 of the spec).
    hooks = hooks_for(cfg)
    if hooks:
        by_after: dict[str, list[Hook]] = {}
        for h in hooks:
            by_after.setdefault(h.after_stage, []).append(h)
        for batch in batches:
            for chain in batch:
                last = chain[-1] if chain else None
                if last in by_after:
                    if len(by_after[last]) > 1:
                        log.warning(
                            "[scheduler] %d hooks share after_stage=%s; they "
                            "will run sequentially in declaration order.",
                            len(by_after[last]), last,
                        )
                    for h in by_after[last]:
                        chain.append(f"{h.name}@hook")

    return batches


def format_stage_chains_text(batches: list[list[list[str]]]) -> str:
    """Serialise nested chain structure for the shell driver.

    Format (one batch per line):
    - Chains separated by ``|``
    - Stages within a chain separated by ``>``

    Example::

        text_gen_gate_a
        del_mesh|flux_2d>trellis_preview
        gate_quality
    """
    lines = []
    for batch in batches:
        chain_strs = [">".join(chain) for chain in batch]
        lines.append("|".join(chain_strs))
    return "\n".join(lines)


def dump_stage_batches(
    cfg: dict,
    stage_names: list[str],
) -> list[list[str]]:
    """Backward-compat wrapper: flatten chains within each batch.

    Prefer :func:`dump_stage_chains` for new callers — it preserves the
    intra-batch chain (sequential) structure that this function loses.

    The output preserves the original relative order of stage_names.

    Example::

        dump_stage_batches(cfg, ["A", "C", "D", "D2", "E"])
        → [["A"], ["C"], ["D", "D2"], ["E"]]
    """
    chains_by_batch = dump_stage_chains(cfg, stage_names)
    out: list[list[str]] = []
    for batch in chains_by_batch:
        flat: list[str] = []
        for chain in batch:
            flat.extend(chain)
        out.append(flat)
    return out


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
    "dump_stage_chains",
    "format_stage_chains_text",
]
