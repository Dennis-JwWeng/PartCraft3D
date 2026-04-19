"""Canonical, pipeline-version-agnostic representation of one promotable edit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PassResult:
    passed: bool
    score: float | None = None
    producer: str = ""
    reason: str = ""
    ts: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "pass": self.passed, "score": self.score,
            "producer": self.producer, "reason": self.reason, "ts": self.ts,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PassResult":
        return cls(
            passed=bool(d["pass"]),
            score=d.get("score"),
            producer=d.get("producer", ""),
            reason=d.get("reason", ""),
            ts=d.get("ts", ""),
            extra=dict(d.get("extra") or {}),
        )


@dataclass
class PromotionRecord:
    obj_id: str
    shard: str
    edit_id: str
    edit_type: str
    source_pipeline: str           # "v2" | "v3"
    source_run_tag: str
    source_run_dir: Path
    spec: dict[str, Any]
    passes: dict[str, PassResult]
    after_glb: Path | None
    after_npz: Path | None
    preview_pngs: list[Path]

    def is_deletion(self) -> bool:
        return self.edit_type == "deletion"

    def is_flux_branch(self) -> bool:
        return self.edit_type in {"modification", "scale", "material",
                                   "color", "glb", "addition"}

    def to_qc_json(self, *, promoted_at: str) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "source": {
                "pipeline_version": self.source_pipeline,
                "run_tag": self.source_run_tag,
                "run_dir": str(self.source_run_dir),
                "promoted_at": promoted_at,
            },
            "passes": {name: pr.to_json() for name, pr in self.passes.items()},
        }


def evaluate_rule(
    passes: dict[str, PassResult],
    rule: dict[str, Any],
    *,
    edit_type: str | None = None,
) -> tuple[bool, str]:
    allowed = rule.get("edit_types_allowed") or []
    if allowed and edit_type is not None and edit_type not in allowed:
        return False, f"disallowed_type: {edit_type}"
    required = list(rule.get("required_passes", []))
    missing = [name for name in required if name not in passes]
    if missing:
        return False, f"missing: {','.join(missing)}"
    failing = [name for name in required if not passes[name].passed]
    if failing:
        return False, f"failed: {','.join(failing)}"
    return True, ""
