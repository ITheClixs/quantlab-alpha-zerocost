from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_research_stack.artifacts import GB, bytes_to_gb, folder_size


class BudgetExceededError(RuntimeError):
    """Raised when a planned artifact operation would exceed the configured cap."""


@dataclass(frozen=True)
class ArtifactBudget:
    max_total_gb: float
    hard_ceiling_gb: float
    counted_paths: tuple[Path, ...]
    used_bytes: int

    @property
    def max_total_bytes(self) -> int:
        return int(self.max_total_gb * GB)

    @property
    def hard_ceiling_bytes(self) -> int:
        return int(self.hard_ceiling_gb * GB)

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.max_total_bytes - self.used_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_total_gb": self.max_total_gb,
            "hard_ceiling_gb": self.hard_ceiling_gb,
            "used_bytes": self.used_bytes,
            "used_gb": bytes_to_gb(self.used_bytes),
            "remaining_bytes": self.remaining_bytes,
            "remaining_gb": bytes_to_gb(self.remaining_bytes),
            "counted_paths": [str(path) for path in self.counted_paths],
        }


def load_artifact_budget(config: dict[str, Any], repo_root: str | Path = ".") -> ArtifactBudget:
    artifact_config = config.get("artifact_budget", {}) or {}
    root = Path(repo_root)
    counted_paths = tuple(root / Path(path) for path in artifact_config.get("counted_paths", []) or [])
    used_bytes = sum(folder_size(path) for path in counted_paths)
    max_total_gb = float(artifact_config.get("max_total_gb", 150))
    hard_ceiling_gb = float(artifact_config.get("hard_ceiling_gb", max_total_gb))
    return ArtifactBudget(
        max_total_gb=max_total_gb,
        hard_ceiling_gb=hard_ceiling_gb,
        counted_paths=counted_paths,
        used_bytes=used_bytes,
    )


def path_size_report(config: dict[str, Any], repo_root: str | Path = ".") -> list[dict[str, Any]]:
    root = Path(repo_root)
    counted = config.get("artifact_budget", {}).get("counted_paths", []) or []
    rows = []
    for raw_path in counted:
        path = root / Path(raw_path)
        size = folder_size(path)
        rows.append(
            {
                "path": raw_path,
                "exists": path.exists(),
                "size_bytes": size,
                "size_gb": bytes_to_gb(size),
            }
        )
    return rows


def ensure_budget_available(budget: ArtifactBudget, planned_bytes: int, *, label: str) -> None:
    projected = budget.used_bytes + planned_bytes
    if projected > budget.max_total_bytes:
        raise BudgetExceededError(
            f"{label} would exceed artifact budget: "
            f"used={bytes_to_gb(budget.used_bytes)} GB, "
            f"planned={bytes_to_gb(planned_bytes)} GB, "
            f"max={budget.max_total_gb} GB"
        )
    if projected > budget.hard_ceiling_bytes:
        raise BudgetExceededError(
            f"{label} would exceed hard ceiling: "
            f"projected={bytes_to_gb(projected)} GB, "
            f"hard_ceiling={budget.hard_ceiling_gb} GB"
        )
