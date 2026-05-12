from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_research_stack.artifacts import GB, bytes_to_gb, folder_size
from quant_research_stack.budget import ArtifactBudget
from quant_research_stack.kaggle_artifacts import KaggleItem, local_path_for


@dataclass(frozen=True)
class KaggleDownloadDecision:
    item: KaggleItem
    local_dir: Path
    local_size_bytes: int
    estimated_size_bytes: int
    decision: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.item.id,
            "resource_type": self.item.resource_type,
            "group": self.item.group,
            "priority": self.item.priority,
            "purpose": self.item.purpose,
            "local_dir": str(self.local_dir),
            "local_size_bytes": self.local_size_bytes,
            "local_size_gb": bytes_to_gb(self.local_size_bytes),
            "estimated_size_bytes": self.estimated_size_bytes,
            "estimated_size_gb": bytes_to_gb(self.estimated_size_bytes),
            "decision": self.decision,
        }


def expected_size_bytes(item: KaggleItem) -> int:
    if item.expected_max_gb is None:
        return 0
    return int(float(item.expected_max_gb) * GB)


def build_kaggle_plan(items: list[KaggleItem], raw_root: str | Path, budget: ArtifactBudget, *, force: bool = False) -> list[KaggleDownloadDecision]:
    remaining = budget.remaining_bytes
    decisions: list[KaggleDownloadDecision] = []
    for item in sorted((item for item in items if item.enabled), key=lambda row: (row.priority, row.id)):
        local_dir = local_path_for(item, raw_root)
        local_size = folder_size(local_dir)
        estimated = expected_size_bytes(item)
        if local_size > 0 and not force:
            decision = "skip_present"
        elif estimated <= 0:
            decision = "skip_unknown_size"
        elif estimated > remaining:
            decision = "skip_budget"
        else:
            decision = "download"
            remaining -= estimated
        decisions.append(
            KaggleDownloadDecision(
                item=item,
                local_dir=local_dir,
                local_size_bytes=local_size,
                estimated_size_bytes=estimated,
                decision=decision,
            )
        )
    return decisions


def kaggle_download_command(decision: KaggleDownloadDecision, *, unzip: bool = False) -> list[str]:
    item = decision.item
    if item.resource_type == "competition":
        cmd = ["kaggle", "competitions", "download", "-c", item.id, "-p", str(decision.local_dir)]
    elif item.resource_type == "dataset":
        cmd = ["kaggle", "datasets", "download", "-d", item.id, "-p", str(decision.local_dir)]
    else:
        raise ValueError(f"Unsupported Kaggle resource_type: {item.resource_type}")
    if unzip:
        cmd.append("--unzip")
    return cmd


def run_kaggle_download(decision: KaggleDownloadDecision, *, unzip: bool = False) -> None:
    decision.local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(kaggle_download_command(decision, unzip=unzip), check=True)
