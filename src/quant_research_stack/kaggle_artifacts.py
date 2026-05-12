from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_research_stack.artifacts import read_yaml, safe_repo_id

VALID_RESOURCE_TYPES = frozenset({"competition", "dataset"})

safe_kaggle_dir_name = safe_repo_id


@dataclass(frozen=True)
class KaggleItem:
    id: str
    resource_type: str
    group: str
    priority: int
    topics: tuple[str, ...]
    purpose: str
    license_hint: str | None
    expected_max_gb: float | None
    enabled: bool


def load_kaggle_items(manifest_path: str | Path) -> list[KaggleItem]:
    manifest = read_yaml(manifest_path)
    defaults = manifest.get("defaults", {}) or {}
    items: list[KaggleItem] = []
    for raw in manifest.get("items", []) or []:
        merged = {**defaults, **raw}
        resource_type = merged.get("resource_type", "competition")
        if resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError(
                f"Invalid resource_type {resource_type!r} for item {merged.get('id')!r}"
            )
        expected_max_gb = merged.get("expected_max_gb")
        items.append(
            KaggleItem(
                id=str(merged["id"]),
                resource_type=resource_type,
                group=str(merged.get("group", "ungrouped")),
                priority=int(merged.get("priority", 9999)),
                topics=tuple(merged.get("topics", []) or []),
                purpose=str(merged.get("purpose", "")),
                license_hint=merged.get("license_hint"),
                expected_max_gb=float(expected_max_gb) if expected_max_gb is not None else None,
                enabled=bool(merged.get("enabled", True)),
            )
        )
    return items


def local_path_for(item: KaggleItem, root: str | Path) -> Path:
    base = Path(root)
    if item.resource_type == "competition":
        return base / "competitions" / item.id
    return base / "datasets" / safe_kaggle_dir_name(item.id)
