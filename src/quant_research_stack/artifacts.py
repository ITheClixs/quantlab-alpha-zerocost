from __future__ import annotations

import fnmatch
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GB = 1024**3


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def safe_repo_id(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id)


def bytes_to_gb(size_bytes: int | None) -> float | None:
    if size_bytes is None:
        return None
    return round(size_bytes / GB, 4)


def folder_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def pattern_match(path: str, patterns: Iterable[str] | None) -> bool:
    patterns = list(patterns or [])
    if not patterns:
        return True
    name = path.replace(os.sep, "/")
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(Path(name).name, pattern) for pattern in patterns)


def should_include(path: str, allow_patterns: Iterable[str] | None, ignore_patterns: Iterable[str] | None) -> bool:
    if ignore_patterns and pattern_match(path, ignore_patterns):
        return False
    return pattern_match(path, allow_patterns)


@dataclass(frozen=True)
class ManifestItem:
    id: str
    repo_type: str
    group: str
    priority: int
    enabled: bool
    allow_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def local_name(self) -> str:
        return safe_repo_id(self.id)


def load_manifest_items(path: str | Path, key: str) -> list[ManifestItem]:
    manifest = read_yaml(path)
    defaults = manifest.get("defaults", {})
    items: list[ManifestItem] = []
    for raw_item in manifest.get(key, []):
        merged = {**defaults, **raw_item}
        items.append(
            ManifestItem(
                id=merged["id"],
                repo_type=merged.get("repo_type", defaults.get("repo_type", key.rstrip("s"))),
                group=merged.get("group", "ungrouped"),
                priority=int(merged.get("priority", 9999)),
                enabled=bool(merged.get("enabled", True)),
                allow_patterns=tuple(merged.get("allow_patterns", defaults.get("allow_patterns", [])) or []),
                ignore_patterns=tuple(merged.get("ignore_patterns", defaults.get("ignore_patterns", [])) or []),
                metadata={k: v for k, v in merged.items() if k not in {"id", "repo_type", "group", "priority", "enabled", "allow_patterns", "ignore_patterns"}},
            )
        )
    return items

