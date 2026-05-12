from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILES = [
    REPO_ROOT / "manifests" / "datasets.yaml",
    REPO_ROOT / "manifests" / "models.yaml",
    REPO_ROOT / "manifests" / "papers.yaml",
    REPO_ROOT / "manifests" / "kaggle.yaml",
]


@pytest.mark.parametrize("manifest_path", MANIFEST_FILES, ids=lambda p: p.name)
def test_manifest_loads(manifest_path: Path) -> None:
    assert manifest_path.exists(), f"missing manifest: {manifest_path}"
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    assert data.get("schema_version") == 1


@pytest.mark.parametrize(
    "manifest_path,key",
    [
        (REPO_ROOT / "manifests" / "datasets.yaml", "datasets"),
        (REPO_ROOT / "manifests" / "models.yaml", "models"),
        (REPO_ROOT / "manifests" / "papers.yaml", "papers"),
        (REPO_ROOT / "manifests" / "kaggle.yaml", "items"),
    ],
    ids=lambda x: str(x),
)
def test_manifest_has_entries(manifest_path: Path, key: str) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    items = data.get(key) or []
    assert isinstance(items, list)
    assert len(items) > 0


def test_no_duplicate_ids_within_manifest() -> None:
    for manifest_path, key in [
        (REPO_ROOT / "manifests" / "datasets.yaml", "datasets"),
        (REPO_ROOT / "manifests" / "models.yaml", "models"),
        (REPO_ROOT / "manifests" / "kaggle.yaml", "items"),
    ]:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        ids = [item["id"] for item in data.get(key, []) if "id" in item]
        assert len(ids) == len(set(ids)), f"duplicate ids in {manifest_path}"
