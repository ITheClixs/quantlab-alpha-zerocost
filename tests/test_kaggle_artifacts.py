from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from quant_research_stack.kaggle_artifacts import (
    KaggleItem,
    load_kaggle_items,
    local_path_for,
    safe_kaggle_dir_name,
)


def write_manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "kaggle.yaml"
    path.write_text(dedent(body), encoding="utf-8")
    return path


def test_load_kaggle_items_parses_competition(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        defaults:
          resource_type: competition
          enabled: true
          expected_max_gb: 5.0
        items:
          - id: jpx-tokyo-stock-exchange-prediction
            group: equity_jpx
            priority: 10
            topics: [equity]
            purpose: test
            license_hint: kaggle_competition
        """,
    )
    items = load_kaggle_items(manifest)
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, KaggleItem)
    assert item.id == "jpx-tokyo-stock-exchange-prediction"
    assert item.resource_type == "competition"
    assert item.priority == 10
    assert item.topics == ("equity",)
    assert item.expected_max_gb == 5.0
    assert item.enabled is True


def test_load_kaggle_items_parses_dataset(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        items:
          - id: jakewright/9000-tickers-of-stock-market-data-full-history
            resource_type: dataset
            group: equity_ohlcv_us
            priority: 200
            topics: [equity]
            purpose: test
            license_hint: cc0
            expected_max_gb: 4.0
            enabled: true
        """,
    )
    items = load_kaggle_items(manifest)
    assert len(items) == 1
    assert items[0].resource_type == "dataset"


def test_load_kaggle_items_skips_disabled_when_filtered(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        defaults:
          resource_type: competition
        items:
          - id: a
            group: g
            priority: 1
            topics: [equity]
            purpose: t
            enabled: false
          - id: b
            group: g
            priority: 2
            topics: [equity]
            purpose: t
            enabled: true
        """,
    )
    all_items = load_kaggle_items(manifest)
    enabled = [item for item in all_items if item.enabled]
    assert {item.id for item in all_items} == {"a", "b"}
    assert {item.id for item in enabled} == {"b"}


def test_safe_kaggle_dir_name_replaces_separators() -> None:
    assert safe_kaggle_dir_name("jakewright/9000-tickers-of-stock-market-data-full-history") == "jakewright__9000-tickers-of-stock-market-data-full-history"
    assert safe_kaggle_dir_name("jpx-tokyo-stock-exchange-prediction") == "jpx-tokyo-stock-exchange-prediction"


def test_local_path_for_competition_uses_competitions_subdir(tmp_path: Path) -> None:
    item = KaggleItem(
        id="jpx-tokyo-stock-exchange-prediction",
        resource_type="competition",
        group="equity_jpx",
        priority=10,
        topics=("equity",),
        purpose="t",
        license_hint=None,
        expected_max_gb=None,
        enabled=True,
    )
    path = local_path_for(item, tmp_path)
    assert path == tmp_path / "competitions" / "jpx-tokyo-stock-exchange-prediction"


def test_local_path_for_dataset_uses_datasets_subdir(tmp_path: Path) -> None:
    item = KaggleItem(
        id="jakewright/9000-tickers-of-stock-market-data-full-history",
        resource_type="dataset",
        group="equity_ohlcv_us",
        priority=200,
        topics=("equity",),
        purpose="t",
        license_hint="cc0",
        expected_max_gb=4.0,
        enabled=True,
    )
    path = local_path_for(item, tmp_path)
    assert path == tmp_path / "datasets" / "jakewright__9000-tickers-of-stock-market-data-full-history"


def test_load_kaggle_items_rejects_invalid_resource_type(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        items:
          - id: bad
            resource_type: notebook
            group: g
            priority: 1
            topics: [equity]
            purpose: t
        """,
    )
    with pytest.raises(ValueError, match="resource_type"):
        load_kaggle_items(manifest)
