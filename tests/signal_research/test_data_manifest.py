"""5-tier data-quality classifier + sha256 manifest (spec §2.2, §2.5)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    ManifestMismatchError,
    load_and_verify_manifest,
    sha256_of_file,
    write_manifest,
)


def test_tier_values_and_ordering() -> None:
    assert DataQualityTier.PIT_SAFE.value == "pit_safe"
    assert DataQualityTier.PARTIAL_PIT_UNIVERSE.value == "partial_pit_universe"
    assert DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT.value == "public_snapshot_not_pit"
    assert DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY.value == "survivorship_prototype_only"
    # 4 explicit tiers + UNKNOWN sentinel = 5 listed in the spec
    assert DataQualityTier.UNKNOWN.value == "unknown"


def test_tier_rejects_unknown_string() -> None:
    with pytest.raises(ValueError):
        DataQualityTier("institutional_grade_marketing_word")


def test_directly_traded_etf_is_not_a_tier() -> None:
    """Per spec wording fix: directly_traded_etf is NOT a separate tier value;
    it is carried by a separate `constituent_survivorship_applicable` flag."""
    with pytest.raises(ValueError):
        DataQualityTier("directly_traded_etf")


def test_manifest_round_trip(tmp_signal_research_root: Path) -> None:
    parquet = tmp_signal_research_root / "demo.parquet"
    pl.DataFrame({"date": ["2024-01-02"], "x": [1.0]}).write_parquet(parquet)
    sha = sha256_of_file(parquet)
    m = DataSourceManifest(
        source_name="demo",
        source_url="https://example.com/demo",
        fetch_timestamp_utc="2026-05-26T12:00:00Z",
        path=str(parquet.name),
        sha256=sha,
        row_count=1,
        symbol_count=0,
        date_range_start="2024-01-02",
        date_range_end="2024-01-02",
        schema_fingerprint="cols:date,x",
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=False,
        vendor_disclosure="yfinance public snapshot — not vendor PIT data",
        timestamp_convention="after_close_t",
        warnings=[],
    )
    out = tmp_signal_research_root / "_manifest.json"
    write_manifest(out, m)
    m2 = load_and_verify_manifest(out, expected_sha256={"demo": sha})
    assert m2.data_quality_tier == DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT
    assert m2.constituent_survivorship_applicable is False


def test_manifest_hash_mismatch_hard_fails(tmp_signal_research_root: Path) -> None:
    parquet = tmp_signal_research_root / "demo.parquet"
    pl.DataFrame({"date": ["2024-01-02"], "x": [1.0]}).write_parquet(parquet)
    out = tmp_signal_research_root / "_manifest.json"
    out.write_text(
        json.dumps(
            {
                "source_name": "demo",
                "source_url": "x",
                "fetch_timestamp_utc": "2026-05-26T12:00:00Z",
                "path": "demo.parquet",
                "sha256": "a" * 64,
                "row_count": 1,
                "symbol_count": 0,
                "date_range_start": "2024-01-02",
                "date_range_end": "2024-01-02",
                "schema_fingerprint": "cols:date,x",
                "data_quality_tier": "public_snapshot_not_pit",
                "constituent_survivorship_applicable": False,
                "vendor_disclosure": "x",
                "timestamp_convention": "after_close_t",
                "warnings": [],
            }
        )
    )
    with pytest.raises(ManifestMismatchError):
        load_and_verify_manifest(out, expected_sha256={"demo": "b" * 64})
