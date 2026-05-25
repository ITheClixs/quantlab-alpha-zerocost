"""Manifest schema, hash, and label tests (spec §2.7, §2.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    EquityManifest,
    ManifestArtifact,
    ManifestMismatchError,
    load_and_verify_manifest,
    write_manifest,
)


def test_data_quality_label_values() -> None:
    assert DataQualityLabel("pit_safe").value == "pit_safe"
    assert DataQualityLabel("partial_pit_universe").value == "partial_pit_universe"
    assert DataQualityLabel("survivorship_prototype_only").value == "survivorship_prototype_only"


def test_data_quality_label_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        DataQualityLabel("institutional_grade_marketing_word")


def test_manifest_artifact_required_fields() -> None:
    art = ManifestArtifact(
        path="sp500_tradable_prices.parquet",
        sha256="a" * 64,
        row_count=10,
        symbol_count=2,
        date_range_start="2020-01-02",
        date_range_end="2020-01-15",
        schema_fingerprint="cols:date,symbol,open,high,low,close,volume",
    )
    assert art.row_count == 10


def test_write_and_load_manifest_round_trip(tmp_equity_root: Path) -> None:
    art = ManifestArtifact(
        path="sp500_tradable_prices.parquet",
        sha256="b" * 64,
        row_count=10,
        symbol_count=2,
        date_range_start="2020-01-02",
        date_range_end="2020-01-15",
        schema_fingerprint="cols:date,symbol,open,high,low,close,volume",
    )
    m = EquityManifest(
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        artifacts={"sp500_tradable_prices": art},
        data_quality_label=DataQualityLabel("partial_pit_universe"),
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1",
        pit_membership_source="wikipedia_fallback",
        delisting_audit_quality="partial_capture",
        delisting_audit_counters={
            "delisted_captured": 12,
            "delisted_missing": 1,
            "merger_captured": 8,
            "merger_missing": 0,
            "ticker_changed": 5,
            "unknown_exit": 0,
        },
        build_command_line="prepare_equity_data.py --config configs/alpha_eq.yaml",
        python_version="3.11.x",
        package_versions={"polars": "x.y", "lightgbm": "x.y"},
        warnings=["dividend feed: public_snapshot_not_vendor_pit"],
    )
    out = tmp_equity_root / "_manifest.json"
    write_manifest(out, m)
    m2 = load_and_verify_manifest(out, expected_sha256={"sp500_tradable_prices": "b" * 64})
    assert m2.data_quality_label.value == "partial_pit_universe"
    assert m2.artifacts["sp500_tradable_prices"].sha256 == "b" * 64


def test_load_and_verify_manifest_hard_fails_on_hash_mismatch(tmp_equity_root: Path) -> None:
    out = tmp_equity_root / "_manifest.json"
    out.write_text(
        json.dumps(
            {
                "pipeline_version": "0.1.0",
                "git_sha": "x",
                "artifacts": {
                    "a": {
                        "path": "a.parquet",
                        "sha256": "a" * 64,
                        "row_count": 1,
                        "symbol_count": 1,
                        "date_range_start": "2020-01-02",
                        "date_range_end": "2020-01-02",
                        "schema_fingerprint": "x",
                    }
                },
                "data_quality_label": "pit_safe",
                "corporate_action_quality": "vendor_total_return",
                "borrow_source_quality": "static_proxy_v1",
                "pit_membership_source": "hf:andyqin18/sp500-historical-membership",
                "delisting_audit_quality": "captured_above_threshold",
                "delisting_audit_counters": {
                    "delisted_captured": 0,
                    "delisted_missing": 0,
                    "merger_captured": 0,
                    "merger_missing": 0,
                    "ticker_changed": 0,
                    "unknown_exit": 0,
                },
                "build_command_line": "x",
                "python_version": "3.11.0",
                "package_versions": {},
                "warnings": [],
            }
        )
    )
    with pytest.raises(ManifestMismatchError):
        load_and_verify_manifest(out, expected_sha256={"a": "b" * 64})


def test_manifest_required_fields_missing(tmp_equity_root: Path) -> None:
    out = tmp_equity_root / "_manifest.json"
    out.write_text(json.dumps({"git_sha": "x"}))
    with pytest.raises(ManifestMismatchError):
        load_and_verify_manifest(out, expected_sha256={})
