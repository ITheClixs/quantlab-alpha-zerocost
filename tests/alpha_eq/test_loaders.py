"""Loaders that hard-fail on manifest hash mismatch (spec §2.8)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha_eq.data.loaders import (
    EquityRootLoader,
    LoaderHashError,
)
from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)


def _write_panel(path: Path) -> None:
    pl.DataFrame(
        {"date": ["2020-01-02"], "symbol": ["AAA"], "close": [100.0]}
    ).write_parquet(path)


def _manifest_for(art_path: Path, sha: str) -> EquityManifest:
    return EquityManifest(
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        artifacts={
            "sp500_tradable_prices": ManifestArtifact(
                path=str(art_path.name),
                sha256=sha,
                row_count=1,
                symbol_count=1,
                date_range_start="2020-01-02",
                date_range_end="2020-01-02",
                schema_fingerprint="cols:date,symbol,close",
            )
        },
        data_quality_label=DataQualityLabel.PARTIAL_PIT_UNIVERSE,
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1",
        pit_membership_source="wikipedia_fallback",
        delisting_audit_quality="partial_capture",
        delisting_audit_counters=DelistingAuditCounters(),
        build_command_line="x",
        python_version="3.11.0",
        package_versions={},
        warnings=[],
    )


def test_loader_succeeds_on_matching_hash(tmp_equity_root: Path) -> None:
    p = tmp_equity_root / "sp500_tradable_prices.parquet"
    _write_panel(p)
    sha = sha256_of_file(p)
    write_manifest(tmp_equity_root / "_manifest.json", _manifest_for(p, sha))
    loader = EquityRootLoader(root=tmp_equity_root)
    df = loader.load_tradable_prices()
    assert df.height == 1


def test_loader_hard_fails_on_corruption(tmp_equity_root: Path) -> None:
    p = tmp_equity_root / "sp500_tradable_prices.parquet"
    _write_panel(p)
    sha = sha256_of_file(p)
    write_manifest(tmp_equity_root / "_manifest.json", _manifest_for(p, sha))
    pl.DataFrame(
        {"date": ["2020-01-02"], "symbol": ["AAA"], "close": [101.0]}
    ).write_parquet(p)
    loader = EquityRootLoader(root=tmp_equity_root)
    with pytest.raises(LoaderHashError):
        loader.load_tradable_prices()


def test_loader_hard_fails_on_missing_artifact(tmp_equity_root: Path) -> None:
    write_manifest(
        tmp_equity_root / "_manifest.json",
        _manifest_for(tmp_equity_root / "sp500_tradable_prices.parquet", "a" * 64),
    )
    loader = EquityRootLoader(root=tmp_equity_root)
    with pytest.raises(FileNotFoundError):
        loader.load_tradable_prices()
