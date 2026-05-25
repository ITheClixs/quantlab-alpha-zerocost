"""CLI smoke for pit_quality_audit.py — re-runs the classifier and prints
a markdown summary."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)


def _subprocess_env() -> dict[str, str]:
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
    }


def _seed_root(root: Path) -> None:
    pl.DataFrame({"date": ["2020-01-02"], "symbol": ["AAA"], "close": [1.0]}).write_parquet(
        root / "sp500_tradable_prices.parquet"
    )
    sha = sha256_of_file(root / "sp500_tradable_prices.parquet")
    m = EquityManifest(
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        artifacts={
            "sp500_tradable_prices": ManifestArtifact(
                path="sp500_tradable_prices.parquet",
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
    write_manifest(root / "_manifest.json", m)


def test_pit_quality_audit_cli_prints_label(tmp_equity_root: Path) -> None:
    _seed_root(tmp_equity_root)
    res = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/pit_quality_audit.py",
            "--equity-root",
            str(tmp_equity_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert "partial_pit_universe" in res.stdout
