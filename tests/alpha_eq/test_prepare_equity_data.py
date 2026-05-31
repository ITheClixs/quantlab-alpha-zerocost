"""End-to-end smoke for the equity-data prep script."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import polars as pl


def _subprocess_env() -> dict[str, str]:
    """Subprocess env that lets `uv run` resolve.  PATH is enriched with
    Homebrew so the test passes on Apple-Silicon dev machines."""
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", ".uv-cache"),
    }


def _write_minimal_inputs(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "symbol": ["A", "A", "A"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_050_000],
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")).write_parquet(
        raw / "panel.parquet"
    )
    pl.DataFrame(
        {"ex_date": ["2020-01-06"], "symbol": ["A"], "dividend_per_share": [0.5]}
    ).with_columns(pl.col("ex_date").str.strptime(pl.Date, "%Y-%m-%d")).write_parquet(
        raw / "dividends.parquet"
    )


def test_prepare_equity_data_writes_manifest_and_artifacts(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    out_root = tmp_path / "processed" / "equities"
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", "scripts/prepare_equity_data.py",
            "--panel", str(tmp_path / "raw" / "panel.parquet"),
            "--dividends", str(tmp_path / "raw" / "dividends.parquet"),
            "--equity-root", str(out_root),
            "--membership-source", "absent_prototype_only",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert (out_root / "_manifest.json").exists()
    manifest = json.loads((out_root / "_manifest.json").read_text())
    assert manifest["data_quality_label"] == "survivorship_prototype_only"
    for key in (
        "sp500_tradable_prices",
        "sp500_split_adjusted_prices",
        "sp500_total_return_prices",
        "sp500_dividends",
        "sp500_adv",
        "sp500_borrow_proxy",
        "sp500_delisting_audit",
    ):
        assert key in manifest["artifacts"]
