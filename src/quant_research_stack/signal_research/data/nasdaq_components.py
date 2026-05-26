"""Current Nasdaq 100 constituents (spec §2.3.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


def _fetch_from_wikipedia() -> pl.DataFrame:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = set(t.columns)
        if {"Ticker", "Company"}.issubset(cols) or {"Symbol", "Company"}.issubset(cols):
            t = t.rename(columns={"Ticker": "symbol", "Symbol": "symbol", "Company": "name"})
            return pl.from_pandas(t[["symbol", "name"]])
    raise RuntimeError("Nasdaq-100 table not found on Wikipedia page")


def load_or_fetch_nasdaq_100(*, parquet_path: Path) -> pl.DataFrame:
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    df = _fetch_from_wikipedia()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    return df


def save_nasdaq_100_manifest(*, parquet_path: Path) -> None:
    df = load_or_fetch_nasdaq_100(parquet_path=parquet_path)
    m = DataSourceManifest(
        source_name="nasdaq_100_current",
        source_url="https://en.wikipedia.org/wiki/Nasdaq-100",
        fetch_timestamp_utc=datetime.now(UTC).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=df.height,
        symbol_count=int(df["symbol"].n_unique()),
        date_range_start="current",
        date_range_end="current",
        schema_fingerprint="cols:" + ",".join(df.columns),
        data_quality_tier=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        constituent_survivorship_applicable=True,
        vendor_disclosure="Wikipedia current-list parse — no historical membership reconstruction",
        timestamp_convention="snapshot_current",
        warnings=[
            "SURVIVORSHIP-WARNED: current Nasdaq 100 constituents only; no PIT membership history",
        ],
    )
    write_manifest(parquet_path.parent / (parquet_path.stem + ".manifest.json"), m)
