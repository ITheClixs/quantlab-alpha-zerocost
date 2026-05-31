"""Current SP500 constituents (Wikipedia parse, cached).

Labelled `survivorship_prototype_only` per spec §6.1 — current-only universe.
"""

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
    """Parse the current SP500 list from Wikipedia.

    Wikipedia blocks the default urllib User-Agent (HTTP 403), so we fetch the
    HTML via requests with a browser-style UA and hand the string to pandas.
    """
    import io

    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    df = df.rename(columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"})
    return pl.from_pandas(df[["symbol", "name", "sector"]])


def load_or_fetch_sp500(*, parquet_path: Path) -> pl.DataFrame:
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    df = _fetch_from_wikipedia()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    return df


def save_sp500_manifest(*, parquet_path: Path) -> None:
    df = load_or_fetch_sp500(parquet_path=parquet_path)
    m = DataSourceManifest(
        source_name="sp500_current",
        source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
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
            "SURVIVORSHIP-WARNED: current S&P 500 constituents only; no PIT membership history",
            "cross-sectional results carry the mandatory survivorship banner per spec §2.8",
        ],
    )
    write_manifest(parquet_path.parent / (parquet_path.stem + ".manifest.json"), m)
