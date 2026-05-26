"""FRED loader via fredapi (spec §2.4).

ALFRED (revision-adjusted) integration is a Phase-2 upgrade; v1 uses plain
FRED which carries `public_snapshot_not_pit` as the data-quality tier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


@dataclass(frozen=True)
class FredConfig:
    start: date
    end: date
    api_key: str | None = None  # falls back to FRED_API_KEY env var


def _fred_get_series(series_id: str, *, api_key: str | None = None,
                     start: str | None = None, end: str | None = None) -> pd.Series:
    """Thin wrapper around fredapi.Fred — extracted to make the loader
    monkeypatchable in tests."""
    from fredapi import Fred  # local import — keeps the module import cheap
    fred = Fred(api_key=api_key or os.environ.get("FRED_API_KEY"))
    return fred.get_series(series_id, observation_start=start, observation_end=end)


def fetch_fred_series(series_id: str, *, config: FredConfig) -> pl.DataFrame:
    s = _fred_get_series(
        series_id,
        api_key=config.api_key,
        start=config.start.isoformat(),
        end=config.end.isoformat(),
    )
    df = s.reset_index()
    df.columns = ["date", series_id]
    out = pl.from_pandas(df).with_columns(pl.col("date").cast(pl.Date))
    return out.sort("date")


def save_fred_panel(*, series_ids: list[str], config: FredConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    panel: pl.DataFrame | None = None
    for sid in series_ids:
        df = fetch_fred_series(sid, config=config)
        panel = df if panel is None else panel.join(df, on="date", how="full", coalesce=True)
    assert panel is not None
    panel = panel.sort("date")
    parquet_path = root / "fred_features.parquet"
    panel.write_parquet(parquet_path)
    m = DataSourceManifest(
        source_name="fred_features",
        source_url="https://api.stlouisfed.org/fred",
        fetch_timestamp_utc=datetime.now(UTC).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=panel.height,
        symbol_count=0,
        date_range_start=str(panel["date"].min()),
        date_range_end=str(panel["date"].max()),
        schema_fingerprint="cols:" + ",".join(panel.columns),
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=False,
        vendor_disclosure=f"FRED public API; v1 plain (not revision-adjusted ALFRED). Series: {series_ids}",
        timestamp_convention="release_date_approximation",
        warnings=[
            "FRED data revisions can affect past values; v1 uses plain FRED not ALFRED",
            "release_date_approximation timestamps may not perfectly align with intra-day publication",
        ],
    )
    write_manifest(root / "fred_features.manifest.json", m)
