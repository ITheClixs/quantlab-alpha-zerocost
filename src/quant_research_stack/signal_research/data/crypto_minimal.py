"""Crypto minimal v1 loader (spec §6.6).

Source: Binance public klines (daily). Spot pairs only in v1 — no perpetuals
or funding rates. Carries `public_snapshot_not_pit` +
`constituent_survivorship_applicable: false`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


@dataclass(frozen=True)
class CryptoMinimalConfig:
    tickers: list[str]
    start: date
    end: date


def _fetch_binance_klines(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Real fetcher (placeholder import path — concrete adapter chosen at
    plan execution time per spec §6.8 open question)."""
    raise NotImplementedError(
        "Concrete Binance public-data adapter is selected at execution time; "
        "see spec §6.8 open question."
    )


def save_crypto_minimal(*, config: CryptoMinimalConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for t in config.tickers:
        df = _fetch_binance_klines(t, config.start.isoformat(), config.end.isoformat())
        parquet_path = root / f"{t}_daily.parquet"
        df.write_parquet(parquet_path)
        m = DataSourceManifest(
            source_name=t,
            source_url=f"binance:public_klines:{t}:1d",
            fetch_timestamp_utc=datetime.now(UTC).isoformat(),
            path=parquet_path.name,
            sha256=sha256_of_file(parquet_path),
            row_count=df.height,
            symbol_count=1,
            date_range_start=str(df["date"].min()),
            date_range_end=str(df["date"].max()),
            schema_fingerprint="cols:" + ",".join(df.columns),
            data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
            constituent_survivorship_applicable=False,
            vendor_disclosure="Binance public klines — spot pair only; no perpetuals or funding in v1",
            timestamp_convention="utc_daily_close",
            warnings=[
                "spot-only v1; perpetual + funding-rate strategies deferred",
            ],
        )
        write_manifest(root / f"{t}_daily.manifest.json", m)
