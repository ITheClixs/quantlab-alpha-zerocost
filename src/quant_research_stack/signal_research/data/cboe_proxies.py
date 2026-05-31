"""CBOE volatility-index proxies via yfinance (spec §2.4).

If a ticker is unavailable (e.g. ^VXN sometimes returns empty from
yfinance), the loader records this in the manifest's warnings list rather
than failing — per spec §3.3 #9 "with documented fallback".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
)
from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


@dataclass(frozen=True)
class CboeProxiesConfig:
    tickers: list[str]
    start: date
    end: date


def fetch_cboe_panel(*, config: CboeProxiesConfig) -> pl.DataFrame:
    long_cfg = LongHistoryConfig(start=config.start, end=config.end)
    out: pl.DataFrame | None = None
    fallbacks: list[str] = []
    for t in config.tickers:
        try:
            df = fetch_one_ticker(t, config=long_cfg)
        except RuntimeError as exc:
            fallbacks.append(f"{t}: {exc}")
            continue
        safe = t.replace("^", "")
        df = df.select(["date", pl.col("close").alias(f"close_{safe}")])
        out = df if out is None else out.join(df, on="date", how="full", coalesce=True)
    if out is None:
        raise RuntimeError(f"all CBOE tickers failed: {fallbacks}")
    out = out.sort("date")
    out.fallback_warnings = fallbacks  # type: ignore[attr-defined]
    return out


def save_cboe_panel(*, config: CboeProxiesConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df = fetch_cboe_panel(config=config)
    fallbacks: list[str] = list(getattr(df, "fallback_warnings", []))
    parquet_path = root / "cboe_proxies.parquet"
    df.write_parquet(parquet_path)
    m = DataSourceManifest(
        source_name="cboe_proxies",
        source_url="yfinance:cboe_proxies",
        fetch_timestamp_utc=datetime.now(UTC).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=df.height,
        symbol_count=0,
        date_range_start=str(df["date"].min()),
        date_range_end=str(df["date"].max()),
        schema_fingerprint="cols:" + ",".join(df.columns),
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=False,
        vendor_disclosure="yfinance CBOE indices (^VIX, ^VVIX, ^SKEW, ^GVZ, ^OVX, ^VXN)",
        timestamp_convention="after_close_t",
        warnings=fallbacks,
    )
    write_manifest(root / "cboe_proxies.manifest.json", m)
