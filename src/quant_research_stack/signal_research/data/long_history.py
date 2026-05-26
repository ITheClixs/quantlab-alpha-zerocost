"""yfinance long-history loader with manifest emission (spec §2.1).

Adapts the pattern already used by `strategy_benchmark.data.fetch_daily_bars`
but emits per-ticker manifests in the signal_research format.
"""

from __future__ import annotations

import datetime as dt
import subprocess
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
class LongHistoryConfig:
    start: date
    end: date | None = None  # None = today


def fetch_one_ticker(ticker: str, *, config: LongHistoryConfig) -> pl.DataFrame:
    import yfinance as yf  # local import — yfinance is heavy

    end = config.end or dt.date.today()
    df = yf.download(
        ticker,
        start=config.start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"empty yfinance result for {ticker} {config.start}..{end}")
    df = df.reset_index()
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    date_col_candidates = [c for c in df.columns if c in ("Date", "Datetime", "index")]
    if not date_col_candidates:
        raise RuntimeError(f"no date column found in yfinance output for {ticker}: {list(df.columns)}")
    date_col = date_col_candidates[0]
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = pl.from_pandas(df).rename(
        {
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            close_col: "close",
            "Volume": "volume",
        }
    )
    if "Close" in out.columns and close_col != "Close":
        out = out.drop("Close")
    out = out.with_columns(
        pl.col("date").cast(pl.Date),
        pl.lit(ticker).alias("symbol"),
    ).select(["date", "symbol", "open", "high", "low", "close", "volume"])
    return out.sort("date")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_with_manifest(
    df: pl.DataFrame,
    *,
    ticker: str,
    root: Path,
    constituent_survivorship_applicable: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("=", "_").replace("^", "")
    parquet_path = root / f"{safe}.parquet"
    df.write_parquet(parquet_path)
    m = DataSourceManifest(
        source_name=ticker,
        source_url=f"yfinance://{ticker}",
        fetch_timestamp_utc=datetime.now(UTC).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=df.height,
        symbol_count=int(df["symbol"].n_unique()),
        date_range_start=str(df["date"].min()),
        date_range_end=str(df["date"].max()),
        schema_fingerprint="cols:" + ",".join(df.columns),
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=constituent_survivorship_applicable,
        vendor_disclosure="yfinance public snapshot — not vendor PIT data",
        timestamp_convention="after_close_t",
        warnings=[
            "yfinance is a public historical snapshot, not vendor-grade PIT data",
        ] + (
            ["constituent_survivorship_applicable=False per spec §2.2 directly-traded note"]
            if not constituent_survivorship_applicable else []
        ),
    )
    write_manifest(root / f"{safe}.manifest.json", m)
