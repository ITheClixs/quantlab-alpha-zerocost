"""VRP data loader — fetches the CBOE volatility-index family and the
underlying SPY/QQQ bars via yfinance.

Non-tradable instruments (^VIX, ^VIX9D, ^VIX3M, ^VVIX, ^SKEW, ^VXN) are
labeled as FEATURE-ONLY in the manifest. Tradable instruments (SPY, QQQ)
carry the directly-traded-instrument flag.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
)

VIX_FAMILY_TICKERS: tuple[str, ...] = (
    "^VIX",
    "^VIX9D",
    "^VIX3M",
    "^VVIX",
    "^SKEW",
    "^VXN",
)

UNDERLYING_TICKERS: tuple[str, ...] = ("SPY", "QQQ")


@dataclass(frozen=True)
class VRPFetchResult:
    underlying: pl.DataFrame  # date, symbol, open, high, low, close, volume
    vol_features: pl.DataFrame  # date + one column per non-tradable index
    fetched_underlying: list[str] = field(default_factory=list)
    fetched_vol: list[str] = field(default_factory=list)
    missing_vol: list[str] = field(default_factory=list)


def _cache_path(root: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "IDX_").replace("=", "_")
    return root / f"{safe}.parquet"


def _load_or_fetch(
    *, ticker: str, start: dt.date, end: dt.date, cache_root: Path
) -> pl.DataFrame | None:
    p = _cache_path(cache_root, ticker)
    if p.exists():
        df = pl.read_parquet(p)
        if df.height > 0:
            return df
    try:
        df = fetch_one_ticker(
            ticker, config=LongHistoryConfig(start=start, end=end)
        )
    except Exception:
        return None
    if df.is_empty():
        return None
    cache_root.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    return df


def _normalize_one(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    def col(name: str) -> str:
        found = cols.get(name) or cols.get(name.replace(" ", ""))
        if found is None:
            raise KeyError(f"column {name!r} missing from {list(df.columns)}")
        return found

    keep = df.select([
        pl.col(col("date")).alias("date"),
        pl.col(col("open")).alias("open"),
        pl.col(col("high")).alias("high"),
        pl.col(col("low")).alias("low"),
        pl.col(col("close")).alias("close"),
        pl.col(col("volume")).alias("volume"),
    ]).with_columns(pl.lit(ticker).alias("symbol"))
    return keep.with_columns(pl.col("date").cast(pl.Date)).drop_nulls(
        subset=["close"]
    )


def fetch_vrp_data(
    *, start: dt.date, end: dt.date, cache_root: Path
) -> VRPFetchResult:
    """Fetch SPY/QQQ underlying + the CBOE vol-index family.

    Missing tickers (no data, network error, or not yet listed) are reported
    in `missing_vol` and `fetched_underlying` / `fetched_vol` lists.
    """
    underlying_frames: list[pl.DataFrame] = []
    fetched_u: list[str] = []
    for tkr in UNDERLYING_TICKERS:
        df = _load_or_fetch(ticker=tkr, start=start, end=end, cache_root=cache_root)
        if df is None:
            continue
        try:
            underlying_frames.append(_normalize_one(df, tkr))
            fetched_u.append(tkr)
        except Exception:
            continue
    if underlying_frames:
        underlying = pl.concat(underlying_frames, how="diagonal_relaxed").drop_nulls(
            subset=["open", "high", "low", "close"]
        )
    else:
        underlying = pl.DataFrame(
            schema={
                "date": pl.Date, "symbol": pl.Utf8,
                "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
            }
        )

    vol_cols: dict[str, pl.DataFrame] = {}
    fetched_v: list[str] = []
    missing_v: list[str] = []
    for tkr in VIX_FAMILY_TICKERS:
        df = _load_or_fetch(ticker=tkr, start=start, end=end, cache_root=cache_root)
        if df is None or df.is_empty():
            missing_v.append(tkr)
            continue
        try:
            norm = _normalize_one(df, tkr)
            col_name = tkr.lstrip("^").lower()  # ^VIX -> "vix"
            vol_cols[col_name] = (
                norm.select(["date", "close"])
                .rename({"close": col_name})
            )
            fetched_v.append(tkr)
        except Exception:
            missing_v.append(tkr)

    if vol_cols:
        vol_features = next(iter(vol_cols.values()))
        for _col_name, frame in list(vol_cols.items())[1:]:
            vol_features = vol_features.join(
                frame, on="date", how="full", coalesce=True
            )
        vol_features = vol_features.sort("date").drop_nulls(subset=["date"])
    else:
        vol_features = pl.DataFrame(schema={"date": pl.Date})

    return VRPFetchResult(
        underlying=underlying,
        vol_features=vol_features,
        fetched_underlying=fetched_u,
        fetched_vol=fetched_v,
        missing_vol=missing_v,
    )
