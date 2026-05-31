"""24-month daily-bar fetcher for the strategy benchmark.

Uses yfinance for free real prices on:
- ES=F (S&P 500 e-mini front-month, auto-rolled)
- NQ=F (Nasdaq 100 e-mini front-month, auto-rolled)
- SPY  (S&P 500 ETF — liquid cash proxy for ES)
- QQQ  (Nasdaq 100 ETF — liquid cash proxy for NQ)

Equal-weighted basket is constructed in-memory from the 4 instruments above.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class BenchmarkUniverse:
    name: str
    tickers: tuple[str, ...]
    description: str


UNIVERSES: tuple[BenchmarkUniverse, ...] = (
    BenchmarkUniverse(name="ES_F", tickers=("ES=F",), description="S&P 500 e-mini front-month"),
    BenchmarkUniverse(name="NQ_F", tickers=("NQ=F",), description="Nasdaq 100 e-mini front-month"),
    BenchmarkUniverse(name="SPY", tickers=("SPY",), description="S&P 500 ETF cash proxy"),
    BenchmarkUniverse(name="QQQ", tickers=("QQQ",), description="Nasdaq 100 ETF cash proxy"),
    BenchmarkUniverse(
        name="EW_BASKET",
        tickers=("SPY", "QQQ", "ES=F", "NQ=F"),
        description="Equal-weighted basket of the four",
    ),
)


def fetch_daily_bars(
    *,
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> pl.DataFrame:
    """Download a single ticker's daily bars.  Returns a polars DataFrame with
    columns: date, symbol, open, high, low, close, volume.

    Uses yfinance with auto_adjust=False so we keep the raw Close and the
    Adj Close separately.  We use Adj Close as the canonical close so that
    splits/dividends are reflected in returns.  Futures (ES=F, NQ=F) have no
    dividend, so Adj Close == Close for those, which is correct.
    """
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"empty data for {ticker} {start}..{end}")
    # Reset index first so the Date index becomes a named column, THEN flatten
    # column levels (yfinance >= 0.2 returns a 2-level MultiIndex even for one ticker).
    df = df.reset_index()
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    # The date column may be named "Date" or "index" depending on yfinance version.
    date_col_candidates = [c for c in df.columns if c in ("Date", "Datetime", "index")]
    if not date_col_candidates:
        raise RuntimeError(f"no date column found in yfinance output: {list(df.columns)}")
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


def fetch_benchmark_panel(
    *,
    universes: tuple[BenchmarkUniverse, ...],
    start: dt.date,
    end: dt.date,
    out_dir: Path,
) -> dict[str, pl.DataFrame]:
    """Download bars for every ticker referenced by any universe.

    Returns a {ticker: bars} dict, also writes one parquet per ticker.
    """
    tickers = sorted({t for u in universes for t in u.tickers})
    out_dir.mkdir(parents=True, exist_ok=True)
    bars_by_ticker: dict[str, pl.DataFrame] = {}
    for t in tickers:
        bars = fetch_daily_bars(ticker=t, start=start, end=end)
        bars_by_ticker[t] = bars
        safe = t.replace("=", "_").replace("^", "")
        bars.write_parquet(out_dir / f"{safe}.parquet")
    return bars_by_ticker


def build_universe_returns(
    *,
    universe: BenchmarkUniverse,
    bars_by_ticker: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Build a single-asset (or equal-weighted) returns series for a universe.

    Returns a polars DataFrame with: date, close (the universe's representative
    close-as-if-one-instrument), and the per-bar columns aggregated for EW.
    """
    if len(universe.tickers) == 1:
        return bars_by_ticker[universe.tickers[0]].clone()

    # Equal-weighted: align on date, compute mean of per-ticker simple returns,
    # cumulative-product into a synthetic price series starting at 100.
    aligned: pl.DataFrame | None = None
    for t in universe.tickers:
        df = bars_by_ticker[t].select(["date", "close"]).rename({"close": f"close_{t}"})
        aligned = df if aligned is None else aligned.join(df, on="date", how="inner")
    assert aligned is not None
    aligned = aligned.sort("date")
    close_cols = [f"close_{t}" for t in universe.tickers]
    # daily simple returns per ticker
    rets = aligned.with_columns(
        [
            (pl.col(c) / pl.col(c).shift(1) - 1.0).alias(f"r_{c}")
            for c in close_cols
        ]
    )
    r_cols = [f"r_{c}" for c in close_cols]
    rets = rets.with_columns(
        pl.mean_horizontal(*[pl.col(c) for c in r_cols]).alias("ret_ew")
    )
    # Build a synthetic price index
    rets = rets.with_columns(
        ((pl.col("ret_ew").fill_null(0.0) + 1.0).cum_prod() * 100.0).alias("close")
    )
    out = rets.select(["date", "close"]).with_columns(
        pl.lit("EW_BASKET").alias("symbol"),
        # Synthetic OHLCV: open=close.shift(1), high/low = close ± 0.1%, volume=1.
        # These are needed by signal generators that consume OHLCV columns.
        pl.col("close").shift(1).alias("open"),
        (pl.col("close") * 1.001).alias("high"),
        (pl.col("close") * 0.999).alias("low"),
        pl.lit(1).alias("volume"),
    ).select(["date", "symbol", "open", "high", "low", "close", "volume"])
    return out.drop_nulls()
