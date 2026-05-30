"""Free, timestamp-safe data loaders + the macro feature registry.

Instruments: SPY/QQQ from disk (vrp bars); BTCUSDT/ETHUSDT from yfinance (BTC-USD/
ETH-USD), cached. Macro: market-priced daily series (VIX/VIX3M, ETFs) + Treasury
yields from FRED (daily CMT, not revised). All series are daily close observed at t,
used at t+1. Network fetches are cached to data/processed/zero_cost/ (gitignored)
and are best-effort (the audit records coverage / unavailability).
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore")
_CACHE = Path("data/processed/zero_cost")
_START = "2010-01-01"

INSTRUMENTS: dict[str, dict[str, str]] = {
    "SPY": {"source": "disk", "path": "data/processed/vrp/bars/SPY.parquet"},
    "QQQ": {"source": "disk", "path": "data/processed/vrp/bars/QQQ.parquet"},
    "BTCUSDT": {"source": "yfinance", "ticker": "BTC-USD"},
    "ETHUSDT": {"source": "yfinance", "ticker": "ETH-USD"},
}


@dataclass(frozen=True)
class MacroSeries:
    name: str
    source: str            # yfinance | fred
    ref: str               # ticker or FRED id
    classification: str    # market_price_clean | daily_next_day_only | revision_risk | reject
    rationale: str


# Allowed: market-priced daily (no revision) OR daily Treasury CMT (published EOD t, not revised).
MACRO_REGISTRY: tuple[MacroSeries, ...] = (
    MacroSeries("vix", "yfinance", "^VIX", "market_price_clean",
                "CBOE VIX daily index close; market-priced, not revised"),
    MacroSeries("vix3m", "yfinance", "^VIX3M", "market_price_clean",
                "CBOE 3-month VIX daily close; with VIX gives the VIX term structure"),
    MacroSeries("bonds_tlt", "yfinance", "TLT", "market_price_clean",
                "20y+ Treasury ETF daily close; market-priced rates proxy"),
    MacroSeries("gold_gld", "yfinance", "GLD", "market_price_clean",
                "gold ETF daily close; cross-asset risk proxy"),
    MacroSeries("credit_hyg", "yfinance", "HYG", "market_price_clean",
                "HY credit ETF daily close; credit-stress proxy"),
    MacroSeries("usd_uup", "yfinance", "UUP", "market_price_clean",
                "USD bull ETF daily close; dollar trend proxy"),
    MacroSeries("ust10y", "fred", "DGS10", "daily_next_day_only",
                "10y Treasury CMT; published EOD t by Treasury, not revised; use at t+1"),
    MacroSeries("ust2y", "fred", "DGS2", "daily_next_day_only",
                "2y Treasury CMT; published EOD t, not revised; use at t+1"),
)

# Explicitly forbidden: revised macro aggregates without point-in-time vintages.
FORBIDDEN_SERIES: dict[str, str] = {
    "GDP": "revised aggregate; needs ALFRED vintage",
    "CPIAUCSL": "revised; release/revision lag; needs PIT vintage",
    "PAYEMS": "nonfarm payrolls; heavily revised; needs PIT vintage",
    "UNRATE": "unemployment; revised; needs PIT vintage",
    "PCE": "revised aggregate; needs PIT vintage",
}

# Derived features (computed in P1; listed here for the availability report).
DERIVED_FEATURES: dict[str, str] = {
    "vix_term_structure": "vix / vix3m (>1 backwardation = stress)",
    "yield_slope": "ust10y - ust2y (inversion = stress)",
    "credit_trend": "HYG trend / drawdown state",
    "usd_trend": "UUP trend state",
    "gold_trend": "GLD trend state",
}


def _read_disk(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path).select(["date", "close"]).sort("date")
    return df.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("date"))


def _fetch_yf(ticker: str) -> pl.DataFrame:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=_START, auto_adjust=True)
    if h.empty:
        return pl.DataFrame({"date": [], "close": []})
    return pl.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in h.index],
                         "close": [float(c) for c in h["Close"].to_list()]})


def _fetch_fred(series_id: str) -> pl.DataFrame:
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed FRED host
        raw = r.read()
    df = pl.read_csv(io.BytesIO(raw))
    cols = df.columns
    df = df.rename({cols[0]: "date", cols[1]: "close"}).with_columns(
        pl.col("date").cast(pl.Utf8).str.slice(0, 10),
        pl.col("close").cast(pl.Float64, strict=False),
    )
    return df.filter(pl.col("close").is_not_null() & (pl.col("date") >= _START)).sort("date")


def _cached(name: str, fetch) -> pl.DataFrame:
    _CACHE.mkdir(parents=True, exist_ok=True)
    p = _CACHE / f"{name}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    df = fetch()
    if df.height:
        df.write_parquet(p)
    return df


def load_instrument(name: str, *, use_cache: bool = True) -> pl.DataFrame:
    spec = INSTRUMENTS[name]
    if spec["source"] == "disk":
        return _read_disk(spec["path"])
    fetch = lambda: _fetch_yf(spec["ticker"])  # noqa: E731
    return _cached(f"inst_{name}", fetch) if use_cache else fetch()


def load_macro(series: MacroSeries, *, use_cache: bool = True) -> pl.DataFrame:
    fetch = (lambda: _fetch_yf(series.ref)) if series.source == "yfinance" else (lambda: _fetch_fred(series.ref))
    return _cached(f"macro_{series.name}", fetch) if use_cache else fetch()
