"""Daily spot + USDT-M perp klines from Binance Vision (free, monthly archives).

For delta-neutral funding carry we need spot and perp closes at the **same instant**
so the basis (perp/spot - 1) is clean. Binance Vision daily klines for both markets
close at UTC midnight, so using one source for both legs avoids a stale-leg basis.

  spot:  data.binance.vision/data/spot/monthly/klines/<SYM>/1d/<SYM>-1d-YYYY-MM.zip
  perp:  data.binance.vision/data/futures/um/monthly/klines/<SYM>/1d/<SYM>-1d-YYYY-MM.zip

Klines CSV (Binance): open_time, open, high, low, close, volume, close_time,
quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore. Older months
have no header row; we sniff it (mirrors funding.data).
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import polars as pl

_SPOT_BASE = "https://data.binance.vision/data/spot/monthly/klines"
_PERP_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
_LIST_HOST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_CACHE = Path("data/processed/funding")
MARKETS = ("spot", "perp")
KLINE_RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
NORMALIZED_COLUMNS = ["symbol", "ts", "open", "close"]


def _fetch(url: str, *, timeout: int = 60, retries: int = 5) -> bytes:
    """GET with exponential backoff — Binance Vision resets connections under load."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed host
                return r.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
            time.sleep(min(2.0 ** attempt, 20.0))
    raise RuntimeError(f"fetch failed after {retries} attempts: {url}") from last


def _base(market: str) -> str:
    if market not in MARKETS:
        raise ValueError(f"market must be one of {MARKETS}, got {market!r}")
    return _SPOT_BASE if market == "spot" else _PERP_BASE


def _prefix(symbol: str, market: str) -> str:
    seg = "spot/monthly/klines" if market == "spot" else "futures/um/monthly/klines"
    return f"data/{seg}/{symbol.upper()}/1d/"


def klines_url(symbol: str, market: str, month: str) -> str:
    sym = symbol.upper()
    return f"{_base(market)}/{sym}/1d/{sym}-1d-{month}.zip"


def available_months(symbol: str, market: str) -> list[str]:
    prefix = _prefix(symbol, market)
    out: list[str] = []
    marker = ""
    while True:
        url = f"{_LIST_HOST}?prefix={prefix}&max-keys=1000" + (f"&marker={marker}" if marker else "")
        root = ET.fromstring(_fetch(url, timeout=30).decode())
        keys = [e.findtext(_NS + "Key") or "" for e in root.iter(_NS + "Contents")]
        out += [k.split("-1d-")[1].replace(".zip", "") for k in keys if k.endswith(".zip") and "-1d-" in k]
        if root.findtext(_NS + "IsTruncated") != "true" or not keys:
            break
        marker = keys[-1]
    return sorted(out)


def normalize_klines(raw: pl.DataFrame, *, symbol: str) -> pl.DataFrame:
    missing = [c for c in ("open_time", "open", "close") if c not in raw.columns]
    if missing:
        raise ValueError(f"klines frame missing columns: {missing}")
    # open_time may be ms or us epoch depending on month; normalize by magnitude.
    ot = pl.col("open_time").cast(pl.Int64)
    ts_us = (
        pl.when(ot > 10_000_000_000_000)
        .then(ot)
        .otherwise(ot * 1000)
        .cast(pl.Datetime("us", "UTC"))
    )
    return (
        raw.with_columns(
            pl.lit(symbol.upper()).alias("symbol"),
            ts_us.alias("ts"),
            pl.col("open").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        )
        .select(NORMALIZED_COLUMNS)
        .sort("ts")
    )


def _read_month(symbol: str, market: str, month: str) -> pl.DataFrame:
    raw = _fetch(klines_url(symbol, market, month))
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        body = z.open(z.namelist()[0]).read()
    has_header = body[:9].lower().startswith(b"open_time")
    return pl.read_csv(
        io.BytesIO(body), has_header=has_header,
        new_columns=None if has_header else KLINE_RAW_COLUMNS,
        schema_overrides={"open_time": pl.Int64} if has_header else None,
    )


def load_daily_klines(symbol: str, market: str, months: list[str] | None = None,
                      *, use_cache: bool = True) -> pl.DataFrame:
    """Load + normalize the full daily-close history for a symbol/market (cached)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache = _CACHE / f"klines_{market}_{symbol.upper()}_1d.parquet"
    if use_cache and cache.exists():
        return pl.read_parquet(cache)
    months = months or available_months(symbol, market)
    frames = []
    for m in months:
        frames.append(normalize_klines(_read_month(symbol, market, m), symbol=symbol))
        time.sleep(0.05)  # gentle pacing — Vision resets under rapid sequential load
    out = pl.concat(frames, how="vertical").unique(subset=["ts"], keep="last").sort("ts")
    if use_cache and out.height:
        out.write_parquet(cache)
    return out


def daily_funding(funding: pl.DataFrame) -> pl.DataFrame:
    """Collapse 8h funding settlements to a per-UTC-day total (the day's carry)."""
    return (
        funding.with_columns(pl.col("funding_time").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("funding_rate").sum().alias("funding_day"),
             pl.len().alias("settlements"))
        .sort("date")
    )


def align_carry(funding: pl.DataFrame, spot: pl.DataFrame, perp: pl.DataFrame) -> pl.DataFrame:
    """Daily panel: spot_close, perp_close, basis, funding_day on a common UTC-day grid.

    `basis = perp/spot - 1` at the simultaneous daily close. `funding_day` is the total
    funding settled that UTC day (collected by a short-perp leg held through the day).
    Inner-join on date so every row has spot, perp, and funding (no implicit fills).
    """
    s = spot.with_columns(pl.col("ts").dt.date().alias("date")).select(
        ["date", pl.col("close").alias("spot_close")])
    p = perp.with_columns(pl.col("ts").dt.date().alias("date")).select(
        ["date", pl.col("close").alias("perp_close")])
    f = daily_funding(funding)
    out = (
        s.join(p, on="date", how="inner")
        .join(f, on="date", how="inner")
        .with_columns((pl.col("perp_close") / pl.col("spot_close") - 1.0).alias("basis"))
        .sort("date")
    )
    return out
