"""Loaders for Binance USDT-M perpetual funding rates (free, monthly archives).

Monthly funding files: data.binance.vision/data/futures/um/monthly/fundingRate/
<SYMBOL>/<SYMBOL>-fundingRate-YYYY-MM.zip with columns
`calc_time` (ms epoch, settlement), `funding_interval_hours` (8), `last_funding_rate`.
Files are tiny (~90 rows/month) so we download whole zips and cache.

Leakage rule: the realized funding at settlement t is known at t; a signal using
funding <= t positions for the next interval and earns the t+1-interval settlement.
"""

from __future__ import annotations

import io
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

FUNDING_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
_LIST_HOST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_CACHE = Path("data/processed/funding")
RAW_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
NORMALIZED_COLUMNS = ["symbol", "funding_time", "funding_rate", "interval_hours"]


def funding_day_url(symbol: str, month: str) -> str:
    sym = symbol.upper()
    return f"{FUNDING_BASE}/{sym}/{sym}-fundingRate-{month}.zip"


def available_months(symbol: str) -> list[str]:
    prefix = f"data/futures/um/monthly/fundingRate/{symbol.upper()}/"
    out: list[str] = []
    marker = ""
    while True:
        url = f"{_LIST_HOST}?prefix={prefix}&max-keys=1000" + (f"&marker={marker}" if marker else "")
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed host
            root = ET.fromstring(r.read().decode())
        keys = [e.findtext(_NS + "Key") or "" for e in root.iter(_NS + "Contents")]
        out += [k.split("-fundingRate-")[1].replace(".zip", "") for k in keys if k.endswith(".zip")]
        if root.findtext(_NS + "IsTruncated") != "true" or not keys:
            break
        marker = keys[-1]
    return sorted(out)


def normalize_funding(raw: pl.DataFrame, *, symbol: str) -> pl.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"funding frame missing columns: {missing}")
    return (
        raw.with_columns(
            pl.lit(symbol.upper()).alias("symbol"),
            (pl.col("calc_time").cast(pl.Int64) * 1000).cast(pl.Datetime("us", "UTC")).alias("funding_time"),
            pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
            pl.col("funding_interval_hours").cast(pl.Int64).alias("interval_hours"),
        )
        .select(NORMALIZED_COLUMNS)
        .sort("funding_time")
    )


def _read_month(symbol: str, month: str) -> pl.DataFrame:
    raw = urllib.request.urlopen(funding_day_url(symbol, month), timeout=60).read()  # noqa: S310 - fixed host
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member = z.namelist()[0]
        body = z.open(member).read()
    has_header = body[:9].lower().startswith(b"calc_time")
    df = pl.read_csv(io.BytesIO(body), has_header=has_header,
                     new_columns=None if has_header else RAW_COLUMNS)
    return df


def load_funding(symbol: str, months: list[str] | None = None, *, use_cache: bool = True) -> pl.DataFrame:
    """Load + normalize the full funding-rate history for a symbol (cached)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache = _CACHE / f"funding_{symbol.upper()}.parquet"
    if use_cache and cache.exists():
        return pl.read_parquet(cache)
    months = months or available_months(symbol)
    frames = [normalize_funding(_read_month(symbol, m), symbol=symbol) for m in months]
    out = pl.concat(frames, how="vertical").unique(subset=["funding_time"], keep="last").sort("funding_time")
    if use_cache and out.height:
        out.write_parquet(cache)
    return out


def annualized_funding(df: pl.DataFrame) -> float:
    """Mean realized funding annualized (8h settlements -> 3/day * 365)."""
    if df.is_empty():
        return 0.0
    mean = df["funding_rate"].mean()
    return (float(mean) * 3.0 * 365.0) if isinstance(mean, (int, float)) else 0.0


def coverage(df: pl.DataFrame) -> dict[str, object]:
    if df.is_empty():
        return {"rows": 0, "start": None, "end": None}
    span_days = df.select(
        (pl.col("funding_time").max() - pl.col("funding_time").min()).dt.total_days()
    ).item()
    span = int(span_days) if isinstance(span_days, (int, float)) else 1
    spd = round(df.height / max(span, 1), 2)
    return {"rows": df.height, "start": str(df["funding_time"].min()),
            "end": str(df["funding_time"].max()), "settlements_per_day": spd,
            "built_utc": datetime.now(UTC).isoformat()}
