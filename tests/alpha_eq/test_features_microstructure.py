"""Microstructure proxies (spec §3.3-3): amihud, roll w/ NaN policy, kyle_proxy."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.microstructure_proxies import (
    build_microstructure_proxies,
)


def _toy(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates_full = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 6, 30), interval="1d", eager=True
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(n).to_list()
    closes = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    opens = closes * (1 + rng.standard_normal(n) * 0.005)
    highs = np.maximum(closes, opens) * 1.005
    lows = np.minimum(closes, opens) * 0.995
    vols = rng.integers(500_000, 2_000_000, size=n)
    return pl.DataFrame(
        {"date": dates, "symbol": ["A"] * n,
         "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}
    )


def test_amihud_roll_kyle_proxy_columns_present() -> None:
    df = build_microstructure_proxies(_toy(), window=20)
    for col in ("amihud_illiq_20", "roll_spread_20", "kyle_proxy_signed_volume_20",
                "overnight_gap", "intraday_return", "close_location_20"):
        assert col in df.columns


def test_roll_spread_is_null_on_positive_autocov() -> None:
    """Spec §3.3-3: when autocov non-negative, Roll is explicitly NaN."""
    n = 60
    closes = np.linspace(100.0, 200.0, n)
    dates_full = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 6, 30), interval="1d", eager=True
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(n).to_list()
    df = pl.DataFrame(
        {"date": dates, "symbol": ["A"] * n,
         "open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": [1_000_000] * n}
    )
    out = build_microstructure_proxies(df, window=20)
    n_null = int(out["roll_spread_20"].is_null().sum())
    assert n_null > 0
