"""Volatility features (spec §3.3-2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.volatility import build_volatility


def _toy(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates_full = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 12, 31), interval="1d", eager=True
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(n).to_list()
    closes = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    highs = closes * (1 + np.abs(rng.standard_normal(n)) * 0.01)
    lows = closes * (1 - np.abs(rng.standard_normal(n)) * 0.01)
    opens = closes * (1 + rng.standard_normal(n) * 0.005)
    return pl.DataFrame(
        {"date": dates, "symbol": ["A"] * n,
         "open": opens, "high": highs, "low": lows, "close": closes}
    )


def test_volatility_columns_present_and_nonneg() -> None:
    df = build_volatility(_toy(), windows=(5, 20, 60), parkinson_window=20, gk_window=20, vov_window=60)
    for col in ("realized_vol_5", "realized_vol_20", "realized_vol_60",
                "parkinson_vol_20", "garman_klass_vol_20", "vol_of_vol_60"):
        assert col in df.columns
        vals = df[col].drop_nulls().to_numpy()
        assert np.all(vals >= 0.0)
