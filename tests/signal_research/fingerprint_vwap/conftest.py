# tests/signal_research/fingerprint_vwap/conftest.py
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def panel() -> pl.DataFrame:
    """Two symbols, 400 trading days. AAA is a clean uptrend (linear log-price),
    BBB is noisy/flat — so fingerprint features have known signs in tests."""
    rng = np.random.default_rng(7)
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    rows = []
    for sym, drift, noise in (("AAA", 0.0010, 0.005), ("BBB", 0.0, 0.02)):
        logp = np.cumsum(np.full(400, drift) + rng.normal(0, noise, 400)) + np.log(100.0)
        close = np.exp(logp)
        high = close * (1.0 + np.abs(rng.normal(0, 0.003, 400)))
        low = close * (1.0 - np.abs(rng.normal(0, 0.003, 400)))
        open_ = close * (1.0 + rng.normal(0, 0.002, 400))
        vol = rng.integers(1_000_000, 5_000_000, 400).astype(float)
        for i, d in enumerate(dates):
            rows.append((d, sym, float(open_[i]), float(high[i]), float(low[i]),
                         float(close[i]), float(vol[i])))
    return pl.DataFrame(
        rows, schema=["date", "symbol", "open", "high", "low", "close", "volume"],
        orient="row",
    ).with_columns(pl.col("date").cast(pl.Date))
