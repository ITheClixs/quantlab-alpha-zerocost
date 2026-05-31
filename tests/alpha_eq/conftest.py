"""Shared fixtures for alpha_eq tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def tmp_equity_root(tmp_path: Path) -> Path:
    """Disposable processed-equities root for tests."""
    root = tmp_path / "data" / "processed" / "equities"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_panel(rng: np.random.Generator) -> pl.DataFrame:
    """Tiny 5-symbol, 50-date synthetic OHLCV panel for unit tests."""
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    dates_full = pl.date_range(
        start=pl.date(2020, 1, 2),
        end=pl.date(2020, 4, 30),
        interval="1d",
        eager=True,
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(50).to_list()
    rows = []
    for s in symbols:
        price = 100.0
        for d in dates:
            ret = float(rng.standard_normal()) * 0.02
            price *= (1.0 + ret)
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "open": price * (1.0 + float(rng.standard_normal()) * 0.005),
                    "high": price * (1.0 + abs(float(rng.standard_normal())) * 0.01),
                    "low": price * (1.0 - abs(float(rng.standard_normal())) * 0.01),
                    "close": price,
                    "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 500_000),
                }
            )
    return pl.DataFrame(rows)
