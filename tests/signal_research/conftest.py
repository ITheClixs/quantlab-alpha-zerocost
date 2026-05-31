"""Shared fixtures for signal_research tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def tmp_signal_research_root(tmp_path: Path) -> Path:
    """Disposable root for signal_research data manifests."""
    root = tmp_path / "data" / "processed" / "signal_research"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def subprocess_env() -> dict[str, str]:
    """Subprocess env that lets `uv run` resolve uv on Apple Silicon dev machines."""
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", ".uv-cache"),
    }


@pytest.fixture()
def synthetic_daily_bars(rng: np.random.Generator) -> pl.DataFrame:
    """Tiny 5-symbol, 250-trading-day OHLCV panel."""
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    base_dates = pl.date_range(
        start=pl.date(2024, 1, 2),
        end=pl.date(2025, 6, 1),
        interval="1d",
        eager=True,
    )
    weekday_mask = base_dates.dt.weekday() < 6
    dates = base_dates.filter(weekday_mask).head(250).to_list()
    rows = []
    for s in symbols:
        price = 100.0
        for d in dates:
            ret = float(rng.standard_normal()) * 0.012
            price *= 1.0 + ret
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "open": price * (1.0 + float(rng.standard_normal()) * 0.003),
                    "high": price * (1.0 + abs(float(rng.standard_normal())) * 0.006),
                    "low": price * (1.0 - abs(float(rng.standard_normal())) * 0.006),
                    "close": price,
                    "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 300_000),
                }
            )
    return pl.DataFrame(rows)
