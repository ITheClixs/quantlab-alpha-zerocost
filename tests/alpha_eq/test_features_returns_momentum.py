"""Returns / momentum features (spec §3.3-1)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.returns_momentum import (
    build_returns_momentum,
)


def _toy() -> pl.DataFrame:
    closes = np.geomspace(100.0, 200.0, num=300)
    dates_full = pl.date_range(
        start=date(2020, 1, 2), end=date(2021, 3, 1), interval="1d", eager=True
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(300).to_list()
    return pl.DataFrame({"date": dates, "symbol": ["A"] * 300, "close": closes})


def test_returns_momentum_emits_expected_columns() -> None:
    df = build_returns_momentum(
        _toy(), horizons=(1, 5, 20, 60, 120, 252), include_skip5=(60, 120, 252)
    )
    expected = {
        "log_return_1", "log_return_5", "log_return_20",
        "log_return_60", "log_return_120", "log_return_252",
        "cumulative_return_60_skip5", "cumulative_return_120_skip5", "cumulative_return_252_skip5",
        "mean_reversion_5",
    }
    assert expected.issubset(set(df.columns))


def test_returns_momentum_no_future_leak_for_after_close_convention() -> None:
    df = _toy()
    out = build_returns_momentum(df, horizons=(1,))
    last = out.tail(2).to_dicts()
    expected = float(np.log(last[1]["close"] / last[0]["close"]))
    assert abs(last[1]["log_return_1"] - expected) < 1e-12
