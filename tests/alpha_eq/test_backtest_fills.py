"""Fill-price selection (spec §5.3)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.fills import (
    FillModel,
    pick_fill_prices,
)


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
        }
    )


def test_fill_open() -> None:
    out = pick_fill_prices(_bars(), model=FillModel.OPEN)
    assert out["fill_price"][0] == 100.0


def test_fill_hlc3_proxy_labeled() -> None:
    out = pick_fill_prices(_bars(), model=FillModel.HLC3_PROXY)
    assert abs(out["fill_price"][0] - (102.0 + 99.0 + 101.0) / 3.0) < 1e-9
    assert out["fill_model"][0] == "vwap_proxy_hlc3"


def test_fill_close() -> None:
    out = pick_fill_prices(_bars(), model=FillModel.CLOSE)
    assert out["fill_price"][0] == 101.0
