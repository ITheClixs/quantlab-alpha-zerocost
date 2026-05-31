"""Borrow + financing (spec §5.7, §5.8)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.borrow import apply_borrow_drag
from quant_research_stack.alpha_eq.backtest.financing import compute_financing_drag


def test_borrow_multiplier_monotonic() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_notional": [-100_000.0],
            "tier": ["general"],
        }
    )
    one = apply_borrow_drag(pos, multiplier=1.0)["borrow_drag"][0]
    two = apply_borrow_drag(pos, multiplier=2.0)["borrow_drag"][0]
    three = apply_borrow_drag(pos, multiplier=3.0)["borrow_drag"][0]
    assert one < two < three


def test_borrow_zero_on_longs() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_notional": [100_000.0],
            "tier": ["general"],
        }
    )
    drag = apply_borrow_drag(pos, multiplier=3.0)
    assert drag["borrow_drag"][0] == 0.0


def test_financing_only_when_gross_above_1() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "gross_notional": [200_000.0],
            "equity": [100_000.0],
        }
    )
    fin = compute_financing_drag(pos, rate_annual=0.02)
    assert abs(fin["financing_drag"][0] - 100_000.0 * 0.02 / 252.0) < 1e-6


def test_financing_zero_when_gross_one() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "gross_notional": [100_000.0],
            "equity": [100_000.0],
        }
    )
    fin = compute_financing_drag(pos, rate_annual=0.02)
    assert fin["financing_drag"][0] == 0.0
