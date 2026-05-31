"""Identity: gross_alpha - (commission + spread + borrow + financing) ≈ net_alpha."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.pnl import decompose_pnl


def _frame(col: str, value: float) -> pl.DataFrame:
    return pl.DataFrame({"date": [date(2020, 1, 3)], "symbol": ["A"], col: [value]})


def test_decomposition_identity_holds() -> None:
    dec = decompose_pnl(
        price_pnl=_frame("price_pnl", 100.0),
        cash_dividend_pnl=_frame("cash_dividend_pnl", 10.0),
        commission_drag=_frame("commission_drag", 5.0),
        spread_drag=_frame("spread_drag", 7.0),
        borrow_drag=_frame("borrow_drag", 3.0),
        financing_drag=_frame("financing_drag", 2.0),
        equity=10_000.0,
        n_days=1,
    )
    # gross = 110 bps; drags = 17 bps → net = 93 bps over equity=10k
    assert abs(dec.gross_alpha_bps_per_day - 110.0) < 1e-6
    assert abs(dec.net_alpha_bps_per_day - (110.0 - 17.0)) < 1e-6
