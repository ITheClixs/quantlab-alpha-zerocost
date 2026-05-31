"""Fill-aligned PnL: new positions PnL from FILL price (not close_t),
existing positions close-to-close MTM (spec §5.4)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.pnl import compute_position_price_pnl


def test_new_position_pnl_is_close_minus_fill_times_shares() -> None:
    new_lots = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_notional_new": [10_000.0],
            "fill_price": [100.0],
            "close_today": [102.0],
        }
    )
    held = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signed_notional_prev": pl.Float64,
            "close_prev": pl.Float64, "close_today": pl.Float64,
        }
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # shares = 10000/100 = 100; price PnL = 100 * (102 - 100) = 200
    assert pnl.height == 1
    assert abs(pnl["price_pnl"][0] - 200.0) < 1e-9


def test_held_position_pnl_is_close_to_close_total_return() -> None:
    held = pl.DataFrame(
        {
            "date": [date(2020, 1, 4)],
            "symbol": ["AAPL"],
            "signed_notional_prev": [10_000.0],
            "close_prev": [102.0],
            "close_today": [104.04],
        }
    )
    new_lots = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signed_notional_new": pl.Float64, "fill_price": pl.Float64,
            "close_today": pl.Float64,
        }
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # ret = 104.04/102 - 1 = 0.02; PnL = 10_000 * 0.02 = 200
    assert pnl.height == 1
    assert abs(pnl["price_pnl"][0] - 200.0) < 1e-9


def test_short_new_position_pnl_signed_correctly() -> None:
    new_lots = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["XYZ"],
            "signed_notional_new": [-10_000.0],
            "fill_price": [100.0],
            "close_today": [102.0],
        }
    )
    held = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signed_notional_prev": pl.Float64,
            "close_prev": pl.Float64, "close_today": pl.Float64,
        }
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # signed_shares = -100; PnL = -100 * (102 - 100) = -200 (short loss)
    assert abs(pnl["price_pnl"][0] - (-200.0)) < 1e-9
