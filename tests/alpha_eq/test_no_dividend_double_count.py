"""Dividends are booked exactly once as cash PnL on ex-date; the MTM path
uses split-adjusted tradable_* (NOT total-return) prices, so there is no
double-count (spec §5.11)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.pnl import (
    compute_cash_dividend_pnl,
    compute_position_price_pnl,
)


def test_long_receives_dividend_cash_once() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "signed_notional": [10_000.0],
            "ref_close": [100.0],
        }
    )
    dividends = pl.DataFrame(
        {
            "ex_date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "dividend_per_share": [0.5],
        }
    )
    div_pnl = compute_cash_dividend_pnl(positions_on_ex_date=positions, dividends=dividends)
    # 100 shares × $0.50 = $50
    assert abs(div_pnl["cash_dividend_pnl"][0] - 50.0) < 1e-9


def test_short_is_debited_dividend_cash_once() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "signed_notional": [-10_000.0],
            "ref_close": [100.0],
        }
    )
    dividends = pl.DataFrame(
        {
            "ex_date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "dividend_per_share": [0.5],
        }
    )
    div_pnl = compute_cash_dividend_pnl(positions_on_ex_date=positions, dividends=dividends)
    assert abs(div_pnl["cash_dividend_pnl"][0] - (-50.0)) < 1e-9


def test_price_pnl_does_not_include_dividend_when_tradable_used() -> None:
    """Held position across ex-date: price PnL uses tradable_close (split-adj),
    which has NOT been bumped by the dividend."""
    held = pl.DataFrame(
        {
            "date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "signed_notional_prev": [10_000.0],
            "close_prev": [100.0],
            "close_today": [99.50],
        }
    )
    new_lots = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8, "signed_notional_new": pl.Float64,
            "fill_price": pl.Float64, "close_today": pl.Float64,
        }
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # price PnL = 10_000 * (99.50/100 - 1) = -50; combined with +50 cash dividend → flat
    assert abs(pnl["price_pnl"][0] - (-50.0)) < 1e-9
