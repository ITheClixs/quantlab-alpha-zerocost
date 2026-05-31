"""Cost model — commission + spread + pre-decimalization (spec §5.6)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.costs import (
    CostConfig,
    compute_commission_drag,
    compute_spread_drag,
)


def test_commission_drag_is_bps_one_way() -> None:
    trades = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "trade_notional_abs": [100_000.0],
        }
    )
    drag = compute_commission_drag(trades, cost=CostConfig())
    assert abs(drag["commission_drag"][0] - 100_000.0 * 0.5 / 10_000.0) < 1e-9


def test_spread_drag_uses_roll_when_available_else_tier() -> None:
    trades = pl.DataFrame(
        {
            "date": [date(2020, 1, 3), date(2020, 1, 3)],
            "symbol": ["A", "B"],
            "trade_notional_abs": [100_000.0, 100_000.0],
            "roll_spread_bps": [10.0, None],
            "tier": ["general", "general"],
        }
    )
    drag = compute_spread_drag(trades, cost=CostConfig())
    # 10 bps roll → 5 bps half; general tier → 15 bps → 7.5 bps half
    assert abs(drag["spread_drag"][0] - 100_000.0 * 5.0 / 10_000.0) < 1e-9
    assert abs(drag["spread_drag"][1] - 100_000.0 * 7.5 / 10_000.0) < 1e-9


def test_pre_decimalization_multiplier_widens_pre_2001() -> None:
    trades = pl.DataFrame(
        {
            "date": [date(2000, 6, 15), date(2002, 6, 15)],
            "symbol": ["A", "A"],
            "trade_notional_abs": [100_000.0, 100_000.0],
            "roll_spread_bps": [None, None],
            "tier": ["general", "general"],
        }
    )
    drag = compute_spread_drag(trades, cost=CostConfig())
    assert drag["spread_drag"][0] > drag["spread_drag"][1]
