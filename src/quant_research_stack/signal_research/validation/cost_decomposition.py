"""Cost decomposition — run the backtest at four cost levels and report
how much of the gross edge survives each layer.

Levels:
- no_cost:    commission=0, spread=0, borrow=0
- fee_only:   commission only
- spread_only: spread only (no commission)
- full_cost:  full hedge-fund cost stack
- stress_2x:  2x full cost stack
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.alpha_eq.backtest.runner import run_backtest
from quant_research_stack.signal_research.backtests._shared import (
    build_backtest_config,
    equity_metrics,
)


@dataclass(frozen=True)
class CostDecomposition:
    no_cost_sharpe: float
    fee_only_sharpe: float
    spread_only_sharpe: float
    full_cost_sharpe: float
    stress_2x_sharpe: float


def cost_decomposition(
    *,
    panel: pl.DataFrame,
    commission_bps_one_way: float,
    spread_bps_one_way: float,
    q_quantile: float,
    target_gross: float,
    equity: float,
    cohort: str,
) -> CostDecomposition:
    def _run(commission: float, spread: float, cost_stress_mult: float) -> float:
        cfg = build_backtest_config(
            commission_bps_one_way=commission,
            spread_bps_one_way=spread,
            cost_stress_mult=cost_stress_mult,
            q_quantile=q_quantile,
            target_gross=target_gross,
            equity=equity,
            cohort=cohort,
        )
        res = run_backtest(signals_with_bars=panel, config=cfg, dividends=None)
        return equity_metrics(res.daily_returns)["sharpe"]

    return CostDecomposition(
        no_cost_sharpe=_run(0.0, 0.0, 1.0),
        fee_only_sharpe=_run(commission_bps_one_way, 0.0, 1.0),
        spread_only_sharpe=_run(0.0, spread_bps_one_way, 1.0),
        full_cost_sharpe=_run(commission_bps_one_way, spread_bps_one_way, 1.0),
        stress_2x_sharpe=_run(commission_bps_one_way, spread_bps_one_way, 2.0),
    )
