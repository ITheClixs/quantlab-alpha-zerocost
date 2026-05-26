"""Strict diagnostics tests for supervised signal_research backtests."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant_research_stack.signal_research.training.backtest_diagnostics import (
    StrictBacktestDiagnosticsConfig,
    render_strict_backtest_report,
    run_strict_backtest_diagnostics,
)


def _prediction_fixture() -> pl.DataFrame:
    rows = []
    start = date(2025, 1, 2)
    symbols = ["AAA", "BBB"]
    for i in range(90):
        current = start + timedelta(days=i)
        if current.weekday() >= 5:
            continue
        for symbol_index, symbol in enumerate(symbols):
            direction = 1.0 if (i + symbol_index) % 4 != 0 else -1.0
            future_return = 0.004 * direction
            model_position = direction if (i + symbol_index) % 5 != 0 else 0.0
            rows.append(
                {
                    "date": current,
                    "fold": i // 20,
                    "symbol": symbol,
                    "primary_position": direction,
                    "future_return_horizon": future_return,
                    "entry_close_proxy": 100.0 + i + symbol_index,
                    "meta_probability": 0.62 if model_position != 0.0 else 0.51,
                    "meta_position": model_position,
                    "gross_return": model_position * future_return,
                    "net_return": model_position * future_return - (0.0002 if model_position != 0.0 else 0.0),
                }
            )
    return pl.DataFrame(rows)


def test_strict_diagnostics_reports_cost_delay_random_and_inversion() -> None:
    result = run_strict_backtest_diagnostics(
        _prediction_fixture(),
        config=StrictBacktestDiagnosticsConfig(
            market_name="unit",
            cost_bps_one_way=1.0,
            bootstrap_resamples=100,
            multiple_testing_trials=12,
        ),
    )

    rows = {row["variant"]: row for row in result.variant_metrics.iter_rows(named=True)}
    assert rows["cost_2x"]["net_total_return"] < rows["cost_1x"]["net_total_return"]
    assert rows["delay_1_bar"]["trade_count"] > 0
    assert rows["random_same_trade_mask"]["trade_count"] == rows["cost_1x"]["trade_count"]
    assert rows["inverted_signal"]["net_total_return"] < rows["cost_1x"]["net_total_return"]
    assert 0.0 <= result.summary["concentration"]["best_day_positive_pnl_share"] <= 1.0
    assert result.summary["bootstrap_sharpe_ci"]["status"] == "computed"
    assert result.summary["deflated_sharpe"]["status"] == "computed_approximation"
    assert result.summary["promotion_eligible"] is False
    assert result.trade_audit.height == rows["cost_1x"]["trade_count"]


def test_dsr_probability_is_penalized_by_larger_trial_count() -> None:
    low_trial = run_strict_backtest_diagnostics(
        _prediction_fixture(),
        config=StrictBacktestDiagnosticsConfig(
            market_name="unit",
            bootstrap_resamples=50,
            multiple_testing_trials=6,
        ),
    )
    high_trial = run_strict_backtest_diagnostics(
        _prediction_fixture(),
        config=StrictBacktestDiagnosticsConfig(
            market_name="unit",
            bootstrap_resamples=50,
            multiple_testing_trials=500,
        ),
    )

    assert (
        high_trial.summary["deflated_sharpe"]["probability"]
        <= low_trial.summary["deflated_sharpe"]["probability"]
    )


def test_strict_report_marks_profile_as_research_only() -> None:
    result = run_strict_backtest_diagnostics(
        _prediction_fixture(),
        config=StrictBacktestDiagnosticsConfig(
            market_name="unit",
            bootstrap_resamples=50,
            multiple_testing_trials=10,
        ),
    )

    report = render_strict_backtest_report(result)

    assert "research_validation_only" in report
    assert "promotion_eligible: `False`" in report
    assert "PBO" in report
    assert "not automatically investment advice" in report
