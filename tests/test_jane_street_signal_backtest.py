from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.backtest.jane_street_signal import (
    evaluate_prediction_column,
    run_grouped_long_short_backtest,
)


def test_evaluate_prediction_column_scores_weighted_competition_metrics() -> None:
    frame = pl.DataFrame(
        {
            "target_actual": [1.0, -1.0, 0.5, -0.5],
            "weight": [1.0, 2.0, 1.0, 2.0],
            "stacked": [0.8, -0.6, 0.1, 0.2],
        }
    )

    metrics = evaluate_prediction_column(frame, "stacked")

    assert metrics["rows"] == 4
    assert metrics["weighted_directional_accuracy"] == pytest.approx(4.0 / 6.0)
    assert metrics["positive_precision"] == pytest.approx(0.5)
    assert metrics["negative_precision"] == pytest.approx(1.0)
    assert metrics["weighted_zero_mean_r2"] > 0.0
    assert metrics["weighted_sign_capture"] > 0.0


def test_grouped_long_short_backtest_uses_top_and_bottom_predictions_per_date() -> None:
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "target_actual": [0.6, 0.4, -0.2, -0.5, 0.3, 0.2, -0.1, -0.4],
            "weight": [1.0] * 8,
            "stacked": [0.9, 0.5, -0.2, -0.8, 0.7, 0.4, -0.1, -0.9],
        }
    )

    result = run_grouped_long_short_backtest(frame, "stacked", selection_fraction=0.25)

    assert result.daily_curve.height == 2
    assert result.metrics["n_groups"] == 2
    assert result.metrics["hit_rate"] == pytest.approx(1.0)
    assert result.metrics["total_pnl_units"] > 0.0
    assert result.metrics["mean_long_short_spread"] > 0.0
