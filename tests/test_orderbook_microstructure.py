from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.backtest.orderbook_signal import (
    OrderBookBacktestConfig,
    OrderBookWalkForwardConfig,
    orderbook_features_from_frame,
    run_orderbook_signal_backtest,
    run_orderbook_walk_forward,
)


def test_orderbook_features_from_frame_builds_targets_and_depth_features() -> None:
    raw = pl.DataFrame(
        {
            "E": [1_000, 2_000, 3_000],
            "T": [900, 1_900, 2_900],
            "u": [10, 11, 12],
            "bids": [
                '[["99.0", "4.0"], ["98.5", "1.0"]]',
                '[["100.0", "1.0"], ["99.5", "1.0"]]',
                '[["101.0", "2.0"], ["100.5", "2.0"]]',
            ],
            "asks": [
                '[["101.0", "2.0"], ["101.5", "2.0"]]',
                '[["102.0", "3.0"], ["102.5", "1.0"]]',
                '[["103.0", "2.0"], ["103.5", "2.0"]]',
            ],
        }
    )

    features = orderbook_features_from_frame(
        raw,
        dataset_id="unit",
        source_file="unit.parquet",
        symbol="BTCUSDT",
        horizons=(1,),
        depth_levels=(1, 2),
    )

    assert features.select("symbol").to_series().to_list() == ["BTCUSDT", "BTCUSDT", "BTCUSDT"]
    assert features["mid_price"].to_list() == [100.0, 101.0, 102.0]
    assert features["spread"].to_list() == [2.0, 2.0, 2.0]
    assert features["bid_depth_2"].to_list() == [5.0, 2.0, 4.0]
    assert features["ask_depth_2"].to_list() == [4.0, 4.0, 4.0]
    assert features["future_mid_return_1"].head(2).round(6).to_list() == [0.01, 0.009901]


def test_orderbook_backtest_charges_spread_and_round_trip_fees() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "event_time": [1, 2, 3],
            "prediction": [0.004, -0.002, 0.0001],
            "future_mid_return_1": [0.006, -0.003, 0.01],
            "relative_spread": [0.001, 0.0015, 0.001],
        }
    )

    result = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            min_signal_abs=0.001,
            fee_bps=1.0,
            starting_equity=100_000.0,
        ),
    )

    assert result.metrics["rows"] == 3
    assert result.metrics["trade_count"] == 2
    assert result.metrics["directional_accuracy"] == 1.0
    assert result.metrics["avg_trade_cost_return"] == pytest.approx(0.00145)
    assert result.trades["net_return"].round(6).to_list() == [0.0048, 0.0013]
    assert result.metrics["total_return"] > 0.006


def test_orderbook_walk_forward_predicts_only_out_of_sample_rows() -> None:
    rows = 80
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * rows,
            "event_time": list(range(rows)),
            "row_index": list(range(rows)),
            "relative_spread": [0.0001] * rows,
            "imbalance_l1": [0.8 if i % 2 == 0 else -0.8 for i in range(rows)],
            "microprice_l1": [100.0 + (i * 0.01) for i in range(rows)],
            "future_mid_return_1": [0.002 if i % 2 == 0 else -0.002 for i in range(rows)],
        }
    )

    result = run_orderbook_walk_forward(
        frame,
        config=OrderBookWalkForwardConfig(
            min_train_rows=40,
            test_rows=10,
            step_rows=10,
            max_folds=2,
            hist_gradient_max_iter=10,
        ),
    )

    assert result.predictions.height == 20
    assert result.fold_specs[0].train_end_row < result.fold_specs[0].test_start_row
    assert set(result.predictions.columns) >= {"pred_ridge", "pred_hist_gradient", "pred_ensemble_mean"}
    assert result.backtest_metrics["ensemble_mean"]["trade_count"] > 0
