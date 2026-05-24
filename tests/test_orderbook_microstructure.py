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


def test_orderbook_backtest_filters_predictions_that_do_not_clear_costs() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "event_time": [1, 2, 3],
            "prediction": [0.0003, 0.0020, -0.0015],
            "future_mid_return_1": [0.0030, 0.0030, -0.0040],
            "relative_spread": [0.0010, 0.0005, 0.0020],
        }
    )

    result = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            min_edge_over_cost=0.0001,
            fee_bps=1.0,
        ),
    )

    assert result.metrics["candidate_count"] == 3
    assert result.metrics["trade_count"] == 1
    assert result.metrics["filtered_count"] == 2
    assert result.trades["event_time"].to_list() == [2]
    assert result.trades["predicted_edge_over_cost"].to_list() == pytest.approx([0.0013])


def test_orderbook_backtest_filters_bad_spread_and_directional_depth() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "event_time": [1, 2, 3],
            "prediction": [0.0020, -0.0020, 0.0020],
            "future_mid_return_1": [0.0030, -0.0030, 0.0030],
            "relative_spread": [0.0003, 0.0002, 0.0020],
            "bid_depth_1": [10.0, 3.0, 10.0],
            "ask_depth_1": [10.0, 10.0, 10.0],
        }
    )

    result = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            max_relative_spread=0.001,
            min_entry_depth=5.0,
            fee_bps=1.0,
        ),
    )

    assert result.metrics["candidate_count"] == 3
    assert result.metrics["trade_count"] == 1
    assert result.metrics["filtered_count"] == 2
    assert result.trades.select(["event_time", "position_side"]).rows() == [(1, 1.0)]


def test_orderbook_walk_forward_predicts_only_out_of_sample_rows() -> None:
    rows = 80
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * rows,
            "event_time": list(range(rows)),
            "row_index": list(range(rows)),
            "relative_spread": [0.0001] * rows,
            "bid_depth_1": [3.0] * rows,
            "ask_depth_1": [4.0] * rows,
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
    assert set(result.predictions.columns) >= {
        "bid_depth_1",
        "ask_depth_1",
        "pred_ridge",
        "pred_hist_gradient",
        "pred_ensemble_mean",
    }
    assert result.backtest_metrics["ensemble_mean"]["trade_count"] > 0
