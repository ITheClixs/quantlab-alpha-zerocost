from __future__ import annotations

import math

import polars as pl
import pytest

from quant_research_stack.backtest.orderbook_signal import (
    OrderBookBacktestConfig,
    OrderBookWalkForwardConfig,
    orderbook_features_from_frame,
    run_orderbook_signal_backtest,
    run_orderbook_walk_forward,
)
from scripts.orderbook_microstructure_benchmark import _best_backtest


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


def test_orderbook_features_include_future_execution_audit_prices() -> None:
    raw = pl.DataFrame(
        {
            "E": [1_000, 2_000],
            "T": [900, 1_900],
            "u": [10, 11],
            "bids": [
                '[["99.0", "4.0"]]',
                '[["101.0", "2.0"]]',
            ],
            "asks": [
                '[["101.0", "2.0"]]',
                '[["103.0", "2.0"]]',
            ],
        }
    )

    features = orderbook_features_from_frame(
        raw,
        dataset_id="unit",
        source_file="unit.parquet",
        symbol="BTCUSDT",
        horizons=(1,),
        depth_levels=(1,),
    )

    assert features.select(["future_mid_price_1", "future_best_bid_1", "future_best_ask_1"]).row(0) == (102.0, 101.0, 103.0)


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


def test_orderbook_backtest_emits_price_audit_and_gross_net_hit_rates() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "event_time": [1_767_225_600_000, 1_767_312_000_000, 1_767_398_400_000],
            "prediction": [0.03, -0.04, 0.02],
            "future_mid_return_1": [0.03, -0.04, 0.005],
            "mid_price": [100.0, 100.0, 100.0],
            "best_bid": [99.0, 99.5, 99.5],
            "best_ask": [101.0, 100.5, 100.5],
            "future_mid_price_1": [103.0, 96.0, 100.5],
            "future_best_bid_1": [102.5, 95.5, 100.0],
            "future_best_ask_1": [103.5, 96.5, 101.0],
            "relative_spread": [0.02, 0.01, 0.01],
        }
    )

    result = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            fee_bps=0.0,
        ),
    )

    assert result.metrics["trade_count"] == 3
    assert result.metrics["gross_hit_rate"] == 1.0
    assert result.metrics["net_hit_rate"] == pytest.approx(2.0 / 3.0)
    assert result.metrics["hit_rate"] == result.metrics["net_hit_rate"]
    assert result.metrics["long_trade_count"] == 2
    assert result.metrics["short_trade_count"] == 1
    assert set(result.trades.columns) >= {
        "timestamp",
        "side",
        "predicted_return",
        "entry_mid",
        "entry_bid",
        "entry_ask",
        "exit_mid",
        "exit_bid",
        "exit_ask",
        "realized_mid_return",
        "gross_return",
        "spread_cost_return",
        "fee_cost_return",
        "slippage_cost_return",
        "estimated_round_trip_cost",
        "edge_to_cost_ratio",
        "net_return",
        "holding_horizon",
        "prediction_direction_correct",
        "gross_hit",
        "net_hit",
    }
    assert result.trades["side"].to_list() == ["long", "short", "long"]
    assert result.trades["spread_cost_return"].round(6).to_list() == [0.015, 0.01, 0.01]
    assert result.trades["net_return"].round(6).to_list() == [0.015, 0.03, -0.005]
    assert result.trades["prediction_direction_correct"].to_list() == [True, True, True]


def test_orderbook_backtest_supports_cost_regimes_and_inverted_signals() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "event_time": [1, 2],
            "prediction": [0.03, 0.03],
            "future_mid_return_1": [0.02, 0.02],
            "mid_price": [100.0, 100.0],
            "best_bid": [99.5, 99.5],
            "best_ask": [100.5, 100.5],
            "future_mid_price_1": [102.0, 102.0],
            "future_best_bid_1": [101.5, 101.5],
            "future_best_ask_1": [102.5, 102.5],
            "relative_spread": [0.01, 0.01],
        }
    )

    no_cost = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            spread_cost_multiplier=0.0,
            fee_bps=0.0,
        ),
    )
    fee_only = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            spread_cost_multiplier=0.0,
            fee_bps=1.0,
        ),
    )
    inverted = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            spread_cost_multiplier=0.0,
            fee_bps=0.0,
            invert_signal=True,
        ),
    )

    assert no_cost.trades["net_return"].to_list() == pytest.approx([0.02, 0.02])
    assert fee_only.trades["fee_cost_return"].to_list() == pytest.approx([0.0002, 0.0002])
    assert fee_only.trades["net_return"].to_list() == pytest.approx([0.0198, 0.0198])
    assert inverted.trades["side"].to_list() == ["short", "short"]
    assert inverted.metrics["gross_hit_rate"] == 0.0
    assert inverted.metrics["total_return"] < 0.0


def test_orderbook_backtest_filters_by_edge_to_cost_ratio() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "event_time": [1, 2],
            "prediction": [0.03, 0.015],
            "future_mid_return_1": [0.04, 0.04],
            "relative_spread": [0.01, 0.01],
        }
    )

    result = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            fee_bps=0.0,
            min_edge_to_cost_ratio=2.0,
        ),
    )

    assert result.metrics["trade_count"] == 1
    assert result.trades["prediction"].to_list() == [0.03]
    assert result.trades["edge_to_cost_ratio"].to_list() == pytest.approx([3.0])


def test_orderbook_backtest_reports_trade_and_daily_sharpe() -> None:
    returns = [0.02, 0.01, -0.005]
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "event_time": [1_767_225_600_000, 1_767_312_000_000, 1_767_398_400_000],
            "prediction": [0.01, 0.01, 0.01],
            "future_mid_return_1": returns,
            "relative_spread": [0.0, 0.0, 0.0],
        }
    )

    result = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            fee_bps=0.0,
        ),
    )

    mean_return = sum(returns) / len(returns)
    std_return = math.sqrt(sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1))
    assert result.metrics["per_trade_sharpe_ratio"] == pytest.approx(mean_return / std_return)
    assert result.metrics["trade_sharpe_ratio"] == pytest.approx(mean_return / std_return * math.sqrt(len(returns)))
    assert result.metrics["daily_sharpe_ratio"] == pytest.approx(mean_return / std_return * math.sqrt(252.0))
    assert result.metrics["daily_return_count"] == 3


def test_orderbook_backtest_penalizes_constant_losing_sharpe() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "event_time": [1_767_225_600_000, 1_767_312_000_000, 1_767_398_400_000],
            "prediction": [0.01, 0.01, 0.01],
            "future_mid_return_1": [-0.001, -0.001, -0.001],
            "relative_spread": [0.0, 0.0, 0.0],
        }
    )

    result = run_orderbook_signal_backtest(
        frame,
        config=OrderBookBacktestConfig(
            prediction_column="prediction",
            target_column="future_mid_return_1",
            fee_bps=0.0,
        ),
    )

    assert result.metrics["trade_sharpe_ratio"] < 0.0
    assert result.metrics["daily_sharpe_ratio"] < 0.0


def test_orderbook_best_backtest_prioritizes_sharpe_before_total_return() -> None:
    best = _best_backtest(
        [
            {
                "model": "high_return",
                "trade_count": 10,
                "daily_sharpe_ratio": 0.2,
                "trade_sharpe_ratio": 0.3,
                "total_return": 0.5,
                "hit_rate": 0.7,
            },
            {
                "model": "high_sharpe",
                "trade_count": 10,
                "daily_sharpe_ratio": 1.2,
                "trade_sharpe_ratio": 1.5,
                "total_return": 0.05,
                "hit_rate": 0.6,
            },
        ]
    )

    assert best["model"] == "high_sharpe"


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
