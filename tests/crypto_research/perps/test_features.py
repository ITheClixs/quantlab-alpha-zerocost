from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_research_stack.crypto_research.perps.features import build_l1_features


def test_l1_features_use_only_current_and_past_rows() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "event_time": [t0 + timedelta(seconds=i) for i in range(4)],
            "best_bid": [100.0, 101.0, 102.0, 103.0],
            "best_ask": [100.2, 101.2, 102.2, 103.2],
            "best_bid_size": [10.0, 20.0, 30.0, 40.0],
            "best_ask_size": [15.0, 25.0, 35.0, 45.0],
        }
    )

    out = build_l1_features(frame, horizons=(1, 2), rolling_windows=(2,))

    assert out["mid_price"][0] == 100.1
    assert out["future_mid_return_1"][0] == out["mid_price"][1] / out["mid_price"][0] - 1.0
    assert out["mid_return_1"][0] is None
    assert out["mid_return_1"][1] == out["mid_price"][1] / out["mid_price"][0] - 1.0
    assert out["future_best_bid_2"][0] == 102.0
    assert out["future_best_ask_2"][0] == 102.2


def test_l1_features_compute_microprice_and_imbalance() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["ETHUSDT"],
            "event_time": [t0],
            "best_bid": [100.0],
            "best_ask": [101.0],
            "best_bid_size": [3.0],
            "best_ask_size": [1.0],
        }
    )

    out = build_l1_features(frame, horizons=(1,), rolling_windows=(2,))

    assert out["spread"][0] == 1.0
    assert out["relative_spread"][0] == 1.0 / 100.5
    assert out["l1_imbalance"][0] == 0.5
    assert out["microprice"][0] == ((101.0 * 3.0) + (100.0 * 1.0)) / 4.0
    assert out["microprice_deviation"][0] == out["microprice"][0] / out["mid_price"][0] - 1.0


def test_l1_features_do_not_cross_symbols() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
            "event_time": [t0, t0, t0 + timedelta(seconds=1), t0 + timedelta(seconds=1)],
            "best_bid": [100.0, 200.0, 101.0, 201.0],
            "best_ask": [100.2, 200.2, 101.2, 201.2],
            "best_bid_size": [10.0, 10.0, 10.0, 10.0],
            "best_ask_size": [10.0, 10.0, 10.0, 10.0],
        }
    )

    out = build_l1_features(frame, horizons=(1,), rolling_windows=(2,))
    btc = out.filter(pl.col("symbol") == "BTCUSDT")
    eth = out.filter(pl.col("symbol") == "ETHUSDT")

    assert btc["future_best_bid_1"][0] == 101.0
    assert eth["future_best_bid_1"][0] == 201.0
    assert btc["mid_return_1"][0] is None
    assert eth["mid_return_1"][0] is None


def test_l1_features_null_microprice_when_top_size_is_zero() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "event_time": [t0],
            "best_bid": [100.0],
            "best_ask": [100.2],
            "best_bid_size": [0.0],
            "best_ask_size": [0.0],
        }
    )

    out = build_l1_features(frame, horizons=(1,), rolling_windows=(2,))

    assert out["l1_imbalance"][0] is None
    assert out["microprice"][0] is None
    assert out["microprice_deviation"][0] is None
