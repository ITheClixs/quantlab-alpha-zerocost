from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_research_stack.crypto_research.perps.training import (
    PerpWalkForwardConfig,
    train_perp_walk_forward,
)


def _feature_frame(rows: int = 160) -> pl.DataFrame:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "event_time": t0 + timedelta(seconds=i),
                "relative_spread": 0.0001,
                "l1_imbalance": (i % 10 - 5) / 10.0,
                "microprice_deviation": 0.00001 * (i % 7),
                "mid_return_1": 0.0001 * ((i % 3) - 1),
                "realized_vol_10": 0.001 + 0.00001 * (i % 5),
                "future_mid_return_5": 0.0002 * ((i % 5) - 2),
                "best_bid": 100.0 + i * 0.01,
                "best_ask": 100.1 + i * 0.01,
                "future_best_bid_5": 100.2 + i * 0.01,
                "future_best_ask_5": 100.3 + i * 0.01,
                "best_bid_size": 10.0,
                "best_ask_size": 10.0,
            }
            for i in range(rows)
        ]
    )


def test_perp_walk_forward_never_trains_on_or_after_test_rows() -> None:
    result = train_perp_walk_forward(
        _feature_frame(),
        config=PerpWalkForwardConfig(
            target_column="future_mid_return_5",
            min_train_rows=60,
            test_rows=25,
            step_rows=25,
            max_folds=2,
        ),
    )

    assert result.predictions.height > 0
    assert {"pred_ridge", "pred_hist_gradient", "pred_ensemble_mean"}.issubset(result.predictions.columns)
    for fold in result.fold_specs:
        assert fold["train_end_time"] < fold["test_start_time"]


def test_perp_walk_forward_respects_embargo_rows() -> None:
    result = train_perp_walk_forward(
        _feature_frame(),
        config=PerpWalkForwardConfig(
            target_column="future_mid_return_5",
            min_train_rows=60,
            test_rows=25,
            step_rows=25,
            embargo_rows=3,
            max_folds=1,
        ),
    )

    fold = result.fold_specs[0]

    assert fold["train_rows"] == 60
    assert fold["test_start_row"] - fold["train_end_row"] == 3


def test_perp_walk_forward_rejects_missing_target() -> None:
    frame = _feature_frame().drop("future_mid_return_5")

    try:
        train_perp_walk_forward(frame, config=PerpWalkForwardConfig(target_column="future_mid_return_5"))
    except ValueError as exc:
        assert "missing target column" in str(exc)
    else:
        raise AssertionError("expected missing target column error")


def test_perp_walk_forward_excludes_train_rows_at_test_timestamp() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    rows = []
    for i in range(80):
        for symbol in ("BTCUSDT", "ETHUSDT"):
            rows.append(
                {
                    "symbol": symbol,
                    "event_time": t0 + timedelta(seconds=i),
                    "relative_spread": 0.0001,
                    "l1_imbalance": (i % 10 - 5) / 10.0,
                    "microprice_deviation": 0.00001 * (i % 7),
                    "mid_return_1": 0.0001 * ((i % 3) - 1),
                    "realized_vol_10": 0.001,
                    "future_mid_return_5": 0.0002 * ((i % 5) - 2),
                    "best_bid": 100.0 + i * 0.01,
                    "best_ask": 100.1 + i * 0.01,
                    "future_best_bid_5": 100.2 + i * 0.01,
                    "future_best_ask_5": 100.3 + i * 0.01,
                    "best_bid_size": 10.0,
                    "best_ask_size": 10.0,
                }
            )

    result = train_perp_walk_forward(
        pl.DataFrame(rows),
        config=PerpWalkForwardConfig(
            target_column="future_mid_return_5",
            min_train_rows=60,
            test_rows=20,
            max_folds=1,
        ),
    )

    assert result.fold_specs[0]["train_end_time"] < result.fold_specs[0]["test_start_time"]
