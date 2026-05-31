from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.backtest.equity_signal import load_signal_model, normalize_equity_ohlcv, predict_signal_frame
from quant_research_stack.backtest.equity_walk_forward import (
    MODEL_NAMES,
    EquityWalkForwardConfig,
    run_equity_walk_forward,
    save_signal_artifacts,
    train_final_equity_models,
    walk_forward_date_splits,
)


def _raw_ohlcv(days: int = 12) -> pl.DataFrame:
    rows = []
    for symbol, base, drift in [("AAA", 10.0, 0.20), ("BBB", 20.0, -0.12), ("CCC", 30.0, 0.08)]:
        for idx in range(days):
            close = base + drift * idx + 0.02 * (idx % 3)
            rows.append(
                {
                    "date": f"2024-01-{idx + 2:02d}",
                    "symbol": symbol,
                    "open": close - 0.03,
                    "high": close + 0.20,
                    "low": close - 0.20,
                    "close": close,
                    "volume": 1000.0 + idx * 10.0 + base,
                }
            )
    return pl.DataFrame(rows)


def _normalized(days: int = 12) -> pl.DataFrame:
    return normalize_equity_ohlcv(
        _raw_ohlcv(days),
        dataset_id="unit",
        date_column="date",
        symbol_column="symbol",
    )


def test_walk_forward_date_splits_use_only_past_dates() -> None:
    frame = _normalized(10).drop_nulls(["future_return_1"])

    folds = walk_forward_date_splits(
        frame,
        min_train_dates=3,
        test_window_dates=2,
        step_dates=2,
        max_folds=2,
    )

    assert len(folds) == 2
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert fold.train_dates >= 3
        assert fold.test_dates <= 2


def test_run_equity_walk_forward_emits_oos_predictions_and_metrics() -> None:
    config = EquityWalkForwardConfig(
        min_train_dates=4,
        test_window_dates=2,
        step_dates=2,
        max_folds=2,
        max_train_rows_per_fold=None,
        hist_gradient_max_iter=5,
        cost_bps=0.0,
        selection_fraction=0.5,
    )

    result = run_equity_walk_forward(_normalized(12), config)

    assert set(result.model_metrics) == set(MODEL_NAMES)
    assert set(result.backtest_metrics) == set(MODEL_NAMES)
    assert result.predictions.height > 0
    assert {"pred_ridge", "pred_hist_gradient", "pred_ensemble_mean"}.issubset(result.predictions.columns)
    assert len(result.fold_metrics) == 2
    assert result.backtest_metrics["ensemble_mean"]["n_days"] > 0


def test_final_artifacts_are_loadable_by_existing_signal_loader(tmp_path: Path) -> None:
    config = EquityWalkForwardConfig(
        min_train_dates=4,
        test_window_dates=2,
        step_dates=2,
        max_folds=1,
        hist_gradient_max_iter=5,
    )
    frame = _normalized(12)
    result = run_equity_walk_forward(frame, config)
    models = train_final_equity_models(frame, config, feature_columns=result.feature_columns)

    paths = save_signal_artifacts(
        models=models,
        feature_columns=result.feature_columns,
        target_column=config.target_column,
        output_dir=tmp_path / "models",
        metadata={"dataset": "unit"},
        compression=0,
    )

    artifact = load_signal_model(paths["ensemble_mean"])
    one_row = frame.drop_nulls([config.target_column]).head(1)
    predicted = predict_signal_frame(one_row, artifact)

    assert predicted["prediction"].to_numpy().shape == (1,)
    assert np.isfinite(predicted["prediction"].to_numpy()[0])
