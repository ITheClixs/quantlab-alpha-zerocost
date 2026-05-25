from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.crypto_research.gresearch import (
    build_gresearch_features,
    chronological_split,
    portfolio_backtest,
)
from scripts.crypto_gresearch_signal_search import (
    GResearchVariant,
    _delayed_execution_frame,
    _gate_row,
    _generate_variants,
    _model_specs,
    _registry_frame,
    _select_validation_candidates,
)


def _synthetic_gresearch(rows: int = 240) -> pl.DataFrame:
    timestamps = np.arange(rows, dtype=np.int64) * 60 + 1_600_000_000
    close_a = 100.0 + np.arange(rows, dtype=np.float64) * 0.05
    close_b = 50.0 - np.arange(rows, dtype=np.float64) * 0.01
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "Asset_ID": [0] * rows,
                    "Count": np.full(rows, 10.0),
                    "Open": close_a,
                    "High": close_a * 1.001,
                    "Low": close_a * 0.999,
                    "Close": close_a,
                    "Volume": np.full(rows, 100.0),
                    "VWAP": close_a,
                    "Target": np.full(rows, 0.001),
                }
            ),
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "Asset_ID": [1] * rows,
                    "Count": np.full(rows, 10.0),
                    "Open": close_b,
                    "High": close_b * 1.001,
                    "Low": close_b * 0.999,
                    "Close": close_b,
                    "Volume": np.full(rows, 100.0),
                    "VWAP": close_b,
                    "Target": np.full(rows, -0.001),
                }
            ),
        ]
    )


def test_build_gresearch_features_uses_past_features_and_forward_return() -> None:
    features = build_gresearch_features(_synthetic_gresearch(), horizon_minutes=15)

    assert {"ret1", "ret5", "vol15", "future_return_15"}.issubset(features.columns)
    assert features["timestamp"].min() >= 1_600_000_000
    assert features.drop_nulls(["future_return_15"]).height < features.height


def test_chronological_split_orders_periods() -> None:
    features = build_gresearch_features(_synthetic_gresearch(), horizon_minutes=15).drop_nulls()

    split = chronological_split(features)

    assert split.development["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.holdout["timestamp"].min()


def test_portfolio_backtest_costs_and_thresholds() -> None:
    frame = build_gresearch_features(_synthetic_gresearch(), horizon_minutes=15).drop_nulls()
    predictions = frame.with_columns(
        pl.when(pl.col("Asset_ID") == 0).then(0.01).otherwise(-0.01).alias("prediction")
    )

    loose = portfolio_backtest(
        predictions,
        threshold=0.0,
        horizon_minutes=15,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    strict = portfolio_backtest(
        predictions,
        threshold=1.0,
        horizon_minutes=15,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    assert loose.metrics["trade_count"] > 0
    assert loose.metrics["net_total_return"] > strict.metrics["net_total_return"]
    assert strict.metrics["trade_count"] == 0


def test_portfolio_backtest_applies_side_policy_and_cost_multiplier() -> None:
    frame = build_gresearch_features(_synthetic_gresearch(), horizon_minutes=15).drop_nulls()
    predictions = frame.with_columns(
        pl.when(pl.col("Asset_ID") == 0).then(0.01).otherwise(-0.01).alias("prediction")
    )

    base = portfolio_backtest(
        predictions,
        threshold=0.0,
        horizon_minutes=15,
        fee_bps=1.0,
        slippage_bps=1.0,
    )
    long_only = portfolio_backtest(
        predictions,
        threshold=0.0,
        horizon_minutes=15,
        fee_bps=1.0,
        slippage_bps=1.0,
        side_policy="long_only",
    )
    stressed = portfolio_backtest(
        predictions,
        threshold=0.0,
        horizon_minutes=15,
        fee_bps=1.0,
        slippage_bps=1.0,
        cost_multiplier=2.0,
    )

    assert set(long_only.trades.get_column("side").unique().to_list()) == {1.0}
    assert long_only.metrics["trade_count"] < base.metrics["trade_count"]
    assert stressed.metrics["net_total_return"] < base.metrics["net_total_return"]
    assert stressed.metrics["cost_multiplier"] == 2.0


def test_gresearch_registry_records_periods_and_label() -> None:
    variant = GResearchVariant(
        strategy_id="ridge_core_h15_q95",
        family="linear",
        model_name="ridge",
        feature_set="core",
        label_name="Target",
        horizon_minutes=15,
        threshold_quantile=0.95,
        threshold=0.001,
        side_policy="both",
        cost_multiplier=1.0,
    )

    registry = _registry_frame(
        [variant],
        periods_payload={
            "development": {"start": "2021-01-01", "end": "2021-06-01"},
            "validation": {"start": "2021-06-02", "end": "2021-10-01"},
            "holdout": {"start": "2021-10-02", "end": "2022-01-01"},
        },
    )

    assert registry.height == 1
    row = registry.row(0, named=True)
    assert row["strategy_id"] == "ridge_core_h15_q95"
    assert row["label_name"] == "Target"
    assert row["pass_fail_status"] == "not_evaluated"


def test_gresearch_gate_requires_sharpe_monthly_trades_and_stress() -> None:
    passing, reasons = _gate_row(
        {
            "net_total_return": 0.5,
            "net_daily_sharpe": 5.5,
            "average_monthly_net_return": 0.12,
            "max_drawdown": -0.08,
            "trade_count": 250,
        },
        pbo=0.05,
        cost_2x_positive=True,
        delay_positive=True,
        best_day_concentration=0.2,
        promotion_sharpe=5.0,
        promotion_monthly_net=0.10,
        promotion_min_trades=100,
        null_baseline_dominates=False,
    )
    failing, failing_reasons = _gate_row(
        {
            "net_total_return": 0.5,
            "net_daily_sharpe": 4.9,
            "average_monthly_net_return": 0.09,
            "max_drawdown": -0.08,
            "trade_count": 25,
        },
        pbo=0.30,
        cost_2x_positive=False,
        delay_positive=False,
        best_day_concentration=0.8,
        promotion_sharpe=5.0,
        promotion_monthly_net=0.10,
        promotion_min_trades=100,
        null_baseline_dominates=True,
    )

    assert passing is True
    assert reasons == []
    assert failing is False
    assert any("Sharpe" in reason for reason in failing_reasons)
    assert any("monthly" in reason for reason in failing_reasons)
    assert any("trade count" in reason for reason in failing_reasons)
    assert any("null baseline" in reason for reason in failing_reasons)


def test_delayed_execution_frame_replaces_horizon_return_with_later_return() -> None:
    frame = build_gresearch_features(_synthetic_gresearch(rows=120), horizon_minutes=15).drop_nulls()

    delayed = _delayed_execution_frame(frame, horizon_minutes=15)

    first_asset = frame.filter(pl.col("Asset_ID") == 0).sort("timestamp")
    delayed_first_asset = delayed.filter(pl.col("Asset_ID") == 0).sort("timestamp")
    assert delayed_first_asset.get_column("future_return_15")[0] == first_asset.get_column("future_return_15")[15]


def test_generate_variants_balances_horizons_before_truncating() -> None:
    variants = _generate_variants(
        horizons=[5, 15],
        feature_sets=["core"],
        model_specs=_model_specs("linear")[:2],
        threshold_quantiles=[0.90, 0.99],
        target_count=6,
    )

    horizons = {variant.horizon_minutes for variant in variants}
    assert horizons == {5, 15}


def test_select_validation_candidates_requires_delay_robustness() -> None:
    rows = [
        {
            "strategy_id": "fast_brittle",
            "period": "validation",
            "trade_count": 1000,
            "net_daily_sharpe": 10.0,
            "average_monthly_net_return": 0.5,
            "net_total_return": 1.0,
            "delay_net_total_return": -0.1,
        },
        {
            "strategy_id": "slower_robust",
            "period": "validation",
            "trade_count": 200,
            "net_daily_sharpe": 3.0,
            "average_monthly_net_return": 0.1,
            "net_total_return": 0.2,
            "delay_net_total_return": 0.01,
        },
    ]

    selected = _select_validation_candidates(rows, limit=2, min_trades=100)

    assert [row["strategy_id"] for row in selected] == ["slower_robust"]
