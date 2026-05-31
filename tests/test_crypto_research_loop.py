from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.crypto_research.backtest import BacktestConfig, run_variant_backtest
from quant_research_stack.crypto_research.data import build_chronological_periods, dataset_manifest_from_frame
from quant_research_stack.crypto_research.pbo import estimate_pbo
from quant_research_stack.crypto_research.reports import write_research_outputs
from quant_research_stack.crypto_research.strategies import StrategyVariant, generate_strategy_variants


def _synthetic_panel(rows: int = 240) -> pl.DataFrame:
    x = np.arange(rows, dtype=np.float64)
    close = 100.0 + (0.03 * x) + np.sin(x / 10.0)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2025, 1, 1),
                end=pl.datetime(2025, 1, 1) + pl.duration(minutes=rows - 1),
                interval="1m",
                eager=True,
            ),
            "symbol": ["BTCUSDT"] * rows,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(rows, 10.0),
            "maker_ratio": np.linspace(0.4, 0.6, rows),
            "no_of_trades": np.arange(rows) + 1,
            "liquidity_sum": np.full(rows, 100_000.0),
        }
    )


def test_dataset_manifest_and_periods_are_chronological() -> None:
    frame = _synthetic_panel(rows=600)

    manifest = dataset_manifest_from_frame(
        frame,
        dataset_id="synthetic",
        source_path=Path("synthetic.parquet"),
        timestamp_column="timestamp",
    )
    periods = build_chronological_periods(frame, timestamp_column="timestamp", holdout_fraction=0.2, validation_fraction=0.2)

    assert manifest.row_count == 600
    assert manifest.symbols == ["BTCUSDT"]
    assert manifest.timestamp_semantics == "bar close timestamp; signal uses only current and past bars"
    assert periods.development.end < periods.validation.start
    assert periods.validation.end < periods.holdout.start


def test_generate_strategy_variants_are_unique_and_broad() -> None:
    variants = generate_strategy_variants(target_count=160)

    ids = [variant.strategy_id for variant in variants]
    families = {variant.family for variant in variants}
    assert len(ids) >= 160
    assert len(ids) == len(set(ids))
    assert {"momentum", "mean_reversion", "breakout", "volatility", "liquidity", "paper_derived"}.issubset(families)
    assert all(variant.entry_rule for variant in variants)


def test_run_variant_backtest_applies_delay_and_turnover_costs() -> None:
    frame = _synthetic_panel(rows=100)
    variant = StrategyVariant(
        strategy_id="unit_momentum",
        family="momentum",
        feature_set="close_return",
        parameters={"lookback": 3, "threshold": 0.0, "vol_filter": "none"},
        horizon=1,
        entry_rule="sign(close / close.shift(lookback) - 1)",
        exit_rule="rebalance each bar",
        execution_assumption="one-bar delayed taker execution",
        cost_assumption="fee + half spread + slippage per unit turnover",
    )

    no_cost = run_variant_backtest(
        frame,
        variant,
        config=BacktestConfig(fee_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0, execution_delay_bars=1),
    )
    costed = run_variant_backtest(
        frame,
        variant,
        config=BacktestConfig(fee_bps=10.0, half_spread_bps=0.0, slippage_bps=0.0, execution_delay_bars=1),
    )
    delayed = run_variant_backtest(
        frame,
        variant,
        config=BacktestConfig(fee_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0, execution_delay_bars=2),
    )

    assert no_cost.metrics["trade_count"] > 0
    assert costed.metrics["net_total_return"] < no_cost.metrics["net_total_return"]
    assert delayed.metrics["execution_delay_bars"] == 2
    assert set(no_cost.trades.columns) >= {"timestamp", "side", "gross_return", "cost_return", "net_return"}


def test_run_variant_backtest_applies_cost_aware_score_filter() -> None:
    frame = _synthetic_panel(rows=120)
    variant = StrategyVariant(
        strategy_id="unit_momentum",
        family="momentum",
        feature_set="close_return",
        parameters={"lookback": 3, "threshold": 0.0, "vol_filter": "none"},
        horizon=1,
        entry_rule="sign(close / close.shift(lookback) - 1)",
        exit_rule="rebalance each bar",
        execution_assumption="one-bar delayed taker execution",
        cost_assumption="fee + half spread + slippage per unit turnover",
    )

    base = run_variant_backtest(frame, variant, config=BacktestConfig(fee_bps=4.0))
    filtered = run_variant_backtest(
        frame,
        variant,
        config=BacktestConfig(fee_bps=4.0, min_edge_to_cost_ratio=10.0),
    )

    assert filtered.metrics["trade_count"] < base.metrics["trade_count"]
    assert filtered.metrics["min_edge_to_cost_ratio"] == 10.0


def test_run_variant_backtest_applies_side_and_cooldown_filters() -> None:
    frame = _synthetic_panel(rows=120)
    variant = StrategyVariant(
        strategy_id="unit_random",
        family="baseline",
        feature_set="deterministic_random",
        parameters={"threshold": 0.0},
        horizon=1,
        entry_rule="deterministic pseudo-random sign",
        exit_rule="rebalance each bar",
        execution_assumption="one-bar delayed taker execution",
        cost_assumption="fee + half spread + slippage per unit turnover",
    )

    base = run_variant_backtest(frame, variant, config=BacktestConfig(fee_bps=0.0))
    long_only = run_variant_backtest(frame, variant, config=BacktestConfig(fee_bps=0.0, allowed_side="long_only"))
    cooled = run_variant_backtest(frame, variant, config=BacktestConfig(fee_bps=0.0, cooldown_bars=20))

    assert set(long_only.trades.get_column("side").unique().to_list()) <= {"long", "flat"}
    assert cooled.metrics["trade_count"] < base.metrics["trade_count"]
    assert cooled.metrics["cooldown_bars"] == 20


def test_estimate_pbo_flags_overfit_winners() -> None:
    scores = pl.DataFrame(
        {
            "strategy_id": ["stable", "lucky"] * 6,
            "block": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "net_sharpe": [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, -2.0, 1.0, -2.0, 1.0, -2.0],
        }
    )

    report = estimate_pbo(scores, score_column="net_sharpe", min_blocks=6)

    assert report.strategy_count == 2
    assert report.split_count > 0
    assert report.pbo > 0.0
    assert any(value < 0.0 for value in report.logit_ranks)


def test_write_research_outputs_smoke(tmp_path: Path) -> None:
    registry = pl.DataFrame(
        {
            "strategy_id": ["s1"],
            "family": ["momentum"],
            "feature_set": ["close_return"],
            "parameters_json": ["{}"],
            "horizon": [1],
            "entry_rule": ["sign"],
            "exit_rule": ["rebalance"],
            "execution_assumption": ["taker"],
            "cost_assumption": ["costed"],
        }
    )
    all_backtests = pl.DataFrame(
        {
            "strategy_id": ["s1"],
            "period": ["validation"],
            "net_total_return": [0.1],
            "net_daily_sharpe": [1.2],
            "max_drawdown": [-0.02],
            "trade_count": [10],
            "pass_gate": [False],
        }
    )

    write_research_outputs(
        output_dir=tmp_path,
        registry=registry,
        all_backtests=all_backtests,
        pbo_payload={"pbo": 0.7, "strategy_count": 1},
        best_candidates=[{"strategy_id": "s1", "net_daily_sharpe": 1.2}],
        holdout_rows=[],
        cost_sensitivity_rows=[],
        failure_reasons=["no strategy passed PBO"],
        commands=["PYTHONPATH=src uv run python scripts/crypto_strategy_research_loop.py"],
    )

    expected = {
        "strategy_registry.parquet",
        "all_backtests.parquet",
        "pbo_report.json",
        "pbo_report.md",
        "best_candidates_report.md",
        "cost_sensitivity_report.md",
        "holdout_report.md",
        "failure_report.md",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert json.loads((tmp_path / "pbo_report.json").read_text())["pbo"] == 0.7
