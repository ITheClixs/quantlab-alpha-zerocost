"""Portfolio construction (spec §5.5)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.portfolio import (
    PortfolioBuildConfig,
    build_target_positions,
)


def _signals(date_: date, n: int = 20) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "execution_date": [date_] * n,
            "symbol": [f"S{i}" for i in range(n)],
            "y_xs_pred": [(i - n / 2) / n for i in range(n)],
            "adv_20d_dollar_lag1": [1e8] * n,
            "tradable": [True] * n,
            "in_pit_universe": [True] * n,
            "fill_price": [100.0] * n,
            "borrow_tier": ["general"] * n,
        }
    )


def test_equal_weight_dollar_neutral_book() -> None:
    cfg = PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0)
    pos = build_target_positions(signals=_signals(date(2020, 1, 3)), config=cfg, cohort="full_universe")
    longs = pos.filter(pl.col("signed_target_notional") > 0)
    shorts = pos.filter(pl.col("signed_target_notional") < 0)
    assert longs.height >= 2
    assert shorts.height >= 2


def test_minimum_bucket_full_universe_skips_when_insufficient() -> None:
    cfg = PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0)
    sig = _signals(date(2020, 1, 3), n=5)
    pos = build_target_positions(signals=sig, config=cfg, cohort="full_universe")
    assert pos.is_empty()


def test_per_name_adv_cap_overrides_equal_weight() -> None:
    cfg = PortfolioBuildConfig(
        q_quantile=0.10, target_gross=1.0, equity=10_000_000.0, adv_participation_pct=0.01,
    )
    sig = _signals(date(2020, 1, 3), n=20).with_columns(pl.lit(1_000_000.0).alias("adv_20d_dollar_lag1"))
    pos = build_target_positions(signals=sig, config=cfg, cohort="full_universe")
    assert pos["signed_target_notional"].abs().max() <= 10_000.0 + 1e-6


def test_out_of_universe_rows_are_dropped() -> None:
    cfg = PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0)
    sig = _signals(date(2020, 1, 3)).with_columns(
        pl.when(pl.col("symbol") == "S0").then(False).otherwise(True).alias("in_pit_universe")
    )
    pos = build_target_positions(signals=sig, config=cfg, cohort="full_universe")
    assert "S0" not in pos["symbol"].to_list()
