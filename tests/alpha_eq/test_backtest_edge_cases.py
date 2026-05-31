"""Edge cases — empty universe, insufficient bucket, ADV-cap (spec §6.2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.portfolio import (
    PortfolioBuildConfig,
    build_target_positions,
)


def test_empty_universe_skips_date() -> None:
    sig = pl.DataFrame(
        schema={
            "execution_date": pl.Date, "symbol": pl.Utf8, "y_xs_pred": pl.Float64,
            "adv_20d_dollar_lag1": pl.Float64, "tradable": pl.Boolean,
            "in_pit_universe": pl.Boolean, "fill_price": pl.Float64,
            "borrow_tier": pl.Utf8,
        }
    )
    pos = build_target_positions(
        signals=sig,
        config=PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0),
        cohort="full_universe",
    )
    assert pos.is_empty()


def test_all_names_adv_capped_results_in_capped_book() -> None:
    sig = pl.DataFrame(
        {
            "execution_date": [date(2020, 1, 3)] * 25,
            "symbol": [f"S{i}" for i in range(25)],
            "y_xs_pred": list(np.linspace(-1, 1, 25)),
            "adv_20d_dollar_lag1": [1_000.0] * 25,
            "tradable": [True] * 25,
            "in_pit_universe": [True] * 25,
            "fill_price": [100.0] * 25,
            "borrow_tier": ["general"] * 25,
        }
    )
    pos = build_target_positions(
        signals=sig,
        config=PortfolioBuildConfig(
            q_quantile=0.10, target_gross=1.0, equity=1_000_000.0, adv_participation_pct=0.01,
        ),
        cohort="full_universe",
    )
    assert pos["signed_target_notional"].abs().max() <= 10.0 + 1e-6
