"""Exposure diagnostics (spec §5.12)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.exposure import (
    compute_daily_exposures,
    rolling_spy_beta,
    top_n_contributors,
)


def test_compute_daily_exposures_shapes() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signed_notional": [50_000.0, 30_000.0, -40_000.0, -40_000.0],
            "sector": ["tech", "tech", "finance", "energy"],
        }
    )
    expo = compute_daily_exposures(positions=positions)
    row = expo.row(0, named=True)
    assert abs(row["net_exposure"] - 0.0) < 1e-6
    assert abs(row["gross_exposure"] - 160_000.0) < 1e-6
    assert row["sector_long_tech"] == 80_000.0
    assert row["sector_short_energy"] == 40_000.0


def test_rolling_spy_beta_returns_one_per_date() -> None:
    n = 80
    rng = np.random.default_rng(0)
    spy = rng.standard_normal(n) * 0.01
    port = spy * 0.5 + rng.standard_normal(n) * 0.005
    dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), interval="1d", eager=True).head(n)
    df = pl.DataFrame({"date": dates, "portfolio_return": port, "spy_return": spy})
    out = rolling_spy_beta(df, window=60)
    assert "rolling_spy_beta" in out.columns
    assert out["rolling_spy_beta"].drop_nulls().to_numpy()[-1] > 0.0


def test_top_n_contributors() -> None:
    pnl = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "net_pnl": [100.0, -50.0, 200.0, -10.0, 5.0],
        }
    )
    top = top_n_contributors(pnl, by="symbol", n=2)
    assert top["symbol"].to_list() == ["C", "A"]
