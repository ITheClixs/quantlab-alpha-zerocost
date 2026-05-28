"""Synthetic smoke for the VRP pipeline."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.signal_research.vrp import (
    VRPSpec,
    render_vrp_report,
    run_vrp_pipeline,
)


def _synthetic_underlying_and_vol(*, n_days: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    start = dt.date(2014, 1, 2)
    dates: list[dt.date] = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d = d + dt.timedelta(days=1)
    # Synthetic SPY-like price series
    rets = rng.standard_normal(n_days) * 0.01 + 0.0003
    price = 100.0 * np.cumprod(1.0 + rets)
    vol = rng.uniform(5e7, 5e8, size=n_days)
    underlying = pl.DataFrame({
        "date": dates,
        "symbol": ["SPY"] * n_days,
        "open": price,
        "high": price * 1.005,
        "low": price * 0.995,
        "close": price,
        "volume": vol,
    })
    # Synthetic VIX-family. VIX is realized-vol-like with a premium.
    realised_21 = (
        pl.Series(rets).pow(2).rolling_sum(window_size=21).fill_null(0.0001).to_numpy()
        * (252.0 / 21.0)
    )
    vix = np.sqrt(np.maximum(realised_21, 1e-6)) * 100.0 + 3.0  # +3 premium
    vix9d = vix * (0.98 + rng.standard_normal(n_days) * 0.02)
    vvix = 80.0 + rng.standard_normal(n_days) * 10.0
    skew = 130.0 + rng.standard_normal(n_days) * 5.0
    vol_features = pl.DataFrame({
        "date": dates,
        "vix": vix,
        "vix9d": vix9d,
        "vvix": vvix,
        "skew": skew,
    })
    return underlying, vol_features


def test_vrp_pipeline_runs_on_synthetic(tmp_path: Path) -> None:
    underlying, vol_features = _synthetic_underlying_and_vol(n_days=1500, seed=0)
    spec = VRPSpec(
        target_symbol="SPY",
        start=dt.date(2014, 1, 2),
        end=dt.date(2020, 1, 1),
        dev_end=dt.date(2018, 12, 31),
        holdout_start=dt.date(2019, 1, 1),
        bootstrap_n_resamples=200,
    )
    report = run_vrp_pipeline(
        underlying=underlying, vol_features=vol_features, spec=spec,
    )
    assert len(report.variant_results) == 6
    assert len(report.baseline_results) == 3
    assert 0.0 <= report.cross.pbo_raw_global <= 1.0
    out = render_vrp_report(report, output_path=tmp_path / "vrp.md")
    body = out.read_text()
    assert "VRP Index Validation" in body
    assert "Pre-registered failure modes" in body
    assert "vrp_long_only" in body
    assert "spy_buy_and_hold" in body
    assert "options_implied_vol" in body
