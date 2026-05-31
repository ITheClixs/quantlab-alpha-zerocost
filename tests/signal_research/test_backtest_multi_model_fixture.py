"""Smoke test for multi-model side-by-side backtest."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.signal_research.backtests.multi_model_fixture import (
    FixtureSpec,
    render_comparison_report,
    run_all_models_on_fixture,
)


def _synthetic_bars(
    *, n_days: int = 800, n_symbols: int = 12, seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_days) * 0.01
    start = dt.date(2018, 1, 2)
    dates: list[dt.date] = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d = d + dt.timedelta(days=1)
    rows = []
    for s in range(n_symbols):
        beta = 0.5 + 0.5 * rng.standard_normal()
        idio = rng.standard_normal(n_days) * 0.008
        rets = beta * factor + idio
        price = 100.0 * np.cumprod(1.0 + rets)
        vol = rng.uniform(5e5, 5e6, size=n_days)
        for t, dd in enumerate(dates):
            rows.append({
                "date": dd, "symbol": f"S{s:02d}",
                "open": float(price[t]),
                "high": float(price[t] * (1.0 + abs(idio[t]) * 0.5)),
                "low": float(price[t] * (1.0 - abs(idio[t]) * 0.5)),
                "close": float(price[t]),
                "volume": float(vol[t]),
            })
    return pl.DataFrame(rows)


def test_all_models_run_and_report_renders(tmp_path: Path) -> None:
    bars = _synthetic_bars(n_days=800, n_symbols=12)
    spec = FixtureSpec(
        universe_tickers=[f"S{i:02d}" for i in range(12)],
        start=dt.date(2018, 1, 2),
        end=dt.date(2022, 1, 1),
        dev_end=dt.date(2020, 12, 31),
        holdout_start=dt.date(2021, 1, 1),
        pca_window=60,
        n_pca_components=2,
        z_entry=1.0,
        gkx_label_horizon=5,
        gkx_n_estimators=50,
        gkx_walk_forward_folds=3,
        gkx_walk_forward_embargo=5,
        equity=100_000.0,
        q_quantile=0.30,
        cohort="focused_basket",
    )
    results = run_all_models_on_fixture(bars=bars, spec=spec)
    assert "raw_avellaneda_lee" in results
    assert "crossectional_momentum_12_1" in results
    assert "gkx_lightgbm" in results
    assert "triple_barrier_meta_av_lee" in results
    for r in results.values():
        assert "sharpe" in r.dev_metrics
        assert "sharpe" in r.holdout_metrics
    report = render_comparison_report(
        results, spec=spec, output_path=tmp_path / "compare.md"
    )
    body = report.read_text()
    assert "Multi-Model Comparison" in body
    assert "raw_avellaneda_lee" in body
    assert "gkx_lightgbm" in body
    assert "Data quality banner" in body
    assert "research_pass" in body
