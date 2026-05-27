"""Synthetic smoke test for the triple-barrier × Avellaneda-Lee backtest."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.signal_research.backtests.triple_barrier_av_lee import (
    TBAvLeeSpec,
    render_report,
    run_triple_barrier_av_lee,
)


def _synthetic_bars(
    *, n_days: int = 1200, n_symbols: int = 15, seed: int = 0
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
                "date": dd,
                "symbol": f"S{s:02d}",
                "open": float(price[t]),
                "high": float(price[t] * (1.0 + abs(idio[t]) * 0.5)),
                "low": float(price[t] * (1.0 - abs(idio[t]) * 0.5)),
                "close": float(price[t]),
                "volume": float(vol[t]),
            })
    return pl.DataFrame(rows)


def test_triple_barrier_av_lee_pipeline_runs_on_synthetic_panel(tmp_path: Path) -> None:
    bars = _synthetic_bars(n_days=900, n_symbols=15)
    spec = TBAvLeeSpec(
        universe_tickers=[f"S{i:02d}" for i in range(15)],
        start=dt.date(2018, 1, 2),
        end=dt.date(2022, 1, 1),
        dev_end=dt.date(2020, 12, 31),
        holdout_start=dt.date(2021, 1, 1),
        pca_window=60,
        n_pca_components=2,
        z_entry=1.0,
        vertical_barrier_days=5,
        vol_estimator_window=10,
        rf_n_estimators=50,
        rf_threshold=0.4,
        equity=100_000.0,
        q_quantile=0.30,
        cohort="focused_basket",
    )
    out = run_triple_barrier_av_lee(bars=bars, spec=spec)
    assert out.n_universe_initial == 15
    assert out.dev_metrics["n_days"] > 0
    assert "sharpe" in out.dev_metrics
    assert "holdout_sharpe_positive" in out.funnel.to_ordered_dict()

    report_path = render_report(out, output_path=tmp_path / "report.md")
    body = report_path.read_text()
    assert "Data quality banner" in body
    assert "survivorship_prototype_only" in body
    assert "Selection funnel" in body
    assert "Disclaimer" in body
