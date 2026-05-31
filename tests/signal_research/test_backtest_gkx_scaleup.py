"""Synthetic smoke for GKX scale-up."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.signal_research.backtests.gkx_scaleup import (
    GKXSpec,
    GKXVariant,
    apply_decision_rule,
    cross_strategy_metrics,
    render_report,
    run_gkx_scaleup,
)


def _synthetic_bars(*, n_days: int, n_symbols: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_days) * 0.01
    start = dt.date(2016, 1, 4)
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


def test_variant_name_format() -> None:
    v = GKXVariant(label_horizon=21, universe_label="top200")
    assert v.name == "gkx_lgb_h21d_top200"


def test_gkx_scaleup_runs_on_synthetic(tmp_path: Path) -> None:
    bars_top100 = _synthetic_bars(n_days=1500, n_symbols=20, seed=0)
    bars_top200 = _synthetic_bars(n_days=1500, n_symbols=25, seed=1)
    spec = GKXSpec(
        start=dt.date(2016, 1, 4),
        end=dt.date(2022, 1, 1),
        dev_end=dt.date(2020, 12, 31),
        holdout_start=dt.date(2021, 1, 1),
        label_horizons=(5, 21),
        universes=("top100", "top200"),
        n_estimators=50, learning_rate=0.05, num_leaves=15,
        walk_forward_folds=3, walk_forward_embargo=5,
        equity=100_000.0,
        q_quantile=0.30, cohort="focused_basket",
    )
    variants, baselines = run_gkx_scaleup(
        bars_per_universe={"top100": bars_top100, "top200": bars_top200},
        spec=spec,
    )
    assert len(variants) == 4  # 2 horizons × 2 universes
    assert len(baselines) == 6  # 3 baselines × 2 universes

    cross = cross_strategy_metrics(variants, baselines)
    assert 0.0 <= cross.pbo_raw_global <= 1.0
    assert cross.n_strategies == 10

    decision, failure_class = apply_decision_rule(
        variants=variants, baselines=baselines, cross=cross,
    )
    assert isinstance(decision, str)

    report = render_report(
        variants=variants, baselines=baselines, cross=cross,
        decision=decision, failure_class=failure_class,
        spec=spec, output_path=tmp_path / "gkx.md",
    )
    body = report.read_text()
    assert "GKX-Style LightGBM" in body
    assert "PBO raw_global" in body
    assert "Decision rule outcome" in body
    assert "gkx_lgb_h5d_top100" in body
