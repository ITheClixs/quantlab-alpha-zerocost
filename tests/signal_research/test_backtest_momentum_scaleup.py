"""Synthetic smoke for momentum scale-up variant matrix."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.signal_research.backtests.momentum_scaleup import (
    MomentumSpec,
    MomentumVariant,
    apply_decision_rule,
    cross_strategy_metrics,
    render_momentum_scaleup_report,
    run_all_momentum_variants,
)


def _synthetic_bars(*, n_days: int, n_symbols: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_days) * 0.01
    start = dt.date(2014, 1, 2)
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


def test_all_momentum_variants_run_and_report_renders(tmp_path: Path) -> None:
    bars_top100 = _synthetic_bars(n_days=1500, n_symbols=20, seed=0)
    bars_top200 = _synthetic_bars(n_days=1500, n_symbols=25, seed=1)
    spec_top100 = MomentumSpec(
        universe_tickers=[f"S{i:02d}" for i in range(20)],
        start=dt.date(2014, 1, 2),
        end=dt.date(2020, 1, 1),
        dev_end=dt.date(2018, 12, 31),
        holdout_start=dt.date(2019, 1, 1),
        equity=100_000.0,
        q_quantile=0.30,
        cohort="focused_basket",
    )
    spec_top200 = MomentumSpec(
        universe_tickers=[f"S{i:02d}" for i in range(25)],
        start=dt.date(2014, 1, 2),
        end=dt.date(2020, 1, 1),
        dev_end=dt.date(2018, 12, 31),
        holdout_start=dt.date(2019, 1, 1),
        equity=100_000.0,
        q_quantile=0.30,
        cohort="focused_basket",
    )
    results = run_all_momentum_variants(
        bars_top100=bars_top100, bars_top200=bars_top200,
        spec_top100=spec_top100, spec_top200=spec_top200,
    )
    assert len(results) == 10
    variants_seen = {r.variant for r in results}
    assert variants_seen == set(MomentumVariant)

    cross = cross_strategy_metrics(results)
    assert 0.0 <= cross.pbo_raw_global <= 1.0
    assert "top100" in cross.pbo_per_profile
    assert "top200" in cross.pbo_per_profile
    # per_family PBO only fills when ≥3 strategies share a family value.
    # With 5 variants × 2 universes, families are at size 2; per_family stays empty.
    assert isinstance(cross.pbo_per_family, dict)

    decision = apply_decision_rule(results)
    assert isinstance(decision, str) and len(decision) > 0

    report = render_momentum_scaleup_report(
        results=results, cross=cross, decision=decision,
        spec_top100=spec_top100, spec_top200=spec_top200,
        output_path=tmp_path / "report.md",
    )
    body = report.read_text()
    assert "Momentum Scale-Up" in body
    assert "PBO raw_global" in body
    assert "DSR for best" in body
    assert "Decision rule outcome" in body
    for v in MomentumVariant:
        assert v.value in body
