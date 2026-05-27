"""Synthetic smoke for sector-conditional AvL pipeline."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.signal_research.backtests.sector_avl import (
    HMMGate,
    SectorAvLSpec,
    SectorAvLVariant,
    apply_decision_rule,
    cross_strategy_metrics,
    filter_sectors,
    render_aggregate_report,
    render_per_sector_report,
    run_all_sector_avl_variants,
    run_sanity_baselines,
)


def _synthetic_bars_two_sectors(*, n_days: int, seed: int) -> tuple[pl.DataFrame, dict[str, str]]:
    """Two synthetic sectors with 15 names each, common factor per sector."""
    rng = np.random.default_rng(seed)
    start = dt.date(2014, 1, 2)
    dates: list[dt.date] = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d = d + dt.timedelta(days=1)
    sector_map: dict[str, str] = {}
    rows = []
    for sector_idx, sector_name in enumerate(["Financials", "Energy"]):
        factor = rng.standard_normal(n_days) * 0.012
        for s in range(15):
            sym = f"{sector_name[:3].upper()}{s:02d}"
            sector_map[sym] = sector_name
            beta = 0.6 + 0.4 * rng.standard_normal()
            idio = rng.standard_normal(n_days) * 0.009
            rets = beta * factor + idio
            price = 100.0 * np.cumprod(1.0 + rets)
            vol = rng.uniform(5e5, 5e6, size=n_days)
            for t, dd in enumerate(dates):
                rows.append({
                    "date": dd, "symbol": sym,
                    "open": float(price[t]),
                    "high": float(price[t] * (1.0 + abs(idio[t]) * 0.5)),
                    "low": float(price[t] * (1.0 - abs(idio[t]) * 0.5)),
                    "close": float(price[t]),
                    "volume": float(vol[t]),
                })
    return pl.DataFrame(rows), sector_map


def test_sector_avl_full_pipeline_runs_on_synthetic(tmp_path: Path) -> None:
    bars, sector_map = _synthetic_bars_two_sectors(n_days=1500, seed=0)
    spec = SectorAvLSpec(
        sectors_to_include=("Financials", "Energy"),
        min_sector_size=15,
        start=dt.date(2014, 1, 2),
        end=dt.date(2020, 1, 1),
        dev_end=dt.date(2018, 12, 31),
        holdout_start=dt.date(2019, 1, 1),
        pca_window=60,
        pca_components_grid=(1, 2),
        z_entry_grid=(1.0, 1.5),
        hmm_gates=(HMMGate.NONE, HMMGate.RISK_ON),
        z_exit_reversion=0.5,
        max_holding_days=5,
        q_quantile_sector=0.30,
        cohort="focused_basket",
        equity=100_000.0,
    )
    baskets = filter_sectors(bars=bars, sector_map=sector_map, spec=spec)
    assert "Financials" in baskets
    assert "Energy" in baskets

    variants = run_all_sector_avl_variants(
        bars=bars, sector_baskets=baskets, spec=spec,
    )
    # 2 pca × 2 z-entry × 2 hmm = 8 variants
    assert len(variants) == 8

    baselines = run_sanity_baselines(
        bars=bars, sector_baskets=baskets, spec=spec,
    )
    assert len(baselines) == 3

    cross = cross_strategy_metrics(variants + baselines)
    assert 0.0 <= cross.pbo_raw_global <= 1.0
    assert cross.n_strategies == 11

    decision, failure_class = apply_decision_rule(
        variants=variants, baselines=baselines, cross=cross,
    )
    assert isinstance(decision, str)

    agg = render_aggregate_report(
        variants=variants, baselines=baselines, cross=cross,
        decision=decision, failure_class=failure_class,
        spec=spec, output_path=tmp_path / "aggregate.md",
    )
    per_sec = render_per_sector_report(
        variants=variants, baselines=baselines,
        spec=spec, output_path=tmp_path / "per_sector.md",
    )
    body_agg = agg.read_text()
    body_sec = per_sec.read_text()
    assert "Sector-Conditional Avellaneda-Lee" in body_agg
    assert "PBO raw_global" in body_agg
    assert "DSR for best" in body_agg
    assert "Decision rule outcome" in body_agg
    assert "Per-Sector Detail" in body_sec


def test_variant_name_is_deterministic() -> None:
    v = SectorAvLVariant(pca_components=2, z_entry=1.5, hmm_gate=HMMGate.RISK_ON)
    assert v.name == "avl_pca2_z1.5_hmmrisk_on"
