"""Synthetic smoke for ValidationPipeline."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.signal_research.status import CandidateStatus
from quant_research_stack.signal_research.validation import (
    InformationSource,
    ValidationSpec,
    render_pipeline_report,
    validate_strategy,
)


def _synthetic_bars(*, n_days: int, n_symbols: int, seed: int) -> pl.DataFrame:
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


def _trivial_mom_signal(bars: pl.DataFrame, spec: ValidationSpec) -> pl.DataFrame:
    df = bars.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias(
            "log_ret"
        )
    )
    df = df.with_columns(
        pl.col("log_ret").rolling_sum(window_size=63).over("symbol").alias("y_xs_pred")
    )
    return df.drop_nulls(subset=["y_xs_pred"]).select(["date", "symbol", "y_xs_pred"])


def test_validate_strategy_runs_and_renders(tmp_path: Path) -> None:
    bars = _synthetic_bars(n_days=1100, n_symbols=15, seed=0)
    spec = ValidationSpec(
        strategy_name="trivial_mom_test",
        hypothesis_statement="63-day cumulative return predicts forward 21-day return.",
        information_sources=(InformationSource.OHLCV,),
        universe_tickers=[f"S{i:02d}" for i in range(15)],
        start=dt.date(2018, 1, 2),
        end=dt.date(2022, 1, 1),
        dev_end=dt.date(2020, 12, 31),
        holdout_start=dt.date(2021, 1, 1),
        equity=100_000.0,
        q_quantile=0.30,
        cohort="focused_basket",
        delay_stress_bars=(1,),
        bootstrap_n_resamples=200,
    )
    report = validate_strategy(
        spec=spec, signal_fn=_trivial_mom_signal, bars=bars,
    )
    assert report.result.dev_net_returns.size > 30
    assert report.assigned_status in {
        CandidateStatus.NONE,
        CandidateStatus.RESEARCH_PASS,
    }
    # OHLCV-only must not reach promotion
    assert report.assigned_status != CandidateStatus.PROMOTION_ELIGIBLE
    assert not report.promotion_eligible

    out = render_pipeline_report(report, output_path=tmp_path / "report.md")
    body = out.read_text()
    assert "Validation Report" in body
    assert "trivial_mom_test" in body
    assert "Cost decomposition" in body
    assert "Delay stress" in body
    assert "Sanity baselines" in body
    assert "Concentration diagnostics" in body
    assert "No-promotion-without-new-information-source rule" in body
    assert "non-OHLCV declared: NO" in body


def test_ohlcv_only_cannot_be_promoted_even_if_metrics_strong(tmp_path: Path) -> None:
    bars = _synthetic_bars(n_days=1100, n_symbols=15, seed=1)

    # Inject a "perfect" signal: future-looking 1-bar shifted log return.
    # Strong dev metrics, but information_sources=OHLCV only → no promotion.
    def perfect_signal(b: pl.DataFrame, _spec: ValidationSpec) -> pl.DataFrame:
        df = b.sort(["symbol", "date"]).with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log())
            .shift(-1)  # look-ahead by 1; intentionally leaky for the test
            .over("symbol")
            .alias("y_xs_pred")
        )
        return df.drop_nulls(subset=["y_xs_pred"]).select(
            ["date", "symbol", "y_xs_pred"]
        )

    spec = ValidationSpec(
        strategy_name="perfect_but_leaky_ohlcv_only",
        hypothesis_statement="(intentionally leaky test signal)",
        information_sources=(InformationSource.OHLCV,),
        universe_tickers=[f"S{i:02d}" for i in range(15)],
        start=dt.date(2018, 1, 2),
        end=dt.date(2022, 1, 1),
        dev_end=dt.date(2020, 12, 31),
        holdout_start=dt.date(2021, 1, 1),
        equity=100_000.0,
        q_quantile=0.30,
        cohort="focused_basket",
        delay_stress_bars=(1,),
        bootstrap_n_resamples=200,
    )
    report = validate_strategy(spec=spec, signal_fn=perfect_signal, bars=bars)
    # No promotion regardless of how strong the metrics look
    assert report.assigned_status != CandidateStatus.PROMOTION_ELIGIBLE
    assert not report.promotion_eligible
