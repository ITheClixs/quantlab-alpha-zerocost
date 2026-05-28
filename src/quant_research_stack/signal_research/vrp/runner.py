"""VRP runner — orchestrates the 6-variant grid + 3 baselines + PBO/DSR
+ 8-criteria gate + 3 pre-registered failure-mode checks per intake.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.signal_research.methodology.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_sharpe_ci,
)
from quant_research_stack.signal_research.methodology.pbo_extensions import (
    compute_three_tier_pbo,
)
from quant_research_stack.signal_research.validation.concentration import (
    ConcentrationReport,
    concentration_by_period,
)
from quant_research_stack.signal_research.vrp.baselines import (
    signal_buy_and_hold,
    signal_hmm_only_gate,
    signal_mom_12_1_single_asset,
)
from quant_research_stack.signal_research.vrp.features import (
    compute_vrp_features,
    vrp_zscore_60d,
)
from quant_research_stack.signal_research.vrp.strategies import (
    vrp_combined,
    vrp_long_only,
    vrp_long_short,
    vrp_with_skew,
    vrp_with_term_structure,
    vrp_with_vvix,
)
from quant_research_stack.signal_research.vrp.timing_backtest import (
    TimingBacktestResult,
    TimingCostConfig,
    run_timing_backtest,
)
from quant_research_stack.strategy_benchmark.dsr import compute_dsr

SignalGenFn = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class VRPSpec:
    target_symbol: str = "SPY"
    start: dt.date = dt.date(2010, 1, 1)
    end: dt.date = dt.date(2026, 5, 26)
    dev_end: dt.date = dt.date(2022, 12, 31)
    holdout_start: dt.date = dt.date(2023, 1, 1)
    realized_window: int = 21
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 0.5
    cost_stress_multipliers: tuple[float, ...] = (2.0, 3.0)
    delay_stress_bars: tuple[int, ...] = (1,)
    bootstrap_n_resamples: int = 2000
    bootstrap_seed: int = 42

    # 8-criteria gate per intake §6 (slightly stricter than the generic
    # pipeline because VRP is a high-conviction strategy)
    gate_dev_sharpe_min: float = 1.5
    gate_holdout_sharpe_min: float = 0.5
    gate_cost_stress_min: float = 0.0
    gate_bootstrap_ci_lower_min: float = 0.0
    gate_pbo_max: float = 0.25
    gate_dsr_min: float = 0.50
    gate_max_month_share: float = 0.5  # pre-registered failure mode #1


@dataclass(frozen=True)
class VRPStrategyResult:
    name: str
    is_vrp_variant: bool
    dev: TimingBacktestResult
    holdout: TimingBacktestResult
    cost_stress: dict[float, TimingBacktestResult]
    delay_stress: dict[int, TimingBacktestResult]
    concentration_dev: ConcentrationReport
    concentration_holdout: ConcentrationReport
    bootstrap_lower_95: float
    bootstrap_upper_95: float


@dataclass(frozen=True)
class VRPCrossMetrics:
    pbo_raw_global: float
    pbo_per_family: dict[str, float]
    best_name: str
    best_dsr: float
    best_psr_zero: float
    n_strategies: int


@dataclass(frozen=True)
class VRPRunReport:
    spec: VRPSpec
    variant_results: list[VRPStrategyResult]
    baseline_results: list[VRPStrategyResult]
    cross: VRPCrossMetrics
    decision: str
    failure_class: str
    failure_mode_1_concentration: bool  # pre-registered: single-period dominance
    failure_mode_2_pbo: bool  # variants are duplicates of each other
    failure_mode_3_dsr: bool  # combined variant inflates the grid


def _bootstrap_ci(
    rets: NDArray[np.float64], *, n_resamples: int, seed: int
) -> tuple[float, float]:
    if rets.size < 30:
        return float("nan"), float("nan")
    bs = bootstrap_sharpe_ci(
        returns=rets, config=BootstrapConfig(n_resamples=n_resamples, seed=seed)
    )
    return bs.ci_lower_95, bs.ci_upper_95


def _split_dev_holdout(
    full: TimingBacktestResult, dev_end: dt.date, holdout_start: dt.date
) -> tuple[TimingBacktestResult, TimingBacktestResult]:
    """Slice a single backtest's daily_returns into dev and holdout phases."""
    dr = full.daily_returns
    dev_daily = dr.filter(pl.col("date") <= dev_end)
    hd_daily = dr.filter(pl.col("date") >= holdout_start)

    def _from(daily: pl.DataFrame) -> TimingBacktestResult:
        rets = daily["net_return"].to_numpy().astype(np.float64)
        if rets.size < 2:
            sharpe = 0.0
            dd = 0.0
            cr = 0.0
        else:
            sd = float(np.std(rets, ddof=1))
            sharpe = (
                float(np.mean(rets)) / sd * float(np.sqrt(252.0)) if sd > 0 else 0.0
            )
            eq = np.cumprod(1.0 + rets)
            peak = np.maximum.accumulate(eq)
            dd = float((eq / peak - 1.0).min())
            cr = float(eq[-1] - 1.0)
        return TimingBacktestResult(
            daily_returns=daily,
            sharpe_annual=sharpe,
            max_drawdown=dd,
            cumulative_return=cr,
            n_days=int(rets.size),
            turnover_total=float(daily["turnover"].sum()) if daily.height > 0 else 0.0,
        )

    return _from(dev_daily), _from(hd_daily)


def _run_one_strategy(
    *,
    name: str,
    signal_df: pl.DataFrame,
    underlying: pl.DataFrame,
    spec: VRPSpec,
    is_vrp_variant: bool,
) -> VRPStrategyResult:
    cost_base = TimingCostConfig(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
    )
    full = run_timing_backtest(
        signals=signal_df, underlying=underlying,
        target_symbol=spec.target_symbol, cost=cost_base,
        cost_stress_mult=1.0, signal_delay_bars=0,
    )
    dev, hd = _split_dev_holdout(full, spec.dev_end, spec.holdout_start)
    cost_stress: dict[float, TimingBacktestResult] = {}
    for mult in spec.cost_stress_multipliers:
        cs_full = run_timing_backtest(
            signals=signal_df, underlying=underlying,
            target_symbol=spec.target_symbol, cost=cost_base,
            cost_stress_mult=mult, signal_delay_bars=0,
        )
        cs_dev, _ = _split_dev_holdout(cs_full, spec.dev_end, spec.holdout_start)
        cost_stress[mult] = cs_dev
    delay_stress: dict[int, TimingBacktestResult] = {}
    for n in spec.delay_stress_bars:
        ds_full = run_timing_backtest(
            signals=signal_df, underlying=underlying,
            target_symbol=spec.target_symbol, cost=cost_base,
            cost_stress_mult=1.0, signal_delay_bars=n,
        )
        ds_dev, _ = _split_dev_holdout(ds_full, spec.dev_end, spec.holdout_start)
        delay_stress[n] = ds_dev
    lo, hi = _bootstrap_ci(
        dev.daily_returns["net_return"].to_numpy().astype(np.float64),
        n_resamples=spec.bootstrap_n_resamples, seed=spec.bootstrap_seed,
    )
    conc_dev = concentration_by_period(dev.daily_returns)
    conc_hd = concentration_by_period(hd.daily_returns)
    return VRPStrategyResult(
        name=name, is_vrp_variant=is_vrp_variant,
        dev=dev, holdout=hd, cost_stress=cost_stress,
        delay_stress=delay_stress,
        concentration_dev=conc_dev, concentration_holdout=conc_hd,
        bootstrap_lower_95=lo, bootstrap_upper_95=hi,
    )


def run_vrp_pipeline(
    *,
    underlying: pl.DataFrame,
    vol_features: pl.DataFrame,
    spec: VRPSpec,
) -> VRPRunReport:
    target = underlying.filter(pl.col("symbol") == spec.target_symbol).sort("date")
    if target.is_empty():
        raise RuntimeError(f"no underlying bars for {spec.target_symbol}")

    feats = compute_vrp_features(
        underlying_single_symbol=target,
        vol_features=vol_features,
        realized_window=spec.realized_window,
    )
    feats = vrp_zscore_60d(feats)

    variant_fns: dict[str, SignalGenFn] = {
        "vrp_long_only": vrp_long_only,
        "vrp_long_short": vrp_long_short,
        "vrp_with_term_structure": vrp_with_term_structure,
        "vrp_with_vvix": vrp_with_vvix,
        "vrp_with_skew": vrp_with_skew,
        "vrp_combined": vrp_combined,
    }
    variant_results: list[VRPStrategyResult] = []
    for name, fn in variant_fns.items():
        signal_df = fn(feats)
        variant_results.append(
            _run_one_strategy(
                name=name, signal_df=signal_df, underlying=underlying,
                spec=spec, is_vrp_variant=True,
            )
        )

    baselines = [
        (
            "spy_buy_and_hold",
            signal_buy_and_hold(target),
        ),
        (
            "hmm_only_gate",
            signal_hmm_only_gate(
                underlying_single_symbol=target, dev_end=spec.dev_end,
            ),
        ),
        (
            "mom_12_1_single_asset",
            signal_mom_12_1_single_asset(target),
        ),
    ]
    baseline_results: list[VRPStrategyResult] = []
    for name, sig in baselines:
        baseline_results.append(
            _run_one_strategy(
                name=name, signal_df=sig, underlying=underlying,
                spec=spec, is_vrp_variant=False,
            )
        )

    cross = _cross_strategy_metrics(variant_results + baseline_results)
    decision, failure_class = _apply_decision_rule(
        variant_results=variant_results, baseline_results=baseline_results,
        cross=cross, spec=spec,
    )

    failure_mode_1 = any(
        r.concentration_dev.max_month_share > spec.gate_max_month_share
        for r in variant_results
    )
    failure_mode_2 = cross.pbo_raw_global > 0.20  # variants are duplicates
    best_idx_within_variants = int(np.argmax([
        r.dev.sharpe_annual for r in variant_results
    ]))
    failure_mode_3 = (
        variant_results[best_idx_within_variants].name == "vrp_combined"
        and cross.best_dsr < 0.5
    )

    return VRPRunReport(
        spec=spec,
        variant_results=variant_results,
        baseline_results=baseline_results,
        cross=cross,
        decision=decision,
        failure_class=failure_class,
        failure_mode_1_concentration=failure_mode_1,
        failure_mode_2_pbo=failure_mode_2,
        failure_mode_3_dsr=failure_mode_3,
    )


def _cross_strategy_metrics(
    all_results: list[VRPStrategyResult],
) -> VRPCrossMetrics:
    names = [r.name for r in all_results]
    series = [
        r.dev.daily_returns["net_return"].to_numpy().astype(np.float64)
        for r in all_results
    ]
    min_len = min(s.size for s in series)
    if min_len < 64 or len(names) < 3:
        return VRPCrossMetrics(
            pbo_raw_global=float("nan"),
            pbo_per_family={},
            best_name=names[0] if names else "",
            best_dsr=float("nan"),
            best_psr_zero=float("nan"),
            n_strategies=len(names),
        )
    dev_matrix = np.column_stack([s[-min_len:] for s in series])
    pbo = compute_three_tier_pbo(
        returns=dev_matrix,
        profile=np.array(["vrp_pool" for _ in names]),
        family=np.array(["vrp_variant" if r.is_vrp_variant else "baseline" for r in all_results]),
        n_partitions=16,
    )
    sharpes = np.array([r.dev.sharpe_annual for r in all_results], dtype=np.float64)
    best_idx = int(np.argmax(sharpes))
    dsr = compute_dsr(
        returns=series[best_idx],
        sharpe_estimates=sharpes,
        selected_idx=best_idx,
    )
    return VRPCrossMetrics(
        pbo_raw_global=pbo.raw_global,
        pbo_per_family=pbo.per_family,
        best_name=names[best_idx],
        best_dsr=float(dsr.dsr),
        best_psr_zero=float(dsr.psr_zero),
        n_strategies=len(names),
    )


def _apply_decision_rule(
    *,
    variant_results: list[VRPStrategyResult],
    baseline_results: list[VRPStrategyResult],
    cross: VRPCrossMetrics,
    spec: VRPSpec,
) -> tuple[str, str]:
    best_variant_idx = int(np.argmax([r.dev.sharpe_annual for r in variant_results]))
    best_v = variant_results[best_variant_idx]
    # Gates
    if cross.pbo_raw_global > spec.gate_pbo_max:
        return (
            f"FAIL — PBO={cross.pbo_raw_global:.3f} > {spec.gate_pbo_max}.",
            "overfit_parameter_grid",
        )
    if cross.best_dsr < spec.gate_dsr_min:
        return (
            f"FAIL — DSR={cross.best_dsr:.3f} < {spec.gate_dsr_min}.",
            "multi_test_penalty_kills_edge",
        )
    if best_v.dev.sharpe_annual < spec.gate_dev_sharpe_min:
        return (
            f"FAIL — best VRP dev Sharpe {best_v.dev.sharpe_annual:+.3f} < "
            f"{spec.gate_dev_sharpe_min}.",
            "no_alpha_at_threshold",
        )
    if best_v.holdout.sharpe_annual < spec.gate_holdout_sharpe_min:
        return (
            f"FAIL — best VRP holdout Sharpe {best_v.holdout.sharpe_annual:+.3f} < "
            f"{spec.gate_holdout_sharpe_min}.",
            "fails_holdout_generalization",
        )
    cs2x = best_v.cost_stress.get(2.0)
    if cs2x is not None and cs2x.sharpe_annual < spec.gate_cost_stress_min:
        return (
            f"FAIL — 2x cost stress Sharpe {cs2x.sharpe_annual:+.3f} < "
            f"{spec.gate_cost_stress_min}.",
            "costs_kill_the_edge",
        )
    if best_v.bootstrap_lower_95 < spec.gate_bootstrap_ci_lower_min:
        return (
            f"FAIL — bootstrap lower CI {best_v.bootstrap_lower_95:+.3f} < "
            f"{spec.gate_bootstrap_ci_lower_min}.",
            "high_variance_noise_signal",
        )
    if best_v.concentration_dev.max_month_share > spec.gate_max_month_share:
        return (
            f"FAIL — max month PnL share "
            f"{best_v.concentration_dev.max_month_share*100:.1f}% > "
            f"{spec.gate_max_month_share*100:.0f}%.",
            "single_period_dominance",
        )
    delay = best_v.delay_stress.get(1)
    if delay is not None and delay.sharpe_annual < best_v.dev.sharpe_annual - 0.5:
        return (
            f"FAIL — 1-bar delay loses {best_v.dev.sharpe_annual - delay.sharpe_annual:+.3f} "
            f"Sharpe (look-ahead suspected).",
            "delay_stress_fail",
        )
    # Beats best baseline
    best_baseline_sharpe = max(b.dev.sharpe_annual for b in baseline_results)
    if best_v.dev.sharpe_annual <= best_baseline_sharpe:
        return (
            f"FAIL — best VRP dev Sharpe {best_v.dev.sharpe_annual:+.3f} ≤ best baseline "
            f"{best_baseline_sharpe:+.3f}.",
            "no_added_value_over_baselines",
        )
    return (
        "PASS — VRP survives all gates; promote to deeper robustness testing.",
        "",
    )


def render_vrp_report(report: VRPRunReport, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in report.variant_results + report.baseline_results:
        cs2 = r.cost_stress.get(2.0)
        cs3 = r.cost_stress.get(3.0)
        dly = r.delay_stress.get(1)
        prefix = "" if r.is_vrp_variant else "_(baseline)_ "
        rows.append(
            f"| {prefix}`{r.name}` | "
            f"{r.dev.sharpe_annual:+.3f} | {r.dev.max_drawdown*100:+.2f}% | "
            f"{r.bootstrap_lower_95:+.3f} | {r.bootstrap_upper_95:+.3f} | "
            f"{r.holdout.sharpe_annual:+.3f} | {r.holdout.max_drawdown*100:+.2f}% | "
            f"{cs2.sharpe_annual if cs2 else float('nan'):+.3f} | "
            f"{cs3.sharpe_annual if cs3 else float('nan'):+.3f} | "
            f"{dly.sharpe_annual if dly else float('nan'):+.3f} | "
            f"{r.concentration_dev.max_month_share*100:.1f}% |"
        )
    header = (
        "| Strategy | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | "
        "holdout Sharpe | holdout DD | cs-2x | cs-3x | delay-1d | max month share |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    body = "\n".join([
        f"# VRP Index Validation — `{report.spec.target_symbol}`",
        "",
        "## Hypothesis",
        "> Implied option-market variance prices a risk premium relative to ",
        "> subsequently-realized variance. Long-vol hedgers pay a structural ",
        "> premium during normal regimes; the short-vol seller earns it but bears ",
        "> crash risk during volatility spikes (Bondarenko 2014).",
        "",
        "## Information sources declared",
        "- `ohlcv` (SPY underlying)",
        "- `options_implied_vol` (^VIX, ^VIX9D, ^VIX3M, ^VVIX, ^SKEW, ^VXN)",
        "- **non-OHLCV source declared: YES** → eligible for promotion if gates pass",
        "",
        "## Fixture",
        f"- target symbol: {report.spec.target_symbol}",
        f"- history: {report.spec.start.isoformat()} → {report.spec.end.isoformat()}",
        f"- dev:     {report.spec.start.isoformat()} → {report.spec.dev_end.isoformat()}",
        f"- holdout: {report.spec.holdout_start.isoformat()} → {report.spec.end.isoformat()}",
        f"- realized-variance window: {report.spec.realized_window} days",
        f"- costs: {report.spec.commission_bps_one_way} bps commission + "
        f"{report.spec.spread_bps_one_way} bps spread one-way",
        f"- cost stress multipliers: {list(report.spec.cost_stress_multipliers)}",
        f"- delay stress: {list(report.spec.delay_stress_bars)} bars",
        "",
        "## All strategies side-by-side",
        "",
        header, sep, *rows,
        "",
        "## Cross-strategy multiple-testing controls",
        "",
        f"- **PBO raw_global**: {report.cross.pbo_raw_global:.3f}  (gate ≤ {report.spec.gate_pbo_max})",
        f"- **Best strategy**: `{report.cross.best_name}`",
        f"- **DSR for best**: {report.cross.best_dsr:.3f}  (gate ≥ {report.spec.gate_dsr_min})",
        f"- **PSR_zero for best**: {report.cross.best_psr_zero:.3f}",
        f"- **n_strategies in DSR deflation**: {report.cross.n_strategies}",
        "",
        "## Pre-registered failure modes (from intake §8)",
        "",
        f"- Mode 1 (single-event concentration, any variant max month > "
        f"{report.spec.gate_max_month_share*100:.0f}%): "
        f"{'TRIGGERED' if report.failure_mode_1_concentration else 'not triggered'}",
        f"- Mode 2 (variant grid is duplicates, PBO > 0.20): "
        f"{'TRIGGERED' if report.failure_mode_2_pbo else 'not triggered'}",
        f"- Mode 3 (combined variant inflates grid, DSR < 0.5 when combined is best): "
        f"{'TRIGGERED' if report.failure_mode_3_dsr else 'not triggered'}",
        "",
        "## Decision rule outcome",
        "",
        f"**{report.decision}**",
        "",
        f"failure_class: `{report.failure_class or 'none'}`",
        "",
        "## Disclaimer",
        "Research output only. Past performance does not guarantee future results. ",
        "No promotion to capital deployment occurs without an explicit promotion record ",
        "(spec §6.5 and the QuantLab promotion runbook).",
        "",
    ])
    output_path.write_text(body)
    return output_path
