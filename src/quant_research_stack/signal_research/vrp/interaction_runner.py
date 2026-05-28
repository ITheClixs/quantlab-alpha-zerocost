"""VRP × HMM interaction runner — Option γ.

Orchestrates the 9 predeclared variants + 3 sanity baselines, runs full
validation discipline, computes attribution analytics, applies the
A/B/C/D decision rules.

Output deliverables (per user spec):
- vrp_hmm_interaction_registry.parquet
- vrp_hmm_validation_report.md
- vrp_hmm_attribution_report.md
- vrp_hmm_orthogonalization_report.md
- vrp_hmm_failure_classification.md (if no combined variant improves on HMM-only)
"""

from __future__ import annotations

import datetime as dt
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
from quant_research_stack.signal_research.vrp.attribution import (
    AttributionMetrics,
    attribution_for_variant,
)
from quant_research_stack.signal_research.vrp.baselines import (
    signal_buy_and_hold,
)
from quant_research_stack.signal_research.vrp.features import (
    compute_vrp_features,
    vrp_zscore_60d,
)
from quant_research_stack.signal_research.vrp.hmm_panel import (
    HMMPanel,
    fit_hmm_panel,
    hmm_signal_from_panel,
)
from quant_research_stack.signal_research.vrp.interaction import (
    signal_additive_ensemble,
    signal_hmm_sized_by_vrp,
    signal_orthogonalized_vrp,
    signal_vrp_sized_by_hmm_prob,
    signal_vrp_when_hmm_risk_off,
    signal_vrp_when_hmm_risk_on,
)
from quant_research_stack.signal_research.vrp.runner import VRPSpec
from quant_research_stack.signal_research.vrp.strategies import vrp_long_only
from quant_research_stack.signal_research.vrp.timing_backtest import (
    TimingBacktestResult,
    TimingCostConfig,
    run_timing_backtest,
)
from quant_research_stack.strategy_benchmark.dsr import compute_dsr


@dataclass(frozen=True)
class InteractionStrategyResult:
    name: str
    category: str  # "anchor" | "interaction" | "sanity"
    dev: TimingBacktestResult
    holdout: TimingBacktestResult
    cost_stress_2x: TimingBacktestResult
    cost_stress_3x: TimingBacktestResult
    delay_1d: TimingBacktestResult
    bootstrap_lower_95: float
    bootstrap_upper_95: float
    attribution: AttributionMetrics
    turnover_total: float
    exposure_time_frac: float


@dataclass(frozen=True)
class InteractionRunReport:
    spec: VRPSpec
    results: list[InteractionStrategyResult]
    pbo_raw_global: float
    best_name: str
    best_dsr: float
    best_psr_zero: float
    n_strategies: int
    decision_branch: str  # A / B / C / D
    decision: str
    failure_class: str
    hmm_panel: HMMPanel


def _bootstrap_ci(
    rets: NDArray[np.float64], *, n_resamples: int, seed: int
) -> tuple[float, float]:
    if rets.size < 30:
        return float("nan"), float("nan")
    bs = bootstrap_sharpe_ci(
        returns=rets, config=BootstrapConfig(n_resamples=n_resamples, seed=seed),
    )
    return bs.ci_lower_95, bs.ci_upper_95


def _split(full: TimingBacktestResult, dev_end: dt.date, holdout_start: dt.date) -> tuple[
    TimingBacktestResult, TimingBacktestResult
]:
    dr = full.daily_returns
    dev_daily = dr.filter(pl.col("date") <= dev_end)
    hd_daily = dr.filter(pl.col("date") >= holdout_start)

    def _from(daily: pl.DataFrame) -> TimingBacktestResult:
        rets = daily["net_return"].to_numpy().astype(np.float64)
        if rets.size < 2:
            return TimingBacktestResult(daily, 0.0, 0.0, 0.0, 0, 0.0)
        sd = float(np.std(rets, ddof=1))
        sr = float(np.mean(rets)) / sd * float(np.sqrt(252.0)) if sd > 0 else 0.0
        eq = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(eq)
        dd = float((eq / peak - 1.0).min())
        return TimingBacktestResult(
            daily_returns=daily,
            sharpe_annual=sr,
            max_drawdown=dd,
            cumulative_return=float(eq[-1] - 1.0),
            n_days=int(rets.size),
            turnover_total=float(daily["turnover"].sum()) if "turnover" in daily.columns else 0.0,
        )

    return _from(dev_daily), _from(hd_daily)


def _exposure_time_frac(signal_df: pl.DataFrame) -> float:
    if signal_df.is_empty():
        return 0.0
    arr = signal_df["signal"].to_numpy().astype(np.float64)
    return float(np.mean(np.abs(arr) > 1e-9))


def _run_strategy(
    *,
    name: str,
    category: str,
    signal_df: pl.DataFrame,
    underlying: pl.DataFrame,
    spec: VRPSpec,
    hmm_only_dev: pl.DataFrame,
    vrp_only_dev: pl.DataFrame,
    hmm_panel_df: pl.DataFrame,
) -> InteractionStrategyResult:
    cost = TimingCostConfig(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
    )
    full = run_timing_backtest(
        signals=signal_df, underlying=underlying,
        target_symbol=spec.target_symbol, cost=cost,
    )
    dev, hd = _split(full, spec.dev_end, spec.holdout_start)
    cs2 = run_timing_backtest(
        signals=signal_df, underlying=underlying,
        target_symbol=spec.target_symbol, cost=cost,
        cost_stress_mult=2.0,
    )
    cs2_dev, _ = _split(cs2, spec.dev_end, spec.holdout_start)
    cs3 = run_timing_backtest(
        signals=signal_df, underlying=underlying,
        target_symbol=spec.target_symbol, cost=cost,
        cost_stress_mult=3.0,
    )
    cs3_dev, _ = _split(cs3, spec.dev_end, spec.holdout_start)
    delay = run_timing_backtest(
        signals=signal_df, underlying=underlying,
        target_symbol=spec.target_symbol, cost=cost,
        signal_delay_bars=1,
    )
    delay_dev, _ = _split(delay, spec.dev_end, spec.holdout_start)
    lo, hi = _bootstrap_ci(
        dev.daily_returns["net_return"].to_numpy().astype(np.float64),
        n_resamples=spec.bootstrap_n_resamples, seed=spec.bootstrap_seed,
    )
    attr = attribution_for_variant(
        name=name,
        daily_dev=dev.daily_returns,
        hmm_only_dev=hmm_only_dev,
        vrp_only_dev=vrp_only_dev,
        hmm_panel=hmm_panel_df,
    )
    return InteractionStrategyResult(
        name=name,
        category=category,
        dev=dev,
        holdout=hd,
        cost_stress_2x=cs2_dev,
        cost_stress_3x=cs3_dev,
        delay_1d=delay_dev,
        bootstrap_lower_95=lo,
        bootstrap_upper_95=hi,
        attribution=attr,
        turnover_total=float(dev.daily_returns["turnover"].sum())
        if "turnover" in dev.daily_returns.columns else 0.0,
        exposure_time_frac=_exposure_time_frac(signal_df),
    )


def _apply_decision_rules(
    *,
    results: list[InteractionStrategyResult],
    spec: VRPSpec,
    pbo_raw_global: float,
    best_dsr: float,
) -> tuple[str, str, str]:
    """Apply user's A/B/C/D criteria.

    Returns (branch_letter, decision_message, failure_class).
    """
    by_name = {r.name: r for r in results}
    hmm_only = by_name.get("hmm_only_baseline")
    vrp_only = by_name.get("vrp_only_baseline")
    if hmm_only is None or vrp_only is None:
        return "?", "FAIL — anchor baselines missing.", "missing_anchors"

    # Interaction variants (not anchors, not sanity)
    interactions = [r for r in results if r.category == "interaction"]
    if not interactions:
        return "?", "FAIL — no interaction variants produced.", "no_interactions"

    # Criterion C: orthogonalized VRP survival check first (most informative)
    ortho = by_name.get("orthogonalized_vrp")
    ortho_survives = False
    if ortho is not None:
        ortho_survives = (
            ortho.bootstrap_lower_95 > 0.0
            and ortho.dev.sharpe_annual > 0.3
            and ortho.cost_stress_2x.sharpe_annual > 0.0
            and ortho.attribution.residual_sharpe_over_hmm_only > 0.3
        )

    # Criterion A: any interaction variant materially beats HMM-only on holdout?
    hmm_holdout = hmm_only.holdout.sharpe_annual
    beats_hmm = [
        r for r in interactions
        if (
            r.holdout.sharpe_annual >= hmm_holdout + 0.25
            and r.bootstrap_lower_95 > 0.0
            and r.cost_stress_2x.sharpe_annual > 0.0
            and r.delay_1d.sharpe_annual > r.dev.sharpe_annual - 0.5
            and r.attribution.sharpe_excl_2020 > 0.0
            and r.attribution.sharpe_excl_2022 > 0.0
        )
    ]
    if pbo_raw_global <= spec.gate_pbo_max and best_dsr >= spec.gate_dsr_min and beats_hmm:
        winner = max(beats_hmm, key=lambda r: r.holdout.sharpe_annual)
        return (
            "A",
            f"PASS-A — `{winner.name}` holdout {winner.holdout.sharpe_annual:+.3f} "
            f"exceeds HMM-only {hmm_holdout:+.3f} by "
            f"{winner.holdout.sharpe_annual - hmm_holdout:+.3f}; VRP carries "
            "independent or complementary information.",
            "",
        )

    if ortho_survives:
        return (
            "C",
            "PASS-C — orthogonalized VRP survives; VRP contains independent "
            "information beyond HMM regime. Proceed to a VRP-specific follow-up.",
            "",
        )

    # B vs D: distinguish "VRP useful but not incremental" from "HMM dominates"
    best_interaction = max(interactions, key=lambda r: r.holdout.sharpe_annual)
    if best_interaction.holdout.sharpe_annual >= vrp_only.holdout.sharpe_annual + 0.1:
        return (
            "B",
            f"PASS-B(no-promo) — best interaction `{best_interaction.name}` "
            f"holdout {best_interaction.holdout.sharpe_annual:+.3f} improves on "
            f"VRP-only ({vrp_only.holdout.sharpe_annual:+.3f}) but does not "
            f"exceed HMM-only ({hmm_holdout:+.3f}) by 0.25. VRP is useful but "
            "NOT incremental over HMM. Keep VRP as a research note.",
            "vrp_useful_but_not_incremental_over_hmm",
        )

    return (
        "D",
        f"HMM-only remains dominant ({hmm_holdout:+.3f}). The OHLCV-only "
        "HMM timing primitive is the strongest result; per the no-OHLCV "
        "rule it cannot be promoted. Create a separate review note "
        "'single-index risk-timing exception proposal' rather than "
        "rewriting the rule.",
        "hmm_only_dominates_vrp_not_incremental",
    )


def run_vrp_hmm_interaction(
    *,
    underlying: pl.DataFrame,
    vol_features: pl.DataFrame,
    spec: VRPSpec,
) -> InteractionRunReport:
    target = underlying.filter(pl.col("symbol") == spec.target_symbol).sort("date")
    if target.is_empty():
        raise RuntimeError(f"no bars for {spec.target_symbol}")

    feats = vrp_zscore_60d(
        compute_vrp_features(
            underlying_single_symbol=target,
            vol_features=vol_features,
            realized_window=spec.realized_window,
        )
    )
    hmm_panel = fit_hmm_panel(
        underlying_single_symbol=target, dev_end=spec.dev_end,
    )
    hmm_signal_df = hmm_signal_from_panel(hmm_panel)

    # Build the two anchors first; their dev daily returns are needed for attribution.
    cost = TimingCostConfig(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
    )
    hmm_full = run_timing_backtest(
        signals=hmm_signal_df, underlying=underlying,
        target_symbol=spec.target_symbol, cost=cost,
    )
    hmm_dev, _ = _split(hmm_full, spec.dev_end, spec.holdout_start)
    vrp_signal_df = vrp_long_only(feats)
    vrp_full = run_timing_backtest(
        signals=vrp_signal_df, underlying=underlying,
        target_symbol=spec.target_symbol, cost=cost,
    )
    vrp_dev, _ = _split(vrp_full, spec.dev_end, spec.holdout_start)

    results: list[InteractionStrategyResult] = []

    def _add(name: str, category: str, sig: pl.DataFrame) -> None:
        results.append(_run_strategy(
            name=name, category=category, signal_df=sig,
            underlying=underlying, spec=spec,
            hmm_only_dev=hmm_dev.daily_returns,
            vrp_only_dev=vrp_dev.daily_returns,
            hmm_panel_df=hmm_panel.panel,
        ))

    # Anchors
    _add("hmm_only_baseline", "anchor", hmm_signal_df)
    _add("vrp_only_baseline", "anchor", vrp_signal_df)

    # Interaction variants
    _add(
        "vrp_when_hmm_risk_on", "interaction",
        signal_vrp_when_hmm_risk_on(vrp_features=feats, hmm_panel=hmm_panel),
    )
    _add(
        "vrp_when_hmm_risk_off", "interaction",
        signal_vrp_when_hmm_risk_off(vrp_features=feats, hmm_panel=hmm_panel),
    )
    _add(
        "hmm_sized_by_vrp", "interaction",
        signal_hmm_sized_by_vrp(vrp_features=feats, hmm_panel=hmm_panel),
    )
    _add(
        "vrp_sized_by_hmm_prob", "interaction",
        signal_vrp_sized_by_hmm_prob(vrp_features=feats, hmm_panel=hmm_panel),
    )
    _add(
        "additive_50_50", "interaction",
        signal_additive_ensemble(vrp_features=feats, hmm_panel=hmm_panel, w_hmm=0.5),
    )
    _add(
        "additive_70_30", "interaction",
        signal_additive_ensemble(vrp_features=feats, hmm_panel=hmm_panel, w_hmm=0.7),
    )
    _add(
        "orthogonalized_vrp", "interaction",
        signal_orthogonalized_vrp(
            vrp_features=feats, hmm_panel=hmm_panel, dev_end=spec.dev_end,
        ),
    )

    # Sanity baselines
    rng = np.random.default_rng(spec.bootstrap_seed + 9999)
    random_signal_df = target.select(["date"]).with_columns(
        pl.Series("signal", (rng.random(target.height) > 0.5).astype(np.float64))
    )
    _add("sanity_random", "sanity", random_signal_df)
    _add("sanity_buy_and_hold", "sanity", signal_buy_and_hold(target))

    # Cross-strategy: PBO across the whole pool
    series = [
        r.dev.daily_returns["net_return"].to_numpy().astype(np.float64)
        for r in results
    ]
    min_len = min(s.size for s in series) if series else 0
    if min_len >= 64 and len(results) >= 3:
        dev_matrix = np.column_stack([s[-min_len:] for s in series])
        pbo = compute_three_tier_pbo(
            returns=dev_matrix,
            profile=np.array(["interaction_pool" for _ in results]),
            family=np.array([r.category for r in results]),
            n_partitions=16,
        )
        pbo_raw = pbo.raw_global
        sharpes = np.array([r.dev.sharpe_annual for r in results], dtype=np.float64)
        best_idx = int(np.argmax(sharpes))
        dsr = compute_dsr(
            returns=series[best_idx],
            sharpe_estimates=sharpes,
            selected_idx=best_idx,
        )
        best_name = results[best_idx].name
        best_dsr = float(dsr.dsr)
        best_psr = float(dsr.psr_zero)
    else:
        pbo_raw = float("nan")
        best_name = ""
        best_dsr = float("nan")
        best_psr = float("nan")

    branch, decision, failure_class = _apply_decision_rules(
        results=results, spec=spec,
        pbo_raw_global=pbo_raw if not np.isnan(pbo_raw) else 1.0,
        best_dsr=best_dsr if not np.isnan(best_dsr) else 0.0,
    )

    return InteractionRunReport(
        spec=spec,
        results=results,
        pbo_raw_global=pbo_raw,
        best_name=best_name,
        best_dsr=best_dsr,
        best_psr_zero=best_psr,
        n_strategies=len(results),
        decision_branch=branch,
        decision=decision,
        failure_class=failure_class,
        hmm_panel=hmm_panel,
    )


def write_interaction_outputs(
    report: InteractionRunReport, *, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Registry parquet
    rows = []
    for r in report.results:
        rows.append({
            "name": r.name,
            "category": r.category,
            "dev_sharpe": r.dev.sharpe_annual,
            "holdout_sharpe": r.holdout.sharpe_annual,
            "cs_2x_sharpe": r.cost_stress_2x.sharpe_annual,
            "cs_3x_sharpe": r.cost_stress_3x.sharpe_annual,
            "delay_1d_sharpe": r.delay_1d.sharpe_annual,
            "boot_lower_95": r.bootstrap_lower_95,
            "boot_upper_95": r.bootstrap_upper_95,
            "max_dd_dev": r.dev.max_drawdown,
            "cum_ret_dev": r.dev.cumulative_return,
            "max_dd_hd": r.holdout.max_drawdown,
            "cum_ret_hd": r.holdout.cumulative_return,
            "turnover_total_dev": r.turnover_total,
            "exposure_time_frac": r.exposure_time_frac,
            "corr_with_hmm_only": r.attribution.corr_dev_with_hmm_only,
            "corr_with_vrp_only": r.attribution.corr_dev_with_vrp_only,
            "incremental_sharpe_over_hmm": r.attribution.incremental_sharpe_over_hmm_only,
            "incremental_sharpe_over_vrp": r.attribution.incremental_sharpe_over_vrp_only,
            "residual_sharpe_over_hmm": r.attribution.residual_sharpe_over_hmm_only,
            "residual_sharpe_over_vrp": r.attribution.residual_sharpe_over_vrp_only,
            "sharpe_excl_2020": r.attribution.sharpe_excl_2020,
            "sharpe_excl_2022": r.attribution.sharpe_excl_2022,
            "sharpe_excl_holdout_period": r.attribution.sharpe_excl_holdout_period,
        })
    registry_df = pl.DataFrame(rows)
    registry_path = output_dir / "vrp_hmm_interaction_registry.parquet"
    registry_df.write_parquet(registry_path)

    # 2. Validation report
    validation_path = output_dir / "vrp_hmm_validation_report.md"
    _write_validation_report(report, validation_path)

    # 3. Attribution report
    attribution_path = output_dir / "vrp_hmm_attribution_report.md"
    _write_attribution_report(report, attribution_path)

    # 4. Orthogonalization report
    orthogonal_path = output_dir / "vrp_hmm_orthogonalization_report.md"
    _write_orthogonalization_report(report, orthogonal_path)

    paths = {
        "registry": registry_path,
        "validation": validation_path,
        "attribution": attribution_path,
        "orthogonalization": orthogonal_path,
    }

    # 5. Failure classification only when decision is B or D
    if report.decision_branch in {"B", "D"}:
        failure_path = output_dir / "vrp_hmm_failure_classification.md"
        failure_path.write_text(
            f"# VRP × HMM Interaction — Failure Classification\n\n"
            f"**decision_branch**: `{report.decision_branch}`\n\n"
            f"**failure_class**: `{report.failure_class}`\n\n"
            f"**decision**: {report.decision}\n\n"
            f"Best strategy in pool: `{report.best_name}` "
            f"(DSR={report.best_dsr:.3f}, PSR_zero={report.best_psr_zero:.3f})\n\n"
            f"Per the user's decision rules (intake §B/§D), no interaction "
            f"variant produced enough incremental value over the HMM-only "
            f"baseline to justify promotion of VRP-augmented strategies.\n"
        )
        paths["failure"] = failure_path

    return paths


def _write_validation_report(
    report: InteractionRunReport, path: Path
) -> None:
    header = (
        "| Strategy | category | dev SR | dev CI_lo | dev CI_hi | holdout SR | "
        "cs-2x SR | cs-3x SR | delay-1d SR | max DD dev | turnover | exposure |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = []
    for r in report.results:
        rows.append(
            f"| `{r.name}` | {r.category} | "
            f"{r.dev.sharpe_annual:+.3f} | "
            f"{r.bootstrap_lower_95:+.3f} | {r.bootstrap_upper_95:+.3f} | "
            f"{r.holdout.sharpe_annual:+.3f} | "
            f"{r.cost_stress_2x.sharpe_annual:+.3f} | "
            f"{r.cost_stress_3x.sharpe_annual:+.3f} | "
            f"{r.delay_1d.sharpe_annual:+.3f} | "
            f"{r.dev.max_drawdown*100:+.2f}% | "
            f"{r.turnover_total:.2f} | "
            f"{r.exposure_time_frac*100:.1f}% |"
        )
    body = "\n".join([
        "# VRP × HMM Interaction — Validation Report",
        "",
        "## Pre-registered variant grid",
        "1. `hmm_only_baseline` (anchor)",
        "2. `vrp_only_baseline` (anchor)",
        "3. `vrp_when_hmm_risk_on` — intersection: long if VRP > 0 AND HMM = risk_on",
        "4. `vrp_when_hmm_risk_off` — intersection: long if VRP > 0 AND HMM = risk_off",
        "5. `hmm_sized_by_vrp` — HMM gate × clip(vrp_z60, 0, 1)",
        "6. `vrp_sized_by_hmm_prob` — VRP gate × p_risk_on (continuous)",
        "7. `additive_50_50` — 0.5 × HMM + 0.5 × VRP",
        "8. `additive_70_30` — 0.7 × HMM + 0.3 × VRP",
        "9. `orthogonalized_vrp` — sign(residual of VRP regressed on HMM, dev-only fit)",
        "",
        "Plus sanity baselines: `sanity_random`, `sanity_buy_and_hold`.",
        "",
        "## Results table",
        "",
        header, sep, *rows,
        "",
        "## Cross-strategy controls",
        "",
        f"- PBO raw_global: {report.pbo_raw_global:.3f}  (gate ≤ {report.spec.gate_pbo_max})",
        f"- Best strategy: `{report.best_name}`",
        f"- DSR for best: {report.best_dsr:.3f}  (gate ≥ {report.spec.gate_dsr_min})",
        f"- PSR_zero for best: {report.best_psr_zero:.3f}",
        f"- n_strategies: {report.n_strategies}",
        "",
        "## Decision",
        "",
        f"**Branch: {report.decision_branch}** — {report.decision}",
        "",
        f"failure_class: `{report.failure_class or 'none'}`",
        "",
    ])
    path.write_text(body)


def _write_attribution_report(
    report: InteractionRunReport, path: Path
) -> None:
    header = (
        "| Strategy | dev SR | ρ(HMM) | ρ(VRP) | incr. over HMM | incr. over VRP | "
        "residual SR vs HMM | residual SR vs VRP | excl-2020 SR | excl-2022 SR | "
        "excl-holdout SR |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = []
    for r in report.results:
        a = r.attribution
        rows.append(
            f"| `{r.name}` | "
            f"{a.sharpe_dev:+.3f} | "
            f"{a.corr_dev_with_hmm_only:+.3f} | "
            f"{a.corr_dev_with_vrp_only:+.3f} | "
            f"{a.incremental_sharpe_over_hmm_only:+.3f} | "
            f"{a.incremental_sharpe_over_vrp_only:+.3f} | "
            f"{a.residual_sharpe_over_hmm_only:+.3f} | "
            f"{a.residual_sharpe_over_vrp_only:+.3f} | "
            f"{a.sharpe_excl_2020:+.3f} | "
            f"{a.sharpe_excl_2022:+.3f} | "
            f"{a.sharpe_excl_holdout_period:+.3f} |"
        )
    # PnL by year breakdown for the interesting strategies (interaction + anchors)
    year_rows = []
    interesting = [r for r in report.results if r.category != "sanity"]
    all_years = sorted({y for r in interesting for y in r.attribution.pnl_by_year})
    year_header = "| Strategy | " + " | ".join(str(y) for y in all_years) + " |"
    year_sep = "|---|" + "|".join("---:" for _ in all_years) + "|"
    for r in interesting:
        cells = [
            f"{r.attribution.pnl_by_year.get(y, 0.0)*100:+.2f}%" for y in all_years
        ]
        year_rows.append(f"| `{r.name}` | " + " | ".join(cells) + " |")
    # PnL by regime
    regime_rows = []
    for r in interesting:
        on = r.attribution.pnl_by_regime.get("risk_on", 0.0)
        off = r.attribution.pnl_by_regime.get("risk_off", 0.0)
        regime_rows.append(
            f"| `{r.name}` | {on*100:+.2f}% | {off*100:+.2f}% |"
        )
    body = "\n".join([
        "# VRP × HMM Interaction — Attribution Report",
        "",
        "## Per-strategy attribution",
        "",
        header, sep, *rows,
        "",
        "## PnL by year (dev only)",
        "",
        year_header, year_sep, *year_rows,
        "",
        "## PnL by HMM regime (dev only)",
        "",
        "| Strategy | risk_on PnL | risk_off PnL |",
        "|---|---:|---:|",
        *regime_rows,
        "",
        "## Interpretation guide",
        "",
        "- `incr. over HMM` is the raw Sharpe difference. Positive means the "
        "strategy *appears* to add value, but does not account for shared regime exposure.",
        "- `residual SR vs HMM` is the Sharpe of the orthogonal component after "
        "regressing the strategy's daily returns on HMM-only's. Positive means "
        "the strategy has real incremental information.",
        "- `excl-X SR` shows how much of the dev edge depends on a specific year. ",
        "A strategy whose Sharpe materially collapses when one crisis is removed "
        "is regime-concentrated.",
        "",
    ])
    path.write_text(body)


def _write_orthogonalization_report(
    report: InteractionRunReport, path: Path
) -> None:
    ortho = next(
        (r for r in report.results if r.name == "orthogonalized_vrp"), None
    )
    hmm_only = next(
        (r for r in report.results if r.name == "hmm_only_baseline"), None
    )
    vrp_only = next(
        (r for r in report.results if r.name == "vrp_only_baseline"), None
    )
    if ortho is None or hmm_only is None or vrp_only is None:
        path.write_text("# Orthogonalization Report\n\nERROR: anchor missing.")
        return
    a = ortho.attribution
    survives = (
        ortho.bootstrap_lower_95 > 0.0
        and ortho.dev.sharpe_annual > 0.3
        and ortho.cost_stress_2x.sharpe_annual > 0.0
        and a.residual_sharpe_over_hmm_only > 0.3
    )
    body = "\n".join([
        "# Orthogonalized VRP — Independence Test",
        "",
        "## Method",
        "",
        "On the dev window, regress the binary `vrp_long_only` signal on the",
        "binary `hmm_risk_on` signal (with intercept). Use the fitted (α, β) to",
        "compute the residual VRP component on the full sample. Trade sign(residual).",
        "",
        "This isolates the VRP information that is NOT explained by the HMM regime.",
        "If this strategy generates positive risk-adjusted returns, VRP carries",
        "independent information beyond HMM risk-timing.",
        "",
        "## Results",
        "",
        f"- orthogonalized_vrp dev Sharpe: {ortho.dev.sharpe_annual:+.3f}",
        f"- bootstrap 95% CI: [{ortho.bootstrap_lower_95:+.3f}, "
        f"{ortho.bootstrap_upper_95:+.3f}]",
        f"- holdout Sharpe: {ortho.holdout.sharpe_annual:+.3f}",
        f"- cost-stress 2× Sharpe: {ortho.cost_stress_2x.sharpe_annual:+.3f}",
        f"- 1-bar delay Sharpe: {ortho.delay_1d.sharpe_annual:+.3f}",
        f"- correlation with HMM-only (dev): {a.corr_dev_with_hmm_only:+.3f}",
        f"- residual Sharpe vs HMM-only (dev): {a.residual_sharpe_over_hmm_only:+.3f}",
        f"- Sharpe excluding 2020: {a.sharpe_excl_2020:+.3f}",
        f"- Sharpe excluding 2022: {a.sharpe_excl_2022:+.3f}",
        f"- exposure time fraction: {ortho.exposure_time_frac*100:.1f}%",
        "",
        "## Survival criteria (all required)",
        "",
        f"- bootstrap CI lower > 0: "
        f"{'PASS' if ortho.bootstrap_lower_95 > 0 else 'FAIL'} "
        f"({ortho.bootstrap_lower_95:+.3f})",
        f"- dev Sharpe > 0.3: "
        f"{'PASS' if ortho.dev.sharpe_annual > 0.3 else 'FAIL'} "
        f"({ortho.dev.sharpe_annual:+.3f})",
        f"- cost-stress 2× Sharpe > 0: "
        f"{'PASS' if ortho.cost_stress_2x.sharpe_annual > 0 else 'FAIL'} "
        f"({ortho.cost_stress_2x.sharpe_annual:+.3f})",
        f"- residual Sharpe vs HMM > 0.3: "
        f"{'PASS' if a.residual_sharpe_over_hmm_only > 0.3 else 'FAIL'} "
        f"({a.residual_sharpe_over_hmm_only:+.3f})",
        "",
        f"## Verdict: {'SURVIVES — VRP carries independent information' if survives else 'FAILS — VRP information is subsumed by HMM regime'}",
        "",
    ])
    path.write_text(body)
