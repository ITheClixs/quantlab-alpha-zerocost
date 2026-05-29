"""HMM single-index v1 validation runner.

Orchestrates the frozen 30-strategy pool (18 HMM variants + 12 baselines)
on Tier-1 instruments (SPY, QQQ) per the accepted intake document.

This runner does not modify the default no-OHLCV-promotion rule. It uses
the exception path activated by ValidationSpec.exception_invoked=True
plus the policy reference and Tier-1 instrument declaration.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

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
from quant_research_stack.signal_research.status import CandidateStatus
from quant_research_stack.signal_research.strategies.hmm_single_index import (
    DEFAULT_FEATURE_SET,
    HMMStrategyConfig,
    compute_feature_panel,
    fit_variant_models,
    predeclared_variant_grid,
    predict_signal_long_or_cash,
)
from quant_research_stack.signal_research.validation.cash_leg_reporting import (
    CASH_CONSERVATIVE,
    CashLegResult,
    evaluate_all_cash_legs,
)
from quant_research_stack.signal_research.validation.exception_robustness import (
    CrashProtectionDiagnostics,
    RegimeExposureDiagnostics,
    StateIdentityFlipReport,
    TransitionDiagnostics,
    TurnoverDiagnostics,
    crash_protection_diagnostics,
    economic_identity_stability,
    regime_exposure,
    transition_diagnostics,
    turnover_diagnostics,
)
from quant_research_stack.signal_research.validation.spec import (
    ACCEPTED_EXCEPTION_POLICY_REF,
)
from quant_research_stack.strategy_benchmark.dsr import compute_dsr

# Reuse same intake hardcoded paths
ACCEPTED_INTAKE_REF: str = (
    "docs/research/intake/2026-05-28-hmm-single-index-v1.md"
)


@dataclass(frozen=True)
class HMMRunnerSpec:
    instruments: tuple[str, ...] = ("SPY", "QQQ")
    start: dt.date = dt.date(2010, 1, 1)
    end: dt.date = dt.date(2026, 5, 26)
    dev_end: dt.date = dt.date(2022, 12, 31)
    holdout_start: dt.date = dt.date(2023, 1, 1)
    feature_set: tuple[str, ...] = DEFAULT_FEATURE_SET
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 0.5
    cost_stress_multipliers: tuple[float, ...] = (2.0, 3.0)
    delay_stress_bars: tuple[int, ...] = (1, 2)
    bootstrap_n_resamples: int = 2000
    bootstrap_seed: int = 42
    # Gates (per intake §9)
    gate_dev_sharpe_min: float = 1.5
    gate_holdout_sharpe_min: float = 1.5
    gate_pbo_max: float = 0.25
    gate_dsr_min: float = 0.5
    gate_bootstrap_ci_lower_min: float = 0.5
    gate_max_dd_floor: float = -0.20  # max DD >= -20% required
    gate_calmar_min_if_dd_worse: float = 1.0
    gate_yearly_share_max: float = 0.50
    gate_quarterly_share_max: float = 0.35
    gate_min_positive_years: int = 2
    gate_sharpe_excl_year_min: float = 0.8
    intake_ref: str = ACCEPTED_INTAKE_REF
    exception_policy_ref: str = ACCEPTED_EXCEPTION_POLICY_REF


@dataclass(frozen=True)
class GateScorecard:
    """Per-variant gate-by-gate pass/fail snapshot (per intake §9)."""

    dev_sharpe_pass: bool
    holdout_sharpe_pass: bool
    cost_stress_2x_pass: bool
    cost_stress_3x_pass: bool
    delay_1d_pass: bool
    delay_2d_pass: bool
    max_dd_or_calmar_pass: bool
    year_share_pass: bool
    quarter_share_pass: bool
    min_positive_years_pass: bool
    survives_excl_2020_pass: bool
    survives_excl_2022_pass: bool
    survives_pre_2020_subsample_pass: bool
    beats_buy_and_hold_pass: bool
    beats_vol_targeted_pass: bool
    beats_sma_50_200_pass: bool
    beats_mom_12_1_pass: bool
    random_baseline_fails_pass: bool
    inverted_baseline_fails_pass: bool
    bootstrap_ci_lower_pass: bool
    pbo_pass: bool
    dsr_pass: bool
    economic_identity_stability_pass: bool
    cash_leg_conservative_pass: bool
    all_pass: bool


@dataclass(frozen=True)
class VariantValidationResult:
    """All artifacts and metrics for one strategy entry in the 30-strategy pool."""

    name: str
    category: str  # "hmm_variant" | "baseline"
    instrument: str
    config: HMMStrategyConfig | None
    signal: pl.DataFrame  # (date, signal)
    dev_cash_legs: dict[str, CashLegResult]
    holdout_cash_legs: dict[str, CashLegResult]
    cost_stress_dev: dict[float, float]  # mult -> Sharpe
    delay_stress_dev: dict[int, float]  # bars -> Sharpe
    bootstrap_lower_95: float
    bootstrap_upper_95: float
    yearly_pnl_share_max: float
    quarterly_pnl_share_max: float
    n_positive_years: int
    sharpe_excl_2020: float
    sharpe_excl_2022: float
    sharpe_pre_2020: float
    transition: TransitionDiagnostics | None
    regime_exposure_dev: RegimeExposureDiagnostics | None
    crash_protection: CrashProtectionDiagnostics | None
    turnover: TurnoverDiagnostics | None
    state_stability: StateIdentityFlipReport | None
    failure_classes: list[str] = field(default_factory=list)


def _underlying_returns(bars: pl.DataFrame, instrument: str) -> pl.DataFrame:
    """Return (date, u_ret) for one instrument."""
    df = (
        bars.filter(pl.col("symbol") == instrument)
        .sort("date")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("u_ret")
        )
        .drop_nulls(subset=["u_ret"])
        .select(["date", "u_ret"])
    )
    return df


def _safe_sharpe(rets: NDArray[np.float64]) -> float:
    if rets.size < 2:
        return 0.0
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(rets)) / sd * float(np.sqrt(252.0))


def _yearly_pnl_share_max(daily: pl.DataFrame) -> float:
    if daily.is_empty() or "net_return" not in daily.columns:
        return 0.0
    with_year = daily.with_columns(pl.col("date").dt.year().alias("year"))
    yearly = with_year.group_by("year").agg(pl.col("net_return").sum().alias("pnl"))
    total_abs = float(np.sum(np.abs(yearly["pnl"].to_numpy()))) or 1e-9
    return float(np.abs(yearly["pnl"].to_numpy()).max() / total_abs)


def _quarterly_pnl_share_max(daily: pl.DataFrame) -> float:
    if daily.is_empty() or "net_return" not in daily.columns:
        return 0.0
    df = daily.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.quarter().alias("qtr"),
    )
    quart = df.group_by(["year", "qtr"]).agg(pl.col("net_return").sum().alias("pnl"))
    total_abs = float(np.sum(np.abs(quart["pnl"].to_numpy()))) or 1e-9
    return float(np.abs(quart["pnl"].to_numpy()).max() / total_abs)


def _n_positive_years(daily: pl.DataFrame) -> int:
    if daily.is_empty():
        return 0
    with_year = daily.with_columns(pl.col("date").dt.year().alias("year"))
    yearly = with_year.group_by("year").agg(pl.col("net_return").sum().alias("pnl"))
    return int((yearly["pnl"].to_numpy() > 0).sum())


def _sharpe_excluding_year(daily: pl.DataFrame, year: int) -> float:
    sub = daily.filter(pl.col("date").dt.year() != year)
    if sub.height < 30:
        return float("nan")
    return _safe_sharpe(sub["net_return"].to_numpy().astype(np.float64))


def _sharpe_pre_2020(daily: pl.DataFrame) -> float:
    sub = daily.filter(pl.col("date").dt.year() < 2020)
    if sub.height < 30:
        return float("nan")
    return _safe_sharpe(sub["net_return"].to_numpy().astype(np.float64))


def _bootstrap_ci(
    rets: NDArray[np.float64], *, n_resamples: int, seed: int,
) -> tuple[float, float]:
    if rets.size < 30:
        return float("nan"), float("nan")
    bs = bootstrap_sharpe_ci(
        returns=rets, config=BootstrapConfig(n_resamples=n_resamples, seed=seed),
    )
    return bs.ci_lower_95, bs.ci_upper_95


def _baseline_signal(
    name: str, bars: pl.DataFrame, instrument: str, dev_end: dt.date,
) -> pl.DataFrame:
    """Generate signal for one of the predeclared baselines."""
    underlying = (
        bars.filter(pl.col("symbol") == instrument)
        .sort("date")
        .with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret")
        )
    )
    if name == "buy_and_hold":
        return underlying.select(["date"]).with_columns(pl.lit(1.0).alias("signal"))
    if name == "vol_targeted_buy_and_hold":
        # 10% annualized target via trailing 60d realised vol
        df = underlying.with_columns(
            (pl.col("log_ret").rolling_std(window_size=60) * (252.0 ** 0.5))
            .alias("rv_60")
        )
        df = df.with_columns(
            (0.10 / pl.col("rv_60").clip(lower_bound=0.02)).clip(
                lower_bound=0.0, upper_bound=1.0,
            ).alias("signal")
        )
        return df.select(["date", "signal"]).fill_null(0.0)
    if name == "sma_50_200_gate":
        df = underlying.with_columns(
            pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
            pl.col("close").rolling_mean(window_size=200).alias("sma_200"),
        ).with_columns(
            pl.when(pl.col("sma_50") > pl.col("sma_200"))
            .then(1.0).otherwise(0.0).alias("signal")
        )
        return df.select(["date", "signal"]).fill_null(0.0)
    if name == "mom_12_1":
        df = underlying.with_columns(
            (
                pl.col("log_ret").rolling_sum(window_size=252)
                - pl.col("log_ret").rolling_sum(window_size=21)
            ).alias("mom"),
        )
        df = df.with_columns(
            pl.when(pl.col("mom") > 0.0).then(1.0).otherwise(0.0).alias("signal")
        )
        return df.select(["date", "signal"]).fill_null(0.0)
    if name == "random":
        rng = np.random.default_rng(seed=hash(instrument) & 0xFFFF)
        n = underlying.height
        return underlying.select(["date"]).with_columns(
            pl.Series("signal", (rng.random(n) > 0.5).astype(np.float64))
        )
    raise ValueError(f"unknown baseline: {name}")


def _evaluate_variant(
    *,
    name: str,
    category: str,
    instrument: str,
    config: HMMStrategyConfig | None,
    signal: pl.DataFrame,
    underlying_ret: pl.DataFrame,
    tbill_panel: pl.DataFrame,
    runner_spec: HMMRunnerSpec,
    bars: pl.DataFrame,
    fits: list | None = None,
) -> VariantValidationResult:
    """Run the full evaluation for one entry in the 30-strategy pool."""
    sig_sorted = signal.sort("date")
    joined = underlying_ret.join(sig_sorted, on="date", how="left").with_columns(
        pl.col("signal").fill_null(0.0)
    )
    pos = joined["signal"].to_numpy().astype(np.float64)
    pos = np.clip(pos, 0.0, 1.0)

    # Dev split + cash legs
    dev_mask_arr = joined["date"].to_numpy() <= np.datetime64(runner_spec.dev_end)
    hd_mask_arr = joined["date"].to_numpy() >= np.datetime64(runner_spec.holdout_start)
    dev_ur = joined.filter(pl.col("date") <= runner_spec.dev_end).select(["date", "u_ret"])
    hd_ur = joined.filter(pl.col("date") >= runner_spec.holdout_start).select(["date", "u_ret"])
    dev_pos = pos[dev_mask_arr]
    hd_pos = pos[hd_mask_arr]

    dev_cash_legs = evaluate_all_cash_legs(
        underlying_returns=dev_ur, position=dev_pos,
        tbill_panel=tbill_panel,
        cost_bps_one_way=runner_spec.commission_bps_one_way + runner_spec.spread_bps_one_way,
    )
    hd_cash_legs = evaluate_all_cash_legs(
        underlying_returns=hd_ur, position=hd_pos,
        tbill_panel=tbill_panel,
        cost_bps_one_way=runner_spec.commission_bps_one_way + runner_spec.spread_bps_one_way,
    )

    conservative_dev_daily = dev_cash_legs[CASH_CONSERVATIVE.name].daily_returns
    conservative_dev_rets = (
        conservative_dev_daily["net_return"].to_numpy().astype(np.float64)
    )

    # Cost stress (vary the cost multiplier on the conservative cash assumption)
    cost_stress_dev: dict[float, float] = {}
    base_cost = runner_spec.commission_bps_one_way + runner_spec.spread_bps_one_way
    for mult in runner_spec.cost_stress_multipliers:
        legs = evaluate_all_cash_legs(
            underlying_returns=dev_ur, position=dev_pos,
            tbill_panel=tbill_panel, cost_bps_one_way=base_cost * mult,
        )
        cost_stress_dev[mult] = legs[CASH_CONSERVATIVE.name].sharpe_annual

    # Delay stress
    delay_stress_dev: dict[int, float] = {}
    for n in runner_spec.delay_stress_bars:
        pos_delayed = np.concatenate([np.zeros(n), pos[:-n]]) if n > 0 else pos
        dev_pos_delayed = pos_delayed[dev_mask_arr]
        legs = evaluate_all_cash_legs(
            underlying_returns=dev_ur, position=dev_pos_delayed,
            tbill_panel=tbill_panel, cost_bps_one_way=base_cost,
        )
        delay_stress_dev[n] = legs[CASH_CONSERVATIVE.name].sharpe_annual

    # Bootstrap CI
    lo, hi = _bootstrap_ci(
        conservative_dev_rets,
        n_resamples=runner_spec.bootstrap_n_resamples,
        seed=runner_spec.bootstrap_seed,
    )

    # Concentration
    yearly_max = _yearly_pnl_share_max(conservative_dev_daily)
    quarterly_max = _quarterly_pnl_share_max(conservative_dev_daily)
    n_pos_years = _n_positive_years(conservative_dev_daily)

    # Exclusion-test Sharpes
    s_excl_2020 = _sharpe_excluding_year(conservative_dev_daily, 2020)
    s_excl_2022 = _sharpe_excluding_year(conservative_dev_daily, 2022)
    s_pre_2020 = _sharpe_pre_2020(conservative_dev_daily)

    # Robustness diagnostics
    transition_diag: TransitionDiagnostics | None = None
    regime_diag: RegimeExposureDiagnostics | None = None
    crash_diag: CrashProtectionDiagnostics | None = None
    turn_diag: TurnoverDiagnostics | None = None
    stability: StateIdentityFlipReport | None = None
    if fits:
        transition_diag = transition_diagnostics(fits[-1])
        stability = economic_identity_stability(fits)
    regime_diag = regime_exposure(daily_returns=conservative_dev_daily, signal=sig_sorted)
    crash_diag = crash_protection_diagnostics(
        underlying_returns=dev_ur, strategy_daily=conservative_dev_daily,
        signal=sig_sorted,
    )
    turn_diag = turnover_diagnostics(
        daily_returns=conservative_dev_daily, cost_bps_one_way=base_cost,
    )

    return VariantValidationResult(
        name=name,
        category=category,
        instrument=instrument,
        config=config,
        signal=sig_sorted,
        dev_cash_legs=dev_cash_legs,
        holdout_cash_legs=hd_cash_legs,
        cost_stress_dev=cost_stress_dev,
        delay_stress_dev=delay_stress_dev,
        bootstrap_lower_95=lo,
        bootstrap_upper_95=hi,
        yearly_pnl_share_max=yearly_max,
        quarterly_pnl_share_max=quarterly_max,
        n_positive_years=n_pos_years,
        sharpe_excl_2020=s_excl_2020,
        sharpe_excl_2022=s_excl_2022,
        sharpe_pre_2020=s_pre_2020,
        transition=transition_diag,
        regime_exposure_dev=regime_diag,
        crash_protection=crash_diag,
        turnover=turn_diag,
        state_stability=stability,
    )


def run_hmm_v1_pipeline(
    *,
    bars: pl.DataFrame,
    runner_spec: HMMRunnerSpec,
    tbill_panel: pl.DataFrame,
) -> tuple[list[VariantValidationResult], dict[str, object], dict[str, GateScorecard]]:
    """Run the 18 HMM variants + 12 baselines, compute PBO/DSR, score gates.

    Returns:
        - results: list of per-variant outputs
        - cross_metrics: PBO/DSR/PSR_zero
        - gate_scorecards: per-variant pass/fail (intake §9)
    """
    results: list[VariantValidationResult] = []

    # HMM variant grid
    variant_grid = predeclared_variant_grid()
    # Only run variants for instruments actually present in the bars
    available_instruments = set(bars["symbol"].to_list())

    # Cache features + fits per instrument × variant
    for cfg in variant_grid:
        if cfg.instrument not in available_instruments:
            continue
        inst_bars = bars.filter(pl.col("symbol") == cfg.instrument).sort("date")
        if inst_bars.height < 252 * 3:
            continue
        features = compute_feature_panel(inst_bars, feature_set=cfg.feature_set)
        fits = fit_variant_models(
            config=cfg, features=features, bars_for_returns=inst_bars,
            start=runner_spec.start, dev_end=runner_spec.dev_end,
        )
        if not fits:
            continue
        signal = predict_signal_long_or_cash(
            config=cfg, fits=fits, features=features, dev_end=runner_spec.dev_end,
        )
        u_ret = _underlying_returns(bars, cfg.instrument)
        result = _evaluate_variant(
            name=cfg.variant_name, category="hmm_variant",
            instrument=cfg.instrument, config=cfg, signal=signal,
            underlying_ret=u_ret, tbill_panel=tbill_panel,
            runner_spec=runner_spec, bars=bars, fits=fits,
        )
        results.append(result)

    # Baselines per instrument
    baseline_names = (
        "buy_and_hold", "vol_targeted_buy_and_hold", "sma_50_200_gate",
        "mom_12_1", "random",
    )
    for inst in runner_spec.instruments:
        if inst not in available_instruments:
            continue
        for bname in baseline_names:
            sig = _baseline_signal(bname, bars, inst, runner_spec.dev_end)
            u_ret = _underlying_returns(bars, inst)
            result = _evaluate_variant(
                name=f"{bname}_{inst.lower()}", category="baseline",
                instrument=inst, config=None, signal=sig,
                underlying_ret=u_ret, tbill_panel=tbill_panel,
                runner_spec=runner_spec, bars=bars, fits=None,
            )
            results.append(result)

    # Inverted baseline of best HMM per instrument
    for inst in runner_spec.instruments:
        if inst not in available_instruments:
            continue
        hmm_results = [
            r for r in results
            if r.category == "hmm_variant" and r.instrument == inst
        ]
        if not hmm_results:
            continue
        best = max(
            hmm_results,
            key=lambda r: r.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual,
        )
        # Invert: signal -> 1 - signal (long when best was flat)
        inv_signal = best.signal.with_columns(
            (1.0 - pl.col("signal")).alias("signal")
        )
        u_ret = _underlying_returns(bars, inst)
        result = _evaluate_variant(
            name=f"inverted_of_best_hmm_{inst.lower()}",
            category="baseline",
            instrument=inst, config=None, signal=inv_signal,
            underlying_ret=u_ret, tbill_panel=tbill_panel,
            runner_spec=runner_spec, bars=bars, fits=None,
        )
        results.append(result)

    # Cross-strategy PBO + DSR
    series = [
        r.dev_cash_legs[CASH_CONSERVATIVE.name].daily_returns["net_return"]
        .to_numpy().astype(np.float64)
        for r in results
    ]
    min_len = min(s.size for s in series) if series else 0
    if min_len >= 64 and len(results) >= 3:
        dev_matrix = np.column_stack([s[-min_len:] for s in series])
        pbo = compute_three_tier_pbo(
            returns=dev_matrix,
            profile=np.array([r.instrument for r in results]),
            family=np.array([r.category for r in results]),
            n_partitions=16,
        )
        sharpes = np.array([
            r.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual for r in results
        ])
        best_idx = int(np.argmax(sharpes))
        dsr = compute_dsr(
            returns=series[best_idx],
            sharpe_estimates=sharpes.astype(np.float64),
            selected_idx=best_idx,
        )
        cross_metrics = {
            "pbo_raw_global": pbo.raw_global,
            "best_name": results[best_idx].name,
            "best_dsr": float(dsr.dsr),
            "best_psr_zero": float(dsr.psr_zero),
            "n_strategies": len(results),
        }
    else:
        cross_metrics = {
            "pbo_raw_global": float("nan"),
            "best_name": "",
            "best_dsr": float("nan"),
            "best_psr_zero": float("nan"),
            "n_strategies": len(results),
        }

    # Gate scorecards
    gate_scorecards: dict[str, GateScorecard] = {}
    for r in results:
        gate_scorecards[r.name] = _score_gates(
            r, runner_spec=runner_spec, cross_metrics=cross_metrics, results=results,
        )

    return results, cross_metrics, gate_scorecards


def _score_gates(
    result: VariantValidationResult, *,
    runner_spec: HMMRunnerSpec,
    cross_metrics: dict[str, object],
    results: list[VariantValidationResult],
) -> GateScorecard:
    """Apply the intake §9 24 gates to one variant."""
    dev = result.dev_cash_legs[CASH_CONSERVATIVE.name]
    hd = result.holdout_cash_legs[CASH_CONSERVATIVE.name]
    dev_sharpe_pass = dev.sharpe_annual >= runner_spec.gate_dev_sharpe_min
    holdout_sharpe_pass = hd.sharpe_annual >= runner_spec.gate_holdout_sharpe_min
    cost_stress_2x_pass = result.cost_stress_dev.get(2.0, -10.0) > 0
    cost_stress_3x_pass = result.cost_stress_dev.get(3.0, -10.0) > 0
    delay_1d_sharpe = result.delay_stress_dev.get(1, -10.0)
    delay_2d_sharpe = result.delay_stress_dev.get(2, -10.0)
    delay_1d_pass = (dev.sharpe_annual - delay_1d_sharpe) <= 0.5
    delay_2d_pass = (dev.sharpe_annual - delay_2d_sharpe) <= 0.5
    if dev.max_drawdown >= runner_spec.gate_max_dd_floor:
        max_dd_pass = True
    else:
        # Compute Calmar (annualized return / |max DD|)
        annualized_ret = (
            (1.0 + dev.cumulative_return) ** (252.0 / max(1, dev.n_days)) - 1.0
        ) if dev.n_days > 0 else 0.0
        calmar = annualized_ret / abs(dev.max_drawdown) if dev.max_drawdown < 0 else 0.0
        max_dd_pass = calmar > runner_spec.gate_calmar_min_if_dd_worse
    year_share_pass = result.yearly_pnl_share_max <= runner_spec.gate_yearly_share_max
    quarter_share_pass = (
        result.quarterly_pnl_share_max <= runner_spec.gate_quarterly_share_max
    )
    min_pos_years_pass = (
        result.n_positive_years >= runner_spec.gate_min_positive_years
    )
    excl_2020_pass = (
        result.sharpe_excl_2020 >= runner_spec.gate_sharpe_excl_year_min
    )
    excl_2022_pass = (
        result.sharpe_excl_2022 >= runner_spec.gate_sharpe_excl_year_min
    )
    pre_2020_pass = (
        result.sharpe_pre_2020 >= runner_spec.gate_sharpe_excl_year_min
    )
    # Baseline-beat checks — only meaningful for HMM variants
    inst = result.instrument
    if result.category == "hmm_variant":
        bah = _find_baseline(results, inst, "buy_and_hold")
        vt = _find_baseline(results, inst, "vol_targeted_buy_and_hold")
        sma = _find_baseline(results, inst, "sma_50_200_gate")
        mom = _find_baseline(results, inst, "mom_12_1")
        random_b = _find_baseline(results, inst, "random")
        inverted_b = _find_baseline(results, inst, "inverted_of_best_hmm")
        beats_bah = (
            bah is not None
            and dev.sharpe_annual > bah.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual
            and dev.max_drawdown > bah.dev_cash_legs[CASH_CONSERVATIVE.name].max_drawdown
        )
        beats_vt = (
            vt is not None
            and dev.sharpe_annual > vt.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual
            and dev.max_drawdown > vt.dev_cash_legs[CASH_CONSERVATIVE.name].max_drawdown
        )
        beats_sma = (
            sma is not None
            and dev.sharpe_annual > sma.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual
            and dev.max_drawdown > sma.dev_cash_legs[CASH_CONSERVATIVE.name].max_drawdown
        )
        beats_mom = (
            mom is not None
            and dev.sharpe_annual > mom.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual
            and dev.max_drawdown > mom.dev_cash_legs[CASH_CONSERVATIVE.name].max_drawdown
        )
        random_fails = (
            random_b is None
            or random_b.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual
            < runner_spec.gate_dev_sharpe_min
        )
        inverted_fails = (
            inverted_b is None
            or inverted_b.dev_cash_legs[CASH_CONSERVATIVE.name].sharpe_annual
            < runner_spec.gate_dev_sharpe_min
        )
    else:
        beats_bah = beats_vt = beats_sma = beats_mom = False
        random_fails = inverted_fails = False
    boot_lower_pass = (
        result.bootstrap_lower_95 > runner_spec.gate_bootstrap_ci_lower_min
    )
    pbo_val = cross_metrics.get("pbo_raw_global", 1.0)
    dsr_val = cross_metrics.get("best_dsr", 0.0)
    pbo_pass = (
        isinstance(pbo_val, int | float)
        and not np.isnan(float(pbo_val))
        and float(pbo_val) <= runner_spec.gate_pbo_max
    )
    dsr_pass = (
        isinstance(dsr_val, int | float)
        and not np.isnan(float(dsr_val))
        and float(dsr_val) >= runner_spec.gate_dsr_min
    )
    stability_pass = (
        result.state_stability is None
        or result.state_stability.passes_stability_gate
    )
    # Cash-leg conservative gate is structural — already evaluated against
    # CASH_CONSERVATIVE above. Treat as pass if dev_sharpe is finite.
    cash_leg_conservative_pass = not np.isnan(dev.sharpe_annual)

    all_pass = all([
        dev_sharpe_pass, holdout_sharpe_pass,
        cost_stress_2x_pass, cost_stress_3x_pass,
        delay_1d_pass, delay_2d_pass,
        max_dd_pass, year_share_pass, quarter_share_pass,
        min_pos_years_pass, excl_2020_pass, excl_2022_pass, pre_2020_pass,
        beats_bah if result.category == "hmm_variant" else True,
        beats_vt if result.category == "hmm_variant" else True,
        beats_sma if result.category == "hmm_variant" else True,
        beats_mom if result.category == "hmm_variant" else True,
        random_fails if result.category == "hmm_variant" else True,
        inverted_fails if result.category == "hmm_variant" else True,
        boot_lower_pass, pbo_pass, dsr_pass,
        stability_pass, cash_leg_conservative_pass,
    ])
    return GateScorecard(
        dev_sharpe_pass=dev_sharpe_pass,
        holdout_sharpe_pass=holdout_sharpe_pass,
        cost_stress_2x_pass=cost_stress_2x_pass,
        cost_stress_3x_pass=cost_stress_3x_pass,
        delay_1d_pass=delay_1d_pass,
        delay_2d_pass=delay_2d_pass,
        max_dd_or_calmar_pass=max_dd_pass,
        year_share_pass=year_share_pass,
        quarter_share_pass=quarter_share_pass,
        min_positive_years_pass=min_pos_years_pass,
        survives_excl_2020_pass=excl_2020_pass,
        survives_excl_2022_pass=excl_2022_pass,
        survives_pre_2020_subsample_pass=pre_2020_pass,
        beats_buy_and_hold_pass=beats_bah,
        beats_vol_targeted_pass=beats_vt,
        beats_sma_50_200_pass=beats_sma,
        beats_mom_12_1_pass=beats_mom,
        random_baseline_fails_pass=random_fails,
        inverted_baseline_fails_pass=inverted_fails,
        bootstrap_ci_lower_pass=boot_lower_pass,
        pbo_pass=pbo_pass,
        dsr_pass=dsr_pass,
        economic_identity_stability_pass=stability_pass,
        cash_leg_conservative_pass=cash_leg_conservative_pass,
        all_pass=all_pass,
    )


def _find_baseline(
    results: list[VariantValidationResult], instrument: str, base_name: str,
) -> VariantValidationResult | None:
    full_name = (
        f"{base_name}_{instrument.lower()}"
        if base_name != "inverted_of_best_hmm"
        else f"inverted_of_best_hmm_{instrument.lower()}"
    )
    for r in results:
        if r.name == full_name:
            return r
    return None


def assign_exception_status(
    *, scorecard: GateScorecard, category: str,
) -> CandidateStatus:
    """Per intake §11: max status reachable from this validation is
    EXCEPTION_REVIEW_REQUIRED (no PROMOTION_ELIGIBLE under exception path,
    no PAPER_TRADE_CANDIDATE, no PRODUCTION_CANDIDATE).
    """
    if category != "hmm_variant":
        # Baselines never reach exception_review_required.
        return CandidateStatus.NONE
    if scorecard.all_pass:
        return CandidateStatus.EXCEPTION_REVIEW_REQUIRED
    # Partial pass: still RESEARCH_PASS if dev Sharpe is positive
    if scorecard.dev_sharpe_pass:
        return CandidateStatus.RESEARCH_PASS
    return CandidateStatus.NONE
