"""ValidationPipeline — the single entrypoint for vetting any new strategy.

Wraps the methodology stack produced by the six-iteration search:
- walk-forward / CPCV training (via the strategy's own signal_fn)
- permanent holdout (dev-only guard)
- bootstrap stationary block CIs
- three-tier PBO across the strategy pool
- deflated Sharpe ratio with multi-test penalty
- cost decomposition (no/fee/spread/full/stress)
- one-bar delay stress
- random + inverted-signal sanity baselines
- concentration diagnostics
- 8-criteria promotion gate with failure-class taxonomy
- 'no promotion without new information source' rule
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.alpha_eq.backtest.runner import run_backtest
from quant_research_stack.signal_research.backtests._shared import (
    build_backtest_config,
    data_quality_banner,
    equity_metrics,
    to_m4_panel,
)
from quant_research_stack.signal_research.methodology.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_sharpe_ci,
)
from quant_research_stack.signal_research.methodology.failure_classifier import (
    FailureCategory,
)
from quant_research_stack.signal_research.methodology.pbo_extensions import (
    compute_three_tier_pbo,
)
from quant_research_stack.signal_research.methodology.selection_funnel import (
    SelectionFunnel,
)
from quant_research_stack.signal_research.status import CandidateStatus
from quant_research_stack.signal_research.validation.concentration import (
    ConcentrationReport,
    concentration_by_period,
)
from quant_research_stack.signal_research.validation.cost_decomposition import (
    CostDecomposition,
    cost_decomposition,
)
from quant_research_stack.signal_research.validation.delay_stress import (
    shift_signal_by_n_bars,
)
from quant_research_stack.signal_research.validation.sanity import (
    inverted_signal,
    random_signal,
)
from quant_research_stack.signal_research.validation.spec import (
    ACCEPTED_EXCEPTION_POLICY_REF,
    TIER_1_INSTRUMENTS,
    ValidationSpec,
    feature_audit_violation,
)
from quant_research_stack.strategy_benchmark.dsr import compute_dsr

SignalFn = Callable[[pl.DataFrame, ValidationSpec], pl.DataFrame]


@dataclass(frozen=True)
class StrategyValidationResult:
    name: str
    dev_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    cost_decomposition: CostDecomposition
    delay_stress: dict[int, dict[str, float]]
    concentration_dev: ConcentrationReport
    concentration_holdout: ConcentrationReport
    dev_net_returns: NDArray[np.float64]
    holdout_net_returns: NDArray[np.float64]
    sanity_random_sharpe: float
    sanity_inverted_sharpe: float
    research_pass: bool
    failure_classes: list[FailureCategory]
    funnel: SelectionFunnel = field(default_factory=SelectionFunnel)


@dataclass(frozen=True)
class PipelineReport:
    spec: ValidationSpec
    result: StrategyValidationResult
    pbo_raw_global: float
    dsr: float
    psr_zero: float
    assigned_status: CandidateStatus
    promotion_eligible: bool


def _bootstrap_ci(
    rets: NDArray[np.float64], *, n_resamples: int, seed: int
) -> tuple[float, float]:
    if rets.size < 30:
        return float("nan"), float("nan")
    bs = bootstrap_sharpe_ci(
        returns=rets,
        config=BootstrapConfig(n_resamples=n_resamples, seed=seed),
    )
    return bs.ci_lower_95, bs.ci_upper_95


def _run_panel_metrics(
    *, panel: pl.DataFrame, spec: ValidationSpec, cost_stress_mult: float
) -> tuple[dict[str, float], NDArray[np.float64], pl.DataFrame]:
    cfg = build_backtest_config(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        cost_stress_mult=cost_stress_mult,
        q_quantile=spec.q_quantile,
        target_gross=spec.target_gross,
        equity=spec.equity,
        cohort=spec.cohort,
    )
    res = run_backtest(signals_with_bars=panel, config=cfg, dividends=None)
    m = equity_metrics(res.daily_returns)
    rets = res.daily_returns["net_return"].to_numpy().astype(np.float64)
    return m, rets, res.daily_returns


def _classify_failures(
    *,
    spec: ValidationSpec,
    dev_m: dict[str, float],
    hd_m: dict[str, float],
    cd: CostDecomposition,
    concentration_dev: ConcentrationReport,
    sanity_random_sharpe: float,
    sanity_inverted_sharpe: float,
    pbo: float,
    dsr: float,
    delay_full: dict[int, dict[str, float]],
) -> list[FailureCategory]:
    cats: list[FailureCategory] = []
    if pbo > spec.gate_pbo_max:
        cats.append(FailureCategory.HIGH_PBO)
    if dsr < spec.gate_dsr_min:
        cats.append(FailureCategory.LOW_DSR)
    if cd.stress_2x_sharpe < spec.gate_cost_stress_min:
        cats.append(FailureCategory.COST_FAILURE)
    if dev_m.get("bootstrap_sharpe_lower_95", -1.0) < spec.gate_bootstrap_ci_lower_min:
        cats.append(FailureCategory.INSUFFICIENT_SAMPLE)
    if any(
        d.get("sharpe", -10.0) - cd.full_cost_sharpe < -0.5 for d in delay_full.values()
    ):
        cats.append(FailureCategory.DELAY_STRESS_FAIL)
    if concentration_dev.max_month_share > 0.5 or concentration_dev.months_above_50pct_share > 0:
        cats.append(FailureCategory.SINGLE_PERIOD_DOMINANCE)
    if spec.gate_must_beat_random and dev_m["sharpe"] <= sanity_random_sharpe + 0.1:
        cats.append(FailureCategory.RANDOMIZATION_FAIL)
    if spec.gate_must_beat_inverted and dev_m["sharpe"] <= sanity_inverted_sharpe:
        cats.append(FailureCategory.OVER_CORRELATED_WITH_BASELINE)
    if hd_m["sharpe"] < spec.gate_holdout_sharpe_min:
        cats.append(FailureCategory.HOLDOUT_FAILURE)
    return cats


def _exception_path_qualifies(spec: ValidationSpec) -> tuple[bool, str]:
    """Return (qualifies, reason). The exception path is restrictive and a
    failure in any precondition falls back to the default rule.
    """
    if not spec.exception_invoked:
        return False, "exception_invoked is False"
    if spec.exception_policy_ref != ACCEPTED_EXCEPTION_POLICY_REF:
        return False, (
            f"exception_policy_ref does not match the accepted policy "
            f"({spec.exception_policy_ref!r})"
        )
    if spec.declared_instrument not in TIER_1_INSTRUMENTS:
        return False, (
            f"declared_instrument {spec.declared_instrument!r} is not Tier-1 "
            f"(allowed: {sorted(TIER_1_INSTRUMENTS)})"
        )
    if not spec.single_instrument_scalar:
        return False, "single_instrument_scalar must be True for exception path"
    violation = feature_audit_violation(spec.feature_audit)
    if violation is not None:
        return False, violation
    return True, ""


def _assign_status(
    *,
    spec: ValidationSpec,
    result: StrategyValidationResult,
    pbo: float,
    dsr: float,
) -> tuple[CandidateStatus, bool]:
    # Exception-path branch — only activates when ALL preconditions match.
    # Any failure falls through to the default rule.
    exception_qualifies, _exc_reason = _exception_path_qualifies(spec)
    if exception_qualifies:
        if result.failure_classes:
            return CandidateStatus.NONE, False
        if (
            result.dev_metrics["sharpe"] >= spec.gate_dev_sharpe_min
            and result.holdout_metrics["sharpe"] >= spec.gate_holdout_sharpe_min
            and pbo <= spec.gate_pbo_max
            and dsr >= spec.gate_dsr_min
        ):
            return CandidateStatus.EXCEPTION_REVIEW_REQUIRED, False
        return CandidateStatus.RESEARCH_PASS, False

    # Default rule (unchanged from prior behavior).
    if not spec.declares_non_ohlcv_source:
        # "No promotion without new information source" rule.
        # OHLCV-only strategies can be RESEARCH_PASS but never beyond.
        if (
            result.dev_metrics["sharpe"] >= spec.gate_dev_sharpe_min
            and result.holdout_metrics["sharpe"] >= spec.gate_holdout_sharpe_min
            and not result.failure_classes
        ):
            return CandidateStatus.RESEARCH_PASS, False
        return CandidateStatus.NONE, False

    if result.failure_classes:
        return CandidateStatus.NONE, False
    if (
        result.dev_metrics["sharpe"] >= spec.gate_dev_sharpe_min
        and result.holdout_metrics["sharpe"] >= spec.gate_holdout_sharpe_min
        and pbo <= spec.gate_pbo_max
        and dsr >= spec.gate_dsr_min
    ):
        return CandidateStatus.PROMOTION_ELIGIBLE, True
    return CandidateStatus.RESEARCH_PASS, False


def validate_strategy(
    *,
    spec: ValidationSpec,
    signal_fn: SignalFn,
    bars: pl.DataFrame,
    pool_dev_returns: dict[str, NDArray[np.float64]] | None = None,
) -> PipelineReport:
    """Run the full validation pipeline on one strategy.

    Args:
        spec: ValidationSpec with hypothesis, information sources, and gates.
        signal_fn: returns (date, symbol, y_xs_pred) given bars + spec.
        bars: long-form OHLCV panel.
        pool_dev_returns: optional dev-period net-return series from OTHER
            strategies in the same validation batch. Required for PBO/DSR
            to be meaningful; a single strategy in isolation gets DSR=PSR_zero.

    Returns:
        PipelineReport with metrics, failure classes, and assigned status.
    """
    funnel = SelectionFunnel()
    funnel.record("universe_initial", int(bars["symbol"].n_unique()))

    signals = signal_fn(bars, spec)
    funnel.record("after_signal_generated", int(signals.height))

    panel = to_m4_panel(
        bars=bars,
        signals=signals,
        spread_bps=spec.spread_bps_one_way * 10.0,
        borrow_tier=spec.borrow_tier,
    )
    funnel.record("universe_used_in_m4", int(panel["symbol"].n_unique()))

    dev_panel = panel.filter(pl.col("execution_date") <= spec.dev_end)
    hd_panel = panel.filter(pl.col("execution_date") >= spec.holdout_start)

    dev_m, dev_rets, dev_daily = _run_panel_metrics(
        panel=dev_panel, spec=spec, cost_stress_mult=1.0
    )
    hd_m, hd_rets, hd_daily = _run_panel_metrics(
        panel=hd_panel, spec=spec, cost_stress_mult=1.0
    )

    if dev_rets.size > 30:
        lo, hi = _bootstrap_ci(
            dev_rets, n_resamples=spec.bootstrap_n_resamples, seed=spec.bootstrap_seed,
        )
        dev_m["bootstrap_sharpe_lower_95"] = lo
        dev_m["bootstrap_sharpe_upper_95"] = hi

    cd = cost_decomposition(
        panel=dev_panel,
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        q_quantile=spec.q_quantile,
        target_gross=spec.target_gross,
        equity=spec.equity,
        cohort=spec.cohort,
    )

    delay_dev: dict[int, dict[str, float]] = {}
    for n in spec.delay_stress_bars:
        delayed_signals = shift_signal_by_n_bars(signals, n_bars=n)
        delayed_panel = to_m4_panel(
            bars=bars,
            signals=delayed_signals,
            spread_bps=spec.spread_bps_one_way * 10.0,
            borrow_tier=spec.borrow_tier,
        ).filter(pl.col("execution_date") <= spec.dev_end)
        delay_dev[n], _, _ = _run_panel_metrics(
            panel=delayed_panel, spec=spec, cost_stress_mult=1.0
        )

    random_panel = to_m4_panel(
        bars=bars,
        signals=random_signal(bars, seed=spec.bootstrap_seed + 7777),
        spread_bps=spec.spread_bps_one_way * 10.0,
        borrow_tier=spec.borrow_tier,
    ).filter(pl.col("execution_date") <= spec.dev_end)
    sanity_random_m, _, _ = _run_panel_metrics(
        panel=random_panel, spec=spec, cost_stress_mult=1.0
    )

    inverted = inverted_signal(signals)
    inverted_panel = to_m4_panel(
        bars=bars,
        signals=inverted,
        spread_bps=spec.spread_bps_one_way * 10.0,
        borrow_tier=spec.borrow_tier,
    ).filter(pl.col("execution_date") <= spec.dev_end)
    sanity_inverted_m, _, _ = _run_panel_metrics(
        panel=inverted_panel, spec=spec, cost_stress_mult=1.0
    )

    concentration_dev = concentration_by_period(dev_daily)
    concentration_hd = concentration_by_period(hd_daily)

    # Compute PBO/DSR with whatever pool is available.
    if pool_dev_returns is None:
        pool = {spec.strategy_name: dev_rets}
    else:
        pool = {spec.strategy_name: dev_rets, **pool_dev_returns}
    names = list(pool.keys())
    series = [pool[n] for n in names]
    min_len = min(s.size for s in series)
    if len(names) >= 3 and min_len >= 64:
        dev_matrix = np.column_stack([s[-min_len:] for s in series])
        pbo_res = compute_three_tier_pbo(
            returns=dev_matrix,
            profile=np.array(["batch" for _ in names]),
            family=np.array(names),
            n_partitions=16,
        )
        pbo = pbo_res.raw_global
        sharpes_list = []
        for s in series:
            window = s[-min_len:]
            sd = float(np.std(window, ddof=1))
            mu = float(np.mean(window))
            sd_safe = sd if sd > 1e-9 else 1e-9
            sharpes_list.append(mu / sd_safe * float(np.sqrt(252.0)))
        sharpes = np.array(sharpes_list, dtype=np.float64)
        best_idx = int(np.argmax(sharpes))
        dsr_res = compute_dsr(
            returns=series[best_idx],
            sharpe_estimates=sharpes.astype(np.float64),
            selected_idx=best_idx,
        )
        dsr = float(dsr_res.dsr)
        psr_zero = float(dsr_res.psr_zero)
    else:
        pbo = float("nan")
        dsr = float("nan")
        psr_zero = float("nan")

    failure_classes = _classify_failures(
        spec=spec,
        dev_m=dev_m,
        hd_m=hd_m,
        cd=cd,
        concentration_dev=concentration_dev,
        sanity_random_sharpe=sanity_random_m["sharpe"],
        sanity_inverted_sharpe=sanity_inverted_m["sharpe"],
        pbo=pbo if not np.isnan(pbo) else 0.0,
        dsr=dsr if not np.isnan(dsr) else 0.0,
        delay_full=delay_dev,
    )

    research_pass = (
        dev_m["sharpe"] >= spec.gate_dev_sharpe_min
        and hd_m["sharpe"] >= spec.gate_holdout_sharpe_min
        and cd.stress_2x_sharpe > spec.gate_cost_stress_min
        and dev_m.get("bootstrap_sharpe_lower_95", -1.0) > spec.gate_bootstrap_ci_lower_min
        and not failure_classes
    )
    funnel.record("research_pass", 1 if research_pass else 0)
    funnel.record("failure_classes_count", len(failure_classes))

    result = StrategyValidationResult(
        name=spec.strategy_name,
        dev_metrics=dev_m,
        holdout_metrics=hd_m,
        cost_decomposition=cd,
        delay_stress=delay_dev,
        concentration_dev=concentration_dev,
        concentration_holdout=concentration_hd,
        dev_net_returns=dev_rets,
        holdout_net_returns=hd_rets,
        sanity_random_sharpe=float(sanity_random_m["sharpe"]),
        sanity_inverted_sharpe=float(sanity_inverted_m["sharpe"]),
        research_pass=research_pass,
        failure_classes=failure_classes,
        funnel=funnel,
    )
    status, promotion = _assign_status(
        spec=spec, result=result,
        pbo=pbo if not np.isnan(pbo) else 1.0,
        dsr=dsr if not np.isnan(dsr) else 0.0,
    )
    return PipelineReport(
        spec=spec,
        result=result,
        pbo_raw_global=pbo,
        dsr=dsr,
        psr_zero=psr_zero,
        assigned_status=status,
        promotion_eligible=promotion,
    )


def render_pipeline_report(report: PipelineReport, *, output_path: Path) -> Path:
    spec = report.spec
    r = report.result
    cd = r.cost_decomposition
    output_path.parent.mkdir(parents=True, exist_ok=True)
    banner = data_quality_banner(
        data_quality_label=spec.data_quality_label,
        constituent_survivorship_applicable=spec.constituent_survivorship_applicable,
    )

    delay_lines = [
        f"- {n}-bar delay dev Sharpe: {m['sharpe']:+.3f} (vs full-cost {cd.full_cost_sharpe:+.3f})"
        for n, m in r.delay_stress.items()
    ]
    failure_lines = (
        [f"- `{c.value}`" for c in r.failure_classes]
        if r.failure_classes
        else ["- (none)"]
    )
    info_sources = ", ".join(s.value for s in spec.information_sources)

    body = "\n".join([
        f"# Validation Report — `{spec.strategy_name}`",
        "",
        "## Hypothesis",
        f"> {spec.hypothesis_statement}",
        "",
        "## Information sources declared",
        f"- {info_sources}",
        f"- non-OHLCV declared: {'YES' if spec.declares_non_ohlcv_source else 'NO'}",
        "",
        "## Fixture",
        f"- universe size: {len(spec.universe_tickers)}",
        f"- history: {spec.start.isoformat()} → {spec.end.isoformat()}",
        f"- dev:     {spec.start.isoformat()} → {spec.dev_end.isoformat()}",
        f"- holdout: {spec.holdout_start.isoformat()} → {spec.end.isoformat()}",
        f"- costs:   {spec.commission_bps_one_way} bps commission + "
        f"{spec.spread_bps_one_way * 10:.1f} bps spread",
        f"- intake doc: {spec.intake_doc_ref or '(none)'}",
        f"- proposer: {spec.proposer or '(unknown)'}",
        "",
        "## Data quality banner",
        "",
        banner,
        "",
        "## Headline metrics",
        f"- dev Sharpe: {r.dev_metrics['sharpe']:+.3f}",
        f"- dev max DD: {r.dev_metrics['max_dd']*100:+.2f}%",
        f"- dev bootstrap 95% CI: "
        f"[{r.dev_metrics.get('bootstrap_sharpe_lower_95', float('nan')):+.3f}, "
        f"{r.dev_metrics.get('bootstrap_sharpe_upper_95', float('nan')):+.3f}]",
        f"- holdout Sharpe: {r.holdout_metrics['sharpe']:+.3f}",
        f"- holdout max DD: {r.holdout_metrics['max_dd']*100:+.2f}%",
        "",
        "## Cost decomposition (dev)",
        f"- no-cost:      {cd.no_cost_sharpe:+.3f}",
        f"- fee-only:     {cd.fee_only_sharpe:+.3f}",
        f"- spread-only:  {cd.spread_only_sharpe:+.3f}",
        f"- full-cost:    {cd.full_cost_sharpe:+.3f}",
        f"- stress-2×:    {cd.stress_2x_sharpe:+.3f}",
        "",
        "## Delay stress (dev)",
        *delay_lines,
        "",
        "## Sanity baselines (dev)",
        f"- random_signal Sharpe:    {r.sanity_random_sharpe:+.3f}",
        f"- inverted_signal Sharpe:  {r.sanity_inverted_sharpe:+.3f}",
        f"- strategy Sharpe vs random margin: "
        f"{r.dev_metrics['sharpe'] - r.sanity_random_sharpe:+.3f}",
        f"- strategy Sharpe vs inverted margin: "
        f"{r.dev_metrics['sharpe'] - r.sanity_inverted_sharpe:+.3f}",
        "",
        "## Concentration diagnostics",
        f"- dev max month PnL share:  {r.concentration_dev.max_month_share*100:.1f}%",
        f"- dev max year PnL share:   {r.concentration_dev.max_year_share*100:.1f}%",
        f"- dev months above 50% share: {r.concentration_dev.months_above_50pct_share}",
        f"- holdout max month PnL share: {r.concentration_holdout.max_month_share*100:.1f}%",
        "",
        "## Cross-strategy controls",
        f"- PBO raw_global: {report.pbo_raw_global:.3f}  (gate ≤ {spec.gate_pbo_max})",
        f"- DSR:            {report.dsr:.3f}  (gate ≥ {spec.gate_dsr_min})",
        f"- PSR_zero:       {report.psr_zero:.3f}",
        "",
        "## Failure classes",
        *failure_lines,
        "",
        "## Status assignment",
        f"- assigned_status: `{report.assigned_status.name_lower}`",
        f"- promotion_eligible: {'YES' if report.promotion_eligible else 'no'}",
        f"- research_pass: {'YES' if r.research_pass else 'no'}",
        "",
        "## Promotion gates (8-criteria)",
        f"- dev Sharpe ≥ {spec.gate_dev_sharpe_min}: "
        f"{'PASS' if r.dev_metrics['sharpe'] >= spec.gate_dev_sharpe_min else 'FAIL'}",
        f"- holdout Sharpe ≥ {spec.gate_holdout_sharpe_min}: "
        f"{'PASS' if r.holdout_metrics['sharpe'] >= spec.gate_holdout_sharpe_min else 'FAIL'}",
        f"- cost-stress 2× ≥ {spec.gate_cost_stress_min}: "
        f"{'PASS' if cd.stress_2x_sharpe >= spec.gate_cost_stress_min else 'FAIL'}",
        f"- bootstrap CI lower ≥ {spec.gate_bootstrap_ci_lower_min}: "
        f"{'PASS' if r.dev_metrics.get('bootstrap_sharpe_lower_95', -1) >= spec.gate_bootstrap_ci_lower_min else 'FAIL'}",
        f"- PBO ≤ {spec.gate_pbo_max}: "
        f"{'PASS' if report.pbo_raw_global <= spec.gate_pbo_max else 'FAIL'}",
        f"- DSR ≥ {spec.gate_dsr_min}: "
        f"{'PASS' if report.dsr >= spec.gate_dsr_min else 'FAIL'}",
        f"- beats random: "
        f"{'PASS' if r.dev_metrics['sharpe'] > r.sanity_random_sharpe + 0.1 else 'FAIL'}",
        f"- beats inverted: "
        f"{'PASS' if r.dev_metrics['sharpe'] > r.sanity_inverted_sharpe else 'FAIL'}",
        "",
        "## No-promotion-without-new-information-source rule",
        "",
        "OHLCV-only strategies cannot reach `promotion_eligible` status. ",
        "This strategy declares: " + info_sources,
        "→ non-OHLCV source present: "
        + ("YES" if spec.declares_non_ohlcv_source else "NO"),
        "",
        "## Disclaimer",
        "Research output only. Past performance does not guarantee future results. ",
        "No promotion to capital deployment occurs without an explicit promotion record ",
        "(spec §6.5).",
        "",
    ])
    output_path.write_text(body)
    return output_path
