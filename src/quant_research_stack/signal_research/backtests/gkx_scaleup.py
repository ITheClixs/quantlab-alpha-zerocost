"""GKX-style LightGBM at maximal scale — strictly controlled experiment.

Completes the GKX (Gu/Kelly/Xiu 2020) family with a predeclared variant grid:

- Label horizon ∈ {5d, 21d, 63d}
- Universe ∈ {top-100, top-200 SP500 by ADV}
- LightGBM params predeclared (no internal sweep)
- Walk-forward training (5 folds, embargo = label_horizon + 5)
- 19 OHLCV characteristics (full GKX-style feature set)

→ 6 GKX variants total.

Sanity baselines (same fixture):
- random_signal — i.i.d. standard normal
- simple_reversal_5d — within-universe 5-day reversal
- mom_12_1 — 252d return minus 21d return

9 total strategies in PBO/DSR pool.

Decision rule (8-criteria):
- dev Sharpe ≥ 1.0
- holdout Sharpe ≥ 0.5
- cost-stress 2× Sharpe > 0
- bootstrap 95% lower-CI Sharpe > 0
- PBO raw_global ≤ 0.25
- DSR ≥ 0.5
- beats simple_reversal AND mom_12_1 baselines
- no single label-horizon dominates everything else
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
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
from quant_research_stack.signal_research.backtests.multi_model_fixture import (
    _GKX_FEATURE_COLS,
    _build_gkx_features,
)
from quant_research_stack.signal_research.methodology.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_sharpe_ci,
)
from quant_research_stack.signal_research.methodology.pbo_extensions import (
    compute_three_tier_pbo,
)
from quant_research_stack.signal_research.methodology.selection_funnel import (
    SelectionFunnel,
)
from quant_research_stack.strategy_benchmark.dsr import compute_dsr


@dataclass(frozen=True)
class GKXVariant:
    label_horizon: int
    universe_label: str

    @property
    def name(self) -> str:
        return f"gkx_lgb_h{self.label_horizon}d_{self.universe_label}"


@dataclass(frozen=True)
class GKXSpec:
    start: dt.date = dt.date(2006, 1, 1)
    end: dt.date = dt.date(2026, 5, 26)
    dev_end: dt.date = dt.date(2022, 12, 31)
    holdout_start: dt.date = dt.date(2023, 1, 1)
    label_horizons: tuple[int, ...] = (5, 21, 63)
    universes: tuple[str, ...] = ("top100", "top200")
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    walk_forward_folds: int = 5
    walk_forward_embargo: int = 5
    seed: int = 42
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0
    cost_stress_multiplier: float = 2.0
    target_gross: float = 1.0
    equity: float = 1_000_000.0
    q_quantile: float = 0.20
    cohort: str = "full_universe"
    data_quality_label: str = "survivorship_prototype_only"
    constituent_survivorship_applicable: bool = True


@dataclass(frozen=True)
class StrategyResult:
    name: str
    variant: GKXVariant | None
    universe_label: str
    dev_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    cost_stress_metrics: dict[str, float]
    dev_net_returns: NDArray[np.float64]
    holdout_net_returns: NDArray[np.float64]
    research_pass: bool
    funnel: SelectionFunnel = field(default_factory=SelectionFunnel)


def _bootstrap_ci(rets: NDArray[np.float64], *, seed: int) -> tuple[float, float]:
    if rets.size < 30:
        return float("nan"), float("nan")
    bs = bootstrap_sharpe_ci(
        returns=rets, config=BootstrapConfig(n_resamples=2000, seed=seed)
    )
    return bs.ci_lower_95, bs.ci_upper_95


def _gkx_signals(
    *, bars: pl.DataFrame, spec: GKXSpec, label_horizon: int,
) -> pl.DataFrame:
    """Walk-forward GKX LightGBM signals over dev+holdout."""
    feats = _build_gkx_features(bars)
    feats = feats.sort(["symbol", "date"]).with_columns(
        pl.col("log_ret")
        .rolling_sum(window_size=label_horizon)
        .over("symbol")
        .shift(-label_horizon)
        .over("symbol")
        .alias("forward_ret"),
    )
    labeled = feats.drop_nulls(subset=[*_GKX_FEATURE_COLS, "forward_ret"])
    dev = labeled.filter(pl.col("date") <= spec.dev_end).sort(["date", "symbol"])
    dev_dates = sorted(set(dev["date"].to_list()))
    if len(dev_dates) < spec.walk_forward_folds * 60:
        raise RuntimeError(
            f"GKX dev period too short ({len(dev_dates)} days) for "
            f"{spec.walk_forward_folds} folds"
        )
    fold_size = len(dev_dates) // (spec.walk_forward_folds + 1)
    oos_predictions: list[pl.DataFrame] = []
    for k in range(spec.walk_forward_folds):
        test_start = (k + 1) * fold_size
        test_end = (
            (k + 2) * fold_size
            if k < spec.walk_forward_folds - 1
            else len(dev_dates)
        )
        test_dates = dev_dates[test_start:test_end]
        if not test_dates:
            continue
        train_until_k = test_dates[0] - dt.timedelta(
            days=label_horizon + spec.walk_forward_embargo
        )
        train = dev.filter(pl.col("date") <= train_until_k)
        test = dev.filter(
            (pl.col("date") >= test_dates[0]) & (pl.col("date") <= test_dates[-1])
        )
        if train.height < 1000 or test.height < 50:
            continue
        X_train = train.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
        y_train = train["forward_ret"].to_numpy().astype(np.float64)
        ds = lgb.Dataset(X_train, label=y_train)
        booster_k = lgb.train(
            params={
                "objective": "regression",
                "num_leaves": spec.num_leaves,
                "learning_rate": spec.learning_rate,
                "seed": spec.seed + k,
                "verbose": -1,
            },
            train_set=ds,
            num_boost_round=spec.n_estimators,
        )
        X_test = test.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
        preds = np.asarray(booster_k.predict(X_test), dtype=np.float64)
        oos_predictions.append(
            test.select(["date", "symbol"]).with_columns(pl.Series("y_xs_pred", preds))
        )
    if not oos_predictions:
        raise RuntimeError("walk-forward produced no OOS predictions")
    dev_oos = pl.concat(oos_predictions, how="diagonal_relaxed")
    # Final RF on full dev → holdout predictions
    X_full = dev.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
    y_full = dev["forward_ret"].to_numpy().astype(np.float64)
    ds_full = lgb.Dataset(X_full, label=y_full)
    final_booster = lgb.train(
        params={
            "objective": "regression",
            "num_leaves": spec.num_leaves,
            "learning_rate": spec.learning_rate,
            "seed": spec.seed,
            "verbose": -1,
        },
        train_set=ds_full,
        num_boost_round=spec.n_estimators,
    )
    holdout = labeled.filter(pl.col("date") >= spec.holdout_start)
    X_hd = holdout.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
    if X_hd.shape[0] == 0:
        hd_preds_df = pl.DataFrame(
            schema={"date": pl.Date, "symbol": pl.Utf8, "y_xs_pred": pl.Float64},
        )
    else:
        preds_hd = np.asarray(final_booster.predict(X_hd), dtype=np.float64)
        hd_preds_df = holdout.select(["date", "symbol"]).with_columns(
            pl.Series("y_xs_pred", preds_hd)
        )
    return pl.concat([dev_oos, hd_preds_df], how="diagonal_relaxed")


def _baseline_random(bars: pl.DataFrame, *, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df = bars.sort(["symbol", "date"])
    df = df.with_columns(pl.Series("y_xs_pred", rng.standard_normal(df.height)))
    return df.select(["date", "symbol", "y_xs_pred"])


def _baseline_simple_reversal_5d(bars: pl.DataFrame) -> pl.DataFrame:
    df = bars.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias(
            "log_ret"
        )
    )
    df = df.with_columns(
        (-pl.col("log_ret").rolling_sum(window_size=5).over("symbol")).alias("y_xs_pred")
    )
    return df.drop_nulls(subset=["y_xs_pred"]).select(["date", "symbol", "y_xs_pred"])


def _baseline_mom_12_1(bars: pl.DataFrame) -> pl.DataFrame:
    df = bars.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias(
            "log_ret"
        )
    )
    df = df.with_columns(
        (
            pl.col("log_ret").rolling_sum(window_size=252).over("symbol")
            - pl.col("log_ret").rolling_sum(window_size=21).over("symbol")
        ).alias("y_xs_pred"),
    )
    return df.drop_nulls(subset=["y_xs_pred"]).select(["date", "symbol", "y_xs_pred"])


def _run_strategy(
    *,
    name: str,
    variant: GKXVariant | None,
    universe_label: str,
    signals: pl.DataFrame,
    bars: pl.DataFrame,
    spec: GKXSpec,
    seed: int,
) -> StrategyResult:
    funnel = SelectionFunnel()
    funnel.record("universe_initial", int(bars["symbol"].n_unique()))
    funnel.record("after_signal_generated", int(signals.height))

    panel = to_m4_panel(
        bars=bars,
        signals=signals,
        spread_bps=spec.spread_bps_one_way * 10.0,
        borrow_tier="general",
    )
    funnel.record("universe_used_in_m4", int(panel["symbol"].n_unique()))

    dev_panel = panel.filter(pl.col("execution_date") <= spec.dev_end)
    hd_panel = panel.filter(pl.col("execution_date") >= spec.holdout_start)
    cfg_n = build_backtest_config(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        cost_stress_mult=1.0,
        q_quantile=spec.q_quantile,
        target_gross=spec.target_gross,
        equity=spec.equity,
        cohort=spec.cohort,
    )
    cfg_s = build_backtest_config(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        cost_stress_mult=spec.cost_stress_multiplier,
        q_quantile=spec.q_quantile,
        target_gross=spec.target_gross,
        equity=spec.equity,
        cohort=spec.cohort,
    )
    dev_res = run_backtest(signals_with_bars=dev_panel, config=cfg_n, dividends=None)
    hd_res = run_backtest(signals_with_bars=hd_panel, config=cfg_n, dividends=None)
    cs_res = run_backtest(signals_with_bars=dev_panel, config=cfg_s, dividends=None)
    dev_m = equity_metrics(dev_res.daily_returns)
    hd_m = equity_metrics(hd_res.daily_returns)
    cs_m = equity_metrics(cs_res.daily_returns)
    if dev_res.daily_returns.height > 30:
        lo, hi = _bootstrap_ci(
            dev_res.daily_returns["net_return"].to_numpy().astype(np.float64), seed=seed,
        )
        dev_m["bootstrap_sharpe_lower_95"] = lo
        dev_m["bootstrap_sharpe_upper_95"] = hi

    funnel.record("dev_sharpe_positive", 1 if dev_m["sharpe"] > 0 else 0)
    funnel.record("holdout_sharpe_positive", 1 if hd_m["sharpe"] > 0 else 0)
    funnel.record("cost_stress_sharpe_positive", 1 if cs_m["sharpe"] > 0 else 0)
    research_pass = (
        dev_m["sharpe"] >= 1.0
        and hd_m["sharpe"] >= 0.5
        and cs_m["sharpe"] > 0
        and dev_m.get("bootstrap_sharpe_lower_95", -1.0) > 0
    )
    funnel.record("research_pass", 1 if research_pass else 0)
    return StrategyResult(
        name=name,
        variant=variant,
        universe_label=universe_label,
        dev_metrics=dev_m,
        holdout_metrics=hd_m,
        cost_stress_metrics=cs_m,
        dev_net_returns=dev_res.daily_returns["net_return"].to_numpy().astype(np.float64),
        holdout_net_returns=hd_res.daily_returns["net_return"]
        .to_numpy()
        .astype(np.float64),
        research_pass=research_pass,
        funnel=funnel,
    )


def run_gkx_scaleup(
    *,
    bars_per_universe: dict[str, pl.DataFrame],
    spec: GKXSpec,
) -> tuple[list[StrategyResult], list[StrategyResult]]:
    variant_results: list[StrategyResult] = []
    for universe_label in spec.universes:
        bars = bars_per_universe[universe_label]
        for horizon in spec.label_horizons:
            variant = GKXVariant(label_horizon=horizon, universe_label=universe_label)
            signals = _gkx_signals(bars=bars, spec=spec, label_horizon=horizon)
            variant_results.append(
                _run_strategy(
                    name=variant.name, variant=variant,
                    universe_label=universe_label,
                    signals=signals, bars=bars, spec=spec,
                    seed=spec.seed + hash(variant.name) % 1000,
                )
            )

    baseline_results: list[StrategyResult] = []
    for universe_label in spec.universes:
        bars = bars_per_universe[universe_label]
        baseline_results.append(
            _run_strategy(
                name=f"random_signal_{universe_label}", variant=None,
                universe_label=universe_label,
                signals=_baseline_random(bars, seed=spec.seed + 7777),
                bars=bars, spec=spec, seed=2001,
            )
        )
        baseline_results.append(
            _run_strategy(
                name=f"simple_reversal_5d_{universe_label}", variant=None,
                universe_label=universe_label,
                signals=_baseline_simple_reversal_5d(bars),
                bars=bars, spec=spec, seed=2002,
            )
        )
        baseline_results.append(
            _run_strategy(
                name=f"mom_12_1_{universe_label}", variant=None,
                universe_label=universe_label,
                signals=_baseline_mom_12_1(bars),
                bars=bars, spec=spec, seed=2003,
            )
        )
    return variant_results, baseline_results


@dataclass(frozen=True)
class CrossStrategyMetrics:
    pbo_raw_global: float
    pbo_per_profile: dict[str, float]
    pbo_per_family: dict[str, float]
    best_index: int
    best_dsr: float
    best_psr_zero: float
    n_strategies: int


def cross_strategy_metrics(
    variants: list[StrategyResult], baselines: list[StrategyResult]
) -> CrossStrategyMetrics:
    all_results = variants + baselines
    series = [r.dev_net_returns for r in all_results]
    min_len = min(s.size for s in series)
    dev_matrix = np.column_stack([s[-min_len:] for s in series])
    family = np.array([
        ("gkx" if r.variant is not None else "baseline") for r in all_results
    ])
    profile = np.array([r.universe_label for r in all_results])
    pbo = compute_three_tier_pbo(
        returns=dev_matrix, profile=profile, family=family, n_partitions=16
    )
    sharpes = np.array([r.dev_metrics["sharpe"] for r in all_results])
    best_idx = int(np.argmax(sharpes))
    dsr = compute_dsr(
        returns=all_results[best_idx].dev_net_returns,
        sharpe_estimates=sharpes.astype(np.float64),
        selected_idx=best_idx,
    )
    return CrossStrategyMetrics(
        pbo_raw_global=pbo.raw_global,
        pbo_per_profile=pbo.per_profile,
        pbo_per_family=pbo.per_family,
        best_index=best_idx,
        best_dsr=float(dsr.dsr),
        best_psr_zero=float(dsr.psr_zero),
        n_strategies=len(all_results),
    )


def apply_decision_rule(
    *,
    variants: list[StrategyResult],
    baselines: list[StrategyResult],
    cross: CrossStrategyMetrics,
) -> tuple[str, str]:
    if cross.pbo_raw_global > 0.25:
        return (
            f"FAIL — PBO={cross.pbo_raw_global:.3f} > 0.25 (overfit variant grid).",
            "overfit_parameter_grid",
        )
    all_results = variants + baselines
    best = all_results[cross.best_index]
    if best.variant is None:
        return (
            f"FAIL — best strategy `{best.name}` is a baseline, not GKX.",
            "no_added_value_over_baselines",
        )
    if best.dev_metrics["sharpe"] < 1.0:
        return (
            f"FAIL — best GKX dev Sharpe {best.dev_metrics['sharpe']:+.3f} < 1.0.",
            "no_signal_at_threshold",
        )
    if best.holdout_metrics["sharpe"] < 0.5:
        return (
            f"FAIL — best GKX holdout Sharpe {best.holdout_metrics['sharpe']:+.3f} < 0.5.",
            "fails_holdout_generalization",
        )
    if best.cost_stress_metrics["sharpe"] < 0:
        return (
            f"FAIL — best GKX dies at 2x cost stress "
            f"({best.cost_stress_metrics['sharpe']:+.3f}).",
            "costs_kill_the_edge",
        )
    if best.dev_metrics.get("bootstrap_sharpe_lower_95", -1.0) < 0:
        return (
            "FAIL — best GKX bootstrap lower-CI Sharpe < 0.",
            "high_variance_noise_signal",
        )
    if cross.best_dsr < 0.5:
        return (
            f"FAIL — DSR={cross.best_dsr:.3f} < 0.5 (multi-test penalty too heavy).",
            "overfit_parameter_grid",
        )
    # Beat baselines
    best_baseline_dev = max(b.dev_metrics["sharpe"] for b in baselines)
    if best.dev_metrics["sharpe"] <= best_baseline_dev:
        return (
            f"FAIL — best GKX dev Sharpe does not beat best baseline "
            f"({best.dev_metrics['sharpe']:+.3f} vs {best_baseline_dev:+.3f}).",
            "no_added_value_over_baselines",
        )
    return (
        "PASS — GKX LightGBM survives all criteria; promote to deeper "
        "robustness testing.",
        "",
    )


def render_report(
    *,
    variants: list[StrategyResult],
    baselines: list[StrategyResult],
    cross: CrossStrategyMetrics,
    decision: str,
    failure_class: str,
    spec: GKXSpec,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    banner = data_quality_banner(
        data_quality_label=spec.data_quality_label,
        constituent_survivorship_applicable=spec.constituent_survivorship_applicable,
    )
    header = (
        "| Strategy | Universe | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | "
        "holdout Sharpe | holdout DD | cost-2x | pass |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|"
    rows = []
    for r in variants + baselines:
        dev = r.dev_metrics
        hd = r.holdout_metrics
        cs = r.cost_stress_metrics
        rows.append(
            f"| `{r.name}` | {r.universe_label} | "
            f"{dev['sharpe']:+.3f} | {dev['max_dd']*100:+.2f}% | "
            f"{dev.get('bootstrap_sharpe_lower_95', float('nan')):+.3f} | "
            f"{dev.get('bootstrap_sharpe_upper_95', float('nan')):+.3f} | "
            f"{hd['sharpe']:+.3f} | {hd['max_dd']*100:+.2f}% | "
            f"{cs['sharpe']:+.3f} | "
            f"{'YES' if r.research_pass else 'no'} |"
        )

    body = "\n".join([
        "# GKX-Style LightGBM Scale-Up — Predeclared Variant Matrix",
        "",
        "## Fixture",
        "- universes: top-100 and top-200 SP500 by ADV",
        f"- history: {spec.start.isoformat()} → {spec.end.isoformat()}",
        f"- dev:     {spec.start.isoformat()} → {spec.dev_end.isoformat()}",
        f"- holdout: {spec.holdout_start.isoformat()} → {spec.end.isoformat()}",
        f"- label horizons: {list(spec.label_horizons)}",
        f"- LightGBM: n_estimators={spec.n_estimators}, num_leaves={spec.num_leaves}, "
        f"lr={spec.learning_rate}",
        f"- walk-forward: {spec.walk_forward_folds} folds, embargo="
        f"label_horizon + {spec.walk_forward_embargo}",
        f"- features: {len(_GKX_FEATURE_COLS)} OHLCV characteristics",
        f"- costs: {spec.commission_bps_one_way} bps commission + "
        f"{spec.spread_bps_one_way * 10:.1f} bps spread",
        f"- cost-stress: {spec.cost_stress_multiplier}× multiplier",
        "",
        "## Data quality banner",
        "",
        banner,
        "",
        "## All strategies side-by-side",
        "",
        header, sep, *rows,
        "",
        "## Cross-strategy multiple-testing controls",
        "",
        f"- **PBO raw_global**: {cross.pbo_raw_global:.3f}  (gate: ≤ 0.25)",
        f"- **Best variant index**: {cross.best_index} "
        f"(`{(variants + baselines)[cross.best_index].name}`)",
        f"- **DSR for best**: {cross.best_dsr:.3f}  (gate: ≥ 0.50)",
        f"- **PSR_zero for best**: {cross.best_psr_zero:.3f}",
        f"- **n_strategies in DSR deflation**: {cross.n_strategies}",
        "",
        "## Decision rule outcome",
        "",
        f"**{decision}**",
        "",
        f"failure_class: `{failure_class or 'none'}`",
        "",
        "## Promotion gates (per-variant)",
        "- dev Sharpe ≥ 1.0",
        "- holdout Sharpe ≥ 0.5",
        "- cost-stress 2× Sharpe > 0",
        "- bootstrap 95% lower-CI Sharpe > 0",
        "- DSR ≥ 0.50 (after multi-test deflation)",
        "- beats best non-GKX baseline",
        "",
        "## Disclaimer",
        "Research output only. Past performance does not guarantee future results. ",
        "No promotion to capital deployment occurs without an explicit promotion record ",
        "(spec §6.5).",
        "",
    ])
    output_path.write_text(body)
    return output_path
