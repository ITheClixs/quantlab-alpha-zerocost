"""Multi-model side-by-side backtest on a shared fixture.

Same data, same costs, same dev/holdout split, same walk-forward gates for
every model. Side-by-side comparison report.

Models:
1. raw_avellaneda_lee — primary signal alone, no meta-labeling
2. crossectional_momentum_12_1 — Jegadeesh-Titman 12-1 cross-sectional momentum
3. gkx_lightgbm — walk-forward GKX-style OHLCV LightGBM ranking
4. triple_barrier_meta_labeled_av_lee — the existing wrapper (re-run for parity)
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
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
from quant_research_stack.signal_research.backtests.triple_barrier_av_lee import (
    TBAvLeeSpec,
    run_triple_barrier_av_lee,
)
from quant_research_stack.signal_research.methodology.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_sharpe_ci,
)
from quant_research_stack.signal_research.methodology.selection_funnel import (
    SelectionFunnel,
)
from quant_research_stack.signal_research.papers.avellaneda_lee import (
    AvellanedaLeeConfig,
    AvellanedaLeeStrategy,
)


@dataclass(frozen=True)
class FixtureSpec:
    universe_tickers: list[str]
    start: dt.date
    end: dt.date
    dev_end: dt.date
    holdout_start: dt.date
    # AvL knobs (used by raw_avl and tb_meta_avl)
    pca_window: int = 252
    n_pca_components: int = 5
    z_entry: float = 1.5
    # GKX knobs
    gkx_label_horizon: int = 5
    gkx_n_estimators: int = 300
    gkx_learning_rate: float = 0.05
    gkx_num_leaves: int = 31
    gkx_seed: int = 42
    gkx_walk_forward_folds: int = 5
    gkx_walk_forward_embargo: int = 10
    # Backtest
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0
    cost_stress_multiplier: float = 2.0
    target_gross: float = 1.0
    equity: float = 1_000_000.0
    q_quantile: float = 0.20
    cohort: str = "full_universe"
    borrow_tier_default: str = "general"
    data_quality_label: str = "survivorship_prototype_only"
    constituent_survivorship_applicable: bool = True


@dataclass(frozen=True)
class ModelResult:
    name: str
    dev_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    cost_stress_metrics: dict[str, float]
    funnel: SelectionFunnel
    research_pass: bool
    dev_net_returns: NDArray[np.float64] | None = None
    holdout_net_returns: NDArray[np.float64] | None = None


def _bootstrap_ci(rets: NDArray[np.float64], *, seed: int) -> tuple[float, float]:
    if rets.size < 30:
        return float("nan"), float("nan")
    bs = bootstrap_sharpe_ci(
        returns=rets, config=BootstrapConfig(n_resamples=2000, seed=seed)
    )
    return bs.ci_lower_95, bs.ci_upper_95


def _run_backtest_phases(
    *,
    panel: pl.DataFrame,
    spec: FixtureSpec,
    seed: int,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    dev_panel = panel.filter(pl.col("execution_date") <= spec.dev_end)
    hd_panel = panel.filter(pl.col("execution_date") >= spec.holdout_start)
    cfg_normal = build_backtest_config(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        cost_stress_mult=1.0,
        q_quantile=spec.q_quantile,
        target_gross=spec.target_gross,
        equity=spec.equity,
        cohort=spec.cohort,
    )
    cfg_stress = build_backtest_config(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        cost_stress_mult=spec.cost_stress_multiplier,
        q_quantile=spec.q_quantile,
        target_gross=spec.target_gross,
        equity=spec.equity,
        cohort=spec.cohort,
    )
    dev_res = run_backtest(signals_with_bars=dev_panel, config=cfg_normal, dividends=None)
    hd_res = run_backtest(signals_with_bars=hd_panel, config=cfg_normal, dividends=None)
    stress_res = run_backtest(
        signals_with_bars=dev_panel, config=cfg_stress, dividends=None
    )
    dev_m = equity_metrics(dev_res.daily_returns)
    hd_m = equity_metrics(hd_res.daily_returns)
    cs_m = equity_metrics(stress_res.daily_returns)
    if dev_res.daily_returns.height > 30:
        lo, hi = _bootstrap_ci(
            dev_res.daily_returns["net_return"].to_numpy().astype(np.float64),
            seed=seed,
        )
        dev_m["bootstrap_sharpe_lower_95"] = lo
        dev_m["bootstrap_sharpe_upper_95"] = hi
    dev_rets = dev_res.daily_returns["net_return"].to_numpy().astype(np.float64)
    hd_rets = hd_res.daily_returns["net_return"].to_numpy().astype(np.float64)
    return dev_m, hd_m, cs_m, dev_rets, hd_rets


def _signals_raw_avellaneda_lee(
    bars: pl.DataFrame, spec: FixtureSpec
) -> pl.DataFrame:
    av = AvellanedaLeeStrategy(
        AvellanedaLeeConfig(
            pca_window=spec.pca_window,
            n_components=spec.n_pca_components,
            z_entry=spec.z_entry,
        )
    )
    preds = av.positions(bars).drop_nulls(subset=["y_xs_pred"])
    return preds.select(["date", "symbol", "y_xs_pred"])


def _signals_crossectional_momentum_12_1(
    bars: pl.DataFrame, spec: FixtureSpec
) -> pl.DataFrame:
    """12-1 cross-sectional momentum: 252-day past return minus 21-day recent return.

    Reversed-sign convention: high momentum → positive signal (long).
    """
    df = bars.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias(
            "log_ret"
        )
    )
    df = df.with_columns(
        pl.col("log_ret")
        .rolling_sum(window_size=252)
        .over("symbol")
        .alias("ret_252d"),
        pl.col("log_ret")
        .rolling_sum(window_size=21)
        .over("symbol")
        .alias("ret_21d"),
    ).with_columns((pl.col("ret_252d") - pl.col("ret_21d")).alias("mom_12_1"))
    return (
        df.drop_nulls(subset=["mom_12_1"])
        .rename({"mom_12_1": "y_xs_pred"})
        .select(["date", "symbol", "y_xs_pred"])
    )


_GKX_FEATURE_COLS: list[str] = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m_skip_1m",
    "reversal_1d",
    "reversal_5d",
    "reversal_1m",
    "realized_vol_20",
    "realized_vol_60",
    "beta_to_market_60",
    "beta_to_market_252",
    "idiosyncratic_vol_60",
    "dollar_volume_20d",
    "amihud_illiq_20",
    "max_daily_return_20",
    "drawdown_60",
    "drawdown_252",
    "volume_shock_zscore_20",
    "close_location_20",
]


def _build_gkx_features(bars: pl.DataFrame) -> pl.DataFrame:
    df = bars.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias(
            "log_ret"
        ),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
    )
    # Equal-weighted market proxy from this universe (no SPY needed)
    market = (
        df.group_by("date")
        .agg(pl.col("log_ret").mean().alias("market_ret"))
        .sort("date")
    )
    df = df.join(market, on="date", how="left")
    df = df.with_columns(
        pl.col("log_ret").rolling_sum(window_size=21).over("symbol").alias("momentum_1m"),
        pl.col("log_ret").rolling_sum(window_size=63).over("symbol").alias("momentum_3m"),
        pl.col("log_ret").rolling_sum(window_size=126).over("symbol").alias("momentum_6m"),
        (
            pl.col("log_ret").rolling_sum(window_size=252).over("symbol")
            - pl.col("log_ret").rolling_sum(window_size=21).over("symbol")
        ).alias("momentum_12m_skip_1m"),
        pl.col("log_ret").alias("reversal_1d"),
        pl.col("log_ret").rolling_sum(window_size=5).over("symbol").alias("reversal_5d"),
        pl.col("log_ret").rolling_sum(window_size=21).over("symbol").alias("reversal_1m"),
        pl.col("log_ret").rolling_std(window_size=20).over("symbol").alias("realized_vol_20"),
        pl.col("log_ret").rolling_std(window_size=60).over("symbol").alias("realized_vol_60"),
        pl.col("dollar_volume").rolling_mean(window_size=20).over("symbol").alias("dollar_volume_20d"),
        (pl.col("log_ret").abs() / pl.col("dollar_volume"))
        .rolling_mean(window_size=20)
        .over("symbol")
        .alias("amihud_illiq_20"),
        pl.col("log_ret").rolling_max(window_size=20).over("symbol").alias("max_daily_return_20"),
        (
            pl.col("close") / pl.col("close").rolling_max(window_size=60).over("symbol") - 1.0
        ).alias("drawdown_60"),
        (
            pl.col("close") / pl.col("close").rolling_max(window_size=252).over("symbol") - 1.0
        ).alias("drawdown_252"),
        (
            (pl.col("volume") - pl.col("volume").rolling_mean(window_size=20).over("symbol"))
            / pl.col("volume").rolling_std(window_size=20).over("symbol")
        ).alias("volume_shock_zscore_20"),
        (
            (pl.col("close") - pl.col("low"))
            / (pl.col("high") - pl.col("low")).clip(lower_bound=1e-9)
        ).rolling_mean(window_size=20).over("symbol").alias("close_location_20"),
    )

    # Rolling beta_to_market and idiosyncratic vol — compute via numpy per-symbol
    out_frames: list[pl.DataFrame] = []
    for sym in sorted(set(df["symbol"].to_list())):
        sym_df = df.filter(pl.col("symbol") == sym).sort("date")
        n = sym_df.height
        r = sym_df["log_ret"].to_numpy().astype(np.float64)
        m = sym_df["market_ret"].to_numpy().astype(np.float64)
        beta60 = np.full(n, np.nan)
        beta252 = np.full(n, np.nan)
        idio60 = np.full(n, np.nan)
        for t in range(60, n):
            window_r = r[t - 60 : t]
            window_m = m[t - 60 : t]
            mask = ~(np.isnan(window_r) | np.isnan(window_m))
            if mask.sum() < 30:
                continue
            var_m = float(np.var(window_m[mask], ddof=1))
            if var_m > 0:
                beta = float(
                    np.cov(window_r[mask], window_m[mask], ddof=1)[0, 1] / var_m
                )
                beta60[t] = beta
                residual = window_r[mask] - beta * window_m[mask]
                idio60[t] = float(np.std(residual, ddof=1))
        for t in range(252, n):
            window_r = r[t - 252 : t]
            window_m = m[t - 252 : t]
            mask = ~(np.isnan(window_r) | np.isnan(window_m))
            if mask.sum() < 100:
                continue
            var_m = float(np.var(window_m[mask], ddof=1))
            if var_m > 0:
                beta252[t] = float(
                    np.cov(window_r[mask], window_m[mask], ddof=1)[0, 1] / var_m
                )
        out_frames.append(
            sym_df.with_columns(
                pl.Series("beta_to_market_60", beta60),
                pl.Series("beta_to_market_252", beta252),
                pl.Series("idiosyncratic_vol_60", idio60),
            )
        )
    return pl.concat(out_frames, how="diagonal_relaxed")


def _signals_gkx_walk_forward(
    bars: pl.DataFrame, spec: FixtureSpec
) -> pl.DataFrame:
    """GKX-style LightGBM, walk-forward training within dev, final model on holdout."""
    feats = _build_gkx_features(bars)
    # Label: forward gkx_label_horizon-day cumulative log return per symbol
    horizon = spec.gkx_label_horizon
    feats = feats.sort(["symbol", "date"]).with_columns(
        pl.col("log_ret")
        .rolling_sum(window_size=horizon)
        .over("symbol")
        .shift(-horizon)
        .over("symbol")
        .alias("forward_ret"),
    )
    labeled = feats.drop_nulls(subset=[*_GKX_FEATURE_COLS, "forward_ret"])

    dev = labeled.filter(pl.col("date") <= spec.dev_end).sort(["date", "symbol"])
    dev_dates = sorted(set(dev["date"].to_list()))
    if len(dev_dates) < spec.gkx_walk_forward_folds * 60:
        raise RuntimeError(
            f"GKX dev period too short ({len(dev_dates)} days) for "
            f"{spec.gkx_walk_forward_folds} folds"
        )

    fold_size = len(dev_dates) // (spec.gkx_walk_forward_folds + 1)
    oos_predictions: list[pl.DataFrame] = []
    for k in range(spec.gkx_walk_forward_folds):
        test_start = (k + 1) * fold_size
        test_end = (
            (k + 2) * fold_size if k < spec.gkx_walk_forward_folds - 1 else len(dev_dates)
        )
        test_dates = dev_dates[test_start:test_end]
        if not test_dates:
            continue
        train_until_k = test_dates[0] - dt.timedelta(
            days=horizon + spec.gkx_walk_forward_embargo
        )
        train = dev.filter(pl.col("date") <= train_until_k)
        test = dev.filter(
            (pl.col("date") >= test_dates[0]) & (pl.col("date") <= test_dates[-1])
        )
        if train.height < 500 or test.height < 20:
            continue
        X_train = train.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
        y_train = train["forward_ret"].to_numpy().astype(np.float64)
        ds = lgb.Dataset(X_train, label=y_train)
        booster_k = lgb.train(
            params={
                "objective": "regression",
                "num_leaves": spec.gkx_num_leaves,
                "learning_rate": spec.gkx_learning_rate,
                "seed": spec.gkx_seed + k,
                "verbose": -1,
            },
            train_set=ds,
            num_boost_round=spec.gkx_n_estimators,
        )
        X_test = test.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
        preds = np.asarray(booster_k.predict(X_test), dtype=np.float64)
        oos_predictions.append(
            test.select(["date", "symbol"]).with_columns(pl.Series("y_xs_pred", preds))
        )

    if not oos_predictions:
        raise RuntimeError("GKX walk-forward produced no OOS predictions")
    dev_oos = pl.concat(oos_predictions, how="diagonal_relaxed")

    # Final booster on full dev → holdout predictions
    X_full = dev.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
    y_full = dev["forward_ret"].to_numpy().astype(np.float64)
    ds_full = lgb.Dataset(X_full, label=y_full)
    final_booster = lgb.train(
        params={
            "objective": "regression",
            "num_leaves": spec.gkx_num_leaves,
            "learning_rate": spec.gkx_learning_rate,
            "seed": spec.gkx_seed,
            "verbose": -1,
        },
        train_set=ds_full,
        num_boost_round=spec.gkx_n_estimators,
    )
    holdout = labeled.filter(pl.col("date") >= spec.holdout_start)
    X_hd = holdout.select(_GKX_FEATURE_COLS).to_numpy().astype(np.float64)
    if X_hd.shape[0] == 0:
        hd_preds_df = pl.DataFrame(
            {"date": [], "symbol": [], "y_xs_pred": []},
            schema={"date": pl.Date, "symbol": pl.Utf8, "y_xs_pred": pl.Float64},
        )
    else:
        preds_hd = np.asarray(final_booster.predict(X_hd), dtype=np.float64)
        hd_preds_df = holdout.select(["date", "symbol"]).with_columns(
            pl.Series("y_xs_pred", preds_hd)
        )

    return pl.concat([dev_oos, hd_preds_df], how="diagonal_relaxed")


def _run_single_model(
    *,
    name: str,
    signals_fn: Callable[[pl.DataFrame, FixtureSpec], pl.DataFrame],
    bars: pl.DataFrame,
    spec: FixtureSpec,
    sectors: dict[str, str] | None,
    seed: int,
) -> ModelResult:
    funnel = SelectionFunnel()
    funnel.record("universe_initial", int(bars["symbol"].n_unique()))
    signals = signals_fn(bars, spec)
    funnel.record("after_signal_generated", int(signals.height))
    panel = to_m4_panel(
        bars=bars,
        signals=signals,
        sectors=sectors,
        spread_bps=spec.spread_bps_one_way * 10.0,
        borrow_tier=spec.borrow_tier_default,
    )
    funnel.record("universe_used_in_m4", int(panel["symbol"].n_unique()))

    dev_m, hd_m, cs_m, dev_rets, hd_rets = _run_backtest_phases(
        panel=panel, spec=spec, seed=seed,
    )
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
    funnel.record("promotion_eligible", 0)
    funnel.record("paper_trade_candidate", 0)
    funnel.record("production_candidate", 0)
    return ModelResult(
        name=name,
        dev_metrics=dev_m,
        holdout_metrics=hd_m,
        cost_stress_metrics=cs_m,
        funnel=funnel,
        research_pass=research_pass,
        dev_net_returns=dev_rets,
        holdout_net_returns=hd_rets,
    )


def _tb_avl_to_fixture(spec: FixtureSpec) -> TBAvLeeSpec:
    return TBAvLeeSpec(
        universe_tickers=spec.universe_tickers,
        start=spec.start,
        end=spec.end,
        dev_end=spec.dev_end,
        holdout_start=spec.holdout_start,
        pca_window=spec.pca_window,
        n_pca_components=spec.n_pca_components,
        z_entry=spec.z_entry,
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        cost_stress_multiplier=spec.cost_stress_multiplier,
        target_gross=spec.target_gross,
        equity=spec.equity,
        q_quantile=spec.q_quantile,
        cohort=spec.cohort,
        borrow_tier_default=spec.borrow_tier_default,
        data_quality_label=spec.data_quality_label,
        constituent_survivorship_applicable=spec.constituent_survivorship_applicable,
    )


def run_all_models_on_fixture(
    *,
    bars: pl.DataFrame,
    spec: FixtureSpec,
    sectors: dict[str, str] | None = None,
) -> dict[str, ModelResult]:
    results: dict[str, ModelResult] = {}
    results["raw_avellaneda_lee"] = _run_single_model(
        name="raw_avellaneda_lee",
        signals_fn=_signals_raw_avellaneda_lee,
        bars=bars,
        spec=spec,
        sectors=sectors,
        seed=spec.gkx_seed,
    )
    results["crossectional_momentum_12_1"] = _run_single_model(
        name="crossectional_momentum_12_1",
        signals_fn=_signals_crossectional_momentum_12_1,
        bars=bars,
        spec=spec,
        sectors=sectors,
        seed=spec.gkx_seed + 1,
    )
    results["gkx_lightgbm"] = _run_single_model(
        name="gkx_lightgbm",
        signals_fn=_signals_gkx_walk_forward,
        bars=bars,
        spec=spec,
        sectors=sectors,
        seed=spec.gkx_seed + 2,
    )
    # tb_meta_av_lee: delegate to the existing walk-forward orchestrator
    tb_spec = _tb_avl_to_fixture(spec)
    tb_out = run_triple_barrier_av_lee(bars=bars, spec=tb_spec, sectors=sectors)
    tb_dev_rets = (
        tb_out.baseline_daily["dev"]["net_return"].to_numpy().astype(np.float64)
        if "dev" in tb_out.baseline_daily
        else None
    )
    tb_hd_rets = (
        tb_out.baseline_daily["holdout"]["net_return"].to_numpy().astype(np.float64)
        if "holdout" in tb_out.baseline_daily
        else None
    )
    results["triple_barrier_meta_av_lee"] = ModelResult(
        name="triple_barrier_meta_av_lee",
        dev_metrics=tb_out.dev_metrics,
        holdout_metrics=tb_out.holdout_metrics,
        cost_stress_metrics=tb_out.cost_stress_metrics,
        funnel=tb_out.funnel,
        research_pass=tb_out.funnel.to_ordered_dict().get("research_pass", 0) == 1,
        dev_net_returns=tb_dev_rets,
        holdout_net_returns=tb_hd_rets,
    )
    return results


@dataclass(frozen=True)
class CrossStrategyMetrics:
    pbo_raw_global: float
    pbo_per_profile: dict[str, float]
    pbo_per_family: dict[str, float]
    best_name: str
    best_dsr: float
    best_psr_zero: float
    n_strategies: int


def cross_strategy_metrics(
    results: dict[str, ModelResult],
) -> CrossStrategyMetrics:
    """Compute three-tier PBO + DSR across the model pool.

    Requires every ModelResult to carry dev_net_returns; raises if any are
    None (i.e. the caller didn't use _run_single_model to populate them).
    """
    from quant_research_stack.signal_research.methodology.pbo_extensions import (
        compute_three_tier_pbo,
    )
    from quant_research_stack.strategy_benchmark.dsr import compute_dsr

    names = list(results.keys())
    series: list[NDArray[np.float64]] = []
    for n in names:
        r = results[n].dev_net_returns
        if r is None:
            raise RuntimeError(f"ModelResult {n!r} has no dev_net_returns")
        series.append(r)
    min_len = min(s.size for s in series)
    dev_matrix = np.column_stack([s[-min_len:] for s in series])
    family = np.array(names)
    profile = np.array(["fixture" for _ in names])
    pbo = compute_three_tier_pbo(
        returns=dev_matrix, profile=profile, family=family, n_partitions=16
    )
    sharpes = np.array([results[n].dev_metrics["sharpe"] for n in names])
    best_idx = int(np.argmax(sharpes))
    dsr = compute_dsr(
        returns=series[best_idx],
        sharpe_estimates=sharpes.astype(np.float64),
        selected_idx=best_idx,
    )
    return CrossStrategyMetrics(
        pbo_raw_global=pbo.raw_global,
        pbo_per_profile=pbo.per_profile,
        pbo_per_family=pbo.per_family,
        best_name=names[best_idx],
        best_dsr=float(dsr.dsr),
        best_psr_zero=float(dsr.psr_zero),
        n_strategies=len(names),
    )


def render_comparison_report(
    results: dict[str, ModelResult], *, spec: FixtureSpec, output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    banner = data_quality_banner(
        data_quality_label=spec.data_quality_label,
        constituent_survivorship_applicable=spec.constituent_survivorship_applicable,
    )
    header = "| Model | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | holdout Sharpe | holdout DD | cost-2x Sharpe | research_pass |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|:---:|"
    rows = []
    for name, r in results.items():
        dev = r.dev_metrics
        hd = r.holdout_metrics
        cs = r.cost_stress_metrics
        rows.append(
            f"| `{name}` | {dev['sharpe']:+.3f} | {dev['max_dd']*100:+.2f}% | "
            f"{dev.get('bootstrap_sharpe_lower_95', float('nan')):+.3f} | "
            f"{dev.get('bootstrap_sharpe_upper_95', float('nan')):+.3f} | "
            f"{hd['sharpe']:+.3f} | {hd['max_dd']*100:+.2f}% | "
            f"{cs['sharpe']:+.3f} | "
            f"{'YES' if r.research_pass else 'no'} |"
        )

    body = "\n".join([
        "# Multi-Model Comparison — Same Fixture",
        "",
        "## Fixture",
        f"- universe: top {len(spec.universe_tickers)} SP500 by ADV",
        f"- history: {spec.start.isoformat()} → {spec.end.isoformat()}",
        f"- dev:     {spec.start.isoformat()} → {spec.dev_end.isoformat()}",
        f"- holdout: {spec.holdout_start.isoformat()} → {spec.end.isoformat()}",
        f"- costs: {spec.commission_bps_one_way} bps commission + {spec.spread_bps_one_way * 10:.1f} bps spread",
        f"- cost-stress multiplier: {spec.cost_stress_multiplier}×",
        "",
        "## Data quality banner",
        "",
        banner,
        "",
        "## Side-by-side results",
        "",
        header,
        sep,
        *rows,
        "",
        "## Promotion gates",
        "All four models tested against the same gates:",
        "- dev Sharpe ≥ 1.0",
        "- holdout Sharpe ≥ 0.5",
        "- cost-stress 2× Sharpe > 0",
        "- bootstrap 95% lower-CI Sharpe > 0",
        "",
        "## Disclaimer",
        "Research output only. Past performance does not guarantee future results. ",
        "No promotion to capital deployment occurs without an explicit promotion record ",
        "(spec §6.5 and the QuantLab promotion runbook).",
        "",
    ])
    output_path.write_text(body)
    return output_path
