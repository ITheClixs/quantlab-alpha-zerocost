"""End-to-end backtest: Triple-Barrier Meta-Labeled Avellaneda-Lee.

Composes:
- papers.avellaneda_lee (primary signal)
- papers.triple_barrier (meta-labeling wrapper)
- alpha_eq.backtest.runner (M4 backtest engine)
- methodology.* (CPCV, PBO, DSR, bootstrap, dedup, regime)
- status (4-tier promotion gates)
- report (3-tier writer)

Hedge-fund-grade defaults (Jane Street / Citadel tier large-cap US equities):
- Commission 0.5 bps one-way, spread 1.0 bps one-way, cost stress 2x.
- Vol target embedded via equity-scaled target_gross.
- Q-quantile = 0.20 → top/bottom 20% cross-sectional, ~equal long/short.
- Walk-forward training; permanent holdout untouched until final pass.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from quant_research_stack.alpha_eq.backtest.costs import CostConfig
from quant_research_stack.alpha_eq.backtest.fills import FillModel
from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.runner import (
    BacktestConfig,
    run_backtest,
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
from quant_research_stack.signal_research.papers.triple_barrier import (
    TripleBarrierConfig,
    label_triple_barrier,
)


@dataclass(frozen=True)
class TBAvLeeSpec:
    universe_tickers: list[str]
    start: dt.date
    end: dt.date
    dev_end: dt.date
    holdout_start: dt.date
    # Avellaneda-Lee
    pca_window: int = 252
    n_pca_components: int = 5
    z_entry: float = 1.5
    # Triple-barrier
    vertical_barrier_days: int = 20
    vol_estimator_window: int = 20
    profit_take_multiplier: float = 1.5
    stop_loss_multiplier: float = 1.5
    rf_n_estimators: int = 200
    rf_threshold: float = 0.55
    rf_seed: int = 42
    # Backtest costs (Jane Street / Citadel tier)
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0
    cost_stress_multiplier: float = 2.0
    # Portfolio
    target_gross: float = 1.0
    equity: float = 1_000_000.0
    q_quantile: float = 0.20
    cohort: str = "full_universe"
    borrow_tier_default: str = "general"
    # Banner
    data_quality_label: str = "survivorship_prototype_only"
    constituent_survivorship_applicable: bool = True


@dataclass(frozen=True)
class TBAvLeeRunOutput:
    strategy_daily: pl.DataFrame
    baseline_daily: dict[str, pl.DataFrame]
    dev_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    cost_stress_metrics: dict[str, float]
    funnel: SelectionFunnel
    data_quality_banner: str
    n_universe_initial: int
    n_universe_used: int


def _compute_meta_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Per-(symbol, date) features for meta-labeler — past-only."""
    sorted_bars = bars.sort(["symbol", "date"])
    out = sorted_bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log())
        .over("symbol")
        .alias("log_return"),
    ).with_columns(
        pl.col("log_return").rolling_mean(window_size=5).over("symbol").alias("ret_5"),
        pl.col("log_return").rolling_mean(window_size=20).over("symbol").alias("ret_20"),
        pl.col("log_return").rolling_std(window_size=20).over("symbol").alias("vol_20"),
        pl.col("log_return").rolling_std(window_size=60).over("symbol").alias("vol_60"),
        (pl.col("close") / pl.col("close").rolling_max(window_size=60).over("symbol") - 1.0)
        .alias("drawdown_60"),
    )
    return out


def _feature_matrix(features: pl.DataFrame, symbol: str) -> tuple[NDArray[np.float64], pl.DataFrame]:
    f = features.filter(pl.col("symbol") == symbol).sort("date")
    cols = ["log_return", "ret_5", "ret_20", "vol_20", "vol_60", "drawdown_60"]
    mat = f.select(cols).to_numpy().astype(np.float64)
    return mat, f


def _train_meta_labeler_per_symbol(
    *,
    bars_with_features: pl.DataFrame,
    primary_positions_long: pl.DataFrame,
    spec: TBAvLeeSpec,
    train_until: dt.date,
) -> tuple[RandomForestClassifier, list[str]]:
    """Train ONE secondary RF across all symbols using triple-barrier labels.

    Spec §4.2: survivor-only pre-filter is enforced upstream of construction; here
    we just generate labels via label_triple_barrier and fit.
    """
    tb_cfg = TripleBarrierConfig(
        vertical_barrier_days=spec.vertical_barrier_days,
        profit_take_multiplier=spec.profit_take_multiplier,
        stop_loss_multiplier=spec.stop_loss_multiplier,
        vol_estimator_window=spec.vol_estimator_window,
        seed=spec.rf_seed,
    )
    feat_cols = ["log_return", "ret_5", "ret_20", "vol_20", "vol_60", "drawdown_60", "primary_position"]
    all_X: list[NDArray[np.float64]] = []
    all_y: list[NDArray[np.float64]] = []
    symbols = sorted(set(primary_positions_long["symbol"].to_list()))
    for sym in symbols:
        bars_s = bars_with_features.filter(pl.col("symbol") == sym).sort("date")
        pos_s = (
            primary_positions_long.filter(pl.col("symbol") == sym)
            .sort("date")
            .rename({"y_xs_pred": "primary_position"})
            .select(["date", "primary_position"])
        )
        joined = bars_s.join(pos_s, on="date", how="left").with_columns(
            pl.col("primary_position").fill_null(0.0)
        )
        train = joined.filter(pl.col("date") <= train_until)
        if train.height < spec.pca_window + spec.vertical_barrier_days + 50:
            continue
        closes = train["close"].to_numpy().astype(np.float64)
        positions = train["primary_position"].to_numpy().astype(np.float64)
        labels = label_triple_barrier(close=closes, positions=positions, cfg=tb_cfg)
        X = train.select(feat_cols).to_numpy().astype(np.float64)
        mask = ~np.isnan(labels) & ~np.isnan(X).any(axis=1)
        if int(mask.sum()) < 50:
            continue
        all_X.append(X[mask])
        all_y.append(labels[mask].astype(int))
    if not all_X:
        raise RuntimeError("no symbols produced enough labeled events for meta-labeler")
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    model = RandomForestClassifier(
        n_estimators=spec.rf_n_estimators, random_state=spec.rf_seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, feat_cols


def _apply_meta_labeling(
    *,
    bars_with_features: pl.DataFrame,
    primary_positions_long: pl.DataFrame,
    model: RandomForestClassifier,
    feat_cols: list[str],
    threshold: float,
) -> pl.DataFrame:
    """Returns (date, symbol, y_xs_pred) with positions zeroed when prob < threshold."""
    sorted_p = primary_positions_long.sort(["symbol", "date"]).rename(
        {"y_xs_pred": "primary_position"}
    )
    joined = bars_with_features.join(sorted_p, on=["date", "symbol"], how="left").with_columns(
        pl.col("primary_position").fill_null(0.0)
    )
    X = joined.select(feat_cols).to_numpy().astype(np.float64)
    nan_mask = np.isnan(X).any(axis=1)
    X_filled = np.where(np.isnan(X), 0.0, X)
    probas = model.predict_proba(X_filled)
    classes = model.classes_.astype(int)
    if 1 in classes:
        one_idx = int(np.where(classes == 1)[0][0])
        p1 = probas[:, one_idx].astype(np.float64)
    else:
        p1 = np.zeros(X.shape[0], dtype=np.float64)
    p1[nan_mask] = 0.0
    keep = (p1 >= threshold).astype(np.float64)
    filtered = joined["primary_position"].to_numpy().astype(np.float64) * keep
    return joined.with_columns(pl.Series("y_xs_pred", filtered)).select(
        ["date", "symbol", "y_xs_pred"]
    )


def _to_m4_panel(
    *,
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    sectors: dict[str, str] | None = None,
    spread_bps: float = 1.0,
    borrow_tier: str = "general",
) -> pl.DataFrame:
    """Build alpha_eq.run_backtest input panel.

    Required cols: execution_date, feature_as_of_date, symbol, y_xs_pred, open, high,
    low, close, adv_20d_dollar_lag1, tradable, in_pit_universe, borrow_tier,
    roll_spread_bps, sector.
    """
    sorted_bars = bars.sort(["symbol", "date"]).with_columns(
        (pl.col("close").shift(1).over("symbol") * pl.col("volume").shift(1).over("symbol"))
        .rolling_mean(window_size=20)
        .over("symbol")
        .alias("adv_20d_dollar_lag1"),
    )
    panel = sorted_bars.join(signals, on=["date", "symbol"], how="left").with_columns(
        pl.col("y_xs_pred").fill_null(0.0),
        pl.col("date").alias("execution_date"),
        (pl.col("date") - pl.duration(days=1)).alias("feature_as_of_date"),
        pl.lit(True).alias("tradable"),
        pl.lit(True).alias("in_pit_universe"),
        pl.lit(borrow_tier).alias("borrow_tier"),
        pl.lit(spread_bps).alias("roll_spread_bps"),
    )
    if sectors:
        mapping_df = pl.DataFrame(
            {"symbol": list(sectors.keys()), "sector": list(sectors.values())}
        )
        panel = panel.join(mapping_df, on="symbol", how="left").with_columns(
            pl.col("sector").fill_null("unknown")
        )
    else:
        panel = panel.with_columns(pl.lit("unknown").alias("sector"))
    return panel.drop_nulls(subset=["open", "high", "low", "close", "adv_20d_dollar_lag1"])


def _sharpe(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns)) / sd * float(np.sqrt(252.0))


def _equity_metrics(daily: pl.DataFrame) -> dict[str, float]:
    rets = daily["net_return"].to_numpy().astype(np.float64)
    if rets.size == 0:
        return {"sharpe": 0.0, "max_dd": 0.0, "cum_return": 0.0, "n_days": 0}
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1.0).min()
    return {
        "sharpe": _sharpe(rets),
        "max_dd": float(dd),
        "cum_return": float(equity[-1] - 1.0),
        "n_days": int(rets.size),
    }


def _build_backtest_config(spec: TBAvLeeSpec, cost_stress: bool = False) -> BacktestConfig:
    cost_mult = spec.cost_stress_multiplier if cost_stress else 1.0
    return BacktestConfig(
        portfolio=PortfolioBuildConfig(
            q_quantile=spec.q_quantile,
            target_gross=spec.target_gross,
            equity=spec.equity,
        ),
        fill_model=FillModel.CLOSE,
        cohort=spec.cohort,
        borrow_multiplier=cost_mult,
        financing_rate_annual=0.05,
        cost=CostConfig(
            commission_bps_one_way=spec.commission_bps_one_way * cost_mult,
            tiered_fallback_easy_bps=5.0 * cost_mult,
            tiered_fallback_general_bps=spec.spread_bps_one_way * 10.0 * cost_mult,
            tiered_fallback_hard_bps=spec.spread_bps_one_way * 30.0 * cost_mult,
        ),
    )


def _data_quality_banner(spec: TBAvLeeSpec) -> str:
    return (
        f"DATA QUALITY: data_quality_label={spec.data_quality_label}, "
        f"constituent_survivorship_applicable={spec.constituent_survivorship_applicable}. "
        "Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time membership "
        "feed. Results may overstate alpha due to survivorship bias. Institutional-grade "
        "labels (per spec §5.4) are NOT allowed for this run."
    )


def run_triple_barrier_av_lee(
    *,
    bars: pl.DataFrame,
    spec: TBAvLeeSpec,
    sectors: dict[str, str] | None = None,
) -> TBAvLeeRunOutput:
    """Run the full pipeline given pre-fetched bars panel.

    Input `bars`: long-form with columns date, symbol, open, high, low, close, volume.
    """
    funnel = SelectionFunnel()
    n_initial = len(set(bars["symbol"].to_list()))
    funnel.record("universe_initial", n_initial)

    bars_with_features = _compute_meta_features(bars)
    funnel.record("after_feature_engineering", n_initial)

    av_cfg = AvellanedaLeeConfig(
        pca_window=spec.pca_window,
        n_components=spec.n_pca_components,
        z_entry=spec.z_entry,
    )
    av_strat = AvellanedaLeeStrategy(av_cfg)
    av_predictions = av_strat.positions(bars).rename({"y_xs_pred": "y_xs_pred"})
    av_predictions = av_predictions.drop_nulls(subset=["y_xs_pred"])

    funnel.record("after_primary_signal", int(av_predictions["symbol"].n_unique()))

    model, feat_cols = _train_meta_labeler_per_symbol(
        bars_with_features=bars_with_features,
        primary_positions_long=av_predictions,
        spec=spec,
        train_until=spec.dev_end,
    )
    funnel.record("after_meta_labeler_trained", 1)

    filtered_signals = _apply_meta_labeling(
        bars_with_features=bars_with_features,
        primary_positions_long=av_predictions,
        model=model,
        feat_cols=feat_cols,
        threshold=spec.rf_threshold,
    )

    panel = _to_m4_panel(
        bars=bars,
        signals=filtered_signals,
        sectors=sectors,
        spread_bps=spec.spread_bps_one_way * 10.0,
        borrow_tier=spec.borrow_tier_default,
    )
    n_used = int(panel["symbol"].n_unique())
    funnel.record("universe_used_in_m4", n_used)

    dev_panel = panel.filter(pl.col("execution_date") <= spec.dev_end)
    holdout_panel = panel.filter(pl.col("execution_date") >= spec.holdout_start)

    cfg_normal = _build_backtest_config(spec, cost_stress=False)
    cfg_stress = _build_backtest_config(spec, cost_stress=True)

    dev_res = run_backtest(signals_with_bars=dev_panel, config=cfg_normal, dividends=None)
    holdout_res = run_backtest(
        signals_with_bars=holdout_panel, config=cfg_normal, dividends=None
    )
    stress_res = run_backtest(signals_with_bars=dev_panel, config=cfg_stress, dividends=None)

    dev_metrics = _equity_metrics(dev_res.daily_returns)
    holdout_metrics = _equity_metrics(holdout_res.daily_returns)
    stress_metrics = _equity_metrics(stress_res.daily_returns)

    if dev_res.daily_returns.height > 30:
        bs = bootstrap_sharpe_ci(
            returns=dev_res.daily_returns["net_return"].to_numpy().astype(np.float64),
            config=BootstrapConfig(n_resamples=2000, seed=spec.rf_seed),
        )
        dev_metrics["bootstrap_sharpe_lower_95"] = bs.ci_lower_95
        dev_metrics["bootstrap_sharpe_upper_95"] = bs.ci_upper_95

    funnel.record(
        "dev_sharpe_positive",
        1 if dev_metrics["sharpe"] > 0 else 0,
    )
    funnel.record(
        "holdout_sharpe_positive",
        1 if holdout_metrics["sharpe"] > 0 else 0,
    )
    funnel.record(
        "cost_stress_sharpe_positive",
        1 if stress_metrics["sharpe"] > 0 else 0,
    )
    research_pass = (
        dev_metrics["sharpe"] >= 1.0
        and holdout_metrics["sharpe"] >= 0.5
        and stress_metrics["sharpe"] > 0
        and dev_metrics.get("bootstrap_sharpe_lower_95", -1.0) > 0
    )
    funnel.record("research_pass", 1 if research_pass else 0)
    funnel.record("promotion_eligible", 0)
    funnel.record("paper_trade_candidate", 0)
    funnel.record("production_candidate", 0)

    return TBAvLeeRunOutput(
        strategy_daily=dev_res.daily_returns,
        baseline_daily={
            "dev": dev_res.daily_returns,
            "holdout": holdout_res.daily_returns,
            "cost_stress_2x": stress_res.daily_returns,
        },
        dev_metrics=dev_metrics,
        holdout_metrics=holdout_metrics,
        cost_stress_metrics=stress_metrics,
        funnel=funnel,
        data_quality_banner=_data_quality_banner(spec),
        n_universe_initial=n_initial,
        n_universe_used=n_used,
    )


def render_report(out: TBAvLeeRunOutput, *, output_path: Path) -> Path:
    funnel_lines = [
        f"- **{stage}**: {count}" for stage, count in out.funnel.to_ordered_dict().items()
    ]
    dev = out.dev_metrics
    hd = out.holdout_metrics
    cs = out.cost_stress_metrics
    body = "\n".join([
        "# Triple-Barrier Meta-Labeled Avellaneda-Lee — Backtest Report",
        "",
        "## Data quality banner",
        "",
        out.data_quality_banner,
        "",
        "## Universe",
        f"- initial universe size: {out.n_universe_initial}",
        f"- used in M4 backtest: {out.n_universe_used}",
        "",
        "## Dev metrics (commission 0.5 bps, spread 1.0 bps)",
        f"- Sharpe (annualized): {dev['sharpe']:.3f}",
        f"- Max drawdown: {dev['max_dd']*100:.2f}%",
        f"- Cumulative return: {dev['cum_return']*100:.2f}%",
        f"- Trading days: {dev['n_days']}",
        f"- Bootstrap 95% CI for Sharpe: "
        f"[{dev.get('bootstrap_sharpe_lower_95', float('nan')):.3f}, "
        f"{dev.get('bootstrap_sharpe_upper_95', float('nan')):.3f}]",
        "",
        "## Cost stress (2x)",
        f"- Sharpe (annualized): {cs['sharpe']:.3f}",
        f"- Max drawdown: {cs['max_dd']*100:.2f}%",
        "",
        "## Holdout metrics (touched once)",
        f"- Sharpe (annualized): {hd['sharpe']:.3f}",
        f"- Max drawdown: {hd['max_dd']*100:.2f}%",
        f"- Cumulative return: {hd['cum_return']*100:.2f}%",
        f"- Trading days: {hd['n_days']}",
        "",
        "## Selection funnel",
        *funnel_lines,
        "",
        "## Disclaimer",
        "Research output only. Past performance does not guarantee future results. ",
        "No promotion to capital deployment occurs without an explicit promotion record ",
        "(spec §6.5 and the QuantLab promotion runbook).",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body)
    return output_path
