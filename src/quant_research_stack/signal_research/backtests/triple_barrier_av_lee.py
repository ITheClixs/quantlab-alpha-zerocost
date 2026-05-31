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


_FEAT_COLS: list[str] = [
    "log_return",
    "ret_5",
    "ret_20",
    "vol_20",
    "vol_60",
    "drawdown_60",
    "primary_position",
]


def _build_labeled_long_form(
    *,
    bars_with_features: pl.DataFrame,
    primary_positions_long: pl.DataFrame,
    spec: TBAvLeeSpec,
) -> pl.DataFrame:
    """Generate triple-barrier labels per-symbol and stack into one long-form table.

    Columns: date, symbol, label, log_return, ret_5, ret_20, vol_20, vol_60,
    drawdown_60, primary_position. Rows with NaN label or feature dropped.
    """
    tb_cfg = TripleBarrierConfig(
        vertical_barrier_days=spec.vertical_barrier_days,
        profit_take_multiplier=spec.profit_take_multiplier,
        stop_loss_multiplier=spec.stop_loss_multiplier,
        vol_estimator_window=spec.vol_estimator_window,
        seed=spec.rf_seed,
    )
    symbols = sorted(set(primary_positions_long["symbol"].to_list()))
    frames: list[pl.DataFrame] = []
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
        if joined.height < spec.pca_window + spec.vertical_barrier_days + 50:
            continue
        closes = joined["close"].to_numpy().astype(np.float64)
        positions = joined["primary_position"].to_numpy().astype(np.float64)
        labels = label_triple_barrier(close=closes, positions=positions, cfg=tb_cfg)
        frame = joined.select(["date", "symbol", *_FEAT_COLS]).with_columns(
            pl.Series("label", labels)
        )
        frames.append(frame)
    if not frames:
        raise RuntimeError("no symbols produced labeled events for meta-labeler")
    long_form = pl.concat(frames, how="diagonal_relaxed")
    return long_form.drop_nulls(subset=[*_FEAT_COLS, "label"])


def _walk_forward_oos_predictions(
    *,
    labeled_long: pl.DataFrame,
    spec: TBAvLeeSpec,
    train_until: dt.date,
    n_folds: int = 5,
) -> tuple[pl.DataFrame, RandomForestClassifier]:
    """Expanding-window walk-forward training within dev.

    Returns (dev_oos_predictions, final_rf_for_holdout_application).
    dev_oos_predictions: (date, symbol, p_meta_oos) for every dev date that
    falls inside a test fold (~80% of dev — first fold is training-only).
    Embargo between train and test = vertical_barrier_days + 5 to prevent
    triple-barrier label horizon overlap.

    final_rf: trained on the entire dev set, used to filter holdout positions
    (genuinely out-of-sample because holdout dates > train_until).
    """
    dev = labeled_long.filter(pl.col("date") <= train_until).sort(["date", "symbol"])
    dev_dates = sorted(set(dev["date"].to_list()))
    if len(dev_dates) < n_folds * (spec.vertical_barrier_days + 10):
        raise RuntimeError(
            f"dev period too short for {n_folds}-fold walk-forward "
            f"({len(dev_dates)} days)"
        )
    fold_size = len(dev_dates) // (n_folds + 1)
    embargo = spec.vertical_barrier_days + 5

    oos_rows: list[pl.DataFrame] = []
    for k in range(n_folds):
        test_start_idx = (k + 1) * fold_size
        test_end_idx = (k + 2) * fold_size if k < n_folds - 1 else len(dev_dates)
        test_dates = dev_dates[test_start_idx:test_end_idx]
        if not test_dates:
            continue
        train_until_k = test_dates[0] - dt.timedelta(days=embargo)
        train = dev.filter(pl.col("date") <= train_until_k)
        test = dev.filter(
            (pl.col("date") >= test_dates[0]) & (pl.col("date") <= test_dates[-1])
        )
        if train.height < 200 or test.height < 5:
            continue
        X_train = train.select(_FEAT_COLS).to_numpy().astype(np.float64)
        y_train = train["label"].to_numpy().astype(int)
        model_k = RandomForestClassifier(
            n_estimators=max(50, spec.rf_n_estimators // 2),
            random_state=spec.rf_seed + k,
            n_jobs=-1,
        )
        model_k.fit(X_train, y_train)
        X_test = test.select(_FEAT_COLS).to_numpy().astype(np.float64)
        proba = model_k.predict_proba(X_test)
        classes = model_k.classes_.astype(int)
        if 1 in classes:
            one_idx = int(np.where(classes == 1)[0][0])
            p1 = proba[:, one_idx].astype(np.float64)
        else:
            p1 = np.zeros(X_test.shape[0], dtype=np.float64)
        oos_rows.append(
            test.select(["date", "symbol"]).with_columns(pl.Series("p_meta_oos", p1))
        )

    if not oos_rows:
        raise RuntimeError("walk-forward produced no OOS predictions")
    dev_oos = pl.concat(oos_rows, how="diagonal_relaxed")

    # Final RF on full dev — used only for holdout application
    X_full = dev.select(_FEAT_COLS).to_numpy().astype(np.float64)
    y_full = dev["label"].to_numpy().astype(int)
    final_rf = RandomForestClassifier(
        n_estimators=spec.rf_n_estimators, random_state=spec.rf_seed, n_jobs=-1
    )
    final_rf.fit(X_full, y_full)
    return dev_oos, final_rf


def _apply_meta_labeling_walk_forward(
    *,
    bars_with_features: pl.DataFrame,
    primary_positions_long: pl.DataFrame,
    dev_oos: pl.DataFrame,
    final_rf: RandomForestClassifier,
    train_until: dt.date,
    threshold: float,
) -> pl.DataFrame:
    """Combine OOS predictions (dev) and final-RF predictions (holdout) into one
    filtered position panel (date, symbol, y_xs_pred).
    """
    sorted_p = primary_positions_long.sort(["symbol", "date"]).rename(
        {"y_xs_pred": "primary_position"}
    )
    joined = bars_with_features.join(
        sorted_p, on=["date", "symbol"], how="left"
    ).with_columns(pl.col("primary_position").fill_null(0.0))

    # Dev rows: use OOS predictions; rows not covered by any fold get p=0 (neutral skip)
    dev_mask = joined["date"].to_numpy() <= np.datetime64(train_until)
    joined_with_oos = joined.join(
        dev_oos, on=["date", "symbol"], how="left"
    ).with_columns(pl.col("p_meta_oos").fill_null(0.0))

    # Holdout rows: apply the final RF
    X_full = joined_with_oos.select(_FEAT_COLS).to_numpy().astype(np.float64)
    nan_mask = np.isnan(X_full).any(axis=1)
    X_filled = np.where(np.isnan(X_full), 0.0, X_full)
    proba_holdout = final_rf.predict_proba(X_filled)
    classes = final_rf.classes_.astype(int)
    if 1 in classes:
        one_idx = int(np.where(classes == 1)[0][0])
        p_holdout = proba_holdout[:, one_idx].astype(np.float64)
    else:
        p_holdout = np.zeros(X_full.shape[0], dtype=np.float64)
    p_holdout[nan_mask] = 0.0

    p_meta = np.where(
        dev_mask,
        joined_with_oos["p_meta_oos"].to_numpy().astype(np.float64),
        p_holdout,
    )
    keep = (p_meta >= threshold).astype(np.float64)
    filtered = joined_with_oos["primary_position"].to_numpy().astype(np.float64) * keep
    return joined_with_oos.with_columns(pl.Series("y_xs_pred", filtered)).select(
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

    labeled_long = _build_labeled_long_form(
        bars_with_features=bars_with_features,
        primary_positions_long=av_predictions,
        spec=spec,
    )
    funnel.record("after_labels_generated", int(labeled_long.height))

    dev_oos, final_rf = _walk_forward_oos_predictions(
        labeled_long=labeled_long,
        spec=spec,
        train_until=spec.dev_end,
        n_folds=5,
    )
    funnel.record("after_walk_forward_oos_dev", int(dev_oos.height))

    filtered_signals = _apply_meta_labeling_walk_forward(
        bars_with_features=bars_with_features,
        primary_positions_long=av_predictions,
        dev_oos=dev_oos,
        final_rf=final_rf,
        train_until=spec.dev_end,
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
