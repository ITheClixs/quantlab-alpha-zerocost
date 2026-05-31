"""Sector-conditional Avellaneda-Lee — strictly controlled experiment.

Tests whether AvL residual mean reversion works when PCA is fit WITHIN
economically coherent sector baskets (Financials, Industrials, Technology,
Healthcare, Consumer Discretionary, Energy) rather than across highly
correlated mega-cap names.

Predeclared variant grid (small, all DoF logged):
- PCA components ∈ {1, 2, 3}
- z-entry threshold ∈ {1.0, 1.5, 2.0}
- HMM gate ∈ {none, risk_on}
→ 18 AvL variants

Sanity baselines:
- random_signal (within-sector i.i.d. standard normal)
- inverted_signal (sign-flip of MOM_12_1 within sector)
- simple_reversal (within-sector 5-day reversal)
→ 21 total strategies in PBO/DSR pool.

Per-sector portfolio: dollar-neutral, q_quantile=0.25, focused_basket cohort.
Aggregate: equal-risk weighting (1 / σ_60d_sector) across sectors active on a
given date. Per-sector PnL also reported.
"""

from __future__ import annotations

import datetime as dt
import enum
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
from quant_research_stack.signal_research.backtests._shared import (
    sharpe as sharpe_fn,
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


class HMMGate(enum.StrEnum):
    NONE = "none"
    RISK_ON = "risk_on"


@dataclass(frozen=True)
class SectorAvLVariant:
    pca_components: int
    z_entry: float
    hmm_gate: HMMGate

    @property
    def name(self) -> str:
        return (
            f"avl_pca{self.pca_components}_z{self.z_entry:.1f}_"
            f"hmm{self.hmm_gate.value}"
        )


@dataclass(frozen=True)
class SectorAvLSpec:
    sectors_to_include: tuple[str, ...] = (
        "Financials",
        "Industrials",
        "Information Technology",
        "Health Care",
        "Consumer Discretionary",
        "Energy",
    )
    min_sector_size: int = 15
    start: dt.date = dt.date(2006, 1, 1)
    end: dt.date = dt.date(2026, 5, 26)
    dev_end: dt.date = dt.date(2022, 12, 31)
    holdout_start: dt.date = dt.date(2023, 1, 1)
    pca_window: int = 252
    pca_components_grid: tuple[int, ...] = (1, 2, 3)
    z_entry_grid: tuple[float, ...] = (1.0, 1.5, 2.0)
    hmm_gates: tuple[HMMGate, ...] = (HMMGate.NONE, HMMGate.RISK_ON)
    z_exit_reversion: float = 0.5
    max_holding_days: int = 10
    q_quantile_sector: float = 0.25
    cohort: str = "focused_basket"
    target_gross: float = 1.0
    equity: float = 1_000_000.0
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0
    cost_stress_multiplier: float = 2.0
    rebalance_vol_window: int = 60
    hmm_n_states: int = 2
    hmm_seed: int = 42
    data_quality_label: str = "survivorship_prototype_only"
    constituent_survivorship_applicable: bool = True


@dataclass(frozen=True)
class SectorBacktestResult:
    sector: str
    strategy_name: str
    daily_returns: pl.DataFrame
    cumulative_metrics: dict[str, float]


@dataclass(frozen=True)
class AggregateResult:
    name: str
    variant: SectorAvLVariant | None
    aggregate_dev: pl.DataFrame
    aggregate_holdout: pl.DataFrame
    aggregate_cost_stress: pl.DataFrame
    per_sector_dev: dict[str, SectorBacktestResult]
    per_sector_holdout: dict[str, SectorBacktestResult]
    dev_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    cost_stress_metrics: dict[str, float]
    research_pass: bool
    funnel: SelectionFunnel = field(default_factory=SelectionFunnel)


def assign_sectors(sp500_df: pl.DataFrame) -> dict[str, str]:
    """symbol → sector (GICS)."""
    return dict(
        zip(
            sp500_df["symbol"].to_list(),
            sp500_df["sector"].to_list(),
            strict=False,
        )
    )


def filter_sectors(
    *,
    bars: pl.DataFrame,
    sector_map: dict[str, str],
    spec: SectorAvLSpec,
) -> dict[str, list[str]]:
    """Return {sector: [tickers]} for sectors meeting min_sector_size in the
    fixture universe."""
    panel_symbols = set(bars["symbol"].to_list())
    out: dict[str, list[str]] = {}
    for s in spec.sectors_to_include:
        members = sorted(
            sym for sym, sec in sector_map.items()
            if sec == s and sym in panel_symbols
        )
        if len(members) >= spec.min_sector_size:
            out[s] = members
    return out


def _log_returns(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias(
            "log_ret"
        )
    )


def _build_returns_wide(
    bars: pl.DataFrame, sector_tickers: list[str]
) -> tuple[NDArray[np.float64], list[dt.date], list[str]]:
    """Return (T, S) returns matrix, sorted date list, and column-symbol list."""
    df = _log_returns(bars.filter(pl.col("symbol").is_in(sector_tickers)))
    wide = (
        df.pivot(values="log_ret", index="date", on="symbol")
        .sort("date")
        .fill_null(0.0)
    )
    cols = [c for c in wide.columns if c != "date"]
    dates = wide["date"].to_list()
    R = wide.select(cols).to_numpy().astype(np.float64)
    return R, dates, cols


def compute_sector_residual_z(
    bars: pl.DataFrame,
    *,
    sector_tickers: list[str],
    pca_window: int,
    n_components: int,
) -> pl.DataFrame:
    """Rolling PCA residual z-scores within a sector, past-only.

    Returns long-form (date, symbol, residual_z). The residual at date T is
    today's return projected onto the orthogonal complement of the past PCA
    basis fit on R[T-pca_window:T], z-scored by trailing-window residual std.
    """
    R, dates, symbols = _build_returns_wide(bars, sector_tickers)
    T, S = R.shape
    z_panel = np.full(R.shape, np.nan, dtype=np.float64)
    for t in range(pca_window, T):
        window = R[t - pca_window : t]
        mean = window.mean(axis=0, keepdims=True)
        centred = window - mean
        try:
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        comps = Vt[:n_components]
        today = R[t] - mean.flatten()
        loadings = today @ comps.T
        reconstruction = loadings @ comps
        residual = today - reconstruction
        window_residuals = centred - centred @ comps.T @ comps
        std = float(np.std(window_residuals, ddof=1)) or 1.0
        z_panel[t] = residual / std
    rows: list[dict[str, object]] = []
    for ti, d in enumerate(dates):
        for si, sym in enumerate(symbols):
            v = z_panel[ti, si]
            rows.append({
                "date": d,
                "symbol": sym,
                "residual_z": float(v) if not np.isnan(v) else None,
            })
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "symbol": pl.Utf8, "residual_z": pl.Float64},
    )


def fit_hmm_risk_on_panel(
    *,
    bars: pl.DataFrame,
    spec: SectorAvLSpec,
) -> pl.DataFrame:
    """Fit HMM on dev market returns, label risk-on (higher mean dev return),
    return (date, regime_id, risk_on) for the full sample.
    """
    from hmmlearn.hmm import GaussianHMM

    market = (
        _log_returns(bars)
        .group_by("date")
        .agg(pl.col("log_ret").mean().alias("market_ret"))
        .sort("date")
    )
    dev_market = market.filter(pl.col("date") <= spec.dev_end)
    dev_ret = dev_market["market_ret"].fill_null(0.0).to_numpy().astype(np.float64)
    model = GaussianHMM(
        n_components=spec.hmm_n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=spec.hmm_seed,
    )
    model.fit(dev_ret.reshape(-1, 1))
    dev_states = model.predict(dev_ret.reshape(-1, 1))
    state_means = {
        int(s): float(np.mean(dev_ret[dev_states == s]))
        for s in range(spec.hmm_n_states)
    }
    risk_on_id = max(state_means, key=lambda k: state_means[k])
    full_ret = market["market_ret"].fill_null(0.0).to_numpy().astype(np.float64)
    full_states = model.predict(full_ret.reshape(-1, 1)).astype(np.int64)
    risk_on_flag = (full_states == risk_on_id).astype(np.int64)
    return market.with_columns(
        pl.Series("regime_id", full_states),
        pl.Series("risk_on", risk_on_flag),
    ).select(["date", "risk_on"])


def _signal_from_residual_z(
    *,
    residual_z: pl.DataFrame,
    z_entry: float,
    z_exit_reversion: float,
    max_holding_days: int,
) -> pl.DataFrame:
    """Convert residual z-scores to mean-reversion positions.

    Position = -sign(z) when |z| >= z_entry, hold until |z| <= z_exit_reversion
    or max_holding_days elapsed.

    Implementation: per (symbol), walk dates and maintain state.
    """
    out_rows: list[dict[str, object]] = []
    df_sorted = residual_z.sort(["symbol", "date"])
    for sym, sym_df in df_sorted.group_by("symbol"):
        sym_str = sym[0] if isinstance(sym, tuple) else sym
        dates = sym_df["date"].to_list()
        z_arr = sym_df["residual_z"].to_numpy()
        position = 0.0
        days_held = 0
        for i, d in enumerate(dates):
            z = z_arr[i] if z_arr[i] is not None and not np.isnan(z_arr[i]) else None
            if position != 0:
                days_held += 1
                # Exit conditions
                if (
                    z is not None
                    and abs(z) <= z_exit_reversion
                ) or days_held >= max_holding_days:
                    position = 0.0
                    days_held = 0
            if position == 0 and z is not None and abs(z) >= z_entry:
                position = -float(np.sign(z))
                days_held = 0
            out_rows.append({
                "date": d, "symbol": sym_str, "y_xs_pred": position,
            })
    return pl.DataFrame(
        out_rows,
        schema={"date": pl.Date, "symbol": pl.Utf8, "y_xs_pred": pl.Float64},
    )


def _apply_hmm_gate(
    signals: pl.DataFrame, *, hmm_panel: pl.DataFrame, gate: HMMGate
) -> pl.DataFrame:
    if gate == HMMGate.NONE:
        return signals
    return (
        signals.join(hmm_panel, on="date", how="left")
        .with_columns(
            pl.when(pl.col("risk_on") == 1)
            .then(pl.col("y_xs_pred"))
            .otherwise(0.0)
            .alias("y_xs_pred"),
        )
        .select(["date", "symbol", "y_xs_pred"])
    )


def _run_sector_backtest(
    *,
    sector: str,
    bars_sector: pl.DataFrame,
    signals: pl.DataFrame,
    spec: SectorAvLSpec,
    cost_stress_mult: float,
) -> SectorBacktestResult:
    panel = to_m4_panel(
        bars=bars_sector,
        signals=signals,
        spread_bps=spec.spread_bps_one_way * 10.0,
        borrow_tier="general",
    )
    cfg = build_backtest_config(
        commission_bps_one_way=spec.commission_bps_one_way,
        spread_bps_one_way=spec.spread_bps_one_way,
        cost_stress_mult=cost_stress_mult,
        q_quantile=spec.q_quantile_sector,
        target_gross=spec.target_gross,
        equity=spec.equity,
        cohort=spec.cohort,
    )
    res = run_backtest(signals_with_bars=panel, config=cfg, dividends=None)
    metrics = equity_metrics(res.daily_returns)
    return SectorBacktestResult(
        sector=sector,
        strategy_name=signals.columns[-1] if "name" not in signals.columns else "",
        daily_returns=res.daily_returns.select(["date", "net_return"]),
        cumulative_metrics=metrics,
    )


def _equal_risk_aggregate(
    *,
    sector_returns: dict[str, pl.DataFrame],
    spec: SectorAvLSpec,
) -> pl.DataFrame:
    """Equal-risk-contribution aggregate: weight each sector by 1/σ_60d(sector).

    Aggregate net_return on date d:
        sum_s w_s(d) * r_s(d), with w_s(d) ∝ 1 / σ_s(d-1..d-60) and Σ_s w_s = 1.
    Sectors that haven't produced returns on date d contribute zero.
    """
    if not sector_returns:
        return pl.DataFrame({"date": [], "net_return": []})
    # Stack to wide (date, ret_S1, ret_S2, ...)
    wide = None
    for sector_name, daily in sector_returns.items():
        renamed = daily.rename({"net_return": f"r__{sector_name}"})
        wide = renamed if wide is None else wide.join(
            renamed, on="date", how="full", coalesce=True
        )
    if wide is None:
        return pl.DataFrame({"date": [], "net_return": []})
    wide = wide.sort("date").fill_null(0.0)
    # Compute trailing vol per sector
    ret_cols = [c for c in wide.columns if c.startswith("r__")]
    vols = {
        c: wide[c]
        .rolling_std(window_size=spec.rebalance_vol_window)
        .fill_null(0.02)
        .clip(lower_bound=0.0005)
        for c in ret_cols
    }
    inv_vols_mat = np.column_stack(
        [(1.0 / vols[c].to_numpy()) for c in ret_cols]
    )
    weights = inv_vols_mat / inv_vols_mat.sum(axis=1, keepdims=True)
    returns_mat = wide.select(ret_cols).to_numpy().astype(np.float64)
    agg = (weights * returns_mat).sum(axis=1)
    return pl.DataFrame({"date": wide["date"], "net_return": agg.astype(np.float64)})


def _bootstrap_ci(rets: NDArray[np.float64], *, seed: int) -> tuple[float, float]:
    if rets.size < 30:
        return float("nan"), float("nan")
    bs = bootstrap_sharpe_ci(
        returns=rets, config=BootstrapConfig(n_resamples=1000, seed=seed)
    )
    return bs.ci_lower_95, bs.ci_upper_95


def _signals_random_within_sector(
    *, bars_sector: pl.DataFrame, seed: int
) -> pl.DataFrame:
    df = _log_returns(bars_sector)
    rng = np.random.default_rng(seed)
    n = df.height
    df = df.with_columns(pl.Series("y_xs_pred", rng.standard_normal(n)))
    return df.select(["date", "symbol", "y_xs_pred"])


def _signals_simple_reversal_within_sector(bars_sector: pl.DataFrame) -> pl.DataFrame:
    df = _log_returns(bars_sector)
    df = df.with_columns(
        (-pl.col("log_ret").rolling_sum(window_size=5).over("symbol")).alias("y_xs_pred")
    )
    return df.drop_nulls(subset=["y_xs_pred"]).select(["date", "symbol", "y_xs_pred"])


def _signals_inverted_mom_within_sector(bars_sector: pl.DataFrame) -> pl.DataFrame:
    df = _log_returns(bars_sector)
    df = df.with_columns(
        (
            -(
                pl.col("log_ret").rolling_sum(window_size=252).over("symbol")
                - pl.col("log_ret").rolling_sum(window_size=21).over("symbol")
            )
        ).alias("y_xs_pred")
    )
    return df.drop_nulls(subset=["y_xs_pred"]).select(["date", "symbol", "y_xs_pred"])


def _run_aggregate(
    *,
    name: str,
    variant: SectorAvLVariant | None,
    bars: pl.DataFrame,
    sector_baskets: dict[str, list[str]],
    sector_signal_fn,
    spec: SectorAvLSpec,
    seed: int,
) -> AggregateResult:
    funnel = SelectionFunnel()
    funnel.record("universe_initial", int(bars["symbol"].n_unique()))
    funnel.record("n_sectors", len(sector_baskets))
    sector_dev: dict[str, SectorBacktestResult] = {}
    sector_hd: dict[str, SectorBacktestResult] = {}
    sector_cs: dict[str, SectorBacktestResult] = {}
    for sector, tickers in sector_baskets.items():
        bars_sec = bars.filter(pl.col("symbol").is_in(tickers))
        signals = sector_signal_fn(sector=sector, bars_sector=bars_sec, seed=seed)
        if signals.is_empty():
            continue
        sector_dev[sector] = _run_sector_backtest(
            sector=sector, bars_sector=bars_sec, signals=signals,
            spec=spec, cost_stress_mult=1.0,
        )
        sector_cs[sector] = _run_sector_backtest(
            sector=sector, bars_sector=bars_sec, signals=signals,
            spec=spec, cost_stress_mult=spec.cost_stress_multiplier,
        )
        sector_hd[sector] = sector_dev[sector]  # will be split below

    def _filter_phase(
        per_sector: dict[str, SectorBacktestResult], lo: dt.date, hi: dt.date
    ) -> dict[str, pl.DataFrame]:
        return {
            s: r.daily_returns.filter(
                (pl.col("date") >= lo) & (pl.col("date") <= hi)
            )
            for s, r in per_sector.items()
        }

    dev_phase = _filter_phase(sector_dev, spec.start, spec.dev_end)
    hd_phase = _filter_phase(sector_dev, spec.holdout_start, spec.end)
    cs_phase = _filter_phase(sector_cs, spec.start, spec.dev_end)

    agg_dev = _equal_risk_aggregate(sector_returns=dev_phase, spec=spec)
    agg_hd = _equal_risk_aggregate(sector_returns=hd_phase, spec=spec)
    agg_cs = _equal_risk_aggregate(sector_returns=cs_phase, spec=spec)

    dev_m = equity_metrics(agg_dev)
    hd_m = equity_metrics(agg_hd)
    cs_m = equity_metrics(agg_cs)
    if agg_dev.height > 30:
        lo, hi = _bootstrap_ci(
            agg_dev["net_return"].to_numpy().astype(np.float64), seed=seed,
        )
        dev_m["bootstrap_sharpe_lower_95"] = lo
        dev_m["bootstrap_sharpe_upper_95"] = hi

    # Per-sector contribution count
    positive_sectors = sum(
        1 for s, df in dev_phase.items()
        if df.height > 30 and sharpe_fn(df["net_return"].to_numpy().astype(np.float64)) > 0
    )
    funnel.record("positive_sectors_in_dev", positive_sectors)
    # Concentration: largest sector PnL fraction
    pnls = {
        s: float(df["net_return"].sum()) for s, df in dev_phase.items() if df.height > 0
    }
    total_pnl = sum(pnls.values()) or 1e-9
    max_share = max(abs(v) / abs(total_pnl) for v in pnls.values()) if pnls else 1.0
    funnel.record("max_sector_pnl_share_x100", int(max_share * 100))

    research_pass = (
        dev_m["sharpe"] >= 0.5
        and positive_sectors >= 2
        and max_share < 0.6
        and dev_m.get("bootstrap_sharpe_lower_95", -1.0) > -0.3
        and cs_m["sharpe"] > -0.5
    )
    funnel.record("research_pass", 1 if research_pass else 0)

    return AggregateResult(
        name=name,
        variant=variant,
        aggregate_dev=agg_dev,
        aggregate_holdout=agg_hd,
        aggregate_cost_stress=agg_cs,
        per_sector_dev={s: r for s, r in sector_dev.items()},
        per_sector_holdout={s: r for s, r in sector_dev.items()},  # same series; split below in report
        dev_metrics=dev_m,
        holdout_metrics=hd_m,
        cost_stress_metrics=cs_m,
        research_pass=research_pass,
        funnel=funnel,
    )


def run_all_sector_avl_variants(
    *,
    bars: pl.DataFrame,
    sector_baskets: dict[str, list[str]],
    spec: SectorAvLSpec,
) -> list[AggregateResult]:
    hmm_panel = fit_hmm_risk_on_panel(bars=bars, spec=spec)

    # Cache residual_z by (sector, n_components)
    residual_cache: dict[tuple[str, int], pl.DataFrame] = {}
    for sector, tickers in sector_baskets.items():
        bars_sec = bars.filter(pl.col("symbol").is_in(tickers))
        for n_pca in spec.pca_components_grid:
            residual_cache[(sector, n_pca)] = compute_sector_residual_z(
                bars_sec,
                sector_tickers=tickers,
                pca_window=spec.pca_window,
                n_components=n_pca,
            )

    variants = [
        SectorAvLVariant(pca_components=p, z_entry=z, hmm_gate=g)
        for p in spec.pca_components_grid
        for z in spec.z_entry_grid
        for g in spec.hmm_gates
    ]
    results: list[AggregateResult] = []
    for variant in variants:
        def signal_fn(
            *, sector: str, bars_sector: pl.DataFrame, seed: int,
            _v: SectorAvLVariant = variant,
        ) -> pl.DataFrame:
            r_z = residual_cache[(sector, _v.pca_components)]
            sig = _signal_from_residual_z(
                residual_z=r_z,
                z_entry=_v.z_entry,
                z_exit_reversion=spec.z_exit_reversion,
                max_holding_days=spec.max_holding_days,
            )
            sig = _apply_hmm_gate(sig, hmm_panel=hmm_panel, gate=_v.hmm_gate)
            return sig

        results.append(
            _run_aggregate(
                name=variant.name, variant=variant,
                bars=bars, sector_baskets=sector_baskets,
                sector_signal_fn=signal_fn, spec=spec, seed=hash(variant.name) & 0xFFFF,
            )
        )
    return results


def run_sanity_baselines(
    *,
    bars: pl.DataFrame,
    sector_baskets: dict[str, list[str]],
    spec: SectorAvLSpec,
) -> list[AggregateResult]:
    def random_fn(*, sector, bars_sector, seed):
        return _signals_random_within_sector(bars_sector=bars_sector, seed=seed)

    def inverted_fn(*, sector, bars_sector, seed):
        return _signals_inverted_mom_within_sector(bars_sector)

    def reversal_fn(*, sector, bars_sector, seed):
        return _signals_simple_reversal_within_sector(bars_sector)

    return [
        _run_aggregate(
            name="random_signal", variant=None, bars=bars,
            sector_baskets=sector_baskets, sector_signal_fn=random_fn,
            spec=spec, seed=1001,
        ),
        _run_aggregate(
            name="inverted_signal_mom", variant=None, bars=bars,
            sector_baskets=sector_baskets, sector_signal_fn=inverted_fn,
            spec=spec, seed=1002,
        ),
        _run_aggregate(
            name="simple_reversal_5d", variant=None, bars=bars,
            sector_baskets=sector_baskets, sector_signal_fn=reversal_fn,
            spec=spec, seed=1003,
        ),
    ]


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
    all_results: list[AggregateResult],
) -> CrossStrategyMetrics:
    series = [r.aggregate_dev["net_return"].to_numpy().astype(np.float64) for r in all_results]
    min_len = min(s.size for s in series)
    dev_matrix = np.column_stack([s[-min_len:] for s in series])
    family = np.array([
        "sector_avl" if r.variant is not None else "sanity"
        for r in all_results
    ])
    profile = np.array(["all_sectors" for _ in all_results])
    pbo = compute_three_tier_pbo(
        returns=dev_matrix, profile=profile, family=family, n_partitions=16
    )
    sharpes = np.array([r.dev_metrics["sharpe"] for r in all_results])
    best_idx = int(np.argmax(sharpes))
    dsr = compute_dsr(
        returns=all_results[best_idx].aggregate_dev["net_return"]
        .to_numpy()
        .astype(np.float64),
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
    variants: list[AggregateResult],
    baselines: list[AggregateResult],
    cross: CrossStrategyMetrics,
) -> tuple[str, str]:
    """Apply the user's 8-criteria decision rule. Returns (decision, failure_class)."""
    if cross.pbo_raw_global >= 0.25:
        return (
            "FAIL — PBO too high (overfit parameter grid).",
            "overfit_parameter_grid",
        )
    if cross.best_dsr < 0.5:
        # Still examine the best variant carefully — DSR alone won't rule it out.
        pass
    best_idx = cross.best_index
    best_results = (variants + baselines)
    best = best_results[best_idx]
    if best.variant is None:
        return (
            "FAIL — best strategy is a sanity baseline, not AvL.",
            "no_residual_meanreversion_edge",
        )
    if best.dev_metrics["sharpe"] < 0.5:
        return (
            "FAIL — aggregate net Sharpe below threshold.",
            "no_residual_meanreversion_edge",
        )
    positive_sectors = best.funnel.to_ordered_dict().get(
        "positive_sectors_in_dev", 0
    )
    max_share = best.funnel.to_ordered_dict().get("max_sector_pnl_share_x100", 100) / 100.0
    if positive_sectors < 2:
        return (
            "FAIL — fewer than 2 sectors contributing positive net PnL.",
            "sector_effect_too_weak",
        )
    if max_share >= 0.6:
        return (
            "FAIL — performance dominated by one sector.",
            "one_sector_concentration",
        )
    if best.cost_stress_metrics["sharpe"] < -0.5:
        return (
            "FAIL — 2x cost stress is catastrophic.",
            "costs_kill_the_edge",
        )
    if best.dev_metrics.get("bootstrap_sharpe_lower_95", -1.0) < -0.3:
        return (
            "FAIL — bootstrap lower bound strongly negative.",
            "no_residual_meanreversion_edge",
        )
    # Check vs simple reversal baseline
    simple_rev = next(
        (r for r in baselines if r.name == "simple_reversal_5d"), None
    )
    if simple_rev is not None and best.dev_metrics["sharpe"] <= simple_rev.dev_metrics["sharpe"]:
        return (
            "FAIL — does not beat simple sector-neutral reversal baseline.",
            "no_added_value_over_reversal_baseline",
        )
    if cross.best_dsr < 0.5:
        return (
            f"FAIL — DSR={cross.best_dsr:.3f} < 0.5 (multi-test penalty too heavy).",
            "overfit_parameter_grid",
        )
    return (
        "PASS — sector-conditional AvL survives all 8 criteria; promote to "
        "deeper robustness testing.",
        "",
    )


def render_aggregate_report(
    *,
    variants: list[AggregateResult],
    baselines: list[AggregateResult],
    cross: CrossStrategyMetrics,
    decision: str,
    failure_class: str,
    spec: SectorAvLSpec,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    banner = data_quality_banner(
        data_quality_label=spec.data_quality_label,
        constituent_survivorship_applicable=spec.constituent_survivorship_applicable,
    )
    header = (
        "| Strategy | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | "
        "holdout Sharpe | holdout DD | cost-2x | pass |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|:---:|"
    rows = []
    for r in variants + baselines:
        dev = r.dev_metrics
        hd = r.holdout_metrics
        cs = r.cost_stress_metrics
        rows.append(
            f"| `{r.name}` | {dev['sharpe']:+.3f} | {dev['max_dd']*100:+.2f}% | "
            f"{dev.get('bootstrap_sharpe_lower_95', float('nan')):+.3f} | "
            f"{dev.get('bootstrap_sharpe_upper_95', float('nan')):+.3f} | "
            f"{hd['sharpe']:+.3f} | {hd['max_dd']*100:+.2f}% | "
            f"{cs['sharpe']:+.3f} | "
            f"{'YES' if r.research_pass else 'no'} |"
        )

    body = "\n".join([
        "# Sector-Conditional Avellaneda-Lee — Aggregate Report",
        "",
        "## Fixture",
        f"- sectors evaluated: {len(set(r.name for r in variants))} variants × "
        f"{len(spec.sectors_to_include)} sectors",
        f"- history: {spec.start.isoformat()} → {spec.end.isoformat()}",
        f"- dev:     {spec.start.isoformat()} → {spec.dev_end.isoformat()}",
        f"- holdout: {spec.holdout_start.isoformat()} → {spec.end.isoformat()}",
        f"- PCA: rolling {spec.pca_window}d window, components ∈ "
        f"{list(spec.pca_components_grid)}",
        f"- z-entry grid: {list(spec.z_entry_grid)}",
        f"- HMM gate: {[g.value for g in spec.hmm_gates]}",
        f"- exit: |z| ≤ {spec.z_exit_reversion} OR held > {spec.max_holding_days}d",
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
        "## Disclaimer",
        "Research output only. Past performance does not guarantee future results. ",
        "No promotion to capital deployment occurs without an explicit promotion record ",
        "(spec §6.5).",
        "",
    ])
    output_path.write_text(body)
    return output_path


def render_per_sector_report(
    *,
    variants: list[AggregateResult],
    baselines: list[AggregateResult],
    spec: SectorAvLSpec,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Sector-Conditional AvL — Per-Sector Detail", ""]
    for r in variants + baselines:
        lines.append(f"## `{r.name}`")
        lines.append("")
        lines.append(
            f"- aggregate dev Sharpe: {r.dev_metrics['sharpe']:+.3f}  "
            f"holdout: {r.holdout_metrics['sharpe']:+.3f}"
        )
        lines.append("")
        lines.append("| Sector | dev Sharpe | dev cum_ret | dev max_dd |")
        lines.append("|---|---:|---:|---:|")
        for sector, sb in r.per_sector_dev.items():
            m = sb.cumulative_metrics
            lines.append(
                f"| {sector} | {m['sharpe']:+.3f} | "
                f"{m['cum_return']*100:+.2f}% | {m['max_dd']*100:+.2f}% |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines))
    return output_path
