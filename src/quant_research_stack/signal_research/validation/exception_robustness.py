"""Exception-path robustness diagnostics — per accepted exception policy §4.

For each HMM variant, compute the §10 diagnostics from the HMM intake:
- 10.1 transition matrix
- 10.2 expected state duration
- 10.3 state persistence (run-length histogram summary)
- 10.4 raw state-label stability across refits (informational only)
- 10.5 economic-identity stability across refits (this IS demotion trigger)
- 10.6 exposure-time fraction by regime
- 10.7 PnL contribution by regime
- 10.8 false de-risking cost
- 10.9 crash-protection contribution during largest 10 B&H drawdowns
- 10.10 re-entry timing quality
- 10.11 turnover
- 10.12 cost drag
- 10.13 delay stress decomposition
- 10.14 bootstrap CI
- 10.15-10.17 PBO/DSR/PSR_zero come from the cross-strategy pool, not per variant
- 10.18 failure classification (from methodology.failure_classifier)
- 10.19 cash-leg sensitivity (handled by cash_leg_reporting module)
- 10.20 cross-instrument summary (handled at the report-renderer level)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.signal_research.strategies.hmm_single_index import (
    FittedHMM,
)


@dataclass(frozen=True)
class StateIdentityFlipReport:
    """Result of the economic-identity stability check across refits."""

    n_refits: int
    n_economic_flips: int
    flip_rate: float  # n_economic_flips / max(1, n_refits - 1)
    raw_label_flips: int  # informational only — does NOT cause demotion
    fits_summary: list[dict[str, float | int]] = field(default_factory=list)
    passes_stability_gate: bool = True  # True if flip_rate <= 0.20


def economic_identity_stability(
    fits: list[FittedHMM], *, mean_tol: float = 0.0002, vol_tol: float = 0.005,
) -> StateIdentityFlipReport:
    """Walk the sequence of refits and count how often the risk-on state's
    (mean, vol) signature changes by more than the tolerance.

    Raw label permutations between refits are NOT flips; only changes in
    the economic identity (mean return, realized vol of the risk-on state)
    count toward the flip-rate.

    Per exception policy §5.4: flip_rate > 0.20 fails the stability gate.
    """
    n_refits = len(fits)
    if n_refits < 2:
        return StateIdentityFlipReport(
            n_refits=n_refits,
            n_economic_flips=0,
            flip_rate=0.0,
            raw_label_flips=0,
            fits_summary=[
                {
                    "fit_idx": 0,
                    "risk_on_state_id": int(f.risk_on_state_id),
                    "risk_on_mean": float(f.risk_on_state_mean_return),
                    "risk_on_vol": float(f.risk_on_state_realized_vol),
                }
                for i, f in enumerate(fits)
            ],
            passes_stability_gate=True,
        )
    n_economic_flips = 0
    raw_label_flips = 0
    summary: list[dict[str, float | int]] = []
    prev = fits[0]
    summary.append({
        "fit_idx": 0,
        "risk_on_state_id": int(prev.risk_on_state_id),
        "risk_on_mean": float(prev.risk_on_state_mean_return),
        "risk_on_vol": float(prev.risk_on_state_realized_vol),
    })
    for i, current in enumerate(fits[1:], start=1):
        # Raw label flip: state ID changed
        if current.risk_on_state_id != prev.risk_on_state_id:
            raw_label_flips += 1
        # Economic identity flip: mean OR vol of risk-on state moved outside tolerance
        mean_delta = abs(
            current.risk_on_state_mean_return - prev.risk_on_state_mean_return
        )
        vol_delta = abs(
            current.risk_on_state_realized_vol - prev.risk_on_state_realized_vol
        )
        if mean_delta > mean_tol or vol_delta > vol_tol:
            n_economic_flips += 1
        summary.append({
            "fit_idx": i,
            "risk_on_state_id": int(current.risk_on_state_id),
            "risk_on_mean": float(current.risk_on_state_mean_return),
            "risk_on_vol": float(current.risk_on_state_realized_vol),
        })
        prev = current
    flip_rate = n_economic_flips / max(1, n_refits - 1)
    return StateIdentityFlipReport(
        n_refits=n_refits,
        n_economic_flips=n_economic_flips,
        flip_rate=flip_rate,
        raw_label_flips=raw_label_flips,
        fits_summary=summary,
        passes_stability_gate=flip_rate <= 0.20,
    )


# ============================================================================
# Per-variant robustness diagnostics (§10.1 - §10.13)
# ============================================================================


@dataclass(frozen=True)
class TransitionDiagnostics:
    """Transition-matrix-derived diagnostics."""

    transition_matrix: NDArray[np.float64]
    risk_on_state_id: int
    expected_state_durations: dict[int, float]  # 1 / (1 − P(stay))
    state_persistence_pct: dict[int, float]  # P(stay) per state


def transition_diagnostics(fit: FittedHMM) -> TransitionDiagnostics:
    """Derive expected duration and persistence from the HMM's transition matrix."""
    P = fit.transition_matrix
    n = P.shape[0]
    persistence = {i: float(P[i, i]) for i in range(n)}
    durations: dict[int, float] = {}
    for i in range(n):
        stay = float(P[i, i])
        durations[i] = 1.0 / max(1e-9, (1.0 - stay))
    return TransitionDiagnostics(
        transition_matrix=P,
        risk_on_state_id=fit.risk_on_state_id,
        expected_state_durations=durations,
        state_persistence_pct=persistence,
    )


@dataclass(frozen=True)
class RegimeExposureDiagnostics:
    """§10.6 + §10.7 — exposure and PnL by regime."""

    exposure_time_risk_on_frac: float
    exposure_time_risk_off_frac: float
    pnl_risk_on: float
    pnl_risk_off: float
    n_days: int


def regime_exposure(
    *, daily_returns: pl.DataFrame, signal: pl.DataFrame,
) -> RegimeExposureDiagnostics:
    """Daily returns DataFrame needs (date, net_return). Signal DataFrame
    needs (date, signal)."""
    joined = daily_returns.join(signal, on="date", how="left").with_columns(
        pl.col("signal").fill_null(0.0)
    )
    n = joined.height
    if n == 0:
        return RegimeExposureDiagnostics(0.0, 0.0, 0.0, 0.0, 0)
    risk_on_mask = joined["signal"].to_numpy() > 0.5
    risk_off_mask = ~risk_on_mask
    rets = joined["net_return"].to_numpy().astype(np.float64)
    return RegimeExposureDiagnostics(
        exposure_time_risk_on_frac=float(risk_on_mask.sum() / n),
        exposure_time_risk_off_frac=float(risk_off_mask.sum() / n),
        pnl_risk_on=float(rets[risk_on_mask].sum()),
        pnl_risk_off=float(rets[risk_off_mask].sum()),
        n_days=int(n),
    )


@dataclass(frozen=True)
class CrashProtectionDiagnostics:
    """§10.8 - §10.10 — crash-protection mechanics."""

    false_derisk_cost_pct: float  # cumulative B&H return missed during risk-off
    crash_periods_top10: list[dict[str, float]]  # one entry per top-10 B&H drawdown
    reentry_quality_avg_20d_return: float  # avg 20d strat return post risk-off→on


def _bah_top_n_drawdowns(
    *, dates: list, bah_returns: NDArray[np.float64], n: int = 10,
) -> list[tuple[int, int]]:
    """Return list of (start_idx, trough_idx) for the top-n peak-to-trough
    drawdowns on buy-and-hold."""
    if bah_returns.size < 2:
        return []
    equity = np.cumprod(1.0 + bah_returns)
    peaks = np.maximum.accumulate(equity)
    drawdown_pct = equity / peaks - 1.0
    # Find local minima as candidate trough points; for simplicity pick the
    # n deepest unique trough indices
    n_take = min(n, drawdown_pct.size)
    trough_indices = np.argsort(drawdown_pct)[:n_take].tolist()
    out = []
    for ti in sorted(trough_indices):
        # Find the most recent peak before ti
        if ti == 0:
            continue
        peak_idx = int(np.argmax(equity[: ti + 1]))
        if peak_idx >= ti:
            continue
        out.append((peak_idx, int(ti)))
    return out


def crash_protection_diagnostics(
    *,
    underlying_returns: pl.DataFrame,  # (date, u_ret)
    strategy_daily: pl.DataFrame,  # (date, net_return)
    signal: pl.DataFrame,  # (date, signal)
) -> CrashProtectionDiagnostics:
    """§10.8: false de-risking cost (missed buy-and-hold return while flat).
    §10.9: strategy return during the 10 largest B&H drawdown windows.
    §10.10: average 20-day strategy return after each risk-off → risk-on transition.
    """
    bah_df = underlying_returns.sort("date")
    strat_df = strategy_daily.sort("date").select(["date", "net_return"])
    sig_df = signal.sort("date").select(["date", "signal"])
    joined = (
        bah_df.join(strat_df, on="date", how="inner")
        .join(sig_df, on="date", how="left")
        .with_columns(pl.col("signal").fill_null(0.0))
    )
    if joined.height == 0:
        return CrashProtectionDiagnostics(0.0, [], 0.0)
    bah_arr = joined["u_ret"].fill_null(0.0).to_numpy().astype(np.float64)
    strat_arr = joined["net_return"].to_numpy().astype(np.float64)
    sig_arr = joined["signal"].to_numpy().astype(np.float64)
    flat_mask = sig_arr <= 0.5
    # False de-risk cost: cumulative B&H return on the days the strategy was flat
    if flat_mask.sum() > 0:
        flat_returns_cum = float(np.prod(1.0 + bah_arr[flat_mask]) - 1.0)
    else:
        flat_returns_cum = 0.0

    # Top-10 B&H drawdowns
    crash_idx = _bah_top_n_drawdowns(
        dates=joined["date"].to_list(), bah_returns=bah_arr,
    )
    crash_summaries: list[dict[str, float]] = []
    for peak_idx, trough_idx in crash_idx:
        bah_window = bah_arr[peak_idx : trough_idx + 1]
        strat_window = strat_arr[peak_idx : trough_idx + 1]
        crash_summaries.append({
            "peak_to_trough_days": float(trough_idx - peak_idx),
            "bah_cum_return": float(np.prod(1.0 + bah_window) - 1.0),
            "strat_cum_return": float(np.prod(1.0 + strat_window) - 1.0),
        })

    # Re-entry quality: find risk-off → risk-on transitions; measure strat
    # cumulative return over next 20 trading days
    re_entry_returns = []
    for i in range(1, joined.height - 20):
        if sig_arr[i - 1] <= 0.5 and sig_arr[i] > 0.5:
            window = strat_arr[i : i + 20]
            re_entry_returns.append(float(np.prod(1.0 + window) - 1.0))
    avg_re_entry = float(np.mean(re_entry_returns)) if re_entry_returns else 0.0

    return CrashProtectionDiagnostics(
        false_derisk_cost_pct=flat_returns_cum,
        crash_periods_top10=crash_summaries,
        reentry_quality_avg_20d_return=avg_re_entry,
    )


@dataclass(frozen=True)
class TurnoverDiagnostics:
    """§10.11 turnover and §10.12 cost drag."""

    turnover_per_year_avg: float
    cost_drag_bps_annual: float


def turnover_diagnostics(
    *, daily_returns: pl.DataFrame, cost_bps_one_way: float, n_days_per_year: float = 252.0,
) -> TurnoverDiagnostics:
    if daily_returns.is_empty() or "turnover" not in daily_returns.columns:
        return TurnoverDiagnostics(0.0, 0.0)
    turnover = daily_returns["turnover"].to_numpy().astype(np.float64)
    n_days = turnover.size
    years = n_days / n_days_per_year if n_days > 0 else 1.0
    total_turnover = float(np.sum(turnover))
    turnover_per_year_avg = total_turnover / max(1e-9, years)
    cost_drag_bps_annual = turnover_per_year_avg * cost_bps_one_way * 2.0
    return TurnoverDiagnostics(
        turnover_per_year_avg=turnover_per_year_avg,
        cost_drag_bps_annual=cost_drag_bps_annual,
    )
