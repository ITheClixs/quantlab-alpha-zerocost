"""zero_cost_riskalloc_v1 — long-flat vol-targeted regime+macro allocator (P1).

Leakage convention (single explicit shift): every signal (vol target, trend,
vol-regime, macro filter) is computed using information through close t; the held
position for day t is the weight decided at close t-1 (`apply_execution_shift`,
delay=1 -> "decision close t, execute t+1"). Long-flat only; weekly rebalance;
equal-risk across active instruments. Pure numpy/polars — no model fit, no future.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

F = NDArray[np.float64]
_ANN = 252.0


def daily_returns(close: F) -> F:
    out = np.zeros_like(close, dtype=np.float64)
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


def _roll(fn, x: F, w: int) -> F:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(len(x)):
        if i + 1 >= w:
            out[i] = fn(x[i + 1 - w : i + 1])
    return out


def realized_vol(returns: F, lookback: int = 20) -> F:
    return _roll(lambda a: float(np.std(a, ddof=1)), returns, lookback) * np.sqrt(_ANN)


def vol_target_weight(returns: F, *, target_ann_vol: float = 0.12, lookback: int = 20, cap: float = 1.5) -> F:
    rv = realized_vol(returns, lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(rv > 0, target_ann_vol / rv, 0.0)
    return np.clip(np.nan_to_num(w, nan=0.0), 0.0, cap)


def trend_on(close: F, *, slow: int = 200) -> NDArray[np.bool_]:
    sma = _roll(np.mean, close, slow)
    return np.where(np.isfinite(sma) & (close > sma), True, False)


def vol_regime_on(returns: F, *, lookback: int = 20, median_window: int = 252) -> NDArray[np.bool_]:
    rv = _roll(lambda a: float(np.std(a, ddof=1)), returns, lookback)
    med = _roll(np.nanmedian, rv, median_window)
    return np.where(np.isfinite(rv) & np.isfinite(med) & (rv <= med), True, False)


def weeklyize(weights: F, dates: list[date]) -> F:
    """Hold the weight constant within an ISO week; update on the week's first day."""
    out = weights.copy()
    last_week: tuple[int, int] | None = None
    held = 0.0
    for i, d in enumerate(dates):
        wk = d.isocalendar()[:2]
        if wk != last_week:
            held = weights[i]
            last_week = wk
        out[i] = held
    return out


def apply_execution_shift(weights: F, delay: int = 1) -> F:
    """Position held during day t = weight decided `delay` days earlier (close t-delay)."""
    out = weights.astype(np.float64).copy()
    for _ in range(delay):
        shifted = np.zeros_like(out)
        shifted[1:] = out[:-1]
        out = shifted
    return out


@dataclass(frozen=True)
class InstrumentSeries:
    name: str
    dates: list[date]
    close: F
    returns: F
    is_crypto: bool


@dataclass(frozen=True)
class PortfolioResult:
    daily_returns: F
    metrics: dict[str, float]
    weights: dict[str, F]


def metrics(net: F) -> dict[str, float]:
    f = net[np.isfinite(net)]
    if f.size < 2:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "total_return": 0.0}
    sd = float(np.std(f, ddof=1))
    sharpe = float(np.mean(f) / sd * np.sqrt(_ANN)) if sd > 0 else 0.0
    eq = np.cumprod(1.0 + f)
    dd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    ann = float(eq[-1] ** (_ANN / f.size) - 1.0) if eq[-1] > 0 else -1.0
    return {"sharpe": sharpe, "ann_return": ann, "ann_vol": sd * np.sqrt(_ANN),
            "max_drawdown": dd, "calmar": float(ann / abs(dd)) if dd < 0 else 0.0,
            "total_return": float(eq[-1] - 1.0)}


def backtest_portfolio(insts: dict[str, InstrumentSeries], target_weights: dict[str, F], *,
                       dates: list[date], cost_bps: dict[str, float], delay: int = 1,
                       weekly: bool = True) -> PortfolioResult:
    """Equal-risk long-flat basket. target_weights[i] are raw (info through t)."""
    names = list(insts)
    n = len(dates)
    raw = {}
    for name in names:
        w = target_weights[name]
        if weekly:
            w = weeklyize(w, dates)
        raw[name] = apply_execution_shift(w, delay)  # held during day t
    # equal-risk: split capital equally among instruments active that day
    active = np.zeros(n)
    for name in names:
        active += (raw[name] > 0).astype(np.float64)
    active = np.where(active > 0, active, 1.0)
    port = np.zeros(n)
    held: dict[str, F] = {}
    for name in names:
        w_eff = raw[name] / active  # equal split across active
        held[name] = w_eff
        r = insts[name].returns
        turn = np.abs(np.diff(w_eff, prepend=w_eff[0]))
        cost = turn * (cost_bps[name] * 1e-4)
        port += w_eff * r - cost
    return PortfolioResult(daily_returns=port, metrics=metrics(port), weights=held)
