"""Frozen event-conditioned strategy positions + a causal daily backtest.

Leakage convention (critical):
  * Event-window flags are known in advance (the schedule is ex-ante), so a
    position for day t MAY use day-t's event flag directly — no shift.
  * Price-derived signals (vol target, SMA, vol regime) use data through day t-1,
    so they are shifted one bar before being applied to day t.
A position array is "the position held during day t"; strategy return = pos_t * r_t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_ANN = 252.0
F = NDArray[np.float64]


def daily_returns(close: F) -> F:
    out = np.zeros_like(close, dtype=np.float64)
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


def _shift1(x: F) -> F:
    out = np.zeros_like(x)
    out[1:] = x[:-1]
    return out


def _roll(fn, x: F, w: int) -> F:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(len(x)):
        if i + 1 >= w:
            out[i] = fn(x[i + 1 - w : i + 1])
    return out


# --- event-flag positions (no shift; flags are ex-ante known) ----------------

def risk_off_gate(flag: NDArray[np.bool_]) -> F:
    """Flat (0) inside the event window, long (1) otherwise."""
    return np.where(flag, 0.0, 1.0).astype(np.float64)


def risk_on_gate(flag: NDArray[np.bool_]) -> F:
    """Long (1) only inside the window, flat (0) otherwise."""
    return np.where(flag, 1.0, 0.0).astype(np.float64)


# --- price-derived positions (shifted one bar) -------------------------------

def buy_and_hold(n: int) -> F:
    return np.ones(n, dtype=np.float64)


def vol_target_position(returns: F, *, lookback: int = 20, target_ann_vol: float = 0.15, cap: float = 1.5) -> F:
    rv = _roll(lambda a: float(np.std(a, ddof=1)), returns, lookback) * np.sqrt(_ANN)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(rv > 0.0, target_ann_vol / rv, 0.0)
    pos = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, cap)
    return _shift1(pos)


def vol_target_event(returns: F, flag: NDArray[np.bool_], *, off_scale: float = 0.0, **kw: float) -> F:
    base = vol_target_position(returns, **kw)  # type: ignore[arg-type]
    return base * np.where(flag, off_scale, 1.0)


def sma_gate(close: F, *, fast: int = 50, slow: int = 200) -> F:
    smaf = _roll(np.mean, close, fast)
    smas = _roll(np.mean, close, slow)
    sig = np.where(np.isfinite(smaf) & np.isfinite(smas) & (smaf > smas), 1.0, 0.0)
    return _shift1(sig)


def vol_regime_gate(returns: F, *, lookback: int = 20, median_window: int = 252) -> F:
    """Risk-on when trailing vol is at/below its trailing median (HMM-only proxy)."""
    rv = _roll(lambda a: float(np.std(a, ddof=1)), returns, lookback)
    med = _roll(np.nanmedian, rv, median_window)
    sig = np.where(np.isfinite(rv) & np.isfinite(med) & (rv <= med), 1.0, 0.0)
    return _shift1(sig)


# --- backtest -----------------------------------------------------------------

@dataclass(frozen=True)
class BacktestResult:
    net_returns: F
    metrics: dict[str, float]


def _metrics(net: F, pos: F) -> dict[str, float]:
    finite = net[np.isfinite(net)]
    if finite.size < 2:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "total_return": 0.0, "turnover": 0.0, "exposure": 0.0}
    mean, sd = float(np.mean(finite)), float(np.std(finite, ddof=1))
    sharpe = mean / sd * np.sqrt(_ANN) if sd > 0 else 0.0
    equity = np.cumprod(1.0 + finite)
    dd = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    ann_ret = float(equity[-1] ** (_ANN / finite.size) - 1.0) if equity[-1] > 0 else -1.0
    turnover = float(np.mean(np.abs(np.diff(pos, prepend=pos[0])))) * _ANN
    return {
        "ann_return": ann_ret,
        "ann_vol": sd * np.sqrt(_ANN),
        "sharpe": float(sharpe),
        "max_drawdown": dd,
        "calmar": float(ann_ret / abs(dd)) if dd < 0 else 0.0,
        "total_return": float(equity[-1] - 1.0),
        "turnover": turnover,
        "exposure": float(np.mean(pos)),
    }


def backtest_positions(returns: F, positions: F, *, cost_oneway_bps: float = 1.0, delay: int = 0) -> BacktestResult:
    pos = positions.astype(np.float64).copy()
    for _ in range(delay):
        pos = _shift1(pos)
    turn = np.abs(np.diff(pos, prepend=pos[0]))
    cost = turn * (cost_oneway_bps * 1e-4)
    net = pos * returns - cost
    return BacktestResult(net_returns=net, metrics=_metrics(net, pos))
