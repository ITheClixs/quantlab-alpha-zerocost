"""Shared zero-cost pipeline: alignment + macro state + weight builders.

Single source of truth for the frozen `zero_cost_riskalloc_v1` construction, reused
by both the v1 runner and the strict-review runner so they cannot drift. No tuning,
no new features — this only factors the v1 logic for reuse.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np

from quant_research_stack.signal_research.zero_cost.data import (
    INSTRUMENTS,
    MACRO_REGISTRY,
    load_instrument,
    load_macro,
)
from quant_research_stack.signal_research.zero_cost.strategy import (
    InstrumentSeries,
    daily_returns,
    trend_on,
    vol_regime_on,
    vol_target_weight,
)

COST: dict[str, float] = {"SPY": 1.0, "QQQ": 1.0, "BTCUSDT": 8.0, "ETHUSDT": 8.0}
CAP: dict[str, float] = {"SPY": 1.5, "QQQ": 1.5, "BTCUSDT": 1.0, "ETHUSDT": 1.0}
_F = np.ndarray


def _roll_mean(x: _F, w: int) -> _F:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(len(x)):
        if i + 1 >= w:
            out[i] = float(np.mean(x[i + 1 - w : i + 1]))
    return out


def aligned(instruments: tuple[str, ...] = tuple(INSTRUMENTS)
            ) -> tuple[list[date], dict[str, InstrumentSeries], dict[str, _F]]:
    """Inner-join instruments on the SPY equity calendar; left-join macro; forward-fill macro."""
    frames = {n: load_instrument(n).rename({"close": n}) for n in instruments}
    master = frames[instruments[0]]
    for n in instruments[1:]:
        master = master.join(frames[n], on="date", how="inner")
    for s in MACRO_REGISTRY:
        m = load_macro(s)
        if m.height:
            master = master.join(m.rename({"close": s.name}), on="date", how="left")
    master = master.sort("date")
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in master["date"].to_list()]
    insts: dict[str, InstrumentSeries] = {}
    for n in instruments:
        close = master[n].to_numpy().astype(np.float64)
        insts[n] = InstrumentSeries(n, dates, close, daily_returns(close), n.endswith("USDT"))
    macro = {s.name: (master[s.name].fill_null(strategy="forward").to_numpy().astype(np.float64)
                      if s.name in master.columns else np.full(master.height, np.nan))
             for s in MACRO_REGISTRY}
    return dates, insts, macro


def macro_risk_on(macro: dict[str, _F], n: int) -> _F:
    vix, vix3m = macro.get("vix"), macro.get("vix3m")
    hyg = macro.get("credit_hyg")
    u10, u2 = macro.get("ust10y"), macro.get("ust2y")
    contango = (vix < vix3m) if vix is not None and vix3m is not None else np.ones(n, bool)
    credit_on = (hyg > _roll_mean(hyg, 100)) if hyg is not None else np.ones(n, bool)
    slope_ok = ((u10 - u2) > -1.0) if u10 is not None and u2 is not None else np.ones(n, bool)
    return (np.nan_to_num(contango, nan=1.0).astype(bool)
            & np.nan_to_num(credit_on, nan=1.0).astype(bool)
            & np.nan_to_num(slope_ok, nan=1.0).astype(bool))


def strategy_weights(insts: dict[str, InstrumentSeries], macro_on: _F) -> dict[str, _F]:
    out = {}
    for n, s in insts.items():
        vt = vol_target_weight(s.returns, target_ann_vol=0.12, lookback=20, cap=CAP[n])
        gate = trend_on(s.close, slow=200) & vol_regime_on(s.returns) & macro_on
        out[n] = vt * gate.astype(np.float64)
    return out


def voltarget_weights(insts: dict[str, InstrumentSeries]) -> dict[str, _F]:
    return {n: vol_target_weight(s.returns, target_ann_vol=0.12, lookback=20, cap=CAP[n]) for n, s in insts.items()}


def bah_weights(insts: dict[str, InstrumentSeries]) -> dict[str, _F]:
    return {n: np.ones(len(s.dates)) for n, s in insts.items()}


def trend_weights(insts: dict[str, InstrumentSeries]) -> dict[str, _F]:
    return {n: trend_on(s.close, slow=200).astype(np.float64) for n, s in insts.items()}


def regime_weights(insts: dict[str, InstrumentSeries]) -> dict[str, _F]:
    return {n: vol_regime_on(s.returns).astype(np.float64) for n, s in insts.items()}


def inverted_weights(insts: dict[str, InstrumentSeries], macro_on: _F) -> dict[str, _F]:
    """Sanity: invert the strategy gate (risk-on when the strategy is risk-off)."""
    out = {}
    base = strategy_weights(insts, macro_on)
    vt = voltarget_weights(insts)
    for n in insts:
        gate_off = (base[n] <= 0).astype(np.float64)
        out[n] = vt[n] * gate_off
    return out
