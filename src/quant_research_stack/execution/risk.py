from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_research_stack.execution.configs import RiskConfig
from quant_research_stack.execution.types import ExecutionTicket
from quant_research_stack.governor.signal_schema import Decision


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    kill_trigger: bool
    reason: str


@dataclass
class RiskState:
    """Snapshot of the world the RiskGate evaluates against."""

    account_equity: float
    peak_equity: float
    daily_realized_pnl: float
    gross_exposure_notional: float
    per_symbol_notional: dict[str, float]
    orders_last_minute: int
    last_tick_ts: dict[str, datetime]
    kill_flag_path: Path
    is_crypto: Callable[[str], bool]
    now: datetime


def kill_flag_check(_ticket: ExecutionTicket, state: RiskState, _cfg: RiskConfig) -> tuple[bool, bool]:
    if state.kill_flag_path.exists():
        return (False, True)
    return (True, False)


def feed_freshness_check(ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    last = state.last_tick_ts.get(ticket.signal.symbol)
    if last is None:
        return (False, True)
    gap_s = (state.now - last).total_seconds()
    threshold = (
        cfg.freshness.crypto_max_gap_seconds
        if state.is_crypto(ticket.signal.symbol)
        else cfg.freshness.equity_max_gap_seconds
    )
    if gap_s > threshold:
        return (False, True)
    return (True, False)


def drawdown_check(_ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    if state.account_equity <= 0:
        return (False, True)
    daily_dd_pct = -state.daily_realized_pnl / state.account_equity if state.daily_realized_pnl < 0 else 0.0
    if daily_dd_pct > cfg.drawdown.daily_realized_dd_kill_pct:
        return (False, True)
    cum_dd_pct = (state.peak_equity - state.account_equity) / state.peak_equity if state.peak_equity > 0 else 0.0
    if cum_dd_pct > cfg.drawdown.cumulative_dd_kill_pct:
        return (False, True)
    return (True, False)


def exposure_check(ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    equity = state.account_equity
    if equity <= 0:
        return (False, False)
    gross_cap = equity * cfg.limits.max_gross_exposure_pct
    per_symbol_cap = equity * cfg.limits.max_per_symbol_pct
    if state.gross_exposure_notional >= gross_cap:
        return (False, False)
    if state.per_symbol_notional.get(ticket.signal.symbol, 0.0) >= per_symbol_cap:
        return (False, False)
    return (True, False)


def rate_limit_check(_ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    if state.orders_last_minute >= cfg.limits.max_orders_per_minute:
        return (False, False)
    return (True, False)


def governor_decision_check(ticket: ExecutionTicket, _state: RiskState, _cfg: RiskConfig) -> tuple[bool, bool]:
    if ticket.primary_verdict.decision != Decision.pass_:
        return (False, False)
    return (True, False)


# Per ADR-0014: kill_flag_check MUST be first. The unit test
# test_gate_order_is_kill_first enforces this invariant.
_GATES = [
    kill_flag_check,
    feed_freshness_check,
    drawdown_check,
    exposure_check,
    rate_limit_check,
    governor_decision_check,
]


class RiskGate:
    """Ordered pre-trade check chain."""

    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg

    def evaluate(self, ticket: ExecutionTicket, state: RiskState) -> RiskDecision:
        for gate in _GATES:
            allowed, kill = gate(ticket, state, self._cfg)
            if not allowed:
                return RiskDecision(allowed=False, kill_trigger=kill, reason=gate.__name__)
        return RiskDecision(allowed=True, kill_trigger=False, reason="")
