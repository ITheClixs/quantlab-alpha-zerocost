from __future__ import annotations

import math
from dataclasses import dataclass

from quant_research_stack.execution.configs import RiskConfig
from quant_research_stack.execution.types import ExecutionTicket
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


@dataclass(frozen=True)
class SizerInput:
    ticket: ExecutionTicket
    account_equity: float
    mid_price: float
    lot_size: float


def _stance_modifier(primary: GovernorVerdict, tier3: GovernorVerdict | None, cfg_pct: float) -> float:
    if tier3 is None or tier3.decision == Decision.insufficient_evidence:
        return 0.0
    if tier3.decision == Decision.veto:
        return -cfg_pct
    if int(tier3.direction) == int(primary.direction):
        return cfg_pct
    return -cfg_pct


def _round_to_lot(qty: float, lot_size: float) -> float:
    if lot_size <= 0:
        return qty
    return math.copysign(math.floor(abs(qty) / lot_size) * lot_size, qty)


class Sizer:
    """Confidence-scaled position sizing with hard caps (ADR-0012)."""

    def __init__(self, cfg: RiskConfig, tier3_stance_pct: float = 0.20) -> None:
        self._cfg = cfg
        self._stance_pct = float(tier3_stance_pct)

    def size(self, inp: SizerInput) -> float:
        primary = inp.ticket.primary_verdict
        if primary.decision != Decision.pass_:
            return 0.0
        direction = int(primary.direction)
        if direction == 0:
            return 0.0
        if inp.mid_price <= 0 or inp.account_equity <= 0:
            return 0.0

        stance_mod = _stance_modifier(primary, inp.ticket.tier3_verdict, self._stance_pct)
        base_pct = self._cfg.limits.base_notional_per_trade_pct
        target_notional = inp.account_equity * base_pct * primary.confidence * (1.0 + stance_mod)
        per_symbol_cap = inp.account_equity * self._cfg.limits.max_per_symbol_pct
        target_notional = min(target_notional, per_symbol_cap)
        qty = direction * target_notional / inp.mid_price
        return _round_to_lot(qty, inp.lot_size)
