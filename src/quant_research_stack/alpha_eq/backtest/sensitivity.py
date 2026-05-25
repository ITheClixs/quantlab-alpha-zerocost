"""Standard + audit sensitivity packs (spec §5.14)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

from quant_research_stack.alpha_eq.backtest.fills import FillModel


@dataclass(frozen=True)
class SensitivityCase:
    borrow_multiplier: float
    fill_model: FillModel
    q_quantile: float
    target_gross: float
    adv_participation_pct: float = 0.01


def enumerate_standard_pack() -> Iterator[SensitivityCase]:
    borrow = (1.0, 3.0)
    fills = (FillModel.OPEN, FillModel.HLC3_PROXY)
    qs = (0.05, 0.10)
    gross = (1.0,)
    for b, f, q, g in product(borrow, fills, qs, gross):
        yield SensitivityCase(borrow_multiplier=b, fill_model=f, q_quantile=q, target_gross=g)


def enumerate_audit_pack() -> Iterator[SensitivityCase]:
    borrow = (1.0, 2.0, 3.0)
    fills = (FillModel.OPEN, FillModel.HLC3_PROXY, FillModel.CLOSE)
    qs = (0.05, 0.10)
    gross = (0.5, 1.0, 2.0)
    adv = (0.01, 0.03)
    for b, f, q, g, a in product(borrow, fills, qs, gross, adv):
        yield SensitivityCase(
            borrow_multiplier=b,
            fill_model=f,
            q_quantile=q,
            target_gross=g,
            adv_participation_pct=a,
        )
