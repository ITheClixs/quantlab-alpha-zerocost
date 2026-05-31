"""Enumerate the 1500-strategy menu.

5 universes × 15 signal families × 4 lookbacks × 5 thresholds = 1500.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES

LOOKBACKS: tuple[int, ...] = (10, 20, 60, 120)
THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str  # canonical, used as column name and PBO key
    universe: str
    signal_family: str
    lookback: int
    threshold: float


def enumerate_strategies(universes: Iterable[str]) -> list[StrategySpec]:
    """Generate every (universe × family × lookback × threshold) tuple."""
    out: list[StrategySpec] = []
    families = sorted(SIGNAL_FAMILIES.keys())
    for u, f, lb, th in product(sorted(universes), families, LOOKBACKS, THRESHOLDS):
        sid = f"{u}|{f}|L{lb}|T{th:.2f}"
        out.append(
            StrategySpec(
                strategy_id=sid,
                universe=u,
                signal_family=f,
                lookback=lb,
                threshold=th,
            )
        )
    return out
