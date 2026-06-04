"""Configurable strategy grid -> up to ~100k single-asset configurations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES


@dataclass(frozen=True)
class GridConfig:
    families: tuple[str, ...] = field(default_factory=lambda: tuple(sorted(SIGNAL_FAMILIES.keys())))
    lookbacks: tuple[int, ...] = (5, 10, 20, 40, 60, 120, 180, 252)
    thresholds: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    vol_estimators: tuple[str, ...] = ("close_to_close", "parkinson", "rogers_satchell")
    position_modes: tuple[str, ...] = ("long_only", "long_short")
    holdings: tuple[int, ...] = (1, 5, 10)
    max_strategies: int | None = None
    seed: int = 42


DEFAULT_GRID = GridConfig()


@dataclass(frozen=True)
class ZooStrategySpec:
    strategy_id: str
    universe: str
    family: str
    lookback: int
    threshold: float
    vol_estimator: str
    position_mode: str
    holding: int


def enumerate_zoo(*, universes: Iterable[str], grid: GridConfig) -> list[ZooStrategySpec]:
    specs: list[ZooStrategySpec] = []
    for u, f, lb, th, ve, pm, hd in product(
        sorted(universes), grid.families, grid.lookbacks, grid.thresholds,
        grid.vol_estimators, grid.position_modes, grid.holdings,
    ):
        sid = f"{u}|{f}|L{lb}|T{th:.2f}|{ve}|{pm}|H{hd}"
        specs.append(ZooStrategySpec(sid, u, f, lb, th, ve, pm, hd))
    if grid.max_strategies is not None and len(specs) > grid.max_strategies:
        rng = np.random.default_rng(grid.seed)
        idx = np.sort(rng.choice(len(specs), size=grid.max_strategies, replace=False))
        specs = [specs[i] for i in idx]
    return specs
