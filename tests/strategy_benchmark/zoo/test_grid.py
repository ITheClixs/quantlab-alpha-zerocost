from __future__ import annotations

from quant_research_stack.strategy_benchmark.zoo.grid import (
    DEFAULT_GRID,
    GridConfig,
    enumerate_zoo,
)


def test_default_grid_cardinality_and_uniqueness() -> None:
    specs = enumerate_zoo(universes=("U1", "U2"), grid=DEFAULT_GRID)
    expected = (
        2 * len(DEFAULT_GRID.families) * len(DEFAULT_GRID.lookbacks)
        * len(DEFAULT_GRID.thresholds) * len(DEFAULT_GRID.vol_estimators)
        * len(DEFAULT_GRID.position_modes) * len(DEFAULT_GRID.holdings)
    )
    assert len(specs) == expected
    assert len({s.strategy_id for s in specs}) == expected


def test_max_strategies_caps_deterministically() -> None:
    grid = GridConfig(max_strategies=100, seed=7)
    a = enumerate_zoo(universes=("U1", "U2"), grid=grid)
    b = enumerate_zoo(universes=("U1", "U2"), grid=grid)
    assert len(a) == 100 and [s.strategy_id for s in a] == [s.strategy_id for s in b]
