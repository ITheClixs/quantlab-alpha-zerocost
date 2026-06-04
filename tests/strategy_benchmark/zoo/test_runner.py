from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.zoo.grid import GridConfig
from quant_research_stack.strategy_benchmark.zoo.runner import ZooResult, run_zoo


def _bars(symbol: str, n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pl.DataFrame({"date": list(range(n)), "symbol": [symbol] * n,
                         "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": [1e6] * n})


def test_run_zoo_shapes_and_split() -> None:
    universes = {"U1": _bars("U1", seed=1), "U2": _bars("U2", seed=2)}
    grid = GridConfig(families=("TS_MOMENTUM", "MA_CROSSOVER"), lookbacks=(10, 20),
                      thresholds=(1.0,), vol_estimators=("close_to_close",),
                      position_modes=("long_short",), holdings=(1,))
    res = run_zoo(universes=universes, grid=grid, oos_fraction=0.3, embargo_days=5)
    assert isinstance(res, ZooResult)
    n = 2 * 2 * 2 * 1 * 1 * 1 * 1
    assert res.metrics.height == n
    assert res.is_returns.shape[1] == n and res.oos_returns.shape[1] == n
    assert res.is_returns.shape[0] > res.oos_returns.shape[0]
    assert 0.0 <= res.pbo["pbo_probability"] <= 1.0
    assert {"strategy_id", "is_sharpe"}.issubset(set(res.metrics.columns))
