from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.zoo.grid import GridConfig
from quant_research_stack.strategy_benchmark.zoo.permutation import (
    permutation_control, permute_prices,
)


def _bars(symbol: str, n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    return pl.DataFrame({"date": list(range(n)), "symbol": [symbol] * n,
                         "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": [1e6] * n})


def test_permute_prices_preserves_length_and_breaks_order() -> None:
    bars = _bars("U1", seed=1)
    out = permute_prices(bars, rng=np.random.default_rng(0))
    assert out.height == bars.height
    assert set(out.columns) == set(bars.columns)
    # same multiset of close-to-close returns (distribution preserved), different order
    r0 = (bars["close"].log().diff().drop_nulls().sort().to_numpy())
    r1 = (out["close"].log().diff().drop_nulls().sort().to_numpy())
    assert np.allclose(r0, r1, atol=1e-9)
    assert not bars["close"].equals(out["close"])  # order changed


def test_mcpt_zero_skill_pool_not_significant() -> None:
    universes = {"U1": _bars("U1", seed=1), "U2": _bars("U2", seed=2)}
    grid = GridConfig(families=("TS_MOMENTUM", "MA_CROSSOVER"), lookbacks=(10, 20),
                      thresholds=(1.0,), vol_estimators=("close_to_close",),
                      position_modes=("long_short",), holdings=(1,))
    out = permutation_control(universes=universes, grid=grid, n_permutations=3, seed=7)
    assert "real_best_sharpe" in out and "permuted_best_sharpe_mean" in out
    assert out["n_permutations"] == 3 and out["seed"] == 7
    assert 0.0 <= out["p_value"] <= 1.0
    # random-walk prices have no exploitable structure → real best is NOT significant
    assert out["p_value"] > 0.05
