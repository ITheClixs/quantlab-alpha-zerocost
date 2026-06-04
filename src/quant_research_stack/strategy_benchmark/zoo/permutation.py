"""Monte-Carlo Permutation Test (MCPT) for the strategy zoo. Permute each universe's
price path (shuffle the *order* of daily return + intraday-shape vectors, preserving the
distribution), re-run the same grid, and compare the best in-sample Sharpe to the real
run. If the real best is not significantly larger, the winner is a noise/selection
artifact rather than exploited time-structure. research_only."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.zoo.grid import GridConfig
from quant_research_stack.strategy_benchmark.zoo.runner import run_zoo


def permute_prices(bars: pl.DataFrame, *, rng: np.random.Generator) -> pl.DataFrame:
    """Per symbol: shuffle the order of daily (close-to-close return, intraday OHLC
    ratios) vectors and rebuild a coherent synthetic OHLCV path. Breaks serial
    dependence; preserves the return distribution and intraday shape."""
    out_frames: list[pl.DataFrame] = []
    for _, g in bars.sort(["symbol", "date"]).group_by("symbol", maintain_order=True):
        c = g["close"].to_numpy().astype(np.float64)
        o = g["open"].to_numpy().astype(np.float64)
        h = g["high"].to_numpy().astype(np.float64)
        low = g["low"].to_numpy().astype(np.float64)
        n = c.size
        # Compute the n-1 actual log returns (index 1..n-1)
        actual_ret = np.log(c[1:] / c[:-1])
        # intraday ratios relative to that day's close
        hi_r = h / c
        lo_r = low / c
        op_r = o / c
        # Permute only the n-1 actual returns; day-0 level stays anchored
        perm_ret = rng.permutation(n - 1)
        # Build a full-length permutation index for intraday ratios:
        # day 0 keeps its own shape; days 1..n-1 get shuffled shapes
        perm_shape = np.empty(n, dtype=np.intp)
        perm_shape[0] = 0
        perm_shape[1:] = perm_ret + 1  # shift back to original indices
        new_ret = np.zeros(n)
        new_ret[1:] = actual_ret[perm_ret]
        new_close = c[0] * np.exp(np.cumsum(new_ret))  # rebuild level path
        new_high = new_close * hi_r[perm_shape]
        new_low = new_close * lo_r[perm_shape]
        new_open = new_close * op_r[perm_shape]
        out_frames.append(g.with_columns(
            pl.Series("open", new_open), pl.Series("high", new_high),
            pl.Series("low", new_low), pl.Series("close", new_close),
        ))
    return pl.concat(out_frames, how="vertical")


def _best_is_sharpe(universes: dict[str, pl.DataFrame], grid: GridConfig) -> float:
    res = run_zoo(universes=universes, grid=grid, oos_fraction=0.3, embargo_days=5)
    col = res.metrics["is_sharpe"].to_numpy()
    return float(np.max(col)) if col.size else 0.0


def permutation_control(*, universes: dict[str, pl.DataFrame], grid: GridConfig,
                        n_permutations: int = 5, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    real_best = _best_is_sharpe(universes, grid)
    perm_bests: list[float] = []
    for _ in range(n_permutations):
        permuted = {name: permute_prices(bars, rng=rng) for name, bars in universes.items()}
        perm_bests.append(_best_is_sharpe(permuted, grid))
    arr = np.asarray(perm_bests, dtype=np.float64)
    p_value = float((np.sum(arr >= real_best) + 1.0) / (n_permutations + 1.0))  # +1 smoothing
    return {"real_best_sharpe": real_best, "permuted_best_sharpe_mean": float(arr.mean()),
            "permuted_best_sharpes": perm_bests, "p_value": p_value,
            "n_permutations": n_permutations, "seed": seed}
