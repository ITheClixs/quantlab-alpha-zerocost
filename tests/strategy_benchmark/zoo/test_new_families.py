from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES

_NEW = ["VOLMANAGED_MOMENTUM", "EWMA_CROSS", "ATR_TRAILING_TREND",
        "ROLLING_SHARPE_MOM", "RANGE_OSCILLATOR", "MOM_SKIP"]


def _bars(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pl.DataFrame({"date": list(range(n)), "symbol": ["X"] * n,
                         "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": [1e6] * n})


def test_new_families_registered_and_valid() -> None:
    bars = _bars()
    for name in _NEW:
        assert name in SIGNAL_FAMILIES, f"{name} not registered"
        s = SIGNAL_FAMILIES[name](bars, lookback=20, threshold=1.0)
        assert isinstance(s, pl.Series) and s.len() == bars.height
        finite = s.drop_nulls().drop_nans()
        assert finite.len() > 0
        assert finite.abs().max() <= 5.0
