from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_research_stack.strategy_benchmark.zoo.vol_estimators import rolling_vol


def _bars(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pl.DataFrame({"date": list(range(n)), "symbol": ["X"] * n,
                         "open": open_, "high": high, "low": low, "close": close,
                         "volume": [1e6] * n})


@pytest.mark.parametrize("est", ["close_to_close", "parkinson", "rogers_satchell"])
def test_rolling_vol_positive_and_asof(est: str) -> None:
    out = rolling_vol(_bars(), window=20, estimator=est)
    assert isinstance(out, pl.Series)
    assert out.len() == 60
    assert out[:19].null_count() == 19
    defined = out.drop_nulls()
    assert (defined > 0).all() and np.isfinite(defined.to_numpy()).all()


def test_unknown_estimator_raises() -> None:
    with pytest.raises(ValueError):
        rolling_vol(_bars(), window=20, estimator="bogus")
