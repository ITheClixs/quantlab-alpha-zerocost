"""Random and inverted-signal sanity checks.

A strategy that doesn't materially beat a random_signal baseline is
indistinguishable from noise. A strategy that beats its own inverted
form is doing real cross-sectional ranking; one that loses to the
inverted form has a sign-flip bug or a structural anti-correlation
with the cost model.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def random_signal(
    bars: pl.DataFrame, *, seed: int = 1001
) -> pl.DataFrame:
    """Independent N(0,1) signal per (date, symbol)."""
    rng = np.random.default_rng(seed)
    df = bars.sort(["symbol", "date"])
    return df.select(["date", "symbol"]).with_columns(
        pl.Series("y_xs_pred", rng.standard_normal(df.height))
    )


def inverted_signal(strategy_signals: pl.DataFrame) -> pl.DataFrame:
    """Sign-flipped version of any strategy's predictions."""
    if "y_xs_pred" not in strategy_signals.columns:
        raise ValueError("strategy_signals must include y_xs_pred")
    return strategy_signals.with_columns(
        (-pl.col("y_xs_pred")).alias("y_xs_pred")
    ).select(["date", "symbol", "y_xs_pred"])
