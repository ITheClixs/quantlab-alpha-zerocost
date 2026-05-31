"""One-bar (or k-bar) execution-delay stress test.

Real fills don't happen at signal-observation close. If a signal loses
all its edge after a 1-bar delay, the apparent alpha was implementable
only with optimistic fill assumptions (or latent leakage from the same-
bar close into the signal).
"""

from __future__ import annotations

import polars as pl


def shift_signal_by_n_bars(
    signals: pl.DataFrame, *, n_bars: int = 1
) -> pl.DataFrame:
    """Shift y_xs_pred forward by n_bars within each symbol (i.e. signal
    observed on day T becomes the position applied on day T+n_bars).

    Rows missing a delayed value (the first n_bars per symbol) are dropped.
    """
    if "y_xs_pred" not in signals.columns:
        raise ValueError("signals must include y_xs_pred")
    if n_bars < 0:
        raise ValueError(f"n_bars must be >= 0, got {n_bars}")
    sorted_sig = signals.sort(["symbol", "date"])
    return (
        sorted_sig.with_columns(
            pl.col("y_xs_pred").shift(n_bars).over("symbol").alias("y_xs_pred")
        )
        .drop_nulls(subset=["y_xs_pred"])
        .select(["date", "symbol", "y_xs_pred"])
    )
