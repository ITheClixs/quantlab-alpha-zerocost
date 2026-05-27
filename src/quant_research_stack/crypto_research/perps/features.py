from __future__ import annotations

import polars as pl


def _required_columns(frame: pl.DataFrame) -> None:
    required = {"symbol", "event_time", "best_bid", "best_ask", "best_bid_size", "best_ask_size"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required L1 columns: {sorted(missing)}")


def build_l1_features(
    book_ticker: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 15, 60, 300),
    rolling_windows: tuple[int, ...] = (10, 50, 200),
) -> pl.DataFrame:
    _required_columns(book_ticker)
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"horizons must be positive; got {horizon}")
    for window in rolling_windows:
        if window <= 0:
            raise ValueError(f"rolling_windows must be positive; got {window}")

    out = (
        book_ticker.sort(["symbol", "event_time"])
        .with_columns(
            [
                pl.col("best_bid").cast(pl.Float64),
                pl.col("best_ask").cast(pl.Float64),
                pl.col("best_bid_size").cast(pl.Float64),
                pl.col("best_ask_size").cast(pl.Float64),
            ]
        )
        .with_columns(
            [
                ((pl.col("best_bid") + pl.col("best_ask")) / 2.0).alias("mid_price"),
                (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
                (pl.col("best_bid_size") + pl.col("best_ask_size")).alias("top_size"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("top_size") > 0.0)
                .then((pl.col("best_bid_size") - pl.col("best_ask_size")) / pl.col("top_size"))
                .otherwise(None)
                .alias("l1_imbalance"),
                pl.when(pl.col("top_size") > 0.0)
                .then(
                    ((pl.col("best_ask") * pl.col("best_bid_size")) + (pl.col("best_bid") * pl.col("best_ask_size")))
                    / pl.col("top_size")
                )
                .otherwise(None)
                .alias("microprice"),
                (pl.col("spread") / pl.col("mid_price")).alias("relative_spread"),
                (pl.col("mid_price") / pl.col("mid_price").shift(1).over("symbol") - 1.0).alias("mid_return_1"),
            ]
        )
        .with_columns((pl.col("microprice") / pl.col("mid_price") - 1.0).alias("microprice_deviation"))
    )

    rolling_exprs: list[pl.Expr] = []
    for window in sorted(set(rolling_windows)):
        rolling_exprs.extend(
            [
                pl.col("mid_return_1")
                .rolling_std(window_size=window, min_samples=2)
                .over("symbol")
                .alias(f"realized_vol_{window}"),
                pl.lit(1)
                .rolling_sum(window_size=window, min_samples=1)
                .over("symbol")
                .alias(f"event_count_{window}"),
            ]
        )
    if rolling_exprs:
        out = out.with_columns(rolling_exprs)

    label_exprs: list[pl.Expr] = []
    for horizon in sorted(set(horizons)):
        future_mid = pl.col("mid_price").shift(-horizon).over("symbol")
        future_bid = pl.col("best_bid").shift(-horizon).over("symbol")
        future_ask = pl.col("best_ask").shift(-horizon).over("symbol")
        label_exprs.extend(
            [
                (future_mid / pl.col("mid_price") - 1.0).alias(f"future_mid_return_{horizon}"),
                future_bid.alias(f"future_best_bid_{horizon}"),
                future_ask.alias(f"future_best_ask_{horizon}"),
                (future_bid / pl.col("best_ask") - 1.0).alias(f"future_taker_long_return_{horizon}"),
                (pl.col("best_bid") / future_ask - 1.0).alias(f"future_taker_short_return_{horizon}"),
                (future_mid > pl.col("mid_price")).cast(pl.Int8).alias(f"mid_direction_up_{horizon}"),
            ]
        )
    return out.with_columns(label_exprs)
