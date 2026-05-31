"""Pure conversion + validation (spec §5.2, §5.10).

NO training, NO tuning, NO backtesting. Just:
- schema validation
- feature_as_of_date < execution_date
- one prediction per (date, symbol)
- NaN handling
- rank-within-tradable-universe
- M4-compatible output
"""

from __future__ import annotations

import polars as pl


class BridgeSchemaError(RuntimeError):
    pass


class BridgeContractError(RuntimeError):
    pass


_REQUIRED_INPUT_COLS: tuple[str, ...] = (
    "date",
    "symbol",
    "feature_as_of_date",
    "execution_date",
    "y_xs_pred",
    "tradable",
    "in_pit_universe",
)


def signal_to_panel(predictions: pl.DataFrame, *, drop_nan: bool = True) -> pl.DataFrame:
    missing = [c for c in _REQUIRED_INPUT_COLS if c not in predictions.columns]
    if missing:
        raise BridgeSchemaError(f"missing required columns: {missing}")

    bad = predictions.filter(pl.col("feature_as_of_date") >= pl.col("execution_date"))
    if not bad.is_empty():
        raise BridgeContractError(
            f"feature_as_of_date >= execution_date on {bad.height} rows"
        )

    duplicate_count = (
        predictions.group_by(["date", "symbol"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count > 0:
        raise BridgeContractError(f"{duplicate_count} (date, symbol) rows duplicated")

    out = predictions
    if drop_nan:
        out = out.drop_nulls(subset=["y_xs_pred"])

    eligible = out.filter(pl.col("tradable") & pl.col("in_pit_universe"))
    n_per_date = eligible.group_by("date").len().rename({"len": "_n"})
    ranks = (
        eligible.with_columns(
            pl.col("y_xs_pred")
            .rank(method="ordinal")
            .over("date")
            .alias("_rank")
        )
        .join(n_per_date, on="date", how="left")
        .with_columns(
            pl.when(pl.col("_n") > 1)
            .then(
                (pl.col("_rank") - 1.0)
                / (pl.col("_n").cast(pl.Float64) - 1.0)
                - 0.5
            )
            .otherwise(None)
            .alias("y_xs_pred_rank")
        )
        .drop(["_rank", "_n"])
    )
    return out.join(
        ranks.select(["date", "symbol", "y_xs_pred_rank"]),
        on=["date", "symbol"],
        how="left",
    )
