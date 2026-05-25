"""Trade-cost model (spec §5.6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class CostConfig:
    commission_bps_one_way: float = 0.5
    roll_spread_cap_bps: float = 50.0
    tiered_fallback_easy_bps: float = 5.0
    tiered_fallback_general_bps: float = 15.0
    tiered_fallback_hard_bps: float = 50.0
    pre_decimalization_cutoff: date = date(2001, 4, 9)
    pre_decimalization_multiplier_fallback: float = 2.5
    pre_decimalization_multiplier_roll: float = 1.5


def compute_commission_drag(trades: pl.DataFrame, *, cost: CostConfig) -> pl.DataFrame:
    return trades.with_columns(
        (pl.col("trade_notional_abs") * cost.commission_bps_one_way / 10_000.0).alias(
            "commission_drag"
        )
    )


def compute_spread_drag(trades: pl.DataFrame, *, cost: CostConfig) -> pl.DataFrame:
    tiered = (
        pl.when(pl.col("tier") == "easy").then(cost.tiered_fallback_easy_bps)
        .when(pl.col("tier") == "hard").then(cost.tiered_fallback_hard_bps)
        .otherwise(cost.tiered_fallback_general_bps)
    )
    raw_spread = pl.when(pl.col("roll_spread_bps").is_not_null()).then(
        pl.min_horizontal(pl.col("roll_spread_bps"), pl.lit(cost.roll_spread_cap_bps))
    ).otherwise(tiered)

    pre_decimal_mult = pl.when(pl.col("date") < cost.pre_decimalization_cutoff).then(
        pl.when(pl.col("roll_spread_bps").is_not_null())
        .then(cost.pre_decimalization_multiplier_roll)
        .otherwise(cost.pre_decimalization_multiplier_fallback)
    ).otherwise(1.0)

    half_spread_bps = (raw_spread * pre_decimal_mult) / 2.0
    return trades.with_columns(
        (pl.col("trade_notional_abs") * half_spread_bps / 10_000.0).alias("spread_drag")
    )
