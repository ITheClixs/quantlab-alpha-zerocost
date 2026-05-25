"""Portfolio construction for the strict backtest (spec §5.5)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


class PortfolioConstructionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PortfolioBuildConfig:
    q_quantile: float
    target_gross: float
    equity: float
    adv_participation_pct: float = 0.01
    min_long_full_universe: int = 10
    min_short_full_universe: int = 10
    min_long_focused_basket: int = 5
    min_short_focused_basket: int = 5
    max_single_name_weight_frac_of_gross: float = 0.05


def _bucket_size(n_total: int, q: float, *, min_required: int) -> int:
    raw = max(1, int(round(n_total * q)))
    return max(raw, min_required)


def _empty_book(signals: pl.DataFrame) -> pl.DataFrame:
    return signals.select(["execution_date", "symbol"]).with_columns(
        pl.lit(0.0).alias("signed_target_notional"),
        pl.lit("").alias("borrow_tier"),
        pl.lit(0.0).alias("fill_price"),
    ).head(0)


def build_target_positions(
    *,
    signals: pl.DataFrame,
    config: PortfolioBuildConfig,
    cohort: str,
) -> pl.DataFrame:
    """Return per-name signed target notional for one execution date.

    Empty DataFrame returned if minimum-bucket cannot be met or if no
    tradable/in-universe rows exist (caller skips the date).
    """
    eligible = signals.filter(pl.col("tradable") & pl.col("in_pit_universe"))
    if eligible.is_empty():
        return _empty_book(signals)

    sorted_sig = eligible.sort("y_xs_pred")
    n = sorted_sig.height
    min_long = (
        config.min_long_full_universe
        if cohort == "full_universe"
        else config.min_long_focused_basket
    )
    min_short = (
        config.min_short_full_universe
        if cohort == "full_universe"
        else config.min_short_focused_basket
    )

    short_size = _bucket_size(n, config.q_quantile, min_required=min_short)
    long_size = _bucket_size(n, config.q_quantile, min_required=min_long)
    if short_size + long_size > n or short_size < min_short or long_size < min_long:
        return _empty_book(signals)

    shorts = sorted_sig.head(short_size)
    longs = sorted_sig.tail(long_size)

    gross_dollars = config.target_gross * config.equity
    per_side_dollars = gross_dollars / 2.0
    equal_long = per_side_dollars / float(long_size)
    equal_short = per_side_dollars / float(short_size)
    weight_cap = gross_dollars * config.max_single_name_weight_frac_of_gross

    def _cap(side: pl.DataFrame, equal_weight: float, sign: int) -> pl.DataFrame:
        adv_cap = pl.col("adv_20d_dollar_lag1") * float(config.adv_participation_pct)
        capped_abs = pl.min_horizontal(
            pl.lit(equal_weight),
            adv_cap,
            pl.lit(float(weight_cap)),
        )
        return side.with_columns(
            (sign * capped_abs).alias("signed_target_notional"),
        )

    book = pl.concat(
        [
            _cap(longs, equal_long, +1),
            _cap(shorts, equal_short, -1),
        ]
    )
    return book.select(
        ["execution_date", "symbol", "signed_target_notional", "borrow_tier", "fill_price"]
    )
