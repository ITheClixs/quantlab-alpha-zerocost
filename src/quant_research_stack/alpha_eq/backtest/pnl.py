"""Fill-aligned PnL accounting + cash dividend booking (spec §5.4).

The v1 invariant: portfolio MTM uses `tradable_*` prices (split-adjusted,
price-only); dividends are booked exactly once on ex-date as cash PnL.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class PnLDecomposition:
    gross_alpha_bps_per_day: float
    cash_dividend_bps_per_day: float
    commission_drag_bps_per_day: float
    spread_drag_bps_per_day: float
    borrow_drag_bps_per_day: float
    financing_drag_bps_per_day: float
    net_alpha_bps_per_day: float


def compute_position_price_pnl(
    *,
    held_positions: pl.DataFrame,
    new_lots: pl.DataFrame,
) -> pl.DataFrame:
    """Per-row price PnL (no dividends, no costs).

    Held positions: PnL = signed_notional_prev * (close_today / close_prev - 1).
    New lots:      PnL = signed_shares_new * (close_today - fill_price), where
                   signed_shares_new = signed_notional_new / fill_price.
    """
    if not held_positions.is_empty():
        held_pnl = held_positions.with_columns(
            (
                pl.col("signed_notional_prev")
                * (pl.col("close_today") / pl.col("close_prev") - 1.0)
            ).alias("price_pnl")
        ).select(["date", "symbol", "price_pnl"])
    else:
        held_pnl = pl.DataFrame(
            schema={"date": pl.Date, "symbol": pl.Utf8, "price_pnl": pl.Float64}
        )

    if not new_lots.is_empty():
        new_pnl = new_lots.with_columns(
            (
                (pl.col("signed_notional_new") / pl.col("fill_price"))
                * (pl.col("close_today") - pl.col("fill_price"))
            ).alias("price_pnl")
        ).select(["date", "symbol", "price_pnl"])
    else:
        new_pnl = pl.DataFrame(
            schema={"date": pl.Date, "symbol": pl.Utf8, "price_pnl": pl.Float64}
        )

    return pl.concat([held_pnl, new_pnl])


def compute_cash_dividend_pnl(
    *,
    positions_on_ex_date: pl.DataFrame,
    dividends: pl.DataFrame,
) -> pl.DataFrame:
    """Cash dividend PnL = signed_shares * dividend_per_share.

    Longs receive; shorts are debited.  `signed_shares = signed_notional /
    ref_close` where `ref_close` is the prior trading day's close.
    """
    if dividends.is_empty() or positions_on_ex_date.is_empty():
        return pl.DataFrame(
            schema={"date": pl.Date, "symbol": pl.Utf8, "cash_dividend_pnl": pl.Float64}
        )
    divs = dividends.rename({"ex_date": "date"})
    joined = positions_on_ex_date.join(divs, on=["date", "symbol"], how="inner")
    return joined.with_columns(
        (
            (pl.col("signed_notional") / pl.col("ref_close"))
            * pl.col("dividend_per_share")
        ).alias("cash_dividend_pnl")
    ).select(["date", "symbol", "cash_dividend_pnl"])


def decompose_pnl(
    *,
    price_pnl: pl.DataFrame,
    cash_dividend_pnl: pl.DataFrame,
    commission_drag: pl.DataFrame,
    spread_drag: pl.DataFrame,
    borrow_drag: pl.DataFrame,
    financing_drag: pl.DataFrame,
    equity: float,
    n_days: int,
) -> PnLDecomposition:
    def _per_day_bps(frame: pl.DataFrame, col: str) -> float:
        if frame.is_empty() or n_days == 0 or equity == 0:
            return 0.0
        return float(frame[col].sum()) / float(equity) / float(n_days) * 10_000.0

    gross_bps = _per_day_bps(price_pnl, "price_pnl") + _per_day_bps(
        cash_dividend_pnl, "cash_dividend_pnl"
    )
    div_bps = _per_day_bps(cash_dividend_pnl, "cash_dividend_pnl")
    comm_bps = _per_day_bps(commission_drag, "commission_drag")
    spread_bps = _per_day_bps(spread_drag, "spread_drag")
    borrow_bps = _per_day_bps(borrow_drag, "borrow_drag")
    fin_bps = _per_day_bps(financing_drag, "financing_drag")
    net_bps = gross_bps - comm_bps - spread_bps - borrow_bps - fin_bps
    return PnLDecomposition(
        gross_alpha_bps_per_day=gross_bps,
        cash_dividend_bps_per_day=div_bps,
        commission_drag_bps_per_day=comm_bps,
        spread_drag_bps_per_day=spread_bps,
        borrow_drag_bps_per_day=borrow_bps,
        financing_drag_bps_per_day=fin_bps,
        net_alpha_bps_per_day=net_bps,
    )
