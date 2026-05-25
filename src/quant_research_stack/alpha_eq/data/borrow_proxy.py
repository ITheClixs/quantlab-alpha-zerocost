"""Borrow proxy (spec §2.4): static 3-tier table + date-aware upgrades.

Borrow charges apply only to SHORT notional and are stress-tested at 1x/2x/3x
in every backtest report.
"""

from __future__ import annotations

import enum
from datetime import date, timedelta

import polars as pl

BORROW_BPS_EASY: int = 25
BORROW_BPS_GENERAL: int = 100
BORROW_BPS_HARD: int = 500

_LOW_ADV_THRESHOLD_USD = 5_000_000.0
_HIGH_VOL_THRESHOLD = 0.80
_LOW_PRICE_THRESHOLD = 5.0
_HIGH_SI_THRESHOLD = 0.10
_RECENT_IPO_WINDOW = timedelta(days=183)  # ~6 months


class BorrowTier(enum.StrEnum):
    EASY = "easy"
    GENERAL = "general"
    HARD = "hard"


def classify_borrow_tier(
    *,
    symbol: str,
    on: date,
    ipo_date: date | None,
    dollar_adv: float | None,
    realized_vol_20: float | None,
    price: float | None,
    recent_index_addition: bool,
    short_interest_ratio: float | None,
    manual_hard_override: bool,
) -> BorrowTier:
    if manual_hard_override:
        return BorrowTier.HARD
    if ipo_date is not None and (on - ipo_date) < _RECENT_IPO_WINDOW:
        return BorrowTier.HARD
    if dollar_adv is not None and dollar_adv < _LOW_ADV_THRESHOLD_USD:
        return BorrowTier.HARD
    if realized_vol_20 is not None and realized_vol_20 > _HIGH_VOL_THRESHOLD:
        return BorrowTier.HARD
    if price is not None and price < _LOW_PRICE_THRESHOLD:
        return BorrowTier.HARD
    if short_interest_ratio is not None and short_interest_ratio > _HIGH_SI_THRESHOLD:
        return BorrowTier.HARD
    # Recent index addition alone is a watchlist flag, not a tier upgrade.
    if recent_index_addition and (
        (dollar_adv is not None and dollar_adv < _LOW_ADV_THRESHOLD_USD * 2)
        or (realized_vol_20 is not None and realized_vol_20 > _HIGH_VOL_THRESHOLD / 2)
    ):
        return BorrowTier.HARD
    return BorrowTier.GENERAL


def build_borrow_proxy(symbols: list[str]) -> pl.DataFrame:
    """Static v1 borrow proxy: every symbol starts at GENERAL until the
    date-aware classifier upgrades them on a given date."""
    return pl.DataFrame(
        {
            "symbol": list(symbols),
            "borrow_tier": ["general"] * len(symbols),
            "annual_bps": [BORROW_BPS_GENERAL] * len(symbols),
        }
    )


_TIER_TO_BPS: dict[str, int] = {
    "easy": BORROW_BPS_EASY,
    "general": BORROW_BPS_GENERAL,
    "hard": BORROW_BPS_HARD,
}


def apply_borrow_charges(positions: pl.DataFrame, *, multiplier: float = 1.0) -> pl.DataFrame:
    """Compute daily borrow cost per row.  Charges apply only to short
    notional (signed_notional < 0); longs are 0.  Stress multiplier is
    applied linearly to the annual_bps."""
    return positions.with_columns(
        pl.col("tier").replace_strict(_TIER_TO_BPS, return_dtype=pl.Int64).alias("_bps")
    ).with_columns(
        pl.when(pl.col("signed_notional") < 0)
        .then(
            (-pl.col("signed_notional"))
            * pl.col("_bps").cast(pl.Float64)
            * float(multiplier)
            / 10_000.0
            / 252.0
        )
        .otherwise(0.0)
        .alias("borrow_cost")
    ).drop("_bps")
