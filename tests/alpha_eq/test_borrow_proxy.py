"""Borrow proxy (spec §2.4)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.data.borrow_proxy import (
    BORROW_BPS_EASY,
    BORROW_BPS_GENERAL,
    BORROW_BPS_HARD,
    BorrowTier,
    apply_borrow_charges,
    build_borrow_proxy,
    classify_borrow_tier,
)


def test_static_tier_defaults() -> None:
    assert BORROW_BPS_EASY == 25
    assert BORROW_BPS_GENERAL == 100
    assert BORROW_BPS_HARD == 500


def test_classify_borrow_tier_handles_recent_ipo() -> None:
    tier = classify_borrow_tier(
        symbol="NEW",
        on=date(2020, 6, 1),
        ipo_date=date(2020, 1, 15),
        dollar_adv=50_000_000,
        realized_vol_20=0.3,
        price=50.0,
        recent_index_addition=False,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier == BorrowTier.HARD


def test_classify_borrow_tier_low_price_is_hard() -> None:
    tier = classify_borrow_tier(
        symbol="LOW",
        on=date(2020, 6, 1),
        ipo_date=None,
        dollar_adv=50_000_000,
        realized_vol_20=0.3,
        price=3.0,
        recent_index_addition=False,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier == BorrowTier.HARD


def test_classify_borrow_tier_low_adv_is_hard() -> None:
    tier = classify_borrow_tier(
        symbol="ILL",
        on=date(2020, 6, 1),
        ipo_date=None,
        dollar_adv=2_000_000,
        realized_vol_20=0.3,
        price=50.0,
        recent_index_addition=False,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier == BorrowTier.HARD


def test_recent_index_addition_alone_is_NOT_hard() -> None:
    """Spec §2.4: index-addition is watchlist flag only, NOT auto-hard."""
    tier = classify_borrow_tier(
        symbol="ADD",
        on=date(2020, 6, 1),
        ipo_date=None,
        dollar_adv=200_000_000,
        realized_vol_20=0.2,
        price=80.0,
        recent_index_addition=True,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier != BorrowTier.HARD


def test_apply_borrow_charges_only_on_shorts() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 6, 1), date(2020, 6, 1)],
            "symbol": ["L", "S"],
            "signed_notional": [100_000.0, -100_000.0],
            "tier": ["general", "general"],
        }
    )
    charges = apply_borrow_charges(positions, multiplier=1.0)
    by_sym = {r["symbol"]: r["borrow_cost"] for r in charges.to_dicts()}
    assert by_sym["L"] == 0.0
    assert by_sym["S"] > 0.0


def test_apply_borrow_charges_monotonic_in_multiplier() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 6, 1)],
            "symbol": ["S"],
            "signed_notional": [-100_000.0],
            "tier": ["general"],
        }
    )
    one = apply_borrow_charges(positions, multiplier=1.0)["borrow_cost"][0]
    two = apply_borrow_charges(positions, multiplier=2.0)["borrow_cost"][0]
    three = apply_borrow_charges(positions, multiplier=3.0)["borrow_cost"][0]
    assert one < two < three
    assert abs(two - 2 * one) < 1e-9
    assert abs(three - 3 * one) < 1e-9


def test_build_borrow_proxy_returns_static_table() -> None:
    symbols = ["AAPL", "MSFT", "RKLB"]
    df = build_borrow_proxy(symbols)
    assert set(df.columns) == {"symbol", "borrow_tier", "annual_bps"}
    assert df.height == 3
