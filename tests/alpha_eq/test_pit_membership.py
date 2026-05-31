"""PIT membership + ticker mapping (spec §2.2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha_eq.data.pit_membership import (
    MembershipSource,
    PITMembership,
    TickerMapping,
    apply_ticker_mapping,
    load_pit_membership,
)


def _toy_membership() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3)],
            "symbol": ["AAPL", "AOL", "AAPL"],
            "in_index": [True, True, True],
            "addition_date": [date(1982, 11, 30), date(1985, 1, 1), date(1982, 11, 30)],
            "removal_date": [None, date(2020, 1, 3), None],
            "removal_reason": [None, "acquired", None],
        }
    )


def test_load_pit_membership_round_trip(tmp_equity_root: Path) -> None:
    df = _toy_membership()
    path = tmp_equity_root / "sp500_pit_membership.parquet"
    df.write_parquet(path)
    mem = load_pit_membership(path, source=MembershipSource.HF_PRIMARY)
    assert isinstance(mem, PITMembership)
    assert mem.is_in_index(symbol="AAPL", on=date(2020, 1, 2))
    assert mem.is_in_index(symbol="AOL", on=date(2020, 1, 2))
    assert not mem.is_in_index(symbol="AOL", on=date(2020, 1, 3))


def test_load_pit_membership_missing_columns(tmp_equity_root: Path) -> None:
    bad = pl.DataFrame({"date": [date(2020, 1, 2)], "symbol": ["AAPL"]})
    p = tmp_equity_root / "bad.parquet"
    bad.write_parquet(p)
    with pytest.raises(ValueError, match="missing column"):
        load_pit_membership(p, source=MembershipSource.HF_PRIMARY)


def test_ticker_mapping_apply() -> None:
    mapping = TickerMapping(
        rows=[
            ("FB", "META", date(2022, 6, 9)),
            ("VIAC", "PARA", date(2022, 2, 16)),
        ]
    )
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2023, 1, 3), date(2022, 6, 9)],
            "symbol": ["FB", "FB", "FB"],
        }
    )
    out = apply_ticker_mapping(df, mapping)
    assert out["symbol"].to_list() == ["FB", "META", "META"]


def test_membership_source_values() -> None:
    assert MembershipSource("hf_primary").value == "hf_primary"
    assert MembershipSource("wikipedia_fallback").value == "wikipedia_fallback"
    assert MembershipSource("absent_prototype_only").value == "absent_prototype_only"
    with pytest.raises(ValueError):
        MembershipSource("guessed")
