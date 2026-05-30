from __future__ import annotations

from quant_research_stack.signal_research.zero_cost.data import (
    FORBIDDEN_SERIES,
    INSTRUMENTS,
    MACRO_REGISTRY,
    load_instrument,
)

_ALLOWED = {"market_price_clean", "daily_next_day_only"}


def test_macro_registry_only_timestamp_safe_classifications() -> None:
    # no revised-aggregate / reject classification may sit in the allowed registry
    assert all(s.classification in _ALLOWED for s in MACRO_REGISTRY)
    assert len(MACRO_REGISTRY) >= 6


def test_forbidden_series_present_and_disjoint() -> None:
    assert {"GDP", "CPIAUCSL", "PAYEMS", "UNRATE"} <= set(FORBIDDEN_SERIES)
    registry_refs = {s.ref for s in MACRO_REGISTRY}
    assert registry_refs.isdisjoint(set(FORBIDDEN_SERIES))


def test_instruments_are_directly_traded_set() -> None:
    assert set(INSTRUMENTS) == {"SPY", "QQQ", "BTCUSDT", "ETHUSDT"}


def test_disk_instrument_loader() -> None:
    df = load_instrument("SPY")  # on-disk, no network
    assert df.columns == ["date", "close"]
    assert df.height > 1000
    assert df["date"].is_sorted()
