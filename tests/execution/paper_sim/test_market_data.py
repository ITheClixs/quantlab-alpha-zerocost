from __future__ import annotations

from quant_research_stack.execution.paper_sim.market_data import (
    MarketSnapshot,
    parse_premium_index,
    parse_spot_price,
)


def test_parse_spot_price() -> None:
    assert parse_spot_price({"symbol": "BTCUSDT", "price": "65000.50"}) == 65000.50


def test_parse_premium_index() -> None:
    mark, funding, next_ts = parse_premium_index({
        "markPrice": "65010.00", "indexPrice": "65005.00",
        "lastFundingRate": "0.0001", "nextFundingTime": 1717200000000,
    })
    assert mark == 65010.0
    assert funding == 0.0001
    assert next_ts == 1717200000000


def test_snapshot_basis() -> None:
    snap = MarketSnapshot(symbol="BTCUSDT", ts_ms=1, spot_price=100.0,
                          perp_mark=101.0, funding_rate=0.0002, next_funding_ms=8)
    assert abs(snap.basis - 0.01) < 1e-9
