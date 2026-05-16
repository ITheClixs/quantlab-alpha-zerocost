from __future__ import annotations

from dataclasses import dataclass

from quant_research_stack.governor.query_builder import build_query


@dataclass(frozen=True)
class _Sig:
    symbol: str
    direction: int
    horizon_minutes: int
    regime_hint: str | None
    recent_vol_label: str


def test_build_query_with_regime_hint() -> None:
    sig = _Sig(symbol="BTCUSDT", direction=1, horizon_minutes=15, regime_hint="trending", recent_vol_label="med")
    assert build_query(sig) == "trending BTCUSDT direction=1 horizon=15m vol=med"


def test_build_query_with_no_regime_hint() -> None:
    sig = _Sig(symbol="ETHUSDT", direction=-1, horizon_minutes=5, regime_hint=None, recent_vol_label="high")
    assert build_query(sig) == "unknown ETHUSDT direction=-1 horizon=5m vol=high"
