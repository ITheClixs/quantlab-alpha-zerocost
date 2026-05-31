from __future__ import annotations

from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.strategy import FundingCarryStrategy, perp_symbol


def _snap(sym: str) -> MarketSnapshot:
    return MarketSnapshot(symbol=sym, ts_ms=0, spot_price=100.0, perp_mark=100.0,
                          funding_rate=0.0001, next_funding_ms=0)


def test_perp_symbol_distinct() -> None:
    assert perp_symbol("BTCUSDT") == "BTCUSDTPERP"


def test_from_flat_opens_both_legs_delta_neutral() -> None:
    cfg = PaperSimConfig(symbols=["BTCUSDT", "ETHUSDT"], total_notional_usd=20000.0)
    strat = FundingCarryStrategy(cfg)
    intents = strat.rebalance_intents(_snap("BTCUSDT"), positions={}, cycle=0)
    # leg_notional = 20000 * 1 / (2 * 2) = 5000 per leg; at price 100 -> 50 units
    by_sym = {i.symbol: i for i in intents}
    assert by_sym["BTCUSDT"].side.value == "buy"
    assert abs(by_sym["BTCUSDT"].quantity - 50.0) < 1e-6
    assert by_sym["BTCUSDTPERP"].side.value == "sell"
    assert abs(by_sym["BTCUSDTPERP"].quantity - 50.0) < 1e-6


def test_no_trade_within_drift_band() -> None:
    cfg = PaperSimConfig(symbols=["BTCUSDT", "ETHUSDT"], total_notional_usd=20000.0,
                         rebalance_drift_bps=50.0)
    strat = FundingCarryStrategy(cfg)
    # already at target (50 long spot, 50 short perp) -> no intents
    pos = {"BTCUSDT": 50.0, "BTCUSDTPERP": -50.0}
    assert strat.rebalance_intents(_snap("BTCUSDT"), positions=pos, cycle=1) == []
