from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant_research_stack.execution.configs import RiskConfig, load_risk_config
from quant_research_stack.execution.risk import _GATES, RiskGate, RiskState
from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import GovernorVerdict


def _cfg() -> RiskConfig:
    return load_risk_config(Path("configs/risk.yaml"))


def _ticket(decision: str = "pass") -> ExecutionTicket:
    sig = S1Signal(
        signal_id="sig-00001111",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    v = GovernorVerdict.model_validate({
        "signal_id": sig.signal_id,
        "decision": decision,
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"] if decision == "pass" else [],
        "contradictions_flagged": [],
    })
    return ExecutionTicket(signal=sig, primary_verdict=v, tier3_verdict=None, ingested_at=datetime.now(UTC))


def test_gate_order_is_kill_first() -> None:
    assert _GATES[0].__name__ == "kill_flag_check"


def test_kill_flag_blocks_before_anything_else(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_TRADING"
    flag.touch()
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=0,
        gross_exposure_notional=0,
        per_symbol_notional={},
        orders_last_minute=0,
        last_tick_ts={},
        kill_flag_path=flag,
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(_cfg())
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is True
    assert decision.reason == "kill_flag_check"


def test_governor_veto_blocks_without_killing() -> None:
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=0,
        gross_exposure_notional=0,
        per_symbol_notional={},
        orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(_cfg())
    decision = gate.evaluate(_ticket(decision="veto"), state)
    assert decision.allowed is False
    assert decision.kill_trigger is False
    assert decision.reason == "governor_decision_check"


def test_drawdown_kill_when_daily_breached() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=-100_000 * cfg.drawdown.daily_realized_dd_kill_pct * 1.1,
        gross_exposure_notional=0,
        per_symbol_notional={},
        orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is True
    assert decision.reason == "drawdown_check"


def test_feed_freshness_kill_when_gap_exceeded() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=0,
        gross_exposure_notional=0,
        per_symbol_notional={},
        orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC) - timedelta(seconds=cfg.freshness.crypto_max_gap_seconds + 10)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is True
    assert decision.reason == "feed_freshness_check"


def test_exposure_blocks_without_killing() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=0,
        gross_exposure_notional=100_000 * cfg.limits.max_gross_exposure_pct,
        per_symbol_notional={"BTCUSDT": 100_000 * cfg.limits.max_per_symbol_pct},
        orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is False
    assert decision.reason == "exposure_check"


def test_rate_limit_blocks_without_killing() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=0,
        gross_exposure_notional=0,
        per_symbol_notional={},
        orders_last_minute=cfg.limits.max_orders_per_minute,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is False
    assert decision.reason == "rate_limit_check"


def test_happy_path_passes_all_gates() -> None:
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=0,
        gross_exposure_notional=0,
        per_symbol_notional={},
        orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(_cfg())
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is True
    assert decision.kill_trigger is False
