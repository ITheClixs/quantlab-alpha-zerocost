from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant_research_stack.execution.configs import load_risk_config
from quant_research_stack.execution.sizing import Sizer, SizerInput
from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import GovernorVerdict


def _ticket(
    direction: int = 1,
    decision: str = "pass",
    confidence: float = 0.7,
    t3_dir: int | None = None,
    t3_dec: str | None = None,
) -> ExecutionTicket:
    sig = S1Signal(
        signal_id="sig-00002222",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=confidence,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    prim = GovernorVerdict.model_validate({
        "signal_id": sig.signal_id,
        "decision": decision,
        "direction": direction,
        "confidence": confidence,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"] if decision == "pass" else [],
        "contradictions_flagged": [],
    })
    t3 = None
    if t3_dir is not None and t3_dec is not None:
        t3 = GovernorVerdict.model_validate({
            "signal_id": sig.signal_id,
            "decision": t3_dec,
            "direction": t3_dir,
            "confidence": 0.8,
            "horizon_minutes": 5,
            "regime_tag": "trending",
            "rationale_short": "ok",
            "cited_paper_chunk_ids": ["paper_pdf:x:0"] if t3_dec == "pass" else [],
            "contradictions_flagged": [],
        })
    return ExecutionTicket(signal=sig, primary_verdict=prim, tier3_verdict=t3, ingested_at=datetime.now(UTC))


def test_veto_yields_zero_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(
        SizerInput(ticket=_ticket(decision="veto"), account_equity=100_000, mid_price=50_000, lot_size=0.0001)
    )
    assert qty == 0.0


def test_neutral_direction_yields_zero_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(
        SizerInput(ticket=_ticket(direction=0), account_equity=100_000, mid_price=50_000, lot_size=0.0001)
    )
    assert qty == 0.0


def test_long_signal_yields_positive_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(
        SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001)
    )
    assert 0 < qty <= 0.02


def test_short_signal_yields_negative_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(
        SizerInput(ticket=_ticket(direction=-1), account_equity=100_000, mid_price=50_000, lot_size=0.0001)
    )
    assert qty < 0


def test_tier3_agreement_increases_size() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    base = sizer.size(
        SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001)
    )
    boosted = sizer.size(
        SizerInput(
            ticket=_ticket(direction=1, t3_dir=1, t3_dec="pass"),
            account_equity=100_000,
            mid_price=50_000,
            lot_size=0.0001,
        )
    )
    assert boosted > base


def test_tier3_disagreement_shrinks_size() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    base = sizer.size(
        SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001)
    )
    shrunk = sizer.size(
        SizerInput(
            ticket=_ticket(direction=1, t3_dir=-1, t3_dec="pass"),
            account_equity=100_000,
            mid_price=50_000,
            lot_size=0.0001,
        )
    )
    assert shrunk < base


def test_tier3_veto_shrinks_size() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    base = sizer.size(
        SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001)
    )
    vetoed = sizer.size(
        SizerInput(
            ticket=_ticket(direction=1, t3_dir=1, t3_dec="veto"),
            account_equity=100_000,
            mid_price=50_000,
            lot_size=0.0001,
        )
    )
    assert vetoed < base


def test_qty_respects_per_symbol_cap() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(
        SizerInput(ticket=_ticket(direction=1, confidence=1.0), account_equity=10_000_000, mid_price=50_000, lot_size=0.0001)
    )
    cap_notional = 10_000_000 * cfg.limits.max_per_symbol_pct
    cap_qty = cap_notional / 50_000
    assert qty <= cap_qty + 1e-9


def test_qty_rounded_to_lot() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(
        SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.001)
    )
    assert abs(qty * 1000 - round(qty * 1000)) < 1e-9
