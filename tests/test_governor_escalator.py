from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.escalator import EscalationConfig, S1Signal, govern_signal
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _v(decision: str = "pass", citations=("paper_pdf:foo:0",)) -> GovernorVerdict:
    return GovernorVerdict.model_validate({
        "signal_id": "sig-12345678",
        "decision": decision,
        "direction": 1,
        "confidence": 0.9,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "stub",
        "cited_paper_chunk_ids": list(citations),
        "contradictions_flagged": [],
    })


@dataclass
class _StubT1:
    next_decision: str = "pass"

    def govern(self, signal, retrieval):  # noqa: D401, ANN001
        return _v(decision=self.next_decision, citations=())


@dataclass
class _StubT2:
    next_decision: str = "pass"

    def govern(self, signal, retrieval):  # noqa: ANN001
        return _v(decision=self.next_decision, citations=("paper_pdf:foo:0",))


@dataclass
class _StubT3:
    scheduled: list = field(default_factory=list)

    def schedule_async(self, signal, chunks):  # noqa: ANN001
        self.scheduled.append((signal.signal_id, len(chunks)))


@dataclass
class _StubRuntimes:
    tier1: _StubT1
    tier2: _StubT2
    tier3: _StubT3


def _corpus_with(ids: list[str]) -> CorpusIndex:
    return CorpusIndex([Chunk(id=cid, source_type="t", source_path="p", chunk_index=i, text="t", sha256="x", n_words=1) for i, cid in enumerate(ids)])


def _signal(confidence=0.7, trade_size_pct=0.5) -> S1Signal:
    return S1Signal(
        signal_id="sig-12345678",
        symbol="BTCUSDT",
        direction=1,
        confidence=confidence,
        horizon_minutes=15,
        regime_hint="trending",
        recent_vol_label="med",
        trade_size_pct=trade_size_pct,
    )


def _retrieval(corpus: CorpusIndex):
    def _retr(signal, k):  # noqa: ANN001
        return [next(iter(corpus))]
    return _retr


def test_tier1_veto_short_circuits() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="veto"), tier2=_StubT2(), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(), cfg, runtimes, corpus, _retrieval(corpus))
    assert out.decision == Decision.veto
    assert runtimes.tier3.scheduled == []


def test_low_confidence_does_not_escalate_to_tier2() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="pass"), tier2=_StubT2(next_decision="veto"), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.3), cfg, runtimes, corpus, _retrieval(corpus))
    # Tier 2 not called -> result is Tier 1's pass (which gets passed through)
    assert out.decision == Decision.pass_


def test_disabled_tier2_returns_tier1_fast_path() -> None:
    cfg = EscalationConfig()

    @dataclass
    class _Runtimes:
        tier1: _StubT1
        tier2 = None
        tier3 = None

    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.9, trade_size_pct=2.0), cfg, _Runtimes(tier1=_StubT1()), corpus, _retrieval(corpus))
    assert out.decision == Decision.pass_


def test_high_confidence_calls_tier2_and_uses_its_decision() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="pass"), tier2=_StubT2(next_decision="veto"), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.9), cfg, runtimes, corpus, _retrieval(corpus))
    assert out.decision == Decision.veto
    assert runtimes.tier3.scheduled == []


def test_large_trade_schedules_tier3_async() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="pass"), tier2=_StubT2(next_decision="pass"), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.9, trade_size_pct=2.0), cfg, runtimes, corpus, _retrieval(corpus))
    assert out.decision == Decision.pass_
    assert len(runtimes.tier3.scheduled) == 1


def test_disabled_tier3_does_not_schedule_large_trade() -> None:
    cfg = EscalationConfig()

    @dataclass
    class _Runtimes:
        tier1: _StubT1
        tier2: _StubT2
        tier3 = None

    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.9, trade_size_pct=2.0), cfg, _Runtimes(tier1=_StubT1(), tier2=_StubT2()), corpus, _retrieval(corpus))
    assert out.decision == Decision.pass_
