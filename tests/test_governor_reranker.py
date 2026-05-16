from __future__ import annotations

from quant_research_stack.governor.reranker import RerankCandidate, Reranker, StubReranker


def test_stub_reranker_orders_by_string_overlap() -> None:
    reranker = StubReranker()
    cands = [
        RerankCandidate(id="a", text="alpha beta"),
        RerankCandidate(id="b", text="beta gamma delta"),
        RerankCandidate(id="c", text="zeta"),
    ]
    out = reranker.rerank("beta gamma", cands)
    assert out[0].id == "b"
    assert out[-1].id == "c"


def test_stub_reranker_returns_same_length() -> None:
    reranker = StubReranker()
    cands = [RerankCandidate(id=str(i), text=f"x {i}") for i in range(5)]
    out = reranker.rerank("anything", cands)
    assert len(out) == len(cands)


def test_reranker_protocol_accepts_stub() -> None:
    def use_reranker(r: Reranker) -> str:
        return r.rerank("q", [RerankCandidate(id="a", text="t")])[0].id
    assert use_reranker(StubReranker()) == "a"
