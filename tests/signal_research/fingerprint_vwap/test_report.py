from quant_research_stack.signal_research.fingerprint_vwap.pipeline import render_report


def test_render_report_contains_verdict_and_disclaimer() -> None:
    md = render_report(
        result={"status": "evaluated", "meta_net_sharpe": 0.4, "baseline_net_sharpe": 0.5,
                "lift": -0.1, "eligibility": {"eligible": True, "reason": "", "event_count": 1234,
                "primary_net_sharpe": 0.3}},
        verdict={"verdict": "DO_NOT_ADVANCE", "failed": ["lift"], "net_sharpe": 0.4,
                 "lift": -0.1, "deflated_sharpe": {"deflated_sharpe_ratio": 0.6}},
        spec_repr="FingerprintVwapSpec(...)",
    )
    assert "DO_NOT_ADVANCE" in md
    assert "research_only" in md.lower() or "not investment advice" in md.lower()
    assert "lift" in md.lower()
