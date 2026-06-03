from quant_research_stack.signal_research.fingerprint_vwap.pipeline import gate_verdict


def test_gate_fails_on_zero_lift() -> None:
    v = gate_verdict(
        meta_net_sharpe=0.5, baseline_net_sharpe=0.5, lift_margin=0.2,
        daily_net_returns=[0.001, -0.002, 0.0015, 0.0, 0.0008] * 60, trials=45,
    )
    assert v["verdict"] == "DO_NOT_ADVANCE"
    assert "lift" in v["failed"]


def test_gate_structure_keys() -> None:
    v = gate_verdict(meta_net_sharpe=1.0, baseline_net_sharpe=0.2, lift_margin=0.2,
                     daily_net_returns=[0.002, -0.001, 0.003] * 80, trials=10)
    assert set(["verdict", "passed", "failed", "deflated_sharpe", "net_sharpe", "lift"]).issubset(v)
