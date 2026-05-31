from __future__ import annotations

from pathlib import Path

from quant_research_stack.execution.paper_sim.config import PaperSimConfig, load_paper_sim_config


def test_defaults_are_one_x_unlevered_and_paper() -> None:
    cfg = PaperSimConfig(symbols=["BTCUSDT", "ETHUSDT"])
    assert cfg.leverage == 1.0
    assert cfg.total_notional_usd > 0
    assert cfg.max_data_gap_seconds == 120


def test_load_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "paper_sim.yaml"
    p.write_text(
        "symbols: [BTCUSDT]\n"
        "total_notional_usd: 20000\n"
        "starting_equity_usd: 100000\n"
        "half_spread_bps: 1.0\n"
        "slippage_bps: 4.0\n"
        "commission_bps: 1.0\n"
        "rebalance_drift_bps: 25.0\n"
        "poll_interval_s: 10.0\n"
    )
    cfg = load_paper_sim_config(p)
    assert cfg.symbols == ["BTCUSDT"]
    assert cfg.slippage_bps == 4.0
    assert cfg.leverage == 1.0  # not in yaml -> default
