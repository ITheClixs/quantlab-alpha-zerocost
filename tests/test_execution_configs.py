from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.execution.configs import (
    ExecConfig,
    PromotionConfig,
    RiskConfig,
    load_exec_config,
    load_promotion_config,
    load_risk_config,
)


def test_risk_config_loads_valid_yaml() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    assert isinstance(cfg, RiskConfig)
    assert 0 < cfg.limits.max_per_symbol_pct < 1
    assert cfg.reconciliation.max_diff_bps > 0


def test_risk_config_rejects_negative_caps(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "limits:\n"
        "  max_per_symbol_pct: -0.1\n"
        "  max_gross_exposure_pct: 0.3\n"
        "  base_notional_per_trade_pct: 0.005\n"
        "  max_orders_per_minute: 10\n"
        "drawdown:\n"
        "  daily_realized_dd_kill_pct: 0.05\n"
        "  cumulative_dd_kill_pct: 0.15\n"
        "freshness:\n"
        "  crypto_max_gap_seconds: 120\n"
        "  equity_max_gap_seconds: 1800\n"
        "reconciliation:\n"
        "  interval_seconds: 60\n"
        "  max_diff_bps: 1.0\n"
    )
    with pytest.raises(ValueError):
        load_risk_config(p)


def test_promotion_config_loads_valid_yaml() -> None:
    cfg = load_promotion_config(Path("configs/promotion.yaml"))
    assert isinstance(cfg, PromotionConfig)
    assert cfg.paper_to_live_shadow.min_days_in_paper >= 1
    assert cfg.live_shadow_to_live.kill_switch_drill_passed in (True, False)


def test_exec_config_loads_valid_yaml() -> None:
    cfg = load_exec_config(Path("configs/exec.yaml"))
    assert isinstance(cfg, ExecConfig)
    assert cfg.ingest.poll_interval_seconds > 0
