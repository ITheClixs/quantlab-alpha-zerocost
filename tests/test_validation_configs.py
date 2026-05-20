from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.validation import ValidationConfig, load_validation_config


def test_loads_valid_yaml() -> None:
    cfg = load_validation_config(Path("configs/validation.yaml"))
    assert isinstance(cfg, ValidationConfig)
    assert cfg.window.min_trading_days >= 1
    assert 0.0 < cfg.thresholds.hit_rate_min < 1.0
    assert cfg.data.forward_return_source == "alpaca_bars"
    assert cfg.data.horizon_alignment == "ceil_to_next_bar"


def test_rejects_hit_rate_out_of_range(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "window:\n"
        "  min_trading_days: 30\n"
        "  rolling_window_days: 14\n"
        "thresholds:\n"
        "  hit_rate_min: 1.5\n"
        "  sharpe_min: 1.0\n"
        "  max_daily_dd_pct: 0.05\n"
        "  governor_block_rate_max: 0.5\n"
        "data:\n"
        "  forward_return_source: alpaca_bars\n"
        "  horizon_alignment: ceil_to_next_bar\n"
        "artifacts:\n"
        "  daily_report_dir: docs/validation\n"
        "  per_signal_parquet_dir: data/validation\n"
    )
    with pytest.raises(ValueError):
        load_validation_config(p)


def test_rejects_unknown_forward_return_source(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "window:\n"
        "  min_trading_days: 30\n"
        "  rolling_window_days: 14\n"
        "thresholds:\n"
        "  hit_rate_min: 0.53\n"
        "  sharpe_min: 1.0\n"
        "  max_daily_dd_pct: 0.05\n"
        "  governor_block_rate_max: 0.5\n"
        "data:\n"
        "  forward_return_source: nonexistent_source\n"
        "  horizon_alignment: ceil_to_next_bar\n"
        "artifacts:\n"
        "  daily_report_dir: docs/validation\n"
        "  per_signal_parquet_dir: data/validation\n"
    )
    with pytest.raises(ValueError):
        load_validation_config(p)
