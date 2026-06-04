from __future__ import annotations

import polars as pl

from quant_research_stack.strategy_benchmark.zoo.transforms import (
    apply_holding,
    apply_position_mode,
    apply_vol_target,
)


def test_position_mode_long_only_clips_negatives() -> None:
    s = pl.Series("s", [1.0, -1.0, 0.5, -0.3])
    assert apply_position_mode(s, mode="long_only").to_list() == [1.0, 0.0, 0.5, 0.0]
    assert apply_position_mode(s, mode="long_short").to_list() == [1.0, -1.0, 0.5, -0.3]


def test_apply_holding_forward_fills_nonzero_for_h_days() -> None:
    s = pl.Series("s", [1.0, 0.0, 0.0, -1.0, 0.0])
    held = apply_holding(s, holding=2)
    assert held.to_list()[0] == 1.0 and held.to_list()[1] == 1.0
    assert held.len() == 5


def test_apply_vol_target_scales_inverse_to_vol() -> None:
    s = pl.Series("s", [1.0, 1.0, 1.0])
    vol = pl.Series("v", [0.01, 0.02, None])
    out = apply_vol_target(s, vol=vol, target_daily_vol=0.01)
    assert abs(out.to_list()[0] - 1.0) < 1e-9
    assert abs(out.to_list()[1] - 0.5) < 1e-9
    assert out.to_list()[2] == 0.0
