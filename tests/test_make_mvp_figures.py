from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_mvp_figures", Path(__file__).resolve().parents[1] / "scripts" / "make_mvp_figures.py")
assert _spec is not None and _spec.loader is not None
mmf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mmf)


def test_s1_fold_model_r2_groups_by_model() -> None:
    metrics = {"fold_metrics": [
        {"fold": 0.0, "ridge_r2": 0.1, "lgb_r2": 0.2, "xgb_r2": 0.0,
         "cat_r2": 0.0, "mlp_r2": 0.0, "seq_r2": 0.0},
        {"fold": 1.0, "ridge_r2": 0.3, "lgb_r2": 0.4, "xgb_r2": 0.0,
         "cat_r2": 0.0, "mlp_r2": 0.0, "seq_r2": 0.0},
    ]}
    out = mmf.s1_fold_model_r2(metrics)
    assert out["ridge"] == [0.1, 0.3]
    assert out["lgb"] == [0.2, 0.4]
    assert set(out) == {"ridge", "lgb", "xgb", "cat", "mlp", "seq"}


def test_leverage_stress_rows_converts_return_to_pct() -> None:
    manifest = {"liquidation_stressed_pooled": {
        "3x": {"sharpe": -0.47, "ann_return": -0.17, "n_liquidations": 10.0},
        "10x": {"sharpe": -5.17, "ann_return": -0.896, "n_liquidations": 284.0},
    }}
    rows = mmf.leverage_stress_rows(manifest)
    assert rows[0] == ("3x", -0.47, -17.0)
    assert rows[1][0] == "10x"
    assert abs(rows[1][2] - (-89.6)) < 1e-6


def test_per_year_bar_rows_from_manifest() -> None:
    manifest = {"honest_pooled_per_year": {
        "2024": {"total_pct": 13.3}, "2026": {"total_pct": -0.16}}}
    rows = mmf.per_year_bar_rows(manifest)
    assert rows == [(2024, 13.3), (2026, -0.16)]


def test_pooled_equity_from_net_compounds() -> None:
    import numpy as np
    net = np.array([0.01, -0.005, 0.02])
    eq = mmf.pooled_equity(net)
    assert eq[0] == 1.0
    assert abs(eq[-1] - (1.01 * 0.995 * 1.02)) < 1e-9


def test_plot_leverage_stress_writes_png(tmp_path) -> None:
    rows = [("3x", -0.47, -17.0), ("5x", -1.5, -37.8), ("10x", -5.17, -89.6)]
    out = tmp_path / "lev.png"
    mmf.plot_leverage_stress(rows, out)
    assert out.exists() and out.stat().st_size > 0
