"""Tests for make_strategy_zoo_figures — F1-F5 figure builders.

TDD Step 1: these tests must FAIL before the module is created.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import polars as pl

_spec = importlib.util.spec_from_file_location(
    "zf",
    pathlib.Path(__file__).resolve().parents[3] / "scripts" / "make_strategy_zoo_figures.py",
)
zf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zf)  # type: ignore[union-attr]


def test_sharpe_distribution_figure_builds() -> None:
    rng = np.random.default_rng(0)
    fig = zf.fig_sharpe_distribution(
        is_sharpe=rng.normal(0, 1, 10_000),
        permuted_sharpe=rng.normal(0, 1, 10_000),
    )
    assert fig is not None
    assert len(fig.axes) >= 1


def test_expected_vs_empirical_figure_builds() -> None:
    tiers = [
        {"n_trials": 1000, "empirical_max": 3.1, "theoretical_max": 3.3},
        {"n_trials": 10000, "empirical_max": 3.5, "theoretical_max": 3.9},
        {"n_trials": 100000, "empirical_max": 4.7, "theoretical_max": 4.4},
    ]
    fig = zf.fig_expected_vs_empirical(tiers=tiers)
    assert fig is not None and len(fig.axes) >= 1


def test_is_oos_decay_figure_builds() -> None:
    rng = np.random.default_rng(1)
    is_sr = rng.normal(2.0, 0.5, 50)
    oos_sr = is_sr * 0.3 + rng.normal(0, 0.4, 50)
    fig = zf.fig_is_oos_decay(is_sharpe=is_sr, oos_sharpe=oos_sr)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_overfitting_panel_figure_builds() -> None:
    fig = zf.fig_overfitting_panel(
        pbo_probability=0.87,
        dsr_pass_count=3,
        n_strategies=500,
        real_best=2.8,
        permuted_best=2.6,
    )
    assert fig is not None
    assert len(fig.axes) >= 1


def test_family_heatmap_figure_builds() -> None:
    rng = np.random.default_rng(2)
    families = ["momentum", "mean_rev", "carry", "vol"]
    lookbacks = [5, 10, 20, 60]
    rows = [
        {"family": f, "lookback": lb, "is_sharpe": float(rng.normal(0.5, 0.3))}
        for f in families
        for lb in lookbacks
    ]
    metrics = pl.DataFrame(rows)
    fig = zf.fig_family_heatmap(metrics=metrics)
    assert fig is not None
    assert len(fig.axes) >= 1
