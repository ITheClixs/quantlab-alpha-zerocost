from __future__ import annotations

import numpy as np
import pytest

from quant_research_stack.alpha.metrics import sharpe_proxy, weighted_zero_mean_r2


def test_weighted_zero_mean_r2_perfect_prediction() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    y_pred = y_true.copy()
    w = np.ones(4, dtype=np.float64)
    score = weighted_zero_mean_r2(y_true, y_pred, w)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_weighted_zero_mean_r2_constant_zero_prediction() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    y_pred = np.zeros_like(y_true)
    w = np.ones_like(y_true)
    score = weighted_zero_mean_r2(y_true, y_pred, w)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_weighted_zero_mean_r2_weights_zero_skips_row() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    y_pred = np.array([0.01, -0.02, 0.03, 100.0], dtype=np.float64)
    w_skip_last = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    score = weighted_zero_mean_r2(y_true, y_pred, w_skip_last)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_sharpe_proxy_basic_shape() -> None:
    returns = np.array([0.001, 0.002, -0.001, 0.0015, -0.0005], dtype=np.float64)
    sharpe = sharpe_proxy(returns)
    assert sharpe > 0


def test_sharpe_proxy_zero_volatility_returns_zero() -> None:
    returns = np.ones(5, dtype=np.float64)
    assert sharpe_proxy(returns) == pytest.approx(0.0, abs=1e-12)
