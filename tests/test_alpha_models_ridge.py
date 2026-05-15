from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig


def test_ridge_fit_predict_shape() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    w = rng.normal(size=5)
    y = X @ w + rng.normal(size=100) * 0.01
    weights = np.ones(100)
    model = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    model.fit(X, y, weights)
    pred = model.predict(X)
    assert pred.shape == (100,)


def test_ridge_fit_uses_weights() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    w = rng.normal(size=3)
    y = X @ w
    weights = np.ones(50)
    weights[:10] = 0.0  # disable first 10 rows
    model = RidgeAlphaModel(RidgeConfig(alpha=0.0))
    model.fit(X, y, weights)
    # weighted ridge with alpha=0 -> exact fit on weighted subset
    pred = model.predict(X[10:])
    assert np.allclose(pred, y[10:], atol=1e-6)


def test_ridge_zero_weights_raises() -> None:
    X = np.zeros((10, 3))
    y = np.zeros(10)
    weights = np.zeros(10)
    model = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    try:
        model.fit(X, y, weights)
    except ValueError as exc:
        assert "weights" in str(exc).lower()
        return
    raise AssertionError("expected ValueError on zero weights")
