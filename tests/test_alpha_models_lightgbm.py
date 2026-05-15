from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig


def test_lgb_fit_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    y = x[:, 0] * 0.5 + rng.normal(size=500) * 0.1
    w = np.ones(500)
    model = LightGBMAlphaModel(LightGBMConfig(n_estimators=50, learning_rate=0.1))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    pred = model.predict(x[:50])
    assert pred.shape == (50,)


def test_lgb_feature_importance() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    y = x[:, 0] * 0.5
    w = np.ones(500)
    model = LightGBMAlphaModel(LightGBMConfig(n_estimators=50, learning_rate=0.1))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    importance = model.feature_importance()
    assert importance.shape == (5,)
    assert importance[0] >= importance[1:].max()
