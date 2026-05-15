from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.catboost_model import CatBoostAlphaModel, CatBoostConfig


def test_catboost_fit_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    y = x[:, 0] * 0.5 + rng.normal(size=500) * 0.1
    w = np.ones(500)
    model = CatBoostAlphaModel(CatBoostConfig(n_estimators=50, learning_rate=0.1, depth=4))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    pred = model.predict(x[:50])
    assert pred.shape == (50,)
