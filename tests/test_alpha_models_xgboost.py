from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel, XGBoostConfig


def test_xgb_fit_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5)).astype(np.float32)
    y = (x[:, 0] * 0.5 + rng.normal(size=500) * 0.1).astype(np.float32)
    w = np.ones(500, dtype=np.float32)
    model = XGBoostAlphaModel(XGBoostConfig(n_estimators=50, learning_rate=0.1))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    pred = model.predict(x[:50])
    assert pred.shape == (50,)
