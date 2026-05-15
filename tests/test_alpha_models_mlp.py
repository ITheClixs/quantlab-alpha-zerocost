from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig


def test_mlp_fit_predict_cpu() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = (x[:, 0] * 0.5 + rng.normal(size=200) * 0.1).astype(np.float32)
    w = np.ones(200, dtype=np.float32)
    cfg = MLPConfig(hidden_dims=[16, 8], max_epochs=3, batch_size=32, mixed_precision=False, device="cpu")
    model = MLPAlphaModel(cfg)
    model.fit(x, y, w, x[150:], y[150:], w[150:])
    pred = model.predict(x[:20])
    assert pred.shape == (20,)
