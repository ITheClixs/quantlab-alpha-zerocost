from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel, Conv1DConfig


def test_conv1d_fit_predict() -> None:
    rng = np.random.default_rng(0)
    n, seq_len, channels = 100, 8, 5
    x = rng.normal(size=(n, seq_len, channels)).astype(np.float32)
    y = (x.sum(axis=(1, 2)) * 0.01 + rng.normal(size=n) * 0.01).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    cfg = Conv1DConfig(max_epochs=3, batch_size=16, device="cpu")
    model = Conv1DAlphaModel(cfg)
    model.fit(x, y, w, x[80:], y[80:], w[80:])
    pred = model.predict(x[:10])
    assert pred.shape == (10,)
