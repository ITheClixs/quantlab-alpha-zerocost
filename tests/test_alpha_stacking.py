from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.stacking import LinearStacker


def test_linear_stacker_fits_oof_predictions() -> None:
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=(n, 3))
    true_w = np.array([0.5, 0.3, 0.2])
    y = base @ true_w + rng.normal(size=n) * 0.01
    weights = np.ones(n)
    stacker = LinearStacker()
    stacker.fit(base, y, weights)
    pred = stacker.predict(base)
    assert pred.shape == (n,)
    # weights should approximately recover true_w under non-negativity + normalization
    recovered = stacker.weights()
    assert np.argmax(recovered) == 0  # column with largest contribution
