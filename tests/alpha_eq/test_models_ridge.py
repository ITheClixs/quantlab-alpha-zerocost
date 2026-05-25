"""Ridge S1-EQ model (target = y_xs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.ridge import (
    RidgeEqConfig,
    RidgeEqModel,
)


def test_ridge_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8))
    y = x[:, 0] - 0.5 * x[:, 1] + rng.standard_normal(200) * 0.1
    m = RidgeEqModel(RidgeEqConfig(alpha=1.0))
    m.fit(x=x, y=y)
    p = m.predict(x)
    assert p.shape == (200,)
    out = tmp_path / "ridge.joblib"
    m.save(out)
    m2 = RidgeEqModel.load(out)
    np.testing.assert_allclose(m.predict(x), m2.predict(x), atol=1e-12)
