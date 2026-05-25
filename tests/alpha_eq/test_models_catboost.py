"""CatBoost S1-EQ model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.catboost_model import (
    CatBoostEqConfig,
    CatBoostEqModel,
)


def test_catboost_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 5))
    y = x[:, 0] * 0.3 + rng.standard_normal(400) * 0.1
    m = CatBoostEqModel(CatBoostEqConfig(iterations=50, depth=4, seed=42))
    m.fit(x=x, y=y, x_val=x[:80], y_val=y[:80])
    out = tmp_path / "cat.cbm"
    cfg = tmp_path / "cat.config.json"
    m.save(out, config_path=cfg)
    m2 = CatBoostEqModel.load(out, config_path=cfg)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-9)
