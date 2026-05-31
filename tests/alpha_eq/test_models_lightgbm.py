"""LightGBM S1-EQ model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.lightgbm_model import (
    LightGBMEqConfig,
    LightGBMEqModel,
)


def test_lightgbm_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 6))
    y = x[:, 0] * 0.5 + rng.standard_normal(500) * 0.1
    m = LightGBMEqModel(LightGBMEqConfig(n_estimators=50, num_leaves=15, seed=42))
    m.fit(x=x, y=y, x_val=x[:100], y_val=y[:100])
    p = m.predict(x[:10])
    assert p.shape == (10,)
    out = tmp_path / "lgb.txt"
    cfg = tmp_path / "lgb.config.json"
    m.save(out, config_path=cfg)
    m2 = LightGBMEqModel.load(out, config_path=cfg)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-9)
