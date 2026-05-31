"""XGBoost S1-EQ model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.xgboost_model import (
    XGBoostEqConfig,
    XGBoostEqModel,
)


def test_xgboost_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 5))
    y = x[:, 0] * 0.3 + rng.standard_normal(400) * 0.1
    m = XGBoostEqModel(XGBoostEqConfig(n_estimators=50, max_depth=4, seed=42))
    m.fit(x=x, y=y, x_val=x[:80], y_val=y[:80])
    out = tmp_path / "xgb.json"
    cfg = tmp_path / "xgb.config.json"
    m.save(out, config_path=cfg)
    m2 = XGBoostEqModel.load(out, config_path=cfg)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-9)
