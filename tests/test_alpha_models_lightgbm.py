from __future__ import annotations

import json

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


def test_lightgbm_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((1000, 8))
    y_tr = x_tr[:, 0] + 0.1 * rng.standard_normal(1000)
    w_tr = np.ones(1000)
    x_val = rng.standard_normal((200, 8))
    y_val = x_val[:, 0] + 0.1 * rng.standard_normal(200)
    w_val = np.ones(200)

    cfg = LightGBMConfig(
        num_leaves=15,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=50,
        early_stopping_rounds=10,
        feature_fraction=1.0,
        bagging_fraction=1.0,
    )
    original = LightGBMAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "lightgbm.txt"
    original.save(path)
    assert path.exists()
    sidecar = path.parent / "lightgbm.config.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["num_leaves"] == 15

    reloaded = LightGBMAlphaModel.load(path)
    np.testing.assert_array_equal(original.predict(x_val), reloaded.predict(x_val))
