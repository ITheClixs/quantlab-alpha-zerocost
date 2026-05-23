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


def test_stacker_save_load_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 6))     # 6 base models post-S0
    true_w = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    y = x @ true_w + 0.05 * rng.standard_normal(500)
    w = np.ones(500)

    feature_order = ["ridge", "lgb", "xgb", "cat", "mlp", "seq"]
    original = LinearStacker(alpha=1e-3, feature_order=feature_order)
    original.fit(x, y, w)

    path = tmp_path / "stacker.joblib"
    original.save(path)
    assert path.exists()

    reloaded = LinearStacker.load(path)
    assert reloaded.feature_order == feature_order
    np.testing.assert_array_equal(original.predict(x), reloaded.predict(x))


def test_stacker_zeros_inactive_noisy_models_and_normalizes_weights() -> None:
    rng = np.random.default_rng(7)
    n = 300
    useful_1 = rng.normal(size=n)
    useful_2 = rng.normal(size=n)
    noisy = rng.normal(size=n)
    x = np.column_stack([useful_1, useful_2, noisy])
    y = 0.7 * useful_1 + 0.3 * useful_2 + rng.normal(scale=0.02, size=n)
    weights = np.ones(n)

    stacker = LinearStacker(alpha=1e-3, feature_order=["ridge", "lgb", "mlp"])
    stacker.fit(x, y, weights, active_feature_order=["ridge", "lgb"])

    recovered = stacker.weights()
    assert stacker.active_feature_order == ["ridge", "lgb"]
    assert recovered[2] == 0.0
    np.testing.assert_allclose(recovered.sum(), 1.0, atol=1e-8)
    assert recovered[0] > recovered[1] > recovered[2]
    assert stacker.residual_scale > 0.0


def test_stacker_save_load_preserves_active_models_and_calibration(tmp_path) -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal((200, 3))
    y = x[:, 0] * 0.6 + x[:, 1] * 0.4
    weights = np.ones(200)

    original = LinearStacker(alpha=1e-3, feature_order=["ridge", "cat", "seq"])
    original.fit(x, y, weights, active_feature_order=["ridge", "cat"])

    path = tmp_path / "stacker.joblib"
    original.save(path)
    reloaded = LinearStacker.load(path)

    assert reloaded.active_feature_order == ["ridge", "cat"]
    assert reloaded.residual_scale == original.residual_scale
    np.testing.assert_array_equal(original.weights(), reloaded.weights())
    np.testing.assert_array_equal(original.predict(x), reloaded.predict(x))
