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
