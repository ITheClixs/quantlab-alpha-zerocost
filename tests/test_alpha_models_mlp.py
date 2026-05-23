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


def test_mlp_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((500, 8)).astype(np.float64)
    y_tr = x_tr[:, 0].astype(np.float64) + 0.1 * rng.standard_normal(500)
    w_tr = np.ones(500)
    x_val = rng.standard_normal((100, 8)).astype(np.float64)
    y_val = x_val[:, 0].astype(np.float64)
    w_val = np.ones(100)

    cfg = MLPConfig(
        hidden_dims=[16, 8],
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=3,
        patience=2,
        mixed_precision=False,
    )
    original = MLPAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "mlp.pt"
    original.save(path)
    assert path.exists()

    reloaded = MLPAlphaModel.load(path)
    np.testing.assert_allclose(
        original.predict(x_val), reloaded.predict(x_val), atol=1e-7, rtol=1e-6
    )

    # Loader puts net in eval() mode — two consecutive forwards must be identical (dropout off).
    first = reloaded.predict(x_val)
    second = reloaded.predict(x_val)
    np.testing.assert_array_equal(first, second)
    assert not reloaded._net.training
