from __future__ import annotations

import numpy as np
import torch

from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel, Conv1DConfig


def test_conv1d_fit_predict() -> None:
    rng = np.random.default_rng(0)
    n, n_features = 100, 8
    x = rng.normal(size=(n, n_features)).astype(np.float64)
    y = (x.sum(axis=1) * 0.01 + rng.normal(size=n) * 0.01).astype(np.float64)
    w = np.ones(n, dtype=np.float64)
    cfg = Conv1DConfig(max_epochs=3, batch_size=16, device="cpu")
    model = Conv1DAlphaModel(cfg)
    model.fit(x, y, w, x[80:], y[80:], w[80:])
    pred = model.predict(x[:10])
    assert pred.shape == (10,)


def test_conv1d_save_load_roundtrip(tmp_path):
    # Conv1DAlphaModel now accepts 2D input: (n, n_features).
    # Internally reshapes to 3D (n, n_features, 1) for Conv1d.
    rng = np.random.default_rng(0)
    n_tr, n_val = 400, 100
    n_features = 8
    x_tr = rng.standard_normal((n_tr, n_features)).astype(np.float64)
    y_tr = x_tr[:, 0].astype(np.float64) + 0.1 * rng.standard_normal(n_tr)
    w_tr = np.ones(n_tr)
    x_val = rng.standard_normal((n_val, n_features)).astype(np.float64)
    y_val = x_val[:, 0].astype(np.float64)
    w_val = np.ones(n_val)

    cfg = Conv1DConfig(
        n_filters=8,
        kernel_sizes=[3, 5],
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=2,
        patience=2,
        device="cpu",
        random_state=0,
    )
    original = Conv1DAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "sequence.pt"
    original.save(path)
    assert path.exists()

    reloaded = Conv1DAlphaModel.load(path)
    np.testing.assert_allclose(
        original.predict(x_val), reloaded.predict(x_val), atol=1e-7, rtol=1e-6
    )
    first = reloaded.predict(x_val)
    second = reloaded.predict(x_val)
    np.testing.assert_array_equal(first, second)
    assert not reloaded._net.training
