from __future__ import annotations

import numpy as np
import torch

from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel, Conv1DConfig


def test_conv1d_fit_predict() -> None:
    rng = np.random.default_rng(0)
    n, seq_len, channels = 100, 8, 5
    x = rng.normal(size=(n, seq_len, channels)).astype(np.float32)
    y = (x.sum(axis=(1, 2)) * 0.01 + rng.normal(size=n) * 0.01).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    cfg = Conv1DConfig(max_epochs=3, batch_size=16, device="cpu")
    model = Conv1DAlphaModel(cfg)
    model.fit(x, y, w, x[80:], y[80:], w[80:])
    pred = model.predict(x[:10])
    assert pred.shape == (10,)


def test_conv1d_save_load_roundtrip(tmp_path):
    # Conv1DAlphaModel takes 3-D input: (batch, seq_len, channels).
    # seq_len=8, channels=4 — same shape the real fit() expects.
    rng = np.random.default_rng(0)
    n_tr, n_val = 400, 100
    seq_len, channels = 8, 4
    x_tr = rng.standard_normal((n_tr, seq_len, channels)).astype(np.float64)
    y_tr = x_tr[:, 0, 0].astype(np.float64) + 0.1 * rng.standard_normal(n_tr)
    w_tr = np.ones(n_tr)
    x_val = rng.standard_normal((n_val, seq_len, channels)).astype(np.float64)
    y_val = x_val[:, 0, 0].astype(np.float64)
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
