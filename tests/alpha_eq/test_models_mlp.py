"""MLP S1-EQ model — fit/predict/save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.mlp import MLPEqConfig, MLPEqModel


def test_mlp_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1]).astype(np.float32) + rng.standard_normal(200).astype(np.float32) * 0.1
    m = MLPEqModel(MLPEqConfig(hidden_dims=(32, 16), max_epochs=2, batch_size=64, seed=42))
    m.fit(x=x.astype(np.float64), y=y.astype(np.float64))
    out = tmp_path / "mlp.pt"
    m.save(out)
    m2 = MLPEqModel.load(out)
    np.testing.assert_allclose(
        m.predict(x.astype(np.float64)[:5]),
        m2.predict(x.astype(np.float64)[:5]),
        atol=1e-5,
    )
