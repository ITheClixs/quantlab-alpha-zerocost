"""Conv1D sequence model — optional under full_v1 (spec §4.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.sequence import (
    Conv1DEqConfig,
    Conv1DEqModel,
)


def test_conv1d_requires_lookback_feature_tensor() -> None:
    cfg = Conv1DEqConfig(lookback=10, feature_channels=4, max_epochs=2)
    m = Conv1DEqModel(cfg)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, cfg.lookback, cfg.feature_channels)).astype(np.float64)
    y = rng.standard_normal(100).astype(np.float64) * 0.1
    m.fit(x=x, y=y)
    p = m.predict(x[:5])
    assert p.shape == (5,)


def test_conv1d_save_load_round_trip(tmp_path: Path) -> None:
    cfg = Conv1DEqConfig(lookback=8, feature_channels=3, max_epochs=2)
    m = Conv1DEqModel(cfg)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, cfg.lookback, cfg.feature_channels))
    y = rng.standard_normal(50) * 0.1
    m.fit(x=x, y=y)
    out = tmp_path / "seq.pt"
    m.save(out)
    m2 = Conv1DEqModel.load(out)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-5)
