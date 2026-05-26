"""HMM regime feature tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def test_hmm_regime_feature_emits_regime_id_per_row() -> None:
    pytest.importorskip("hmmlearn")
    from quant_research_stack.signal_research.papers.hmm_regime import (
        HMMRegimeConfig,
        HMMRegimeFeature,
    )

    rng = np.random.default_rng(0)
    n = 600
    rets = np.concatenate([
        rng.standard_normal(n // 2) * 0.005,
        rng.standard_normal(n // 2) * 0.025,
    ])
    panel = pl.DataFrame({
        "date": list(range(n)),
        "market_close": (100.0 * np.cumprod(1 + rets)).tolist(),
    })
    out = HMMRegimeFeature(HMMRegimeConfig()).features(panel)
    assert "regime_id" in out.columns
    assert out["regime_id"].n_unique() >= 2
