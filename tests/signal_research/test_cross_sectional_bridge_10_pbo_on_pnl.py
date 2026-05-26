"""Bridge contract #10: cross-sectional PBO uses L/S PnL series, not rank IC."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.pbo_extensions import (
    compute_three_tier_pbo,
)


def test_cs_pbo_consumes_pnl_series_not_rank_ic() -> None:
    rng = np.random.default_rng(0)
    pnl = rng.standard_normal((480, 12)) * 0.01
    profile = np.array(["sp500_cs"] * 12)
    family = np.array(["AVL"] * 6 + ["GKX"] * 6)
    res = compute_three_tier_pbo(returns=pnl, profile=profile, family=family)
    assert 0.0 <= res.raw_global <= 1.0
