"""Three-tier PBO reporting (spec §4.7)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.pbo_extensions import (
    PBOMultiResult,
    compute_three_tier_pbo,
)


def test_three_tier_pbo_reports_all_three_values() -> None:
    rng = np.random.default_rng(0)
    T, S = 480, 100
    returns = rng.standard_normal((T, S)) * 0.01
    profile = np.array(["sp500"] * 50 + ["nasdaq"] * 50)
    family = np.array(["MOM"] * 25 + ["MR"] * 25 + ["MOM"] * 25 + ["MR"] * 25)
    res = compute_three_tier_pbo(returns=returns, profile=profile, family=family)
    assert isinstance(res, PBOMultiResult)
    assert 0.0 <= res.raw_global <= 1.0
    assert "sp500" in res.per_profile
    assert "nasdaq" in res.per_profile
    assert "MOM" in res.per_family
    assert "MR" in res.per_family
