"""Triple-barrier wrapper tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_research_stack.signal_research.methodology.meta_labeling import (
    PrimarySignalStats,
    check_eligibility,
)
from quant_research_stack.signal_research.papers.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierWrapper,
    label_triple_barrier,
)


def test_label_triple_barrier_shape() -> None:
    rng = np.random.default_rng(0)
    closes = 100.0 * np.cumprod(1.0 + rng.standard_normal(500) * 0.01)
    positions = (rng.random(500) > 0.5).astype(float) * 2 - 1
    labels = label_triple_barrier(
        close=closes, positions=positions, cfg=TripleBarrierConfig()
    )
    assert labels.size == 500


def test_wrapper_refuses_ineligible_primary() -> None:
    bad = PrimarySignalStats(
        validation_net_sharpe=-0.1,
        validation_hit_rate=0.45,
        validation_expectancy=-0.001,
        event_count=300,
        single_asset_or_cross_sectional="single_asset",
        is_inverted_superior=False,
        is_near_duplicate=False,
    )
    elig = check_eligibility(bad)
    with pytest.raises(RuntimeError):
        TripleBarrierWrapper(TripleBarrierConfig(), elig)
