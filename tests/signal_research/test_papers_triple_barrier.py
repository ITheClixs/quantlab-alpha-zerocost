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


def test_wrapper_trains_secondary_and_filters_positions() -> None:
    rng = np.random.default_rng(7)
    closes = 100.0 * np.cumprod(1.0 + rng.standard_normal(700) * 0.012)
    primary_positions = np.where(rng.random(700) > 0.5, 1.0, -1.0)
    features = np.column_stack(
        [
            np.r_[0.0, np.diff(np.log(closes))],
            np.abs(np.r_[0.0, np.diff(np.log(closes))]),
            primary_positions,
        ]
    )
    eligibility = check_eligibility(
        PrimarySignalStats(
            validation_net_sharpe=0.6,
            validation_hit_rate=0.54,
            validation_expectancy=0.001,
            event_count=700,
            single_asset_or_cross_sectional="single_asset",
            is_inverted_superior=False,
            is_near_duplicate=False,
        )
    )
    wrapper = TripleBarrierWrapper(
        TripleBarrierConfig(vertical_barrier_days=5, vol_estimator_window=10),
        eligibility,
    )

    wrapper.train_secondary(
        primary_positions=primary_positions,
        closes=closes,
        features_at_event=features,
    )

    probabilities = wrapper.predict_trade_probability(features)
    assert probabilities.shape == (700,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.all(
        wrapper.filter_positions(
            primary_positions=primary_positions,
            features_at_event=features,
            probability_threshold=1.01,
        )
        == 0.0
    )
