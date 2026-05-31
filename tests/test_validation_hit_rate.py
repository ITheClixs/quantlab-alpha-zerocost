from __future__ import annotations

import math

import pytest

from quant_research_stack.validation.hit_rate import (
    HitRateResult,
    ScoredSignal,
    compute_hit_rate,
)


def _s(signal_id: str, pred_dir: int, real_dir: int, weight: float = 1.0,
       s2_decision: str = "pass") -> ScoredSignal:
    return ScoredSignal(
        signal_id=signal_id,
        predicted_direction=pred_dir,
        realized_direction=real_dir,
        weight=weight,
        s2_decision=s2_decision,
    )


def test_empty_signals_yields_zero_hit_rate_and_zero_block_rate() -> None:
    result = compute_hit_rate([])
    assert isinstance(result, HitRateResult)
    assert result.hit_rate == 0.0
    assert result.n_signals == 0
    assert result.n_hits == 0
    assert result.governor_block_rate == 0.0


def test_all_correct_direction_yields_hit_rate_one() -> None:
    signals = [_s("a", 1, 1), _s("b", -1, -1), _s("c", 1, 1)]
    result = compute_hit_rate(signals)
    assert result.hit_rate == 1.0
    assert result.n_signals == 3
    assert result.n_hits == 3
    assert result.governor_block_rate == 0.0


def test_half_correct_yields_hit_rate_half() -> None:
    signals = [_s("a", 1, 1), _s("b", 1, -1)]
    result = compute_hit_rate(signals)
    assert result.hit_rate == 0.5
    assert result.n_signals == 2
    assert result.n_hits == 1


def test_weighted_hit_rate_respects_weights() -> None:
    signals = [_s("a", 1, 1, weight=9.0), _s("b", 1, -1, weight=1.0)]
    result = compute_hit_rate(signals)
    assert math.isclose(result.hit_rate, 0.9, abs_tol=1e-9)


def test_veto_excluded_from_hit_rate_numerator_and_denominator() -> None:
    signals = [
        _s("a", 1, 1, s2_decision="pass"),
        _s("b", 0, 1, s2_decision="veto"),
    ]
    result = compute_hit_rate(signals)
    assert result.hit_rate == 1.0
    assert result.n_signals == 1
    assert result.governor_block_rate == 0.5


def test_insufficient_evidence_counts_in_block_rate_not_hit_rate() -> None:
    signals = [
        _s("a", 1, 1, s2_decision="pass"),
        _s("b", 0, 0, s2_decision="insufficient_evidence"),
    ]
    result = compute_hit_rate(signals)
    assert result.hit_rate == 1.0
    assert result.governor_block_rate == 0.5


def test_zero_realized_direction_excluded() -> None:
    signals = [_s("a", 1, 0, s2_decision="pass"), _s("b", 1, 1, s2_decision="pass")]
    result = compute_hit_rate(signals)
    assert math.isclose(result.hit_rate, 0.5, abs_tol=1e-9)
    assert result.n_signals == 2
    assert result.n_hits == 1


def test_zero_weight_signals_ignored() -> None:
    signals = [_s("a", 1, 1, weight=0.0), _s("b", 1, -1, weight=1.0)]
    result = compute_hit_rate(signals)
    assert result.hit_rate == 0.0
    assert result.n_signals == 1
    assert result.n_hits == 0


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="weight must be non-negative"):
        ScoredSignal(
            signal_id="x", predicted_direction=1, realized_direction=1,
            weight=-1.0, s2_decision="pass",
        )


def test_all_veto_returns_zero_hit_rate_and_full_block_rate() -> None:
    signals = [_s("a", 0, 0, s2_decision="veto"), _s("b", 0, 0, s2_decision="veto")]
    result = compute_hit_rate(signals)
    assert result.hit_rate == 0.0
    assert result.n_signals == 0
    assert result.governor_block_rate == 1.0
