"""Mandatory tests for the HMM single-index v1 exception-path implementation.

These tests must pass before the dedicated HMM validation can run.

Coverage (per user requirements list, items 9.i through 9.xi):
1. default no-OHLCV behavior unchanged when exception_invoked=False
2. exception path only activates for accepted policy reference
3. non-Tier-1 instruments fail the exception path
4. forbidden features hard-fail
5. current-constituent / cross-sectional strategies cannot invoke the exception
6. raw HMM label swaps do not fail stability
7. economic identity flips do fail stability
8. cash-leg conservative assumption is used for gate evaluation
9. 2-bar delay stress is computed
10. status cannot exceed paper_trade_candidate from this validation
11. no production_candidate emitted by this run
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quant_research_stack.signal_research.status import CandidateStatus
from quant_research_stack.signal_research.strategies.hmm_runner import (
    HMMRunnerSpec,
    assign_exception_status,
)
from quant_research_stack.signal_research.strategies.hmm_single_index import (
    DEFAULT_FEATURE_SET,
    FitScheme,
    ForbiddenFeatureError,
    HMMStrategyConfig,
    predeclared_variant_grid,
)
from quant_research_stack.signal_research.validation.cash_leg_reporting import (
    CASH_CONSERVATIVE,
    CASH_TBILL,
    DEFAULT_CASH_FEE_BPS_ANNUAL,
    GATE_ASSUMPTION,
    compute_long_or_cash_returns,
)
from quant_research_stack.signal_research.validation.exception_robustness import (
    economic_identity_stability,
)
from quant_research_stack.signal_research.validation.pipeline import (
    _exception_path_qualifies,
)
from quant_research_stack.signal_research.validation.spec import (
    ACCEPTED_EXCEPTION_POLICY_REF,
    EXCEPTION_FORBIDDEN_FEATURE_TOKENS,
    InformationSource,
    ValidationSpec,
    feature_audit_violation,
)

# ============================================================================
# Test fixtures — minimal frozen spec with everything turned off
# ============================================================================


def _base_spec(**overrides) -> ValidationSpec:
    defaults: dict[str, object] = dict(
        strategy_name="test_strategy",
        hypothesis_statement="test",
        information_sources=(InformationSource.OHLCV,),
        universe_tickers=["SPY"],
        start=dt.date(2010, 1, 1),
        end=dt.date(2026, 5, 26),
        dev_end=dt.date(2022, 12, 31),
        holdout_start=dt.date(2023, 1, 1),
    )
    defaults.update(overrides)
    return ValidationSpec(**defaults)  # type: ignore[arg-type]


# ============================================================================
# Test 1: default behavior unchanged when exception_invoked=False
# ============================================================================


def test_default_behavior_unchanged_when_exception_not_invoked() -> None:
    """When exception_invoked=False, _exception_path_qualifies returns False
    even if other fields look exception-shaped."""
    spec = _base_spec(
        exception_invoked=False,
        exception_policy_ref=ACCEPTED_EXCEPTION_POLICY_REF,
        declared_instrument="SPY",
        single_instrument_scalar=True,
    )
    qualifies, reason = _exception_path_qualifies(spec)
    assert qualifies is False
    assert "exception_invoked is False" in reason


# ============================================================================
# Test 2: exception path only activates for accepted policy reference
# ============================================================================


def test_exception_path_requires_exact_policy_ref() -> None:
    spec = _base_spec(
        exception_invoked=True,
        exception_policy_ref="docs/research/intake/some-other-policy.md",
        declared_instrument="SPY",
        single_instrument_scalar=True,
    )
    qualifies, reason = _exception_path_qualifies(spec)
    assert qualifies is False
    assert "exception_policy_ref does not match" in reason


def test_exception_path_qualifies_with_correct_policy_ref() -> None:
    spec = _base_spec(
        exception_invoked=True,
        exception_policy_ref=ACCEPTED_EXCEPTION_POLICY_REF,
        declared_instrument="SPY",
        single_instrument_scalar=True,
        feature_audit=("log_return", "realized_vol_21"),
    )
    qualifies, _reason = _exception_path_qualifies(spec)
    assert qualifies is True


# ============================================================================
# Test 3: non-Tier-1 instruments fail the exception path
# ============================================================================


@pytest.mark.parametrize("instrument", ["BTCUSDT", "ETHUSDT", "ES", "NQ", "AAPL", ""])
def test_non_tier_1_instruments_fail_exception_path(instrument: str) -> None:
    spec = _base_spec(
        exception_invoked=True,
        exception_policy_ref=ACCEPTED_EXCEPTION_POLICY_REF,
        declared_instrument=instrument,
        single_instrument_scalar=True,
    )
    qualifies, reason = _exception_path_qualifies(spec)
    assert qualifies is False
    assert "not Tier-1" in reason


# ============================================================================
# Test 4: forbidden features hard-fail
# ============================================================================


@pytest.mark.parametrize("forbidden_feature", [
    "vix_term_structure",
    "vrp_premium",
    "vvix_ratio",
    "skew_z60",
    "vxn_to_vix",
    "implied_vol_30d",
    "macro_dgs10",
    "yield_curve_slope",
    "sentiment_finbert_score",
    "news_polarity",
    "earnings_drift",
    "fundamental_pe",
    "cross_asset_bond_yield",
    "microstructure_imbalance",
    "tick_intensity",
    "book_depth",
    "fomc_window_flag",
    "cpi_window_flag",
    "event_window_indicator",
    "calendar_quarter_end",
])
def test_forbidden_feature_audit_fails(forbidden_feature: str) -> None:
    violation = feature_audit_violation((forbidden_feature,))
    assert violation is not None
    assert forbidden_feature in violation


def test_forbidden_feature_hardfails_in_strategy_config() -> None:
    """The HMM strategy config must reject forbidden tokens at construction."""
    with pytest.raises((ForbiddenFeatureError, ValueError)):
        HMMStrategyConfig(
            instrument="SPY",
            state_count=2,
            fit_scheme=FitScheme.FULL_DEV,
            feature_set=("log_return", "vix_term_structure"),  # forbidden
        )


def test_exception_path_fails_when_feature_audit_has_forbidden_token() -> None:
    spec = _base_spec(
        exception_invoked=True,
        exception_policy_ref=ACCEPTED_EXCEPTION_POLICY_REF,
        declared_instrument="SPY",
        single_instrument_scalar=True,
        feature_audit=("log_return", "vix_proxy"),
    )
    qualifies, reason = _exception_path_qualifies(spec)
    assert qualifies is False
    assert "forbidden" in reason


# ============================================================================
# Test 5: current-constituent / cross-sectional strategies cannot invoke exception
# ============================================================================


def test_cross_sectional_strategy_cannot_invoke_exception() -> None:
    """A strategy with single_instrument_scalar=False fails the exception path
    regardless of how other fields look."""
    spec = _base_spec(
        exception_invoked=True,
        exception_policy_ref=ACCEPTED_EXCEPTION_POLICY_REF,
        declared_instrument="SPY",
        single_instrument_scalar=False,  # this is the cross-sectional indicator
        universe_tickers=["AAPL", "MSFT", "NVDA"],
    )
    qualifies, reason = _exception_path_qualifies(spec)
    assert qualifies is False
    assert "single_instrument_scalar must be True" in reason


# ============================================================================
# Test 6: raw HMM label swaps do not fail stability
# ============================================================================


def test_raw_label_swap_without_economic_change_does_not_fail_stability() -> None:
    """Two refits whose risk-on state has different raw ID but identical
    (mean, vol) signature should NOT register as an economic flip."""
    from quant_research_stack.signal_research.strategies.hmm_single_index import (
        FittedHMM,
    )

    transition = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    means = np.array([[0.001], [-0.001]], dtype=np.float64)
    fit_a = FittedHMM(
        state_count=2,
        fit_window_start=dt.date(2010, 1, 1),
        fit_window_end=dt.date(2015, 12, 31),
        transition_matrix=transition,
        state_means_per_feature=means,
        risk_on_state_id=0,
        risk_on_state_mean_return=0.0005,
        risk_on_state_realized_vol=0.010,
        raw_label_to_economic_order=(0, 1),
        model=object(),
    )
    fit_b = FittedHMM(
        state_count=2,
        fit_window_start=dt.date(2010, 1, 1),
        fit_window_end=dt.date(2016, 12, 31),
        transition_matrix=transition,
        state_means_per_feature=means,
        risk_on_state_id=1,  # different raw label
        risk_on_state_mean_return=0.0005,  # same economic mean
        risk_on_state_realized_vol=0.010,  # same economic vol
        raw_label_to_economic_order=(1, 0),
        model=object(),
    )
    report = economic_identity_stability([fit_a, fit_b])
    assert report.raw_label_flips == 1  # raw label flip recorded informationally
    assert report.n_economic_flips == 0  # but NOT counted as economic flip
    assert report.passes_stability_gate is True


# ============================================================================
# Test 7: economic identity flips DO fail stability
# ============================================================================


def test_economic_identity_flip_counts_as_failure() -> None:
    """A material change in the risk-on state's mean return between refits
    IS an economic flip and counts toward the flip-rate gate."""
    from quant_research_stack.signal_research.strategies.hmm_single_index import (
        FittedHMM,
    )

    transition = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    means = np.array([[0.001], [-0.001]], dtype=np.float64)
    fit_a = FittedHMM(
        state_count=2,
        fit_window_start=dt.date(2010, 1, 1),
        fit_window_end=dt.date(2015, 12, 31),
        transition_matrix=transition,
        state_means_per_feature=means,
        risk_on_state_id=0,
        risk_on_state_mean_return=0.0010,
        risk_on_state_realized_vol=0.010,
        raw_label_to_economic_order=(0, 1),
        model=object(),
    )
    fit_b = FittedHMM(
        state_count=2,
        fit_window_start=dt.date(2010, 1, 1),
        fit_window_end=dt.date(2016, 12, 31),
        transition_matrix=transition,
        state_means_per_feature=means,
        risk_on_state_id=0,  # same raw label
        risk_on_state_mean_return=0.0050,  # but mean jumped by 0.004 (~> tol)
        risk_on_state_realized_vol=0.010,
        raw_label_to_economic_order=(0, 1),
        model=object(),
    )
    report = economic_identity_stability([fit_a, fit_b])
    assert report.raw_label_flips == 0
    assert report.n_economic_flips == 1
    assert report.flip_rate == 1.0
    assert report.passes_stability_gate is False


def test_economic_stability_flip_rate_above_20pct_fails_gate() -> None:
    from quant_research_stack.signal_research.strategies.hmm_single_index import (
        FittedHMM,
    )

    transition = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    means = np.array([[0.001], [-0.001]], dtype=np.float64)

    def _fit(mean: float) -> FittedHMM:
        return FittedHMM(
            state_count=2,
            fit_window_start=dt.date(2010, 1, 1),
            fit_window_end=dt.date(2015, 12, 31),
            transition_matrix=transition,
            state_means_per_feature=means,
            risk_on_state_id=0,
            risk_on_state_mean_return=mean,
            risk_on_state_realized_vol=0.010,
            raw_label_to_economic_order=(0, 1),
            model=object(),
        )

    # 5 fits, 2 economic flips → flip rate = 2/4 = 50% > 20%
    fits = [
        _fit(0.001),
        _fit(0.001),
        _fit(0.010),  # flip
        _fit(0.010),
        _fit(0.001),  # flip
    ]
    report = economic_identity_stability(fits)
    assert report.flip_rate >= 0.20
    assert report.passes_stability_gate is False


# ============================================================================
# Test 8: cash-leg conservative assumption is the gating assumption
# ============================================================================


def test_gate_assumption_is_conservative_after_fee() -> None:
    assert GATE_ASSUMPTION.name == CASH_CONSERVATIVE.name


def test_conservative_cash_leg_uses_tbill_minus_fee() -> None:
    """Verify the conservative assumption produces a return lower than
    or equal to the T-bill assumption for the same underlying flow."""
    rng = np.random.default_rng(0)
    n = 60
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    u_ret_df = pl.DataFrame({"date": dates, "u_ret": rng.standard_normal(n) * 0.01})
    tbill_panel = pl.DataFrame({
        "date": dates,
        "tbill_rate_pct": np.full(n, 5.0),  # 5% T-bill
    })
    # Fully cash position
    position = np.zeros(n, dtype=np.float64)

    tbill_result = compute_long_or_cash_returns(
        underlying_returns=u_ret_df, position=position,
        tbill_panel=tbill_panel, assumption=CASH_TBILL,
        cost_bps_one_way=0.5,
    )
    cons_result = compute_long_or_cash_returns(
        underlying_returns=u_ret_df, position=position,
        tbill_panel=tbill_panel, assumption=CASH_CONSERVATIVE,
        cost_bps_one_way=0.5,
    )
    assert cons_result.cumulative_return < tbill_result.cumulative_return


# ============================================================================
# Test 9: 2-bar delay stress is computed
# ============================================================================


def test_2_bar_delay_in_runner_spec() -> None:
    runner_spec = HMMRunnerSpec()
    assert 1 in runner_spec.delay_stress_bars
    assert 2 in runner_spec.delay_stress_bars


# ============================================================================
# Test 10: status cannot exceed exception_review_required from this validation
# ============================================================================


def test_assign_exception_status_never_emits_paper_trade_candidate() -> None:
    """The HMM v1 validation may emit EXCEPTION_REVIEW_REQUIRED at most.
    PAPER_TRADE_CANDIDATE and PRODUCTION_CANDIDATE are blocked here."""
    from quant_research_stack.signal_research.strategies.hmm_runner import (
        GateScorecard,
    )

    scorecard_all_pass = GateScorecard(
        dev_sharpe_pass=True, holdout_sharpe_pass=True,
        cost_stress_2x_pass=True, cost_stress_3x_pass=True,
        delay_1d_pass=True, delay_2d_pass=True,
        max_dd_or_calmar_pass=True, year_share_pass=True,
        quarter_share_pass=True, min_positive_years_pass=True,
        survives_excl_2020_pass=True, survives_excl_2022_pass=True,
        survives_pre_2020_subsample_pass=True,
        beats_buy_and_hold_pass=True, beats_vol_targeted_pass=True,
        beats_sma_50_200_pass=True, beats_mom_12_1_pass=True,
        random_baseline_fails_pass=True, inverted_baseline_fails_pass=True,
        bootstrap_ci_lower_pass=True, pbo_pass=True, dsr_pass=True,
        economic_identity_stability_pass=True,
        cash_leg_conservative_pass=True,
        all_pass=True,
    )
    status = assign_exception_status(scorecard=scorecard_all_pass, category="hmm_variant")
    assert status == CandidateStatus.EXCEPTION_REVIEW_REQUIRED
    assert status < CandidateStatus.PAPER_TRADE_CANDIDATE
    assert status < CandidateStatus.PRODUCTION_CANDIDATE


# ============================================================================
# Test 11: no PRODUCTION_CANDIDATE emitted by this validation pipeline
# ============================================================================


def test_no_production_candidate_emitted_by_pipeline() -> None:
    from quant_research_stack.signal_research.strategies.hmm_runner import (
        GateScorecard,
    )

    # Even with all gates pass, status stays at EXCEPTION_REVIEW_REQUIRED
    sc_pass = GateScorecard(
        dev_sharpe_pass=True, holdout_sharpe_pass=True,
        cost_stress_2x_pass=True, cost_stress_3x_pass=True,
        delay_1d_pass=True, delay_2d_pass=True,
        max_dd_or_calmar_pass=True, year_share_pass=True,
        quarter_share_pass=True, min_positive_years_pass=True,
        survives_excl_2020_pass=True, survives_excl_2022_pass=True,
        survives_pre_2020_subsample_pass=True,
        beats_buy_and_hold_pass=True, beats_vol_targeted_pass=True,
        beats_sma_50_200_pass=True, beats_mom_12_1_pass=True,
        random_baseline_fails_pass=True, inverted_baseline_fails_pass=True,
        bootstrap_ci_lower_pass=True, pbo_pass=True, dsr_pass=True,
        economic_identity_stability_pass=True,
        cash_leg_conservative_pass=True,
        all_pass=True,
    )
    for category in ("hmm_variant", "baseline"):
        status = assign_exception_status(scorecard=sc_pass, category=category)
        assert status != CandidateStatus.PRODUCTION_CANDIDATE
        assert status != CandidateStatus.PAPER_TRADE_CANDIDATE


# ============================================================================
# Bonus: variant registry shape (sanity)
# ============================================================================


def test_predeclared_variant_grid_is_18_per_instrument_set() -> None:
    grid = predeclared_variant_grid()
    assert len(grid) == 18  # 2 instruments × 3 states × 3 schemes
    spy_count = sum(1 for c in grid if c.instrument == "SPY")
    qqq_count = sum(1 for c in grid if c.instrument == "QQQ")
    assert spy_count == 9
    assert qqq_count == 9


def test_default_feature_set_is_intake_specified() -> None:
    assert DEFAULT_FEATURE_SET == (
        "log_return", "realized_vol_21", "drawdown_60", "range_pct_20",
    )


def test_default_fee_is_25_bps() -> None:
    """Per accepted exception policy §4.14(c) amendment 5."""
    assert DEFAULT_CASH_FEE_BPS_ANNUAL == 25.0


def test_forbidden_tokens_cover_intake_categories() -> None:
    """Quick sanity: forbidden tokens must include vix, macro, sentiment etc."""
    required = {"vix", "macro", "sentiment", "tick", "cross_asset", "fomc", "cpi", "calendar"}
    assert required.issubset(set(EXCEPTION_FORBIDDEN_FEATURE_TOKENS))
