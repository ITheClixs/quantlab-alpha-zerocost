"""Failure classifier (spec §4.10, §6.3) — 13 categories."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FailureCategory(enum.StrEnum):
    HIGH_PBO = "high_pbo"
    LOW_DSR = "low_dsr"
    COST_FAILURE = "cost_failure"
    REGIME_CONCENTRATION = "regime_concentration"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    TOO_FEW_TRADES = "too_few_trades"
    DELAY_STRESS_FAIL = "delay_stress_fail"
    SINGLE_PERIOD_DOMINANCE = "single_period_dominance"
    OVER_CORRELATED_WITH_BASELINE = "over_correlated_with_baseline"
    RANDOMIZATION_FAIL = "randomization_fail"
    DATA_QUALITY_FAIL = "data_quality_fail"
    HOLDOUT_FAILURE = "holdout_failure"
    CAPACITY_FAILURE = "capacity_failure"


@dataclass(frozen=True)
class CandidateFailureRecord:
    strategy_id: str
    categories: list[FailureCategory]


def all_failure_categories() -> list[FailureCategory]:
    return list(FailureCategory)
