"""Validation spec — the contract a strategy proposal must satisfy.

The information-source declaration is the most important field. Per the
'no promotion without new information source' rule, strategies that
declare only `ohlcv` cannot reach `promotion_eligible` status without
explicit operator override.
"""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass, field


class InformationSource(enum.StrEnum):
    """Information channels a strategy may consume.

    The list is intentionally narrow: every new source either adds a
    fundamentally different data stream or must be argued in the intake.
    """

    OHLCV = "ohlcv"
    OPTIONS_IMPLIED_VOL = "options_implied_vol"
    OPTIONS_VOLUME = "options_volume"
    SENTIMENT_NEWS = "sentiment_news"
    SENTIMENT_SOCIAL = "sentiment_social"
    EARNINGS_FUNDAMENTALS = "earnings_fundamentals"
    MACRO_RATES = "macro_rates"
    MACRO_FX = "macro_fx"
    MACRO_COMMODITY = "macro_commodity"
    MICROSTRUCTURE_TICK = "microstructure_tick"
    MICROSTRUCTURE_BOOK = "microstructure_book"
    CROSS_ASSET = "cross_asset"
    EVENT_WINDOW = "event_window"
    ALTERNATIVE = "alternative"  # satellite, web-scraped, etc.


@dataclass(frozen=True)
class ValidationSpec:
    """Spec for a single strategy under validation."""

    strategy_name: str
    hypothesis_statement: str
    information_sources: tuple[InformationSource, ...]
    universe_tickers: list[str]
    start: dt.date
    end: dt.date
    dev_end: dt.date
    holdout_start: dt.date

    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0
    cost_stress_multipliers: tuple[float, ...] = (2.0, 3.0)
    delay_stress_bars: tuple[int, ...] = (1,)

    target_gross: float = 1.0
    equity: float = 1_000_000.0
    q_quantile: float = 0.20
    cohort: str = "full_universe"
    borrow_tier: str = "general"

    cpcv_n_partitions: int = 8
    cpcv_test_partitions: int = 2
    cpcv_label_horizon: int = 10
    cpcv_embargo: int = 5

    bootstrap_n_resamples: int = 2000
    bootstrap_seed: int = 42

    # 8-criteria promotion gates (per spec §6.1)
    gate_dev_sharpe_min: float = 1.5
    gate_holdout_sharpe_min: float = 0.5
    gate_cost_stress_min: float = 0.0
    gate_bootstrap_ci_lower_min: float = 0.0
    gate_pbo_max: float = 0.25
    gate_dsr_min: float = 0.50
    gate_must_beat_random: bool = True
    gate_must_beat_inverted: bool = True
    gate_no_single_regime_dominance: bool = True
    gate_no_single_period_dominance: bool = True

    # Banner / labels
    data_quality_label: str = "survivorship_prototype_only"
    constituent_survivorship_applicable: bool = True

    # Audit fields
    proposer: str = ""
    intake_date: dt.date = field(default_factory=lambda: dt.date(1970, 1, 1))
    intake_doc_ref: str = ""

    # Exception-path fields (single-index risk-timing exception policy)
    # See docs/research/intake/2026-05-28-single-index-risk-timing-exception.md
    # When exception_invoked=False, every default behavior is unchanged.
    exception_invoked: bool = False
    exception_policy_ref: str = ""
    declared_instrument: str = ""  # Tier-1 instrument for exception path
    single_instrument_scalar: bool = False  # required True for exception path
    feature_audit: tuple[str, ...] = ()  # honest declaration of features used

    @property
    def declares_non_ohlcv_source(self) -> bool:
        return any(s != InformationSource.OHLCV for s in self.information_sources)


# Accepted exception policy reference (commit 74ca502, 2026-05-28).
# A spec must match this exact string in exception_policy_ref to be eligible
# for the exception path.
ACCEPTED_EXCEPTION_POLICY_REF: str = (
    "docs/research/intake/2026-05-28-single-index-risk-timing-exception.md"
)

# Tier-1 allowed instruments (per accepted exception policy §1, amendment 6).
# Tier-2 instruments (BTCUSDT, ETHUSDT, ES, NQ) are NOT yet eligible.
TIER_1_INSTRUMENTS: frozenset[str] = frozenset({"SPY", "QQQ"})

# Forbidden feature substrings — any feature in spec.feature_audit whose
# lower-cased name contains one of these tokens triggers a hard fail under
# the exception path. The intent: the exception path is OHLCV-only on the
# instrument's own bars; any feature that hints at off-OHLCV information
# is rejected at validation time.
EXCEPTION_FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "vix",
    "vrp",
    "vvix",
    "skew",
    "vxn",
    "implied",
    "macro",
    "rates",
    "yield",
    "sentiment",
    "finbert",
    "news",
    "earnings",
    "fundamental",
    "cross_asset",
    "microstructure",
    "tick",
    "book",
    "fomc",
    "cpi",
    "event_window",
    "calendar",
)


def feature_audit_violation(feature_audit: tuple[str, ...]) -> str | None:
    """Return the first forbidden token found, or None if all features are OK."""
    for feature in feature_audit:
        lower = feature.lower()
        for token in EXCEPTION_FORBIDDEN_FEATURE_TOKENS:
            if token in lower:
                return f"feature {feature!r} contains forbidden token {token!r}"
    return None
