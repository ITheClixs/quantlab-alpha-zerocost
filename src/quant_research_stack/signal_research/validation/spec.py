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

    @property
    def declares_non_ohlcv_source(self) -> bool:
        return any(s != InformationSource.OHLCV for s in self.information_sources)
