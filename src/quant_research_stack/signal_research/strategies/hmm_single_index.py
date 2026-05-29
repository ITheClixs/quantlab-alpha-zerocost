"""HMM single-index strategy module for the v1 exception-path validation.

Pre-registered per docs/research/intake/2026-05-28-hmm-single-index-v1.md.

Variant grid (frozen at intake submission, 18 entries per instrument):
- state count ∈ {2, 3, 4}
- fit scheme ∈ {full_dev, expanding, rolling_5y}
- instrument ∈ {SPY, QQQ}

Allowed features (closed list):
- log_return
- realized_vol_21
- realized_vol_63
- drawdown_60
- drawdown_252
- trend_slope_50
- trend_slope_200
- range_pct_20
- volume_zscore_20

Default feature set: log_return, realized_vol_21, drawdown_60, range_pct_20.

Forbidden features hard-fail in HMMStrategyConfig.__post_init__.

Risk-on state is identified economically (highest mean return on fitting
window, tie-broken by lower realized vol when means differ by < 0.0001
daily). Raw HMM label permutations are not flips; economic-identity flips
are flagged separately and may demote per exception policy §4.5.
"""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.signal_research.validation.spec import (
    EXCEPTION_FORBIDDEN_FEATURE_TOKENS,
)


class FitScheme(enum.StrEnum):
    FULL_DEV = "full_dev"
    EXPANDING = "expanding"
    ROLLING_5Y = "rolling_5y"


ALLOWED_FEATURES: frozenset[str] = frozenset({
    "log_return",
    "realized_vol_21",
    "realized_vol_63",
    "drawdown_60",
    "drawdown_252",
    "trend_slope_50",
    "trend_slope_200",
    "range_pct_20",
    "volume_zscore_20",
})

DEFAULT_FEATURE_SET: tuple[str, ...] = (
    "log_return",
    "realized_vol_21",
    "drawdown_60",
    "range_pct_20",
)

TIER_1_INSTRUMENTS: tuple[str, ...] = ("SPY", "QQQ")
STATE_COUNTS: tuple[int, ...] = (2, 3, 4)
FIT_SCHEMES: tuple[FitScheme, ...] = (
    FitScheme.FULL_DEV,
    FitScheme.EXPANDING,
    FitScheme.ROLLING_5Y,
)

# Risk-on tie-breaker tolerance (per intake §5.2)
RISK_ON_MEAN_TIE_TOL_DAILY: float = 0.0001


class ForbiddenFeatureError(ValueError):
    """Raised when a feature name contains a forbidden information-source token."""


@dataclass(frozen=True)
class HMMStrategyConfig:
    """Configuration for one HMM variant on one instrument."""

    instrument: str
    state_count: int
    fit_scheme: FitScheme
    feature_set: tuple[str, ...] = DEFAULT_FEATURE_SET
    seed: int = 42

    def __post_init__(self) -> None:
        if self.instrument not in TIER_1_INSTRUMENTS:
            raise ValueError(
                f"instrument {self.instrument!r} is not Tier-1 "
                f"(allowed: {TIER_1_INSTRUMENTS})"
            )
        if self.state_count not in STATE_COUNTS:
            raise ValueError(
                f"state_count {self.state_count} not in {STATE_COUNTS}"
            )
        if not self.feature_set:
            raise ValueError("feature_set must not be empty")
        for feature in self.feature_set:
            if feature not in ALLOWED_FEATURES:
                raise ValueError(
                    f"feature {feature!r} not in allowed list "
                    f"{sorted(ALLOWED_FEATURES)}"
                )
            lower = feature.lower()
            for token in EXCEPTION_FORBIDDEN_FEATURE_TOKENS:
                if token in lower:
                    raise ForbiddenFeatureError(
                        f"feature {feature!r} contains forbidden token "
                        f"{token!r} — exception-path strategies must use "
                        "OHLCV-derived features only"
                    )

    @property
    def variant_name(self) -> str:
        return (
            f"hmm_{self.state_count}_{self.fit_scheme.value}_"
            f"{self.instrument.lower()}"
        )


def predeclared_variant_grid() -> list[HMMStrategyConfig]:
    """The frozen 18-variant grid (3 states × 3 fits × 2 instruments).

    Per intake §4.1: 'No variants may be added after seeing results.'
    """
    out: list[HMMStrategyConfig] = []
    for inst in TIER_1_INSTRUMENTS:
        for sc in STATE_COUNTS:
            for fs in FIT_SCHEMES:
                out.append(
                    HMMStrategyConfig(
                        instrument=inst, state_count=sc, fit_scheme=fs,
                    )
                )
    return out


# ============================================================================
# Feature computation — all past-only, no look-ahead
# ============================================================================


def compute_feature_panel(
    bars: pl.DataFrame, *, feature_set: tuple[str, ...] = DEFAULT_FEATURE_SET,
) -> pl.DataFrame:
    """Compute the requested feature set on a single-instrument bars DataFrame.

    Returns (date, *features) long-form. Rows with any null feature dropped.
    """
    df = bars.sort("date").with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return"),
    )
    if "realized_vol_21" in feature_set:
        df = df.with_columns(
            (
                pl.col("log_return").rolling_std(window_size=21) * (252.0 ** 0.5)
            ).alias("realized_vol_21")
        )
    if "realized_vol_63" in feature_set:
        df = df.with_columns(
            (
                pl.col("log_return").rolling_std(window_size=63) * (252.0 ** 0.5)
            ).alias("realized_vol_63")
        )
    if "drawdown_60" in feature_set:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").rolling_max(window_size=60) - 1.0)
            .alias("drawdown_60")
        )
    if "drawdown_252" in feature_set:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").rolling_max(window_size=252) - 1.0)
            .alias("drawdown_252")
        )
    if "trend_slope_50" in feature_set:
        df = df.with_columns(
            (
                (pl.col("close").log() - pl.col("close").shift(50).log()) / 50.0
            ).alias("trend_slope_50")
        )
    if "trend_slope_200" in feature_set:
        df = df.with_columns(
            (
                (pl.col("close").log() - pl.col("close").shift(200).log()) / 200.0
            ).alias("trend_slope_200")
        )
    if "range_pct_20" in feature_set:
        df = df.with_columns(
            (
                ((pl.col("high") - pl.col("low")) / pl.col("close"))
                .rolling_mean(window_size=20)
            ).alias("range_pct_20")
        )
    if "volume_zscore_20" in feature_set:
        df = df.with_columns(
            (
                (pl.col("volume") - pl.col("volume").rolling_mean(window_size=20))
                / pl.col("volume").rolling_std(window_size=20).clip(lower_bound=1e-9)
            ).alias("volume_zscore_20")
        )
    keep = ["date", *feature_set]
    return df.select(keep).drop_nulls()


# ============================================================================
# HMM fit + risk-on identification (per intake §5)
# ============================================================================


@dataclass(frozen=True)
class FittedHMM:
    """Result of one HMM fit on a window."""

    state_count: int
    fit_window_start: dt.date
    fit_window_end: dt.date
    transition_matrix: NDArray[np.float64]
    state_means_per_feature: NDArray[np.float64]  # (n_states, n_features)
    risk_on_state_id: int  # economic identity per §5
    risk_on_state_mean_return: float
    risk_on_state_realized_vol: float
    raw_label_to_economic_order: tuple[int, ...]
    model: object  # the hmmlearn model, for prediction


def _identify_risk_on_state(
    *,
    state_means_return: NDArray[np.float64],
    state_vols_return: NDArray[np.float64],
) -> int:
    """Apply the predeclared §5 risk-on identification rule.

    Primary: argmax mean return on fitting window.
    Tie-breaker: if two states have mean returns within RISK_ON_MEAN_TIE_TOL_DAILY,
    choose the lower-volatility state.
    """
    n_states = state_means_return.size
    if n_states == 0:
        raise ValueError("no states")
    if n_states == 1:
        return 0
    # Sort states by mean return descending
    order = np.argsort(-state_means_return)
    top = int(order[0])
    runner_up = int(order[1])
    if (
        state_means_return[top] - state_means_return[runner_up]
        < RISK_ON_MEAN_TIE_TOL_DAILY
    ):
        # Tie: choose lower vol
        if state_vols_return[runner_up] < state_vols_return[top]:
            return runner_up
    return top


def fit_hmm_window(
    *,
    features: pl.DataFrame,
    bars_for_returns: pl.DataFrame,
    config: HMMStrategyConfig,
    fit_start: dt.date,
    fit_end: dt.date,
) -> FittedHMM:
    """Fit one HMM on the [fit_start, fit_end] window of features and identify
    the risk-on state per §5.

    `features` must contain `date` and the columns in config.feature_set.
    `bars_for_returns` must contain (date, close) for the same instrument
    to compute per-state return statistics for risk-on identification.
    """
    from hmmlearn.hmm import GaussianHMM

    window = features.filter(
        (pl.col("date") >= fit_start) & (pl.col("date") <= fit_end)
    ).sort("date")
    if window.height < 60:
        raise ValueError(
            f"fit window too short ({window.height} days, need ≥ 60)"
        )
    X = window.select(list(config.feature_set)).to_numpy().astype(np.float64)
    model = GaussianHMM(
        n_components=config.state_count,
        covariance_type="diag",
        n_iter=200,
        random_state=config.seed,
    )
    model.fit(X)
    states = model.predict(X)

    # Compute per-state return statistics using log_return from bars
    log_returns_df = bars_for_returns.sort("date").with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret")
    ).drop_nulls(subset=["log_ret"])
    returns_window = log_returns_df.filter(
        (pl.col("date") >= fit_start) & (pl.col("date") <= fit_end)
    )
    if returns_window.height != window.height:
        # Align dates conservatively
        joined = window.select(["date"]).join(
            returns_window.select(["date", "log_ret"]),
            on="date",
            how="left",
        )
        returns_arr = joined["log_ret"].to_numpy().astype(np.float64)
    else:
        returns_arr = returns_window["log_ret"].to_numpy().astype(np.float64)

    state_means_return = np.zeros(config.state_count, dtype=np.float64)
    state_vols_return = np.zeros(config.state_count, dtype=np.float64)
    for s in range(config.state_count):
        mask = (states == s) & ~np.isnan(returns_arr)
        if mask.sum() > 1:
            state_means_return[s] = float(np.mean(returns_arr[mask]))
            state_vols_return[s] = float(np.std(returns_arr[mask], ddof=1))
        else:
            state_means_return[s] = -np.inf  # invalid → won't win argmax
            state_vols_return[s] = np.inf

    risk_on = _identify_risk_on_state(
        state_means_return=state_means_return,
        state_vols_return=state_vols_return,
    )

    # Economic-ordering: states sorted by mean return descending
    economic_order = tuple(int(i) for i in np.argsort(-state_means_return))

    return FittedHMM(
        state_count=config.state_count,
        fit_window_start=fit_start,
        fit_window_end=fit_end,
        transition_matrix=model.transmat_.astype(np.float64),
        state_means_per_feature=model.means_.astype(np.float64),
        risk_on_state_id=risk_on,
        risk_on_state_mean_return=float(state_means_return[risk_on]),
        risk_on_state_realized_vol=float(state_vols_return[risk_on]),
        raw_label_to_economic_order=economic_order,
        model=model,
    )


# ============================================================================
# Fit-scheme orchestration
# ============================================================================


def _calendar_year_ends_in_dev(
    *, features: pl.DataFrame, dev_end: dt.date,
) -> list[dt.date]:
    """Return the last available trading date in each calendar year within dev."""
    dev_dates = (
        features.filter(pl.col("date") <= dev_end)
        .sort("date")
        .with_columns(pl.col("date").dt.year().alias("year"))
    )
    if dev_dates.is_empty():
        return []
    grouped = dev_dates.group_by("year").agg(pl.col("date").max().alias("year_end"))
    return sorted(grouped["year_end"].to_list())


def fit_variant_models(
    *,
    config: HMMStrategyConfig,
    features: pl.DataFrame,
    bars_for_returns: pl.DataFrame,
    start: dt.date,
    dev_end: dt.date,
) -> list[FittedHMM]:
    """Fit the HMM(s) for the variant per its fit_scheme.

    - FULL_DEV: 1 fit on [start, dev_end]
    - EXPANDING: 1 fit per calendar year-end within dev, using [start, year_end]
    - ROLLING_5Y: 1 fit per calendar year-end within dev, using last 5 years
    """
    if config.fit_scheme == FitScheme.FULL_DEV:
        return [
            fit_hmm_window(
                features=features, bars_for_returns=bars_for_returns,
                config=config, fit_start=start, fit_end=dev_end,
            )
        ]
    year_ends = _calendar_year_ends_in_dev(features=features, dev_end=dev_end)
    fits: list[FittedHMM] = []
    if config.fit_scheme == FitScheme.EXPANDING:
        for ye in year_ends:
            try:
                fits.append(fit_hmm_window(
                    features=features, bars_for_returns=bars_for_returns,
                    config=config, fit_start=start, fit_end=ye,
                ))
            except ValueError:
                continue
    elif config.fit_scheme == FitScheme.ROLLING_5Y:
        for ye in year_ends:
            window_start = max(start, dt.date(ye.year - 5, 1, 1))
            try:
                fits.append(fit_hmm_window(
                    features=features, bars_for_returns=bars_for_returns,
                    config=config, fit_start=window_start, fit_end=ye,
                ))
            except ValueError:
                continue
    return fits


def predict_signal_long_or_cash(
    *,
    config: HMMStrategyConfig,
    fits: list[FittedHMM],
    features: pl.DataFrame,
    dev_end: dt.date,
) -> pl.DataFrame:
    """Generate (date, signal) using the variant's fit(s).

    Long-or-cash binary signal: 1 when the fit applicable to the date
    classifies it as risk-on, else 0. For multi-fit schemes the model
    in effect on date T is the one whose fit_window_end is the latest
    one ≤ T (no using-future-fit allowed).
    """
    if not fits:
        return features.select(["date"]).with_columns(pl.lit(0.0).alias("signal"))
    if len(fits) == 1:
        # Single fit covers everything
        fit = fits[0]
        df_sorted = features.sort("date")
        X = df_sorted.select(list(config.feature_set)).to_numpy().astype(np.float64)
        states = fit.model.predict(X)  # type: ignore[attr-defined]
        signal = (states == fit.risk_on_state_id).astype(np.float64)
        return df_sorted.select(["date"]).with_columns(pl.Series("signal", signal))

    # Multi-fit: pick the fit by date
    fits_sorted = sorted(fits, key=lambda f: f.fit_window_end)
    df_sorted = features.sort("date")
    dates = df_sorted["date"].to_list()
    X = df_sorted.select(list(config.feature_set)).to_numpy().astype(np.float64)
    signal = np.zeros(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        # Find latest fit whose fit_window_end ≤ d
        applicable: FittedHMM | None = None
        for f in fits_sorted:
            if f.fit_window_end <= d:
                applicable = f
            else:
                break
        if applicable is None:
            # No fit available yet (early dates before first year-end). Stay flat.
            signal[i] = 0.0
            continue
        if d <= dev_end and applicable.fit_window_end > d:
            # Should not happen by construction, but guard.
            signal[i] = 0.0
            continue
        # Predict with the applicable fit
        state = int(
            applicable.model.predict(X[i : i + 1])[0]  # type: ignore[attr-defined]
        )
        signal[i] = 1.0 if state == applicable.risk_on_state_id else 0.0
    return df_sorted.select(["date"]).with_columns(pl.Series("signal", signal))


# ============================================================================
# Variant outputs container
# ============================================================================


@dataclass(frozen=True)
class VariantOutputs:
    """All artifacts produced by one variant after fit + signal generation."""

    config: HMMStrategyConfig
    fits: list[FittedHMM]
    signal: pl.DataFrame  # (date, signal)
    feature_set_used: tuple[str, ...] = field(default_factory=tuple)
