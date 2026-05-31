"""Attribution analytics for the VRP × HMM interaction test.

For each variant, compute:
- correlation with HMM-only daily net returns
- correlation with VRP-only daily net returns
- incremental Sharpe over HMM-only (raw + hedged-residual)
- incremental Sharpe over VRP-only (raw + hedged-residual)
- PnL by year
- PnL by HMM regime
- crisis-exclusion Sharpes: remove {2020, 2022, 2023-2026}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class AttributionMetrics:
    name: str
    sharpe_dev: float
    corr_dev_with_hmm_only: float
    corr_dev_with_vrp_only: float
    incremental_sharpe_over_hmm_only: float
    incremental_sharpe_over_vrp_only: float
    residual_sharpe_over_hmm_only: float
    residual_sharpe_over_vrp_only: float
    pnl_by_year: dict[int, float] = field(default_factory=dict)
    pnl_by_regime: dict[str, float] = field(default_factory=dict)
    sharpe_excl_2020: float = float("nan")
    sharpe_excl_2022: float = float("nan")
    sharpe_excl_holdout_period: float = float("nan")


def _sharpe(rets: NDArray[np.float64]) -> float:
    if rets.size < 2:
        return 0.0
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(rets)) / sd * float(np.sqrt(252.0))


def _correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    a2, b2 = a[-n:], b[-n:]
    if np.std(a2) == 0 or np.std(b2) == 0:
        return 0.0
    return float(np.corrcoef(a2, b2)[0, 1])


def _residual_sharpe(
    target: NDArray[np.float64], explanator: NDArray[np.float64]
) -> float:
    """Sharpe of `target` after regressing on `explanator` (intercept + slope)
    over the overlapping window. Captures the orthogonal component."""
    n = min(target.size, explanator.size)
    if n < 30:
        return float("nan")
    y = target[-n:]
    x = explanator[-n:]
    X = np.column_stack([np.ones_like(x), x])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - (coefs[0] + coefs[1] * x)
    return _sharpe(residual)


def attribution_for_variant(
    *,
    name: str,
    daily_dev: pl.DataFrame,
    hmm_only_dev: pl.DataFrame,
    vrp_only_dev: pl.DataFrame,
    hmm_panel: pl.DataFrame,
) -> AttributionMetrics:
    rets = daily_dev["net_return"].to_numpy().astype(np.float64)
    hmm = hmm_only_dev["net_return"].to_numpy().astype(np.float64)
    vrp = vrp_only_dev["net_return"].to_numpy().astype(np.float64)
    sharpe_self = _sharpe(rets)
    sharpe_hmm = _sharpe(hmm)
    sharpe_vrp = _sharpe(vrp)

    pnl_by_year: dict[int, float] = {}
    if daily_dev.height > 0:
        with_year = daily_dev.with_columns(pl.col("date").dt.year().alias("year"))
        for row in with_year.group_by("year").agg(
            pl.col("net_return").sum().alias("pnl")
        ).iter_rows(named=True):
            pnl_by_year[int(row["year"])] = float(row["pnl"])

    pnl_by_regime: dict[str, float] = {}
    if hmm_panel.height > 0 and daily_dev.height > 0:
        joined = daily_dev.join(
            hmm_panel.select(["date", "risk_on"]), on="date", how="left"
        )
        for row in joined.group_by("risk_on").agg(
            pl.col("net_return").sum().alias("pnl")
        ).iter_rows(named=True):
            key = "risk_on" if row["risk_on"] == 1 else "risk_off"
            pnl_by_regime[key] = float(row["pnl"])

    def _sharpe_excluding(daily: pl.DataFrame, predicate: pl.Expr) -> float:
        sub = daily.filter(predicate)
        if sub.height < 30:
            return float("nan")
        return _sharpe(sub["net_return"].to_numpy().astype(np.float64))

    s_excl_2020 = _sharpe_excluding(daily_dev, pl.col("date").dt.year() != 2020)
    s_excl_2022 = _sharpe_excluding(daily_dev, pl.col("date").dt.year() != 2022)
    s_excl_holdout = _sharpe_excluding(daily_dev, pl.col("date").dt.year() < 2023)

    return AttributionMetrics(
        name=name,
        sharpe_dev=sharpe_self,
        corr_dev_with_hmm_only=_correlation(rets, hmm),
        corr_dev_with_vrp_only=_correlation(rets, vrp),
        incremental_sharpe_over_hmm_only=sharpe_self - sharpe_hmm,
        incremental_sharpe_over_vrp_only=sharpe_self - sharpe_vrp,
        residual_sharpe_over_hmm_only=_residual_sharpe(rets, hmm),
        residual_sharpe_over_vrp_only=_residual_sharpe(rets, vrp),
        pnl_by_year=pnl_by_year,
        pnl_by_regime=pnl_by_regime,
        sharpe_excl_2020=s_excl_2020,
        sharpe_excl_2022=s_excl_2022,
        sharpe_excl_holdout_period=s_excl_holdout,
    )
