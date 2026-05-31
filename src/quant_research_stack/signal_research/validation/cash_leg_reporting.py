"""Cash-leg reporting for long-or-cash strategies.

Per the accepted exception policy §3.25, §4.14:

For long-or-cash strategies, results must be reported under three explicit
cash-leg assumptions:
- zero cash return
- T-bill / cash proxy return (via FRED DTB3 or equivalent)
- conservative after-fee cash return (T-bill minus 25 bps default fee)

The §3 gate is evaluated against the conservative after-fee assumption only.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

DEFAULT_CASH_FEE_BPS_ANNUAL: float = 25.0


@dataclass(frozen=True)
class CashLegAssumption:
    """One cash-leg assumption identifier."""

    name: str  # "zero" | "tbill" | "conservative_after_fee"
    description: str


CASH_ZERO = CashLegAssumption(
    name="zero",
    description="Risk-off days earn 0%. Stress floor.",
)
CASH_TBILL = CashLegAssumption(
    name="tbill",
    description="Risk-off days earn the prevailing 3-month T-bill rate (FRED DTB3).",
)
CASH_CONSERVATIVE = CashLegAssumption(
    name="conservative_after_fee",
    description=(
        "Risk-off days earn T-bill minus the prime-broker/cash-sweep fee "
        "(25 bps annualized default)."
    ),
)

ALL_ASSUMPTIONS: tuple[CashLegAssumption, ...] = (
    CASH_ZERO,
    CASH_TBILL,
    CASH_CONSERVATIVE,
)
GATE_ASSUMPTION: CashLegAssumption = CASH_CONSERVATIVE


@dataclass(frozen=True)
class CashLegResult:
    """Daily returns + Sharpe under one cash-leg assumption."""

    assumption: CashLegAssumption
    daily_returns: pl.DataFrame  # (date, gross_return, cash_return, net_return)
    sharpe_annual: float
    max_drawdown: float
    cumulative_return: float
    n_days: int


def _safe_sharpe(rets: NDArray[np.float64]) -> float:
    if rets.size < 2:
        return 0.0
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(rets)) / sd * float(np.sqrt(252.0))


def _max_dd(rets: NDArray[np.float64]) -> float:
    if rets.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def load_tbill_panel(
    *,
    start: dt.date,
    end: dt.date,
    cache_root: Path,
    fred_series_id: str = "DTB3",
) -> pl.DataFrame:
    """Load FRED DTB3 (3-month T-bill, secondary market rate, %) as a daily
    panel: (date, tbill_rate_pct).

    Carries forward across non-trading days. Returns an empty DataFrame if
    FRED is unreachable (the caller should fall back to zero cash assumption
    and document).
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_root / f"fred_{fred_series_id.lower()}.parquet"
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path)
        if df.height > 0:
            return df.filter(
                (pl.col("date") >= start) & (pl.col("date") <= end)
            ).sort("date")
    try:
        from quant_research_stack.signal_research.data.fred import (
            FredConfig,
            fetch_fred_series,
        )

        cfg = FredConfig(start=start, end=end)
        df = fetch_fred_series(series_id=fred_series_id, config=cfg)
    except Exception:
        return pl.DataFrame(
            schema={"date": pl.Date, "tbill_rate_pct": pl.Float64}
        )
    if df.is_empty():
        return pl.DataFrame(
            schema={"date": pl.Date, "tbill_rate_pct": pl.Float64}
        )
    df = (
        df.rename({fred_series_id: "tbill_rate_pct"})
        .with_columns(pl.col("date").cast(pl.Date))
        .sort("date")
    )
    df.write_parquet(parquet_path)
    return df


def _cash_daily_yield(
    *, tbill_panel: pl.DataFrame, dates: pl.Series, assumption: CashLegAssumption,
) -> NDArray[np.float64]:
    """Compute daily cash yield per the given assumption, aligned to `dates`."""
    if assumption.name == "zero":
        return np.zeros(len(dates), dtype=np.float64)
    if tbill_panel.is_empty():
        # Fallback to zero with a documented warning at the caller level
        return np.zeros(len(dates), dtype=np.float64)
    target = pl.DataFrame({"date": dates}).join(
        tbill_panel, on="date", how="left"
    ).with_columns(pl.col("tbill_rate_pct").fill_null(strategy="forward"))
    annual_pct = target["tbill_rate_pct"].fill_null(0.0).to_numpy().astype(np.float64)
    annual_decimal = annual_pct / 100.0
    if assumption.name == "conservative_after_fee":
        annual_decimal = annual_decimal - (DEFAULT_CASH_FEE_BPS_ANNUAL / 10_000.0)
    return annual_decimal / 365.0


def compute_long_or_cash_returns(
    *,
    underlying_returns: pl.DataFrame,  # (date, u_ret)
    position: NDArray[np.float64],  # gross exposure in [0, 1]
    tbill_panel: pl.DataFrame,
    assumption: CashLegAssumption,
    cost_bps_one_way: float,
) -> CashLegResult:
    """Compute daily returns: position(t)*u_ret(t) + (1 - position(t))*cash(t)
    minus turnover cost on position changes.
    """
    sorted_u = underlying_returns.sort("date")
    if sorted_u.height != position.size:
        raise ValueError(
            f"position size {position.size} != underlying dates {sorted_u.height}"
        )
    dates_series = sorted_u["date"]
    cash_yield = _cash_daily_yield(
        tbill_panel=tbill_panel, dates=dates_series, assumption=assumption,
    )
    u_ret = sorted_u["u_ret"].fill_null(0.0).to_numpy().astype(np.float64)
    gross_long = position * u_ret
    cash_leg = (1.0 - position) * cash_yield
    gross_ret = gross_long + cash_leg
    pos_change = np.abs(np.diff(position, prepend=0.0))
    turnover_cost = pos_change * (cost_bps_one_way * 2.0) / 10_000.0
    net_ret = gross_ret - turnover_cost
    daily = sorted_u.select(["date"]).with_columns(
        pl.Series("gross_return", gross_ret),
        pl.Series("cash_return", cash_leg),
        pl.Series("net_return", net_ret),
        pl.Series("turnover", pos_change),
    )
    return CashLegResult(
        assumption=assumption,
        daily_returns=daily,
        sharpe_annual=_safe_sharpe(net_ret),
        max_drawdown=_max_dd(net_ret),
        cumulative_return=float(np.prod(1.0 + net_ret) - 1.0),
        n_days=int(net_ret.size),
    )


def evaluate_all_cash_legs(
    *,
    underlying_returns: pl.DataFrame,
    position: NDArray[np.float64],
    tbill_panel: pl.DataFrame,
    cost_bps_one_way: float,
) -> dict[str, CashLegResult]:
    """Run all three cash-leg assumptions and return them keyed by name.

    The gate must be evaluated against `results[GATE_ASSUMPTION.name]`.
    """
    out: dict[str, CashLegResult] = {}
    for a in ALL_ASSUMPTIONS:
        out[a.name] = compute_long_or_cash_returns(
            underlying_returns=underlying_returns,
            position=position,
            tbill_panel=tbill_panel,
            assumption=a,
            cost_bps_one_way=cost_bps_one_way,
        )
    return out
