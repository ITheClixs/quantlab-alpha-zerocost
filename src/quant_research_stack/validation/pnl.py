from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean, stdev

import polars as pl

from quant_research_stack.validation.daily_report import PerSignalRow


@dataclass(frozen=True)
class DailyPnlMetrics:
    daily_pnl: float
    daily_pnl_pct: float
    daily_dd_pct: float
    sharpe_rolling: float


def _row_pnl(
    *,
    predicted_direction: int,
    fill_price: float | None,
    weight: float,
    realized_return: float,
    fee: float,
) -> float:
    if fill_price is None or predicted_direction == 0 or math.isnan(realized_return):
        return 0.0
    notional = abs(weight * fill_price)
    return float(predicted_direction) * notional * realized_return - fee


def _annualized_sharpe(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    sigma = stdev(daily_returns)
    if sigma == 0.0:
        return 0.0
    return fmean(daily_returns) / sigma * math.sqrt(252.0)


def compute_daily_pnl_metrics(
    rows: list[PerSignalRow],
    *,
    starting_equity: float,
    historical_daily_pnl_pct: list[float] | None = None,
) -> DailyPnlMetrics:
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be positive; got {starting_equity}")
    daily_pnl = sum(
        _row_pnl(
            predicted_direction=r.predicted_direction,
            fill_price=r.fill_price,
            weight=r.weight,
            realized_return=r.realized_return,
            fee=r.fee,
        )
        for r in rows
    )
    daily_pnl_pct = daily_pnl / starting_equity
    returns = [*(historical_daily_pnl_pct or []), daily_pnl_pct]
    return DailyPnlMetrics(
        daily_pnl=float(daily_pnl),
        daily_pnl_pct=float(daily_pnl_pct),
        daily_dd_pct=float(abs(min(daily_pnl_pct, 0.0))),
        sharpe_rolling=float(_annualized_sharpe(returns)),
    )


def _parse_report_date(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d")
    except ValueError:
        return None


def _pnl_pct_from_parquet(path: Path, *, starting_equity: float) -> float:
    df = pl.read_parquet(path)
    if df.height == 0:
        return 0.0
    fee_values = df["fee"].to_list() if "fee" in df.columns else [0.0] * df.height
    daily_pnl = 0.0
    for row, fee in zip(df.iter_rows(named=True), fee_values, strict=True):
        daily_pnl += _row_pnl(
            predicted_direction=int(row.get("predicted_dir") or 0),
            fill_price=row.get("fill_price"),
            weight=float(row.get("weight") or 0.0),
            realized_return=float(row.get("realized_return") or math.nan),
            fee=float(fee or 0.0),
        )
    return daily_pnl / starting_equity


def load_historical_daily_pnl_pct(
    parquet_dir: Path,
    *,
    before_date: str,
    window_days: int,
    starting_equity: float,
) -> list[float]:
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be positive; got {starting_equity}")
    if window_days <= 0:
        raise ValueError(f"window_days must be positive; got {window_days}")
    if not parquet_dir.exists():
        return []

    end = datetime.strptime(before_date, "%Y-%m-%d")
    start = end - timedelta(days=window_days)
    dated_paths: list[tuple[datetime, Path]] = []
    for path in parquet_dir.glob("*.parquet"):
        day = _parse_report_date(path)
        if day is None or not (start <= day < end):
            continue
        dated_paths.append((day, path))

    return [
        _pnl_pct_from_parquet(path, starting_equity=starting_equity)
        for _, path in sorted(dated_paths)
    ]
