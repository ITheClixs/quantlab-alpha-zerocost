from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class ForwardReturnRequest:
    signal_id: str
    symbol: str
    fill_ts_utc: datetime
    horizon_minutes: int


@dataclass(frozen=True)
class ForwardReturnResult:
    signal_id: str
    symbol: str
    fill_ts_utc: datetime
    horizon_ts_utc: datetime
    realized_return: float  # NaN if either bar missing
    realized_direction: int  # in {-1, 0, 1}; 0 when realized_return is NaN or exactly 0


BarLoader = Callable[[str, datetime], "Bar | None"]


def align_horizon_to_bar(
    fill_ts: datetime,
    horizon_minutes: int,
    bar_interval_minutes: int,
    mode: Literal["ceil_to_next_bar", "floor_to_next_bar"],
) -> datetime:
    """Return the UTC bar timestamp where the horizon return is measured.

    ceil_to_next_bar:  the next bar boundary strictly after fill_ts + horizon
                       (or that boundary itself if target is exactly on one).
    floor_to_next_bar: the bar boundary at or before fill_ts + horizon.
    """
    if fill_ts.tzinfo is None:
        fill_ts = fill_ts.replace(tzinfo=UTC)
    target = fill_ts + timedelta(minutes=horizon_minutes)
    interval = timedelta(minutes=bar_interval_minutes)
    epoch_minutes = int(target.timestamp() // 60)
    floored_minutes = (epoch_minutes // bar_interval_minutes) * bar_interval_minutes
    floored = datetime.fromtimestamp(floored_minutes * 60, tz=UTC)
    if mode == "floor_to_next_bar":
        return floored
    if floored == target:
        return floored
    return floored + interval


def _bar_interval_for(_symbol: str) -> int:
    return 1


def fetch_forward_returns(
    requests: Iterable[ForwardReturnRequest],
    bar_loader: BarLoader,
    horizon_alignment: Literal["ceil_to_next_bar", "floor_to_next_bar"],
) -> list[ForwardReturnResult]:
    out: list[ForwardReturnResult] = []
    for req in requests:
        interval = _bar_interval_for(req.symbol)
        entry_ts = datetime.fromtimestamp(
            (int(req.fill_ts_utc.timestamp()) // (interval * 60)) * (interval * 60),
            tz=UTC,
        )
        horizon_ts = align_horizon_to_bar(
            req.fill_ts_utc, req.horizon_minutes, interval, horizon_alignment,
        )
        entry_bar = bar_loader(req.symbol, entry_ts)
        horizon_bar = bar_loader(req.symbol, horizon_ts)
        if entry_bar is None or horizon_bar is None or entry_bar.close <= 0:
            out.append(ForwardReturnResult(
                signal_id=req.signal_id,
                symbol=req.symbol,
                fill_ts_utc=req.fill_ts_utc,
                horizon_ts_utc=horizon_ts,
                realized_return=math.nan,
                realized_direction=0,
            ))
            continue
        ret = (horizon_bar.close - entry_bar.close) / entry_bar.close
        direction = 1 if ret > 0 else (-1 if ret < 0 else 0)
        out.append(ForwardReturnResult(
            signal_id=req.signal_id,
            symbol=req.symbol,
            fill_ts_utc=req.fill_ts_utc,
            horizon_ts_utc=horizon_ts,
            realized_return=float(ret),
            realized_direction=direction,
        ))
    return out
