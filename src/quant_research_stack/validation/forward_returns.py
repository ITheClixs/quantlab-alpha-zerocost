from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx


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


def _load_alpaca_credentials(path: Path | str) -> tuple[str, str]:
    payload = json.loads(Path(path).expanduser().read_text())
    return str(payload["api_key"]), str(payload["api_secret"])


@dataclass
class AlpacaBarsLoader:
    credentials_path: str = "~/.alpaca/paper_keys.json"
    base_url: str = "https://data.alpaca.markets/v2/stocks/bars"
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        api_key, api_secret = _load_alpaca_credentials(self.credentials_path)
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        self._client = self.client or httpx.Client(timeout=10.0)
        self._cache: dict[tuple[str, datetime], Bar | None] = {}

    def __call__(self, symbol: str, ts_utc: datetime) -> Bar | None:
        ts_utc = ts_utc.astimezone(UTC).replace(second=0, microsecond=0)
        key = (symbol, ts_utc)
        if key in self._cache:
            return self._cache[key]

        end = ts_utc + timedelta(minutes=1)
        response = self._client.get(
            self.base_url,
            params={
                "symbols": symbol,
                "timeframe": "1Min",
                "start": ts_utc.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "limit": 10,
            },
            headers=self._headers,
        )
        response.raise_for_status()
        for row in (response.json().get("bars") or {}).get(symbol, []):
            row_ts = datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00")).astimezone(UTC)
            if row_ts != ts_utc:
                continue
            bar = Bar(
                symbol=symbol,
                ts_utc=row_ts,
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=int(row["v"]),
            )
            self._cache[key] = bar
            return bar

        self._cache[key] = None
        return None


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
