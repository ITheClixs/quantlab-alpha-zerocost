from __future__ import annotations

from typing import Protocol


class _SignalShape(Protocol):
    symbol: str
    direction: int
    horizon_minutes: int
    regime_hint: str | None
    recent_vol_label: str


def build_query(signal: _SignalShape) -> str:
    regime = signal.regime_hint or "unknown"
    return f"{regime} {signal.symbol} direction={signal.direction} horizon={signal.horizon_minutes}m vol={signal.recent_vol_label}"
