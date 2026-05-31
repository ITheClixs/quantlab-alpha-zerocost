from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from quant_research_stack.brokers.order_types import Fill, OrderSide


@dataclass
class Position:
    symbol: str
    qty: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")


@dataclass
class PositionBook:
    snapshot_root: Path
    stage: str
    starting_equity: Decimal
    _positions: dict[str, Position] = field(default_factory=dict)
    _daily_realized_pnl: Decimal = Decimal("0")
    _peak_equity: Decimal | None = None
    _last_snap_day: str | None = None
    stage_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.snapshot_root = Path(self.snapshot_root)
        self.stage_dir = self.snapshot_root / self.stage
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        if self._peak_equity is None:
            self._peak_equity = self.starting_equity

    @property
    def daily_realized_pnl(self) -> Decimal:
        return self._daily_realized_pnl

    @property
    def peak_equity(self) -> Decimal:
        assert self._peak_equity is not None
        return self._peak_equity

    def position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol=symbol))

    def per_symbol_notional(self, mid: dict[str, Decimal]) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym, pos in self._positions.items():
            if pos.qty == 0:
                continue
            price = mid.get(sym, pos.avg_price)
            out[sym] = float(abs(pos.qty) * price)
        return out

    def gross_exposure(self, mid: dict[str, Decimal]) -> float:
        return sum(self.per_symbol_notional(mid).values())

    def apply_fill(self, fill: Fill) -> None:
        pos = self._positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        fill_qty = Decimal(str(fill.quantity))
        fill_price = Decimal(str(fill.price))
        signed_qty = fill_qty if fill.side == OrderSide.buy else -fill_qty
        new_qty = pos.qty + signed_qty

        if pos.qty == 0 or (pos.qty > 0 and signed_qty > 0) or (pos.qty < 0 and signed_qty < 0):
            total_cost = pos.qty * pos.avg_price + signed_qty * fill_price
            pos.qty = new_qty
            pos.avg_price = total_cost / pos.qty if pos.qty != 0 else Decimal("0")
            return

        closing_qty = min(abs(signed_qty), abs(pos.qty))
        direction = Decimal("1") if pos.qty > 0 else Decimal("-1")
        self._daily_realized_pnl += direction * closing_qty * (fill_price - pos.avg_price)
        pos.qty = new_qty
        if pos.qty == 0:
            self._positions.pop(fill.symbol, None)
        elif abs(signed_qty) > abs(pos.qty - signed_qty):
            pos.avg_price = fill_price

    def snapshot(self) -> Path:
        now = datetime.now(UTC)
        day = now.strftime("%Y-%m-%d")
        path = self.stage_dir / f"{day}.parquet"
        if self._last_snap_day is not None and self._last_snap_day != day:
            prev = self.stage_dir / f"{self._last_snap_day}.parquet"
            if prev.exists():
                self._chmod_a_w(prev)
        self._last_snap_day = day
        rows = [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_price": float(p.avg_price),
                "snapshot_ts_utc": now.isoformat(),
            }
            for p in self._positions.values()
        ]
        if not rows:
            rows = [{"symbol": "_empty", "qty": 0.0, "avg_price": 0.0, "snapshot_ts_utc": now.isoformat()}]
        pl.DataFrame(rows).write_parquet(path, compression="zstd")
        return path

    def load_latest_snapshot(self) -> bool:
        files = sorted(self.stage_dir.glob("*.parquet"))
        if not files:
            return False
        df = pl.read_parquet(files[-1])
        for row in df.iter_rows(named=True):
            if row["symbol"] == "_empty":
                continue
            self._positions[row["symbol"]] = Position(
                symbol=row["symbol"],
                qty=Decimal(str(row["qty"])),
                avg_price=Decimal(str(row["avg_price"])),
            )
        return True

    def _chmod_a_w(self, path: Path) -> None:
        current = path.stat().st_mode
        os.chmod(path, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
