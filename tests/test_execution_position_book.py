from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from quant_research_stack.brokers.order_types import Fill, OrderSide
from quant_research_stack.execution.position_book import PositionBook


def _fill(symbol: str, side: str, qty: str, price: str) -> Fill:
    return Fill(
        client_order_id="c-1",
        fill_id="f-1",
        symbol=symbol,
        side=OrderSide(side),
        quantity=float(Decimal(qty)),
        price=float(Decimal(price)),
        timestamp_utc=datetime.now(UTC),
        commission=0.0,
    )


def test_apply_buy_fill_increases_position(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.01", "50000"))
    pos = book.position("BTCUSDT")
    assert pos.qty == Decimal("0.01")
    assert pos.avg_price == Decimal("50000.0")


def test_apply_sell_fill_decreases_position(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.02", "50000"))
    book.apply_fill(_fill("BTCUSDT", "sell", "0.01", "55000"))
    pos = book.position("BTCUSDT")
    assert pos.qty == Decimal("0.01")
    assert book.daily_realized_pnl > 0


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.01", "50000"))
    book.snapshot()
    files = list((tmp_path / "paper").glob("*.parquet"))
    assert len(files) == 1
    df = pl.read_parquet(files[0])
    assert "symbol" in df.columns and "qty" in df.columns
    assert df.height >= 1


def test_load_latest_snapshot_recovers_book(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.01", "50000"))
    book.snapshot()
    book2 = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book2.load_latest_snapshot()
    assert book2.position("BTCUSDT").qty == Decimal("0.01")
