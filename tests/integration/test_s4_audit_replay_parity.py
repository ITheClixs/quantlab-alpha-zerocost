from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = pytest.mark.s4_integration


def test_replay_reconstructs_same_position_book(tmp_path: Path) -> None:
    from quant_research_stack.execution.position_book import PositionBook
    from scripts.audit_replay_check import replay_audit_to_book

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    rec = {
        "event": "trade_fill",
        "not_investment_advice": True,
        "payload": {
            "fill_id": "f-1",
            "client_order_id": "c-1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "qty": 0.01,
            "price": 50000.0,
            "fee": 0.0,
            "ts_utc": "2026-05-20T00:00:00+00:00",
        },
        "timestamp_utc": "2026-05-20T00:00:00+00:00",
    }
    (audit_dir / "2026-05-20.jsonl").write_text(json.dumps(rec) + "\n")
    book = PositionBook(snapshot_root=tmp_path / "positions", stage="paper", starting_equity=Decimal("100000"))
    replay_audit_to_book(audit_dir, book)
    assert book.position("BTCUSDT").qty == Decimal("0.01")
