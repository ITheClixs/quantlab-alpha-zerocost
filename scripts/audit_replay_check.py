"""Replay an S4 audit JSONL log and reconstruct the position book."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quant_research_stack.brokers.order_types import Fill, OrderSide
from quant_research_stack.execution.position_book import PositionBook


def replay_audit_to_book(audit_dir: Path, book: PositionBook) -> int:
    """Apply every trade_fill event in chronological order. Returns fills applied."""
    n = 0
    for path in sorted(audit_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") != "trade_fill":
                continue
            payload = rec["payload"]
            fill = Fill(
                client_order_id=payload["client_order_id"],
                fill_id=str(payload.get("fill_id", payload.get("order_id", payload["client_order_id"]))),
                symbol=payload["symbol"],
                side=OrderSide(payload["side"]),
                price=float(payload["price"]),
                quantity=float(payload["qty"]),
                timestamp_utc=datetime.fromisoformat(payload["ts_utc"]),
                commission=float(payload.get("fee", 0.0)),
            )
            book.apply_fill(fill)
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay S4 audit logs into a position book")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--snapshot-root", default="data/positions")
    parser.add_argument("--stage", choices=["paper", "live_shadow", "live"], default="paper")
    parser.add_argument("--starting-equity", type=Decimal, default=Decimal("100000"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    book = PositionBook(
        snapshot_root=Path(args.snapshot_root),
        stage=args.stage,
        starting_equity=args.starting_equity,
    )
    n = replay_audit_to_book(Path(args.audit_dir), book)
    print(f"Applied {n} fills.")
    for sym, pos in book._positions.items():
        print(f"  {sym}: qty={pos.qty} avg={pos.avg_price}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
