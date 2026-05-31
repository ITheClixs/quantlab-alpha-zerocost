"""Replay an S4 audit JSONL log and reconstruct the position book.

Also supports `equity-backtest` subcommand for S1-EQ JSONL replay (spec §5.17).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quant_research_stack.brokers.order_types import Fill, OrderSide
from quant_research_stack.execution.position_book import PositionBook


def _verify_equity_backtest(audit_log: Path) -> int:
    if not audit_log.exists():
        print(f"missing audit log: {audit_log}")
        return 2
    with audit_log.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    h_all = hashlib.sha256()
    for r in rows:
        h_all.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
    print(f"rows={len(rows)} sha256={h_all.hexdigest()}")
    return 0


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
                fill_id=str(payload["fill_id"]),
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
    # equity-backtest subcommand dispatch (spec §5.17)
    if len(sys.argv) > 1 and sys.argv[1] == "equity-backtest":
        sub = argparse.ArgumentParser(prog="audit_replay_check.py equity-backtest")
        sub.add_argument("--audit-log", required=True)
        sub_args = sub.parse_args(sys.argv[2:])
        return _verify_equity_backtest(Path(sub_args.audit_log))

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
