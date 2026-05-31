from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.execution.audit import AuditLog


def test_audit_log_appends_jsonl(tmp_path: Path) -> None:
    log = AuditLog(root=tmp_path, rotation="daily", chmod_after_close=False)
    log.append("signal_ingested", {"signal_id": "sig-1", "symbol": "BTCUSDT"})
    log.append("trade_placed", {"signal_id": "sig-1", "order_id": "o-1", "qty": 0.01})
    log.close_current()
    files = sorted(tmp_path.iterdir())
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text().splitlines() if line]
    assert len(lines) == 2
    assert lines[0]["event"] == "signal_ingested"
    assert lines[0]["not_investment_advice"] is True
    assert "timestamp_utc" in lines[0]


def test_audit_log_chmod_a_w_on_close(tmp_path: Path) -> None:
    log = AuditLog(root=tmp_path, rotation="daily", chmod_after_close=True)
    log.append("test", {})
    log.close_current()
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    mode = files[0].stat().st_mode & 0o222
    assert mode == 0, f"expected no write bits, got {oct(mode)}"
