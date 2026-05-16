from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.audit import AuditWriter, replay_audit


def test_audit_writer_appends_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    w = AuditWriter(out)
    w.record(event="signal_received", payload={"signal_id": "abc"})
    w.record(event="governor_verdict", payload={"signal_id": "abc", "decision": "veto"})
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "signal_received"
    assert "timestamp_utc" in first


def test_replay_audit_yields_records_in_order(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    w = AuditWriter(out)
    w.record(event="a", payload={"i": 1})
    w.record(event="b", payload={"i": 2})
    rows = list(replay_audit(out))
    assert [r["event"] for r in rows] == ["a", "b"]


def test_audit_writes_not_investment_advice_flag(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    w = AuditWriter(out)
    w.record(event="x", payload={})
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["not_investment_advice"] is True
