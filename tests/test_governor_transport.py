from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.signal_schema import GovernorVerdict
from quant_research_stack.governor.transport import VerdictWriter, tail_verdicts


def _v() -> GovernorVerdict:
    return GovernorVerdict.model_validate({
        "signal_id": "sig-12345678",
        "decision": "veto",
        "direction": 0,
        "confidence": 0.9,
        "horizon_minutes": 15,
        "regime_tag": "high_vol",
        "rationale_short": "x",
        "cited_paper_chunk_ids": [],
        "contradictions_flagged": [],
    })


def test_writer_appends_one_line_per_call(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    w.write(_v())
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["signal_id"] == "sig-12345678"


def test_writer_creates_parent_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    assert out.exists()


def test_tail_verdicts_yields_appended_records(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    w.write(_v())
    rows = list(tail_verdicts(out))
    assert len(rows) == 2
    assert rows[0]["signal_id"] == "sig-12345678"


def test_writer_chmod_when_requested(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    w.close_and_lock()
    assert not (out.stat().st_mode & 0o222)  # no write bits anywhere
