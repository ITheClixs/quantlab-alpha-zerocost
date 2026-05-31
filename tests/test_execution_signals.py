from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.signals import SignalIngestor


def _write_pred(dir_: Path, ts: datetime, sig_id: str) -> None:
    df = pl.DataFrame({
        "signal_id": [sig_id],
        "symbol": ["BTCUSDT"],
        "predicted_score": [0.05],
        "confidence": [0.7],
        "horizon_minutes": [5],
        "ts_utc": [ts.isoformat()],
    })
    dir_.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dir_ / f"{ts.strftime('%Y-%m-%d')}.parquet")


def _write_verdict(dir_: Path, ts: datetime, sig_id: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    rec = {
        "signal_id": sig_id,
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"],
        "contradictions_flagged": [],
    }
    with (dir_ / f"{ts.strftime('%Y-%m-%d')}.jsonl").open("a") as h:
        h.write(json.dumps(rec) + "\n")


@pytest.mark.asyncio
async def test_signal_ingestor_pairs_within_window(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    verdicts_dir = tmp_path / "verdicts"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)
    ts = datetime.now(UTC)
    _write_pred(preds_dir, ts, "sig-00000001")
    _write_verdict(verdicts_dir, ts, "sig-00000001")
    ing = SignalIngestor(
        preds_dir=preds_dir,
        verdicts_dir=verdicts_dir,
        poll_interval_s=0.05,
        pair_window_s=5,
        audit=audit,
    )
    tickets: list = []

    async def drain() -> None:
        async for t in ing.stream():
            tickets.append(t)
            if len(tickets) >= 1:
                ing.stop()

    await asyncio.wait_for(drain(), timeout=3.0)
    assert len(tickets) == 1
    assert tickets[0].signal.signal_id == "sig-00000001"


@pytest.mark.asyncio
async def test_signal_ingestor_audits_verdict_timeout(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    verdicts_dir = tmp_path / "verdicts"
    audit_dir = tmp_path / "audit"
    audit = AuditLog(root=audit_dir, chmod_after_close=False)
    ts = datetime.now(UTC)
    _write_pred(preds_dir, ts, "sig-00000099")
    ing = SignalIngestor(
        preds_dir=preds_dir,
        verdicts_dir=verdicts_dir,
        poll_interval_s=0.05,
        pair_window_s=1,
        audit=audit,
    )

    async def drain() -> None:
        async for _ in ing.stream():
            ing.stop()

    try:
        await asyncio.wait_for(drain(), timeout=3.0)
    except TimeoutError:
        pass
    ing.stop()
    audit.close_current()
    files = list(audit_dir.iterdir())
    assert files, "audit log empty"
    lines = files[0].read_text().splitlines()
    events = [json.loads(line)["event"] for line in lines if line]
    assert "verdict_timeout" in events
