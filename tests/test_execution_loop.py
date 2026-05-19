from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.configs import ExecConfig, load_risk_config
from quant_research_stack.execution.loop import S4Loop


def _write_pred(dir_: Path, sig_id: str) -> None:
    df = pl.DataFrame({
        "signal_id": [sig_id],
        "symbol": ["BTCUSDT"],
        "predicted_score": [0.05],
        "confidence": [0.7],
        "horizon_minutes": [5],
        "ts_utc": [datetime.now(UTC).isoformat()],
    })
    dir_.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.parquet")


def _write_verdict(dir_: Path, sig_id: str, decision: str = "pass") -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    rec = {
        "signal_id": sig_id,
        "decision": decision,
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"] if decision == "pass" else [],
        "contradictions_flagged": [],
    }
    with (dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl").open("a") as h:
        h.write(json.dumps(rec) + "\n")


@pytest.mark.asyncio
async def test_loop_processes_one_signal_end_to_end(tmp_path: Path) -> None:
    risk_cfg = load_risk_config(Path("configs/risk.yaml"))
    exec_cfg_data = {
        "ingest": {
            "s1_predictions_dir": str(tmp_path / "preds"),
            "s2_verdicts_dir": str(tmp_path / "verdicts"),
            "poll_interval_seconds": 0.05,
            "pair_window_seconds": 5,
        },
        "position_book": {"snapshot_root": str(tmp_path / "positions"), "snapshot_interval_seconds": 60},
        "audit": {"root": str(tmp_path / "audit"), "rotation": "daily", "chmod_after_close": False},
        "kill_switch": {
            "repo_root_marker": str(tmp_path / "KILL_TRADING_NEVER"),
            "poll_interval_seconds": 0.05,
            "emergency_snapshot_root": str(tmp_path / "snaps"),
        },
    }
    exec_cfg = ExecConfig.model_validate(exec_cfg_data)
    audit = AuditLog(root=Path(exec_cfg.audit.root), chmod_after_close=False)
    broker = NullBroker(fill_model=FillModel(FillModelConfig()))
    loop = S4Loop(
        stage="paper",
        risk_cfg=risk_cfg,
        exec_cfg=exec_cfg,
        broker=broker,
        audit=audit,
        starting_equity=Decimal("100000"),
        mid_price_lookup=lambda _s: Decimal("50000"),
        is_crypto=lambda _s: True,
        tier3_stance_pct=0.20,
    )

    _write_pred(Path(exec_cfg.ingest.s1_predictions_dir), "sig-00003333")
    _write_verdict(Path(exec_cfg.ingest.s2_verdicts_dir), "sig-00003333")

    task = asyncio.create_task(loop.run(max_tickets=1))
    await asyncio.wait_for(task, timeout=5.0)

    audit.close_current()
    events = []
    for p in Path(exec_cfg.audit.root).iterdir():
        for line in p.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line)["event"])
    assert "signal_ingested" in events
    assert "trade_placed" in events
