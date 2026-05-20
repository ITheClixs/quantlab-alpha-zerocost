from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

pytestmark = pytest.mark.validation_integration


def test_report_produced_for_synthetic_day(tmp_path: Path) -> None:
    date_str = "2026-05-20"
    preds_dir = tmp_path / "preds"
    verdicts_dir = tmp_path / "verdicts"
    audit_root = tmp_path / "audit"
    md_dir = tmp_path / "md"
    pq_dir = tmp_path / "pq"
    preds_dir.mkdir()
    verdicts_dir.mkdir()
    (audit_root / "paper").mkdir(parents=True)

    pl.DataFrame({
        "signal_id": ["sig-int-0001"],
        "symbol": ["AAPL"],
        "predicted_score": [0.05],
        "confidence": [0.7],
        "horizon_minutes": [5],
        "ts_utc": [datetime(2026, 5, 20, 13, 35, tzinfo=UTC).isoformat()],
    }).write_parquet(preds_dir / f"{date_str}.parquet")

    with (verdicts_dir / f"{date_str}.jsonl").open("w") as h:
        h.write(json.dumps({
            "signal_id": "sig-int-0001",
            "decision": "pass", "direction": 1, "confidence": 0.7,
            "horizon_minutes": 5, "regime_tag": "trending", "rationale_short": "ok",
            "cited_paper_chunk_ids": ["paper_pdf:x:0"], "contradictions_flagged": [],
        }) + "\n")

    with (audit_root / "paper" / f"{date_str}.jsonl").open("w") as h:
        h.write(json.dumps({
            "event": "trade_fill",
            "not_investment_advice": True,
            "payload": {
                "fill_id": "f-1", "client_order_id": "sig-int-0001",
                "symbol": "AAPL", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.0,
                "ts_utc": datetime(2026, 5, 20, 13, 35, tzinfo=UTC).isoformat(),
            },
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }) + "\n")

    cfg = tmp_path / "validation.yaml"
    cfg.write_text(
        f"window:\n"
        f"  min_trading_days: 30\n"
        f"  rolling_window_days: 14\n"
        f"thresholds:\n"
        f"  hit_rate_min: 0.53\n"
        f"  sharpe_min: 1.0\n"
        f"  max_daily_dd_pct: 0.05\n"
        f"  governor_block_rate_max: 0.5\n"
        f"data:\n"
        f"  forward_return_source: alpaca_bars\n"
        f"  horizon_alignment: ceil_to_next_bar\n"
        f"artifacts:\n"
        f"  daily_report_dir: {md_dir}\n"
        f"  per_signal_parquet_dir: {pq_dir}\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    rc = subprocess.run(
        [sys.executable, "-u", "scripts/tv_validation_report.py",
         "--date", date_str, "--config", str(cfg),
         "--predictions-dir", str(preds_dir),
         "--verdicts-dir", str(verdicts_dir),
         "--audit-root", str(audit_root),
         "--stage", "paper"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    md_path = md_dir / f"{date_str}.md"
    pq_path = pq_dir / f"{date_str}.parquet"
    assert md_path.exists()
    assert pq_path.exists()
    md = md_path.read_text()
    assert "QuantLab paper validation" in md
    assert "## Headline" in md
    assert "## Per-signal table" in md
    df = pl.read_parquet(pq_path)
    assert df.height == 1
    assert df["signal_id"].to_list() == ["sig-int-0001"]
