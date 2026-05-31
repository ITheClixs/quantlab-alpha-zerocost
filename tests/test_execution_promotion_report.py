from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_promotion_report import build_report


def test_build_report_marks_each_gate(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    for i in range(30):
        f = audit_dir / f"2026-04-{i + 1:02d}.jsonl"
        f.write_text(
            json.dumps({
                "event": "trade_placed",
                "not_investment_advice": True,
                "payload": {},
                "timestamp_utc": "2026-04-01T00:00:00+00:00",
            })
            + "\n"
        )
    report = build_report(
        from_stage="paper",
        to_stage="live_shadow",
        promotion_config_path=Path("configs/promotion.yaml"),
        audit_root=audit_dir,
        s1_metrics_path=None,
    )
    assert isinstance(report, dict)
    assert "gates" in report
    assert any(g["name"] == "min_days_in_paper" for g in report["gates"])
