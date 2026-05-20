from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from scripts.generate_promotion_report import build_report


def _write_per_signal_parquet(d: Path, name: str, hit_rate: float) -> None:
    """Write a per-signal parquet with the requested weighted hit-rate."""
    n_hits = int(round(10 * hit_rate))
    rows = []
    for i in range(10):
        rows.append({
            "signal_id": f"sig-{i:04d}",
            "symbol": "AAPL",
            "predicted_score": 0.05,
            "confidence": 0.7,
            "predicted_dir": 1,
            "s2_decision": "pass",
            "fill_price": 100.0,
            "horizon_minutes": 5,
            "realized_return": 0.005 if i < n_hits else -0.005,
            "realized_dir": 1 if i < n_hits else -1,
            "hit": i < n_hits,
            "weight": 1.0,
            "fill_ts_utc": "2026-05-20T13:35:00+00:00",
        })
    pl.DataFrame(rows).write_parquet(d / f"{name}.parquet")


def test_promotion_report_includes_hit_rate_min_row(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    parquet_dir = tmp_path / "validation"
    parquet_dir.mkdir()
    for i in range(1, 31):
        _write_per_signal_parquet(parquet_dir, f"2026-04-{i:02d}", hit_rate=0.6)

    report = build_report(
        from_stage="paper",
        to_stage="live_shadow",
        promotion_config_path=Path("configs/promotion.yaml"),
        audit_root=audit,
        s1_metrics_path=None,
        validation_parquet_dir=parquet_dir,
        validation_config_path=Path("configs/validation.yaml"),
    )
    names = [g["name"] for g in report["gates"]]
    assert "hit_rate_min" in names
    hit_gate = next(g for g in report["gates"] if g["name"] == "hit_rate_min")
    assert hit_gate["observed"] == pytest.approx(0.6, abs=0.05)
    assert hit_gate["passed"] is True
