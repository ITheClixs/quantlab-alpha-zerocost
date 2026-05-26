"""Report writer tests (spec §6.2, §7)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from quant_research_stack.signal_research.report import write_reports
from quant_research_stack.signal_research.runner import run_enhanced_benchmark


def _seed_profile(root: Path) -> Path:
    p = root / "p.yaml"
    p.write_text(
        yaml.safe_dump({
            "profile": "p",
            "asset_class": "equity",
            "universes": [
                {
                    "name": "u",
                    "tickers": ["AAPL"],
                    "data_quality_label": "survivorship_prototype_only",
                    "constituent_survivorship_applicable": True,
                },
            ],
            "benchmarks": ["SPY"],
            "context_features": [],
            "cost_model": {"commission_bps_one_way": 0.5, "spread_bps_one_way": 0.5},
        })
    )
    return p


def test_report_writes_master_with_funnel_and_disclaimer(tmp_path: Path) -> None:
    profile_path = _seed_profile(tmp_path)

    def fn(_p) -> pl.DataFrame:
        return pl.DataFrame({"strategy_id": ["X"], "status": [1]})

    res = run_enhanced_benchmark(
        profile_path=profile_path,
        strategy_run_fns=[fn],
        output_dir=tmp_path / "out",
    )
    master = write_reports(res, output_dir=tmp_path / "reports")
    body = master.read_text()
    assert "Selection funnel" in body
    assert "Disclaimer" in body
    assert "Past performance does not guarantee" in body
