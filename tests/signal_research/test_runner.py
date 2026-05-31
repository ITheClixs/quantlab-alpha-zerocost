"""Runner orchestration tests (spec §6.2, §6.4)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from quant_research_stack.signal_research.runner import run_enhanced_benchmark


def _seed_profile(root: Path) -> Path:
    p = root / "test_profile.yaml"
    p.write_text(
        yaml.safe_dump({
            "profile": "test_profile",
            "asset_class": "equity",
            "universes": [
                {
                    "name": "u1",
                    "tickers": ["AAPL", "MSFT"],
                    "data_quality_label": "survivorship_prototype_only",
                    "constituent_survivorship_applicable": True,
                },
            ],
            "benchmarks": ["SPY"],
            "context_features": ["vix"],
            "cost_model": {"commission_bps_one_way": 0.5, "spread_bps_one_way": 0.5},
        })
    )
    return p


def test_runner_collects_strategy_frames_and_records_funnel(tmp_path: Path) -> None:
    profile_path = _seed_profile(tmp_path)

    def fn1(_p) -> pl.DataFrame:
        return pl.DataFrame({"strategy_id": ["A"], "status": [1]})

    def fn2(_p) -> pl.DataFrame:
        return pl.DataFrame({"strategy_id": ["B"], "status": [0]})

    res = run_enhanced_benchmark(
        profile_path=profile_path,
        strategy_run_fns=[fn1, fn2],
        output_dir=tmp_path / "out",
    )
    assert res.metrics.height == 2
    counts = res.funnel.to_ordered_dict()
    assert counts["total_raw_candidates"] == 2
    assert counts["research_pass"] == 1


def test_runner_continues_on_individual_strategy_failure(tmp_path: Path) -> None:
    profile_path = _seed_profile(tmp_path)

    def good(_p) -> pl.DataFrame:
        return pl.DataFrame({"strategy_id": ["G"]})

    def bad(_p) -> pl.DataFrame:
        raise RuntimeError("boom")

    res = run_enhanced_benchmark(
        profile_path=profile_path,
        strategy_run_fns=[good, bad],
        output_dir=tmp_path / "out",
    )
    assert res.metrics.height == 2
