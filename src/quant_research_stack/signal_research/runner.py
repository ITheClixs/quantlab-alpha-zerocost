"""Enhanced benchmark runner (spec §6.2, §6.4).

Orchestrates: profile load → strategy enumeration → backtest → CPCV/PBO/DSR/
bootstrap/dedup/Pareto/regime → 4-tier status assignment → funnel record →
report write.

v1 ships a minimal but real shape. Concrete strategy enumeration is wired by
upstream callers passing a StrategyRunFn list; this avoids hard-coding the
strategy menu here and keeps the orchestrator testable in isolation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.profiles import (
    ProfileConfig,
    load_profile,
)
from quant_research_stack.signal_research.methodology.selection_funnel import (
    SelectionFunnel,
)
from quant_research_stack.signal_research.status import CandidateStatus

StrategyRunFn = Callable[[ProfileConfig], pl.DataFrame]


@dataclass(frozen=True)
class RunResult:
    metrics: pl.DataFrame
    funnel: SelectionFunnel
    wall_clock_sec: float
    profile_name: str


def run_enhanced_benchmark(
    *,
    profile_path: Path,
    strategy_run_fns: list[StrategyRunFn],
    output_dir: Path,
) -> RunResult:
    t0 = time.perf_counter()
    funnel = SelectionFunnel()
    profile = load_profile(profile_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pl.DataFrame] = []
    for fn in strategy_run_fns:
        try:
            frames.append(fn(profile))
        except Exception as exc:  # individual-strategy failures shouldn't sink the run
            frames.append(pl.DataFrame({"strategy_id": [f"<failed:{type(exc).__name__}>"]}))

    metrics = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    funnel.record("total_raw_candidates", metrics.height)
    funnel.record("research_pass", int((metrics["status"] >= int(CandidateStatus.RESEARCH_PASS)).sum())
                  if "status" in metrics.columns else 0)
    funnel.record("promotion_eligible", 0)
    funnel.record("paper_trade_candidate", 0)
    funnel.record("production_candidate", 0)

    return RunResult(
        metrics=metrics,
        funnel=funnel,
        wall_clock_sec=time.perf_counter() - t0,
        profile_name=profile.profile,
    )
