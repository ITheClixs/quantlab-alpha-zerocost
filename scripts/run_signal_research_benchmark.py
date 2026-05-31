"""Run the signal_research enhanced benchmark.

Usage:
    PYTHONPATH=src uv run python scripts/run_signal_research_benchmark.py \\
        --profile configs/signal_research_profiles/nasdaq.yaml \\
        --out reports/signal_research/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from quant_research_stack.signal_research.report import write_reports
from quant_research_stack.signal_research.runner import run_enhanced_benchmark

console = Console()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--out", default="reports/signal_research")
    args = p.parse_args()

    res = run_enhanced_benchmark(
        profile_path=Path(args.profile),
        strategy_run_fns=[],
        output_dir=Path(args.out),
    )
    master = write_reports(res, output_dir=Path(args.out))
    console.print(f"[green]ok[/green] wrote {master}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
