"""Run the VRP × HMM interaction test (Option γ).

Predeclared 9-variant grid + 2 anchors + 2 sanity = 13 strategies.

Usage:
    PYTHONPATH=src uv run python scripts/run_vrp_hmm_interaction.py
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from rich.console import Console

from quant_research_stack.signal_research.vrp import (
    VRPSpec,
    fetch_vrp_data,
)
from quant_research_stack.signal_research.vrp.interaction_runner import (
    run_vrp_hmm_interaction,
    write_interaction_outputs,
)

console = Console()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="SPY")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument("--dev-end", default="2022-12-31")
    p.add_argument("--holdout-start", default="2023-01-01")
    p.add_argument("--cache-root", default="data/processed/vrp/bars")
    p.add_argument("--out", default="reports/signal_research/vrp_hmm_interaction")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    dev_end = dt.date.fromisoformat(args.dev_end)
    holdout_start = dt.date.fromisoformat(args.holdout_start)

    fetched = fetch_vrp_data(start=start, end=end, cache_root=Path(args.cache_root))
    console.print(
        f"[green]Fetched[/green] underlying={fetched.fetched_underlying}, "
        f"vol={fetched.fetched_vol}"
    )

    spec = VRPSpec(
        target_symbol=args.target,
        start=start, end=end, dev_end=dev_end, holdout_start=holdout_start,
        realized_window=21,
        commission_bps_one_way=0.5,
        spread_bps_one_way=0.5,
        cost_stress_multipliers=(2.0, 3.0),
        delay_stress_bars=(1,),
        bootstrap_n_resamples=2000,
        bootstrap_seed=42,
    )
    console.print(
        "[cyan]Running[/cyan] 13-strategy interaction test on SPY..."
    )
    report = run_vrp_hmm_interaction(
        underlying=fetched.underlying,
        vol_features=fetched.vol_features,
        spec=spec,
    )

    console.print("[bold]Results:[/bold]")
    for r in report.results:
        prefix = (
            "[green]" if r.category == "anchor"
            else "[cyan]" if r.category == "interaction"
            else "[yellow]"
        )
        console.print(
            f"  {prefix}{r.name:30s}[/] "
            f"dev={r.dev.sharpe_annual:+6.3f}  "
            f"hd={r.holdout.sharpe_annual:+6.3f}  "
            f"cs2x={r.cost_stress_2x.sharpe_annual:+6.3f}  "
            f"delay1d={r.delay_1d.sharpe_annual:+6.3f}  "
            f"ρHMM={r.attribution.corr_dev_with_hmm_only:+.3f}  "
            f"resid_vs_HMM={r.attribution.residual_sharpe_over_hmm_only:+.3f}"
        )
    console.print(
        f"[cyan]PBO[/cyan]={report.pbo_raw_global:.3f}  "
        f"best=`{report.best_name}`  "
        f"DSR={report.best_dsr:.3f}  PSR_zero={report.best_psr_zero:.3f}"
    )
    console.print(
        f"[bold yellow]BRANCH {report.decision_branch}:[/bold yellow] "
        f"{report.decision}"
    )
    if report.failure_class:
        console.print(f"[bold red]failure_class:[/bold red] {report.failure_class}")

    paths = write_interaction_outputs(report, output_dir=Path(args.out))
    for k, v in paths.items():
        console.print(f"[green]{k:14s}[/green] {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
