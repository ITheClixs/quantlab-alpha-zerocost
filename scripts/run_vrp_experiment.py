"""Run the VRP index validation experiment on SPY.

Per intake docs/research/intake/2026-05-28-vrp-index-v1.md.

Usage:
    PYTHONPATH=src uv run python scripts/run_vrp_experiment.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from rich.console import Console

from quant_research_stack.signal_research.vrp import (
    VRPSpec,
    fetch_vrp_data,
    render_vrp_report,
    run_vrp_pipeline,
)

console = Console()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="SPY", help="SPY or QQQ")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument("--dev-end", default="2022-12-31")
    p.add_argument("--holdout-start", default="2023-01-01")
    p.add_argument("--cache-root", default="data/processed/vrp/bars")
    p.add_argument("--out", default="reports/signal_research/vrp")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    dev_end = dt.date.fromisoformat(args.dev_end)
    holdout_start = dt.date.fromisoformat(args.holdout_start)

    console.print(
        f"[cyan]Fetching[/cyan] VIX family + {args.target} from {start} to {end}..."
    )
    fetched = fetch_vrp_data(
        start=start, end=end, cache_root=Path(args.cache_root),
    )
    console.print(
        f"[green]Underlying[/green] fetched: {fetched.fetched_underlying}"
    )
    console.print(
        f"[green]Vol family[/green] fetched: {fetched.fetched_vol}"
    )
    if fetched.missing_vol:
        console.print(
            f"[yellow]Missing[/yellow] vol tickers: {fetched.missing_vol}"
        )

    if fetched.underlying.is_empty() or "vix" not in fetched.vol_features.columns:
        console.print("[red]Cannot proceed: missing required data.[/red]")
        return 1

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
        gate_dev_sharpe_min=1.5,
        gate_holdout_sharpe_min=0.5,
        gate_cost_stress_min=0.0,
        gate_bootstrap_ci_lower_min=0.0,
        gate_pbo_max=0.25,
        gate_dsr_min=0.50,
        gate_max_month_share=0.5,
    )
    console.print("[cyan]Running[/cyan] 6 VRP variants + 3 baselines on real fixture...")
    report = run_vrp_pipeline(
        underlying=fetched.underlying,
        vol_features=fetched.vol_features,
        spec=spec,
    )

    console.print("[bold]Results:[/bold]")
    for r in report.variant_results:
        cs2 = r.cost_stress.get(2.0)
        cs2_sr = cs2.sharpe_annual if cs2 else float("nan")
        console.print(
            f"  [green]{r.name:30s}[/green] "
            f"dev={r.dev.sharpe_annual:+6.3f}  "
            f"hd={r.holdout.sharpe_annual:+6.3f}  "
            f"cs2x={cs2_sr:+6.3f}  "
            f"max_month={r.concentration_dev.max_month_share*100:5.1f}%"
        )
    for r in report.baseline_results:
        cs2 = r.cost_stress.get(2.0)
        cs2_sr = cs2.sharpe_annual if cs2 else float("nan")
        console.print(
            f"  [yellow]{r.name:30s}[/yellow] "
            f"dev={r.dev.sharpe_annual:+6.3f}  "
            f"hd={r.holdout.sharpe_annual:+6.3f}  "
            f"cs2x={cs2_sr:+6.3f}  "
            f"max_month={r.concentration_dev.max_month_share*100:5.1f}%"
        )
    console.print(
        f"[cyan]PBO[/cyan] raw_global={report.cross.pbo_raw_global:.3f}  "
        f"DSR for `{report.cross.best_name}`={report.cross.best_dsr:.3f}  "
        f"PSR_zero={report.cross.best_psr_zero:.3f}"
    )
    console.print(f"[bold yellow]DECISION:[/bold yellow] {report.decision}")
    if report.failure_class:
        console.print(f"[bold red]failure_class:[/bold red] {report.failure_class}")
    console.print(
        f"[bold]Pre-registered failure modes:[/bold] "
        f"concentration={'YES' if report.failure_mode_1_concentration else 'no'}, "
        f"pbo={'YES' if report.failure_mode_2_pbo else 'no'}, "
        f"dsr_inflation={'YES' if report.failure_mode_3_dsr else 'no'}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = render_vrp_report(report, output_path=out_dir / "report.md")
    (out_dir / "vrp_pbo.json").write_text(json.dumps({
        "target_symbol": args.target,
        "pbo_raw_global": report.cross.pbo_raw_global,
        "pbo_per_family": report.cross.pbo_per_family,
        "best_name": report.cross.best_name,
        "best_dsr": report.cross.best_dsr,
        "best_psr_zero": report.cross.best_psr_zero,
        "n_strategies": report.cross.n_strategies,
        "decision": report.decision,
        "failure_class": report.failure_class,
        "failure_mode_1_concentration": report.failure_mode_1_concentration,
        "failure_mode_2_pbo": report.failure_mode_2_pbo,
        "failure_mode_3_dsr": report.failure_mode_3_dsr,
    }, indent=2))
    if report.failure_class:
        (out_dir / "failure_classification.md").write_text(
            f"# VRP — Failure Classification\n\n"
            f"**failure_class**: `{report.failure_class}`\n\n"
            f"**decision**: {report.decision}\n\n"
            f"Best variant: `{report.cross.best_name}`\n\n"
            f"Pre-registered failure modes triggered:\n"
            f"- concentration (max month > 50% PnL share): "
            f"{report.failure_mode_1_concentration}\n"
            f"- variant grid is duplicates (PBO > 0.20): "
            f"{report.failure_mode_2_pbo}\n"
            f"- combined-variant inflates grid (DSR < 0.5 when combined is best): "
            f"{report.failure_mode_3_dsr}\n"
        )
    console.print(f"[green]Report[/green] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
