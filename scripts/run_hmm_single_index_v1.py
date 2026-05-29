"""Run the HMM single-index v1 validation under the accepted exception policy.

Refuses to run if:
- the intake document is missing
- the accepted exception policy document is missing
- bars data for Tier-1 instruments cannot be loaded

Usage:
    PYTHONPATH=src uv run python scripts/run_hmm_single_index_v1.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.strategies.hmm_runner import (
    ACCEPTED_INTAKE_REF,
    HMMRunnerSpec,
    assign_exception_status,
    run_hmm_v1_pipeline,
)
from quant_research_stack.signal_research.validation.cash_leg_reporting import (
    ALL_ASSUMPTIONS,
    CASH_CONSERVATIVE,
    load_tbill_panel,
)
from quant_research_stack.signal_research.validation.spec import (
    ACCEPTED_EXCEPTION_POLICY_REF,
)
from quant_research_stack.signal_research.vrp.data import fetch_vrp_data

console = Console()


def _verify_required_docs() -> None:
    """Refuse to run if the accepted exception policy doc or the intake are
    missing on disk."""
    repo_root = Path(__file__).resolve().parent.parent
    policy_path = repo_root / ACCEPTED_EXCEPTION_POLICY_REF
    intake_path = repo_root / ACCEPTED_INTAKE_REF
    if not policy_path.exists():
        raise SystemExit(
            f"REFUSING TO RUN: accepted exception policy not found at "
            f"{policy_path}. Cannot invoke the exception path."
        )
    if not intake_path.exists():
        raise SystemExit(
            f"REFUSING TO RUN: intake document not found at "
            f"{intake_path}. Cannot pre-register this validation."
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument("--dev-end", default="2022-12-31")
    p.add_argument("--holdout-start", default="2023-01-01")
    p.add_argument("--cache-root", default="data/processed/vrp/bars")
    p.add_argument("--tbill-cache-root", default="data/processed/fred")
    p.add_argument(
        "--out", default="reports/signal_research/hmm_single_index"
    )
    args = p.parse_args()

    _verify_required_docs()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    dev_end = dt.date.fromisoformat(args.dev_end)
    holdout_start = dt.date.fromisoformat(args.holdout_start)

    # SPY and QQQ bars (reuse the VRP cache)
    fetched = fetch_vrp_data(
        start=start, end=end, cache_root=Path(args.cache_root),
    )
    bars = fetched.underlying
    if bars.is_empty():
        raise SystemExit(
            "REFUSING TO RUN: no underlying bars for SPY/QQQ available."
        )
    console.print(
        f"[green]Bars loaded[/green]: "
        f"{sorted(set(bars['symbol'].to_list()))}"
    )

    # T-bill panel for cash leg
    tbill = load_tbill_panel(
        start=start, end=end, cache_root=Path(args.tbill_cache_root),
    )
    if tbill.is_empty():
        console.print(
            "[yellow]WARNING[/yellow] T-bill series unavailable; cash leg "
            "will fall back to zero rate (still reported under all three "
            "assumptions per intake §8)."
        )

    runner_spec = HMMRunnerSpec(
        instruments=("SPY", "QQQ"),
        start=start, end=end, dev_end=dev_end, holdout_start=holdout_start,
    )

    console.print(
        "[cyan]Running[/cyan] 18 HMM variants + 12 baselines on SPY/QQQ..."
    )
    results, cross, scorecards = run_hmm_v1_pipeline(
        bars=bars, runner_spec=runner_spec, tbill_panel=tbill,
    )

    console.print("[bold]Per-variant headlines (conservative cash leg):[/bold]")
    for r in results:
        cons = r.dev_cash_legs[CASH_CONSERVATIVE.name]
        hd_cons = r.holdout_cash_legs[CASH_CONSERVATIVE.name]
        sc = scorecards[r.name]
        status = assign_exception_status(scorecard=sc, category=r.category)
        prefix = "[green]" if r.category == "hmm_variant" else "[yellow]"
        console.print(
            f"  {prefix}{r.name:34s}[/] "
            f"dev={cons.sharpe_annual:+6.3f}  "
            f"hd={hd_cons.sharpe_annual:+6.3f}  "
            f"cs2x={r.cost_stress_dev.get(2.0, float('nan')):+6.3f}  "
            f"delay1d={r.delay_stress_dev.get(1, float('nan')):+6.3f}  "
            f"max_year_share={r.yearly_pnl_share_max*100:5.1f}%  "
            f"status={status.name_lower}"
        )
    console.print(
        f"[cyan]Cross-strategy[/cyan]: "
        f"PBO={cross.get('pbo_raw_global', float('nan'))}  "
        f"best=`{cross.get('best_name')}`  "
        f"DSR={cross.get('best_dsr', float('nan'))}  "
        f"PSR_zero={cross.get('best_psr_zero', float('nan'))}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(
        results=results, cross=cross, scorecards=scorecards,
        runner_spec=runner_spec, output_dir=out_dir,
    )
    console.print(f"[green]Reports[/green] written to {out_dir}")
    return 0


def _write_outputs(
    *,
    results,
    cross,
    scorecards,
    runner_spec: HMMRunnerSpec,
    output_dir: Path,
) -> None:
    # Registry parquet
    rows = []
    for r in results:
        cons_dev = r.dev_cash_legs[CASH_CONSERVATIVE.name]
        cons_hd = r.holdout_cash_legs[CASH_CONSERVATIVE.name]
        sc = scorecards[r.name]
        status = assign_exception_status(scorecard=sc, category=r.category)
        rows.append({
            "name": r.name,
            "category": r.category,
            "instrument": r.instrument,
            "dev_sharpe_conservative": cons_dev.sharpe_annual,
            "holdout_sharpe_conservative": cons_hd.sharpe_annual,
            "dev_max_dd": cons_dev.max_drawdown,
            "boot_lower_95": r.bootstrap_lower_95,
            "boot_upper_95": r.bootstrap_upper_95,
            "cs_2x_sharpe": r.cost_stress_dev.get(2.0, float("nan")),
            "cs_3x_sharpe": r.cost_stress_dev.get(3.0, float("nan")),
            "delay_1d_sharpe": r.delay_stress_dev.get(1, float("nan")),
            "delay_2d_sharpe": r.delay_stress_dev.get(2, float("nan")),
            "yearly_pnl_share_max": r.yearly_pnl_share_max,
            "quarterly_pnl_share_max": r.quarterly_pnl_share_max,
            "n_positive_years": r.n_positive_years,
            "sharpe_excl_2020": r.sharpe_excl_2020,
            "sharpe_excl_2022": r.sharpe_excl_2022,
            "sharpe_pre_2020": r.sharpe_pre_2020,
            "all_gates_pass": sc.all_pass,
            "status": status.name_lower,
        })
    pl.DataFrame(rows).write_parquet(
        output_dir / "hmm_single_index_registry.parquet"
    )

    # Validation report
    header = (
        "| Variant | instrument | dev SR | dev DD | dev CI_lo | "
        "holdout SR | cs-2x SR | cs-3x SR | delay-1d SR | delay-2d SR | "
        "year share | status |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    body_rows = []
    for r in results:
        cons_dev = r.dev_cash_legs[CASH_CONSERVATIVE.name]
        cons_hd = r.holdout_cash_legs[CASH_CONSERVATIVE.name]
        sc = scorecards[r.name]
        status = assign_exception_status(scorecard=sc, category=r.category)
        body_rows.append(
            f"| `{r.name}` | {r.instrument} | "
            f"{cons_dev.sharpe_annual:+.3f} | "
            f"{cons_dev.max_drawdown*100:+.2f}% | "
            f"{r.bootstrap_lower_95:+.3f} | "
            f"{cons_hd.sharpe_annual:+.3f} | "
            f"{r.cost_stress_dev.get(2.0, float('nan')):+.3f} | "
            f"{r.cost_stress_dev.get(3.0, float('nan')):+.3f} | "
            f"{r.delay_stress_dev.get(1, float('nan')):+.3f} | "
            f"{r.delay_stress_dev.get(2, float('nan')):+.3f} | "
            f"{r.yearly_pnl_share_max*100:.1f}% | "
            f"{status.name_lower} |"
        )
    pbo = cross.get("pbo_raw_global", float("nan"))
    pbo_str = f"{float(pbo):.3f}" if isinstance(pbo, int | float) else str(pbo)
    dsr_val = cross.get("best_dsr", float("nan"))
    dsr_str = f"{float(dsr_val):.3f}" if isinstance(dsr_val, int | float) else str(dsr_val)
    psr_val = cross.get("best_psr_zero", float("nan"))
    psr_str = f"{float(psr_val):.3f}" if isinstance(psr_val, int | float) else str(psr_val)
    md = "\n".join([
        "# HMM Single-Index v1 — Validation Report",
        "",
        f"Intake: `{runner_spec.intake_ref}`",
        f"Exception policy: `{runner_spec.exception_policy_ref}`",
        "",
        "## Fixture",
        f"- instruments: {list(runner_spec.instruments)}",
        f"- history: {runner_spec.start} → {runner_spec.end}",
        f"- dev:     {runner_spec.start} → {runner_spec.dev_end}",
        f"- holdout: {runner_spec.holdout_start} → {runner_spec.end}",
        f"- costs: {runner_spec.commission_bps_one_way} bps commission + "
        f"{runner_spec.spread_bps_one_way} bps spread one-way",
        "- evaluated against CONSERVATIVE after-fee cash leg "
        "(T-bill DTB3 minus 25 bps prime-broker default)",
        "",
        "## Side-by-side results (conservative cash leg)",
        "",
        header, sep, *body_rows,
        "",
        "## Cross-strategy controls",
        "",
        f"- PBO raw_global: {pbo_str}  (gate ≤ {runner_spec.gate_pbo_max})",
        f"- Best strategy: `{cross.get('best_name')}`",
        f"- DSR for best: {dsr_str}  (gate ≥ {runner_spec.gate_dsr_min})",
        f"- PSR_zero for best: {psr_str}",
        f"- n_strategies: {cross.get('n_strategies')}",
        "",
        "## Status outcomes",
        "",
        "Per intake §11, the maximum status reachable from this validation is",
        "`exception_review_required`. No `paper_trade_candidate` or",
        "`production_candidate` status is emitted by this run.",
        "",
    ])
    (output_dir / "hmm_single_index_validation_report.md").write_text(md)

    # Cash-leg report
    cash_rows = []
    for r in results:
        for a in ALL_ASSUMPTIONS:
            dev = r.dev_cash_legs[a.name]
            hd = r.holdout_cash_legs[a.name]
            cash_rows.append(
                f"| `{r.name}` | {a.name} | "
                f"{dev.sharpe_annual:+.3f} | {dev.max_drawdown*100:+.2f}% | "
                f"{hd.sharpe_annual:+.3f} | {hd.max_drawdown*100:+.2f}% |"
            )
    cash_md = "\n".join([
        "# HMM Single-Index v1 — Cash-Leg Report",
        "",
        "Per intake §8 / accepted exception policy §3.25, §4.14:",
        "",
        "Long-or-cash strategies must report under three cash assumptions.",
        "The §9 gate is evaluated against `conservative_after_fee` ONLY.",
        "",
        "| Variant | assumption | dev Sharpe | dev DD | holdout Sharpe | holdout DD |",
        "|---|---|---:|---:|---:|---:|",
        *cash_rows,
        "",
    ])
    (output_dir / "hmm_cash_leg_report.md").write_text(cash_md)

    # State stability report
    stability_rows = []
    for r in results:
        if r.category != "hmm_variant" or r.state_stability is None:
            continue
        s = r.state_stability
        stability_rows.append(
            f"| `{r.name}` | {s.n_refits} | {s.n_economic_flips} | "
            f"{s.flip_rate*100:.1f}% | {s.raw_label_flips} | "
            f"{'PASS' if s.passes_stability_gate else 'FAIL'} |"
        )
    stability_md = "\n".join([
        "# HMM Single-Index v1 — State Stability Report",
        "",
        "Per intake §5.4 / exception policy §4.5: economic-identity flips",
        "drive demotion; raw HMM label permutations do NOT.",
        "",
        "Gate: economic-identity flip rate ≤ 20% across refits.",
        "",
        "| Variant | n_refits | n_economic_flips | flip_rate | raw_label_flips | gate |",
        "|---|---:|---:|---:|---:|---|",
        *stability_rows,
        "",
    ])
    (output_dir / "hmm_state_stability_report.md").write_text(stability_md)

    # Baseline comparison report (HMM variant vs same-instrument baselines)
    base_rows = []
    for r in results:
        if r.category != "hmm_variant":
            continue
        cons_dev = r.dev_cash_legs[CASH_CONSERVATIVE.name]
        sc = scorecards[r.name]
        base_rows.append(
            f"| `{r.name}` | "
            f"{cons_dev.sharpe_annual:+.3f} | "
            f"{'P' if sc.beats_buy_and_hold_pass else 'F'} | "
            f"{'P' if sc.beats_vol_targeted_pass else 'F'} | "
            f"{'P' if sc.beats_sma_50_200_pass else 'F'} | "
            f"{'P' if sc.beats_mom_12_1_pass else 'F'} | "
            f"{'P' if sc.random_baseline_fails_pass else 'F'} | "
            f"{'P' if sc.inverted_baseline_fails_pass else 'F'} |"
        )
    base_md = "\n".join([
        "# HMM Single-Index v1 — Baseline Comparison Report",
        "",
        "Per intake §9.16-§9.21. Each HMM variant must beat ALL non-sanity",
        "baselines on Sharpe AND max drawdown, AND the random / inverted",
        "sanity baselines must fail the §9.3 gate.",
        "",
        "| Variant | dev SR | beats BAH | beats VT | beats SMA | beats MOM | "
        "random fails | inverted fails |",
        "|---|---:|---|---|---|---|---|---|",
        *base_rows,
        "",
    ])
    (output_dir / "hmm_baseline_comparison_report.md").write_text(base_md)

    # Exception-gate scorecard
    gate_rows = []
    for r in results:
        if r.category != "hmm_variant":
            continue
        sc = scorecards[r.name]
        # Concise gate flags
        flags = [
            ("dev_sharpe", sc.dev_sharpe_pass),
            ("holdout_sharpe", sc.holdout_sharpe_pass),
            ("cs2x", sc.cost_stress_2x_pass),
            ("cs3x", sc.cost_stress_3x_pass),
            ("delay1d", sc.delay_1d_pass),
            ("delay2d", sc.delay_2d_pass),
            ("dd/calmar", sc.max_dd_or_calmar_pass),
            ("year_share", sc.year_share_pass),
            ("qtr_share", sc.quarter_share_pass),
            ("npos_years", sc.min_positive_years_pass),
            ("excl2020", sc.survives_excl_2020_pass),
            ("excl2022", sc.survives_excl_2022_pass),
            ("pre2020", sc.survives_pre_2020_subsample_pass),
            ("beats_bah", sc.beats_buy_and_hold_pass),
            ("beats_vt", sc.beats_vol_targeted_pass),
            ("beats_sma", sc.beats_sma_50_200_pass),
            ("beats_mom", sc.beats_mom_12_1_pass),
            ("random_fail", sc.random_baseline_fails_pass),
            ("inverted_fail", sc.inverted_baseline_fails_pass),
            ("boot_lo", sc.bootstrap_ci_lower_pass),
            ("pbo", sc.pbo_pass),
            ("dsr", sc.dsr_pass),
            ("stability", sc.economic_identity_stability_pass),
            ("cash_cons", sc.cash_leg_conservative_pass),
        ]
        passes = sum(1 for _, p in flags if p)
        gate_rows.append(
            f"| `{r.name}` | {passes}/{len(flags)} | "
            f"{'YES' if sc.all_pass else 'no'} | "
            f"{', '.join(n for n, p in flags if not p)} |"
        )
    gate_md = "\n".join([
        "# HMM Single-Index v1 — Exception-Gate Scorecard",
        "",
        "Per accepted exception policy §3 and intake §9 (24 gates total).",
        "",
        "| Variant | passed | all pass? | failed gates |",
        "|---|---:|---|---|",
        *gate_rows,
        "",
    ])
    (output_dir / "hmm_exception_gate_report.md").write_text(gate_md)

    # Failure classification — only if any HMM variant misses a gate
    any_fail = any(
        not scorecards[r.name].all_pass
        for r in results if r.category == "hmm_variant"
    )
    if any_fail:
        fail_lines = ["# HMM Single-Index v1 — Failure Classification", ""]
        for r in results:
            if r.category != "hmm_variant":
                continue
            sc = scorecards[r.name]
            if sc.all_pass:
                continue
            fail_lines.append(f"## `{r.name}`")
            fail_lines.append("")
            fail_lines.append("Failed gates:")
            for attr, val in sc.__dict__.items():
                if attr == "all_pass":
                    continue
                if not val:
                    fail_lines.append(f"- `{attr}` = False")
            fail_lines.append("")
        (output_dir / "hmm_failure_classification.md").write_text(
            "\n".join(fail_lines)
        )

    # JSON summary
    (output_dir / "hmm_cross_metrics.json").write_text(json.dumps({
        "pbo_raw_global": cross.get("pbo_raw_global"),
        "best_name": cross.get("best_name"),
        "best_dsr": cross.get("best_dsr"),
        "best_psr_zero": cross.get("best_psr_zero"),
        "n_strategies": cross.get("n_strategies"),
    }, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
