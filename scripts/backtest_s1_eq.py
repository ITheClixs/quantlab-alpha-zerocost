"""Backtest CLI (spec §5).

Usage:
    PYTHONPATH=src uv run python scripts/backtest_s1_eq.py \
        --config configs/backtest_eq.yaml --mode standard \
        --equity-root data/processed/equities \
        --run-dir experiments/alpha_eq/<run_id> \
        --out-dir experiments/alpha_eq/<run_id>/backtest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.report import ReportInputs, write_report
from quant_research_stack.alpha_eq.backtest.runner import (
    BacktestConfig,
    run_backtest,
)
from quant_research_stack.alpha_eq.backtest.sensitivity import (
    enumerate_audit_pack,
    enumerate_standard_pack,
)
from quant_research_stack.alpha_eq.data.loaders import EquityRootLoader

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/backtest_eq.yaml")
    p.add_argument("--mode", default="standard", choices=["standard", "audit"])
    p.add_argument("--equity-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    yaml.safe_load(Path(args.config).read_text())  # parse for future extension
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = EquityRootLoader(root=Path(args.equity_root))
    bars = loader.load_tradable_prices()
    bars = bars.with_columns(pl.col("date").alias("execution_date"))
    bars = bars.join(loader.load_adv(), on=["date", "symbol"], how="left")

    # Placeholder predictions (zero) — real prediction integration is a follow-up
    bars = bars.with_columns(
        pl.lit(0.0).alias("y_xs_pred"),
        pl.lit(True).alias("tradable"),
        pl.lit(True).alias("in_pit_universe"),
        pl.lit("general").alias("borrow_tier"),
        pl.lit(10.0).alias("roll_spread_bps"),
        pl.lit("tech").alias("sector"),
    )
    cases = list(enumerate_standard_pack() if args.mode == "standard" else enumerate_audit_pack())
    sensitivity_rows: list[dict[str, str | float]] = []
    res = None
    for case in cases:
        res = run_backtest(
            signals_with_bars=bars,
            config=BacktestConfig(
                portfolio=PortfolioBuildConfig(
                    q_quantile=case.q_quantile, target_gross=case.target_gross, equity=100_000.0,
                    adv_participation_pct=case.adv_participation_pct,
                ),
                fill_model=case.fill_model,
                cohort="full_universe",
                borrow_multiplier=case.borrow_multiplier,
                financing_rate_annual=0.0,
            ),
            dividends=loader.load_dividends() if (
                Path(args.equity_root) / "sp500_dividends.parquet"
            ).exists() else None,
        )
        sensitivity_rows.append(
            {
                "borrow": case.borrow_multiplier,
                "fill": case.fill_model.value,
                "q": case.q_quantile,
                "gross": case.target_gross,
                "net_alpha_bps_per_day": res.decomposition.net_alpha_bps_per_day,
            }
        )

    eq_manifest = json.loads((Path(args.equity_root) / "_manifest.json").read_text())
    label = eq_manifest["data_quality_label"]
    data_sha = hashlib.sha256((Path(args.equity_root) / "_manifest.json").read_bytes()).hexdigest()
    daily = res.daily_returns if res is not None else pl.DataFrame()
    decomposition = res.decomposition if res is not None else None
    inputs = ReportInputs(
        run_id=Path(args.run_dir).name,
        git_sha="filled-by-ci",
        data_manifest_sha256=data_sha,
        data_quality_label=label,
        cohort="full_universe",
        daily_returns=daily,
        decomposition_bps={
            "gross_alpha": decomposition.gross_alpha_bps_per_day if decomposition else 0.0,
            "cash_dividend": decomposition.cash_dividend_bps_per_day if decomposition else 0.0,
            "commission": decomposition.commission_drag_bps_per_day if decomposition else 0.0,
            "spread": decomposition.spread_drag_bps_per_day if decomposition else 0.0,
            "borrow": decomposition.borrow_drag_bps_per_day if decomposition else 0.0,
            "financing": decomposition.financing_drag_bps_per_day if decomposition else 0.0,
            "net_alpha": decomposition.net_alpha_bps_per_day if decomposition else 0.0,
        },
        sensitivity_rows=sensitivity_rows,
    )
    write_report(out_dir / "report.md", inputs)
    console.print(f"[bold green]Backtest report written:[/bold green] {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
