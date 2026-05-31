"""Run the sector-conditional Avellaneda-Lee experiment.

Reuses the cached bars from data/processed/momentum_scaleup/bars (300 SP500
candidates × 2006-2026). Loads SP500 sector info, partitions into 6 sectors,
runs 18 AvL variants × 6 sectors + 3 sanity baselines, computes PBO/DSR,
applies the 8-criteria decision rule, writes per-sector and aggregate reports.

Usage:
    PYTHONPATH=src uv run python scripts/run_sector_avl_experiment.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.backtests.sector_avl import (
    HMMGate,
    SectorAvLSpec,
    apply_decision_rule,
    assign_sectors,
    cross_strategy_metrics,
    filter_sectors,
    render_aggregate_report,
    render_per_sector_report,
    run_all_sector_avl_variants,
    run_sanity_baselines,
)
from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
)
from quant_research_stack.signal_research.data.sp500_components import (
    load_or_fetch_sp500,
)

console = Console()


def _cache_path(root: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "IDX_").replace("=", "_")
    return root / f"{safe}.parquet"


def _load_or_fetch(
    *, ticker: str, start: dt.date, end: dt.date, cache_root: Path
) -> pl.DataFrame | None:
    p = _cache_path(cache_root, ticker)
    if p.exists():
        df = pl.read_parquet(p)
        if df.height > 0:
            return df
    try:
        df = fetch_one_ticker(ticker, config=LongHistoryConfig(start=start, end=end))
    except Exception as exc:
        console.print(f"[yellow]skip[/yellow] {ticker}: {exc}")
        return None
    if df.is_empty():
        return None
    cache_root.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    return df


def _normalize_one(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    def col(name: str) -> str:
        found = cols.get(name) or cols.get(name.replace(" ", ""))
        if found is None:
            raise KeyError(f"column {name!r} missing from {list(df.columns)}")
        return found

    keep = df.select([
        pl.col(col("date")).alias("date"),
        pl.col(col("open")).alias("open"),
        pl.col(col("high")).alias("high"),
        pl.col(col("low")).alias("low"),
        pl.col(col("close")).alias("close"),
        pl.col(col("volume")).alias("volume"),
    ]).with_columns(pl.lit(ticker).alias("symbol"))
    return keep.with_columns(pl.col("date").cast(pl.Date)).drop_nulls()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument("--dev-end", default="2022-12-31")
    p.add_argument("--holdout-start", default="2023-01-01")
    p.add_argument("--max-candidates", type=int, default=300)
    p.add_argument(
        "--cache-root", default="data/processed/momentum_scaleup/bars"
    )
    p.add_argument("--out", default="reports/signal_research/sector_avl")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    dev_end = dt.date.fromisoformat(args.dev_end)
    holdout_start = dt.date.fromisoformat(args.holdout_start)
    cache_root = Path(args.cache_root)

    sp500_parquet = Path("data/processed/signal_research/sp500/sp500_current.parquet")
    sp500_df = load_or_fetch_sp500(parquet_path=sp500_parquet)
    raw_candidates = sp500_df["symbol"].to_list()[: args.max_candidates]
    candidates = [t.replace(".", "-") for t in raw_candidates]
    console.print(f"[cyan]Pool[/cyan] of {len(candidates)} SP500 candidates")

    frames: list[pl.DataFrame] = []
    fetched_count = 0
    for tkr in candidates:
        df = _load_or_fetch(ticker=tkr, start=start, end=end, cache_root=cache_root)
        if df is None:
            continue
        try:
            frames.append(_normalize_one(df, tkr))
            fetched_count += 1
        except Exception as exc:
            console.print(f"[yellow]normalize-fail[/yellow] {tkr}: {exc}")
    console.print(f"[green]Fetched[/green] {fetched_count} tickers")

    panel_all = pl.concat(frames, how="diagonal_relaxed").drop_nulls(
        subset=["open", "high", "low", "close", "volume"]
    )
    panel_all = panel_all.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    # Build sector map. Note the yfinance hyphen-substitution (BRK.B -> BRK-B); we
    # need to also map back when looking up sectors. sp500_df uses dot form.
    sector_map_raw = assign_sectors(sp500_df)
    sector_map = {k.replace(".", "-"): v for k, v in sector_map_raw.items()}

    spec = SectorAvLSpec(
        sectors_to_include=(
            "Financials",
            "Industrials",
            "Information Technology",
            "Health Care",
            "Consumer Discretionary",
            "Energy",
        ),
        min_sector_size=15,
        start=start, end=end, dev_end=dev_end, holdout_start=holdout_start,
        pca_window=252,
        pca_components_grid=(1, 2, 3),
        z_entry_grid=(1.0, 1.5, 2.0),
        hmm_gates=(HMMGate.NONE, HMMGate.RISK_ON),
        z_exit_reversion=0.5,
        max_holding_days=10,
        q_quantile_sector=0.25,
        cohort="focused_basket",
        equity=1_000_000.0,
    )

    baskets = filter_sectors(bars=panel_all, sector_map=sector_map, spec=spec)
    console.print(
        f"[cyan]Sectors[/cyan] meeting min_size={spec.min_sector_size}: "
        + ", ".join(f"{s} ({len(t)} names)" for s, t in baskets.items())
    )
    if not baskets:
        console.print("[red]No sectors pass the min-size filter; aborting.[/red]")
        return 1

    console.print(
        f"[cyan]Running[/cyan] "
        f"{len(spec.pca_components_grid) * len(spec.z_entry_grid) * len(spec.hmm_gates)} "
        f"AvL variants on {len(baskets)} sectors ..."
    )
    variants = run_all_sector_avl_variants(
        bars=panel_all, sector_baskets=baskets, spec=spec,
    )
    console.print(f"[green]Variants done[/green] ({len(variants)})")
    for r in variants:
        console.print(
            f"  [green]{r.name:34s}[/green] "
            f"dev={r.dev_metrics['sharpe']:+6.3f}  "
            f"hd={r.holdout_metrics['sharpe']:+6.3f}  "
            f"cs2x={r.cost_stress_metrics['sharpe']:+6.3f}"
        )

    console.print("[cyan]Running 3 sanity baselines...[/cyan]")
    baselines = run_sanity_baselines(
        bars=panel_all, sector_baskets=baskets, spec=spec,
    )
    for r in baselines:
        console.print(
            f"  [yellow]{r.name:34s}[/yellow] "
            f"dev={r.dev_metrics['sharpe']:+6.3f}  "
            f"hd={r.holdout_metrics['sharpe']:+6.3f}  "
            f"cs2x={r.cost_stress_metrics['sharpe']:+6.3f}"
        )

    cross = cross_strategy_metrics(variants + baselines)
    console.print(
        f"[cyan]PBO[/cyan] raw_global={cross.pbo_raw_global:.3f}  "
        f"DSR={cross.best_dsr:.3f}  PSR_zero={cross.best_psr_zero:.3f}"
    )

    decision, failure_class = apply_decision_rule(
        variants=variants, baselines=baselines, cross=cross,
    )
    console.print(f"[bold yellow]DECISION:[/bold yellow] {decision}")
    if failure_class:
        console.print(f"[bold red]failure_class:[/bold red] {failure_class}")

    out_dir = Path(args.out)
    agg_path = render_aggregate_report(
        variants=variants, baselines=baselines, cross=cross,
        decision=decision, failure_class=failure_class,
        spec=spec, output_path=out_dir / "aggregate_sector_avl_report.md",
    )
    sec_path = render_per_sector_report(
        variants=variants, baselines=baselines,
        spec=spec, output_path=out_dir / "per_sector_report.md",
    )
    console.print(f"[green]Reports[/green] {agg_path} | {sec_path}")

    # Machine-readable artefacts
    out_dir.mkdir(parents=True, exist_ok=True)
    pbo_json = out_dir / "sector_avl_pbo.json"
    pbo_json.write_text(json.dumps({
        "pbo_raw_global": cross.pbo_raw_global,
        "pbo_per_profile": cross.pbo_per_profile,
        "pbo_per_family": cross.pbo_per_family,
        "best_index": cross.best_index,
        "best_dsr": cross.best_dsr,
        "best_psr_zero": cross.best_psr_zero,
        "n_strategies": cross.n_strategies,
        "decision": decision,
        "failure_class": failure_class,
    }, indent=2))
    console.print(f"[green]Wrote[/green] {pbo_json}")

    if failure_class:
        fail_path = out_dir / "failure_classification.md"
        fail_path.write_text(
            f"# Sector-Conditional AvL — Failure Classification\n\n"
            f"**failure_class**: `{failure_class}`\n\n"
            f"**decision**: {decision}\n\n"
            f"Per spec §4.10 failure taxonomy.\n"
        )
        console.print(f"[yellow]Wrote[/yellow] {fail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
