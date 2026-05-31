"""Maximal momentum-only scale-up.

Fetches up to N candidates from current SP500, screens by 20-day median
dollar volume, runs the predeclared variant matrix (5 variants × 2 universes)
through the same hedge-fund-grade pipeline + PBO/DSR multiple-testing
controls.

Usage:
    PYTHONPATH=src uv run python scripts/run_momentum_scaleup.py \\
        --start 2006-01-01 --end 2026-05-26 \\
        --dev-end 2022-12-31 --holdout-start 2023-01-01 \\
        --max-candidates 300 \\
        --cache-root data/processed/momentum_scaleup/bars \\
        --out reports/signal_research/momentum_scaleup
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.backtests.momentum_scaleup import (
    MomentumSpec,
    apply_decision_rule,
    cross_strategy_metrics,
    render_momentum_scaleup_report,
    run_all_momentum_variants,
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


def _top_n_by_dollar_volume(panel: pl.DataFrame, *, n: int) -> list[str]:
    dvol = (
        panel.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))
        .group_by("symbol")
        .agg(pl.col("dollar_volume").median().alias("med_dvol"))
        .sort("med_dvol", descending=True)
        .head(n)
    )
    return dvol["symbol"].to_list()


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
    p.add_argument("--out", default="reports/signal_research/momentum_scaleup")
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
    top_200 = _top_n_by_dollar_volume(panel_all, n=200)
    top_100 = top_200[:100]
    bars_top200 = panel_all.filter(pl.col("symbol").is_in(top_200))
    bars_top100 = panel_all.filter(pl.col("symbol").is_in(top_100))
    console.print(
        f"[cyan]Top-100[/cyan]: {len(top_100)} symbols, {bars_top100.height} rows, "
        f"{bars_top100['date'].n_unique()} dates"
    )
    console.print(
        f"[cyan]Top-200[/cyan]: {len(top_200)} symbols, {bars_top200.height} rows, "
        f"{bars_top200['date'].n_unique()} dates"
    )

    spec_top100 = MomentumSpec(
        universe_tickers=top_100,
        start=start, end=end, dev_end=dev_end, holdout_start=holdout_start,
        equity=1_000_000.0,
        q_quantile=0.20,
        cohort="full_universe",
    )
    spec_top200 = MomentumSpec(
        universe_tickers=top_200,
        start=start, end=end, dev_end=dev_end, holdout_start=holdout_start,
        equity=1_000_000.0,
        q_quantile=0.20,
        cohort="full_universe",
    )

    console.print("[cyan]Running 10 momentum variants (5 × 2 universes)...[/cyan]")
    results = run_all_momentum_variants(
        bars_top100=bars_top100, bars_top200=bars_top200,
        spec_top100=spec_top100, spec_top200=spec_top200,
    )
    for r in results:
        console.print(
            f"  [green]{r.variant.value:30s}[/green] @{r.universe_label:6s}  "
            f"dev={r.dev_metrics['sharpe']:+6.3f}  "
            f"hd={r.holdout_metrics['sharpe']:+6.3f}  "
            f"cs2x={r.cost_stress_metrics['sharpe']:+6.3f}  "
            f"pass={'YES' if r.research_pass else 'no'}"
        )
    cross = cross_strategy_metrics(results)
    console.print(
        f"[cyan]PBO[/cyan] raw_global={cross.pbo_raw_global:.3f}  "
        f"DSR={cross.best_variant_dsr:.3f}  "
        f"PSR_zero={cross.best_variant_psr_zero:.3f}"
    )
    decision = apply_decision_rule(results)
    console.print(f"[bold yellow]DECISION:[/bold yellow] {decision}")

    out_dir = Path(args.out)
    report = render_momentum_scaleup_report(
        results=results, cross=cross, decision=decision,
        spec_top100=spec_top100, spec_top200=spec_top200,
        output_path=out_dir / "report.md",
    )
    console.print(f"[green]Report[/green] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
