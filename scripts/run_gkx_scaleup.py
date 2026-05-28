"""Run the GKX LightGBM scale-up experiment.

Reuses the cached bars from data/processed/momentum_scaleup/bars.

Usage:
    PYTHONPATH=src uv run python scripts/run_gkx_scaleup.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.backtests.gkx_scaleup import (
    GKXSpec,
    apply_decision_rule,
    cross_strategy_metrics,
    render_report,
    run_gkx_scaleup,
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
    return (
        panel.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))
        .group_by("symbol")
        .agg(pl.col("dollar_volume").median().alias("med_dvol"))
        .sort("med_dvol", descending=True)
        .head(n)["symbol"]
        .to_list()
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument("--dev-end", default="2022-12-31")
    p.add_argument("--holdout-start", default="2023-01-01")
    p.add_argument("--max-candidates", type=int, default=300)
    p.add_argument("--cache-root", default="data/processed/momentum_scaleup/bars")
    p.add_argument("--out", default="reports/signal_research/gkx_scaleup")
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
    for tkr in candidates:
        df = _load_or_fetch(ticker=tkr, start=start, end=end, cache_root=cache_root)
        if df is None:
            continue
        try:
            frames.append(_normalize_one(df, tkr))
        except Exception as exc:
            console.print(f"[yellow]normalize-fail[/yellow] {tkr}: {exc}")
    console.print(f"[green]Loaded[/green] {len(frames)} ticker frames")

    panel_all = pl.concat(frames, how="diagonal_relaxed").drop_nulls(
        subset=["open", "high", "low", "close", "volume"]
    ).filter((pl.col("date") >= start) & (pl.col("date") <= end))

    top_200 = _top_n_by_dollar_volume(panel_all, n=200)
    top_100 = top_200[:100]
    bars_top100 = panel_all.filter(pl.col("symbol").is_in(top_100))
    bars_top200 = panel_all.filter(pl.col("symbol").is_in(top_200))
    console.print(
        f"[cyan]Top-100[/cyan] {bars_top100.height} rows × "
        f"{bars_top100['symbol'].n_unique()} symbols"
    )
    console.print(
        f"[cyan]Top-200[/cyan] {bars_top200.height} rows × "
        f"{bars_top200['symbol'].n_unique()} symbols"
    )

    spec = GKXSpec(
        start=start, end=end, dev_end=dev_end, holdout_start=holdout_start,
        label_horizons=(5, 21, 63),
        universes=("top100", "top200"),
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        walk_forward_folds=5, walk_forward_embargo=5,
        equity=1_000_000.0,
        q_quantile=0.20, cohort="full_universe",
    )

    console.print(
        f"[cyan]Running[/cyan] "
        f"{len(spec.universes) * len(spec.label_horizons)} GKX variants + "
        f"{3 * len(spec.universes)} baselines..."
    )
    variants, baselines = run_gkx_scaleup(
        bars_per_universe={"top100": bars_top100, "top200": bars_top200},
        spec=spec,
    )
    for r in variants:
        console.print(
            f"  [green]{r.name:30s}[/green] "
            f"dev={r.dev_metrics['sharpe']:+6.3f}  "
            f"hd={r.holdout_metrics['sharpe']:+6.3f}  "
            f"cs2x={r.cost_stress_metrics['sharpe']:+6.3f}  "
            f"pass={'YES' if r.research_pass else 'no'}"
        )
    for r in baselines:
        console.print(
            f"  [yellow]{r.name:30s}[/yellow] "
            f"dev={r.dev_metrics['sharpe']:+6.3f}  "
            f"hd={r.holdout_metrics['sharpe']:+6.3f}  "
            f"cs2x={r.cost_stress_metrics['sharpe']:+6.3f}"
        )

    cross = cross_strategy_metrics(variants, baselines)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = render_report(
        variants=variants, baselines=baselines, cross=cross,
        decision=decision, failure_class=failure_class,
        spec=spec, output_path=out_dir / "report.md",
    )
    (out_dir / "gkx_pbo.json").write_text(json.dumps({
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
    if failure_class:
        (out_dir / "failure_classification.md").write_text(
            f"# GKX Scale-Up — Failure Classification\n\n"
            f"**failure_class**: `{failure_class}`\n\n"
            f"**decision**: {decision}\n\n"
            f"Per spec §4.10 failure taxonomy.\n"
        )
    console.print(f"[green]Report[/green] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
