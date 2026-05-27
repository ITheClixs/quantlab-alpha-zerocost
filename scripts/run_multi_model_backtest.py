"""Run the multi-model side-by-side backtest on the cached real-data fixture.

Reuses the bars cache from the triple-barrier run (data/processed/triple_barrier_av_lee/bars/).

Usage:
    PYTHONPATH=src uv run python scripts/run_multi_model_backtest.py \\
        --top-n 50 --start 2015-01-01 --dev-end 2022-12-31 \\
        --holdout-start 2023-01-01 --end 2026-05-26
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.backtests.multi_model_fixture import (
    FixtureSpec,
    render_comparison_report,
    run_all_models_on_fixture,
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
            raise KeyError(f"column {name!r} not found in {list(df.columns)}")
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


def _top_by_dollar_volume(panel: pl.DataFrame, *, top_n: int) -> list[str]:
    dvol = (
        panel.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))
        .group_by("symbol")
        .agg(pl.col("dollar_volume").median().alias("med_dvol"))
        .sort("med_dvol", descending=True)
        .head(top_n)
    )
    return dvol["symbol"].to_list()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--dev-end", default="2022-12-31")
    p.add_argument("--holdout-start", default="2023-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument(
        "--out", default="reports/signal_research/multi_model_fixture/focused"
    )
    p.add_argument(
        "--cache-root", default="data/processed/triple_barrier_av_lee/bars"
    )
    p.add_argument("--max-candidates", type=int, default=90)
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    dev_end = dt.date.fromisoformat(args.dev_end)
    holdout_start = dt.date.fromisoformat(args.holdout_start)
    end = dt.date.fromisoformat(args.end)
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
    selected = _top_by_dollar_volume(panel_all, top_n=args.top_n)
    console.print(
        f"[cyan]Selected[/cyan] top {len(selected)} by median dollar volume: "
        f"{', '.join(selected[:10])}..."
    )
    bars = panel_all.filter(pl.col("symbol").is_in(selected))
    console.print(
        f"[green]Panel[/green] {bars.height} rows × "
        f"{bars['symbol'].n_unique()} symbols × {bars['date'].n_unique()} dates"
    )

    spec = FixtureSpec(
        universe_tickers=selected,
        start=start,
        end=end,
        dev_end=dev_end,
        holdout_start=holdout_start,
        pca_window=252,
        n_pca_components=5,
        z_entry=1.5,
        gkx_label_horizon=5,
        gkx_n_estimators=300,
        gkx_walk_forward_folds=5,
        gkx_walk_forward_embargo=10,
        equity=1_000_000.0,
        q_quantile=0.20,
        cohort="full_universe" if args.top_n >= 30 else "focused_basket",
    )
    console.print("[cyan]Running 4 models on shared fixture...[/cyan]")
    results = run_all_models_on_fixture(bars=bars, spec=spec)
    for name, r in results.items():
        console.print(
            f"  [green]{name:30s}[/green] "
            f"dev={r.dev_metrics['sharpe']:+6.3f}  "
            f"hd={r.holdout_metrics['sharpe']:+6.3f}  "
            f"cs2x={r.cost_stress_metrics['sharpe']:+6.3f}  "
            f"pass={'YES' if r.research_pass else 'no'}"
        )
    report = render_comparison_report(
        results, spec=spec, output_path=Path(args.out) / "report.md"
    )
    console.print(f"[green]Report[/green] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
