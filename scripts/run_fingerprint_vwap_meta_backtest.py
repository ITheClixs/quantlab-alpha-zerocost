"""Run the Fingerprint-VWAP meta-labeling backtest on a real SP500 universe.

Usage:
    PYTHONPATH=src uv run python scripts/run_fingerprint_vwap_meta_backtest.py \\
        --top-n 30 --start 2018-01-01 --end 2024-12-31 \\
        --band 0.0 --horizon 3 --trials 50 \\
        --out reports/signal_research/fingerprint_vwap_meta_v1/run01

The script fetches yfinance bars (cached to
data/processed/fingerprint_vwap_meta_v1/bars/), selects top-N tickers by
20-day median dollar volume, then hands off to `run_fingerprint_vwap_meta`
for the full pipeline.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
)
from quant_research_stack.signal_research.data.sp500_components import (
    load_or_fetch_sp500,
)
from quant_research_stack.signal_research.fingerprint_vwap.pipeline import (
    FingerprintVwapSpec,
    gate_verdict,
    render_report,
    run_fingerprint_vwap_meta,
)

console = Console()

_CACHE_ROOT_DEFAULT = "data/processed/fingerprint_vwap_meta_v1/bars"
_SP500_PARQUET = "data/processed/signal_research/sp500/sp500_current.parquet"
_MAX_CANDIDATES_DEFAULT = 120


# ---------------------------------------------------------------------------
# Data helpers — mirrors run_triple_barrier_av_lee_backtest.py
# ---------------------------------------------------------------------------


def _cache_path(root: Path, ticker: str) -> Path:
    return root / f"{ticker.replace('/', '_').replace('^', 'IDX_').replace('=', '_')}.parquet"


def _load_or_fetch(
    *,
    ticker: str,
    start: dt.date,
    end: dt.date,
    cache_root: Path,
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
    if df is None or df.is_empty():
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

    keep = df.select(
        [
            pl.col(col("date")).alias("date"),
            pl.col(col("open")).alias("open"),
            pl.col(col("high")).alias("high"),
            pl.col(col("low")).alias("low"),
            pl.col(col("close")).alias("close"),
            pl.col(col("volume")).alias("volume"),
        ]
    ).with_columns(pl.lit(ticker).alias("symbol"))
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


def _build_panel(
    *,
    start: dt.date,
    end: dt.date,
    top_n: int,
    max_candidates: int,
    cache_root: Path,
) -> pl.DataFrame | None:
    sp500_parquet = Path(_SP500_PARQUET)
    sp500_df = load_or_fetch_sp500(parquet_path=sp500_parquet)
    raw_candidates = sp500_df["symbol"].to_list()[:max_candidates]
    candidates = [t.replace(".", "-") for t in raw_candidates]
    console.print(
        f"[cyan]Pool[/cyan] of {len(candidates)} SP500 candidates "
        f"(top {max_candidates} by Wikipedia order)"
    )

    frames: list[pl.DataFrame] = []
    for tkr in candidates:
        df = _load_or_fetch(ticker=tkr, start=start, end=end, cache_root=cache_root)
        if df is None:
            continue
        try:
            frames.append(_normalize_one(df, tkr))
        except Exception as exc:
            console.print(f"[yellow]normalize-fail[/yellow] {tkr}: {exc}")

    console.print(f"[green]Fetched[/green] {len(frames)} tickers")
    if not frames:
        return None

    panel_all = pl.concat(frames, how="diagonal_relaxed").drop_nulls(
        subset=["open", "high", "low", "close", "volume"]
    )
    panel_all = panel_all.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    selected_tickers = _top_by_dollar_volume(panel_all, top_n=top_n)
    console.print(
        f"[cyan]Selected[/cyan] top {len(selected_tickers)} by median dollar volume: "
        f"{', '.join(selected_tickers[:10])}..."
    )
    panel = panel_all.filter(pl.col("symbol").is_in(selected_tickers))
    console.print(
        f"[green]Panel[/green] shape: {panel.height} rows × "
        f"{panel['symbol'].n_unique()} symbols × "
        f"{panel['date'].n_unique()} dates"
    )
    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fingerprint-VWAP meta-labeling backtest on real SP500 universe."
    )
    parser.add_argument("--top-n", type=int, default=30, help="Universe size after liquidity screen")
    parser.add_argument("--start", default="2018-01-01", help="Bar history start (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Bar history end (YYYY-MM-DD)")
    parser.add_argument("--band", type=float, default=0.0, help="VWAP band half-width (fraction)")
    parser.add_argument("--horizon", type=int, default=3, help="Forward return horizon in days")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials for deflated-Sharpe adjustment")
    parser.add_argument("--out", default="reports/signal_research/fingerprint_vwap_meta_v1/run", help="Output directory")
    parser.add_argument("--cache-root", default=_CACHE_ROOT_DEFAULT, help="Per-ticker parquet cache root")
    parser.add_argument("--max-candidates", type=int, default=_MAX_CANDIDATES_DEFAULT, help="Initial fetch pool size")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    cache_root = Path(args.cache_root)
    out_dir = Path(args.out)

    # ------------------------------------------------------------------
    # 1. Build universe panel
    # ------------------------------------------------------------------
    panel = _build_panel(
        start=start,
        end=end,
        top_n=args.top_n,
        max_candidates=args.max_candidates,
        cache_root=cache_root,
    )
    if panel is None:
        console.print("[red]No data fetched — aborting[/red]")
        return 1

    # ------------------------------------------------------------------
    # 2. Run pipeline
    # ------------------------------------------------------------------
    spec = FingerprintVwapSpec(
        windows=(20, 60, 120, 252),
        band=args.band,
        horizon_days=args.horizon,
    )
    console.print(f"[cyan]Running[/cyan] pipeline with spec: {spec!r}")
    result = run_fingerprint_vwap_meta(panel=panel, spec=spec)
    console.print(f"[green]Pipeline status[/green]: {result.get('status')}")

    # ------------------------------------------------------------------
    # 3. Gate verdict
    # ------------------------------------------------------------------
    if result.get("status") == "evaluated":
        predictions: pl.DataFrame = result["predictions"]
        # The predictions DataFrame contains a 'net_return' column per row
        # (per-event net return after round-trip cost). We aggregate to daily
        # means before passing to gate_verdict, consistent with how
        # _daily_returns() works in meta_label_walk_forward.
        daily_net_returns: list[float] = (
            predictions.group_by("date")
            .agg(pl.col("net_return").mean().alias("daily_net"))
            .sort("date")["daily_net"]
            .to_list()
        )
        verdict = gate_verdict(
            meta_net_sharpe=result["meta_net_sharpe"],
            baseline_net_sharpe=result["baseline_net_sharpe"],
            lift_margin=0.2,
            daily_net_returns=daily_net_returns,
            trials=args.trials,
        )
        console.print(f"[bold]Verdict[/bold]: {verdict['verdict']}")
        if verdict["failed"]:
            console.print(f"  failed gates: {verdict['failed']}")
    else:
        # Primary signal ineligible — build a terminal DO_NOT_ADVANCE verdict
        verdict = {
            "verdict": "DO_NOT_ADVANCE",
            "passed": False,
            "failed": ["primary_ineligible"],
            "net_sharpe": 0.0,
            "lift": 0.0,
            "deflated_sharpe": {},
        }
        elig = result.get("eligibility", {})
        console.print(
            f"[red]Primary ineligible[/red]: {elig.get('reason') or 'unknown'}"
        )

    # ------------------------------------------------------------------
    # 4. Write report
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    report_text = render_report(result=result, verdict=verdict, spec_repr=repr(spec))
    report_path = out_dir / "report.md"
    report_path.write_text(report_text)
    console.print(f"[green]Report[/green] written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
