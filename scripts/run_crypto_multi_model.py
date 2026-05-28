"""Crypto top-30 multi-model backtest — independent test of M3 model families
on a fundamentally different microstructure.

Fetches a curated pool of top-50 crypto candidates from yfinance (-USD pairs),
screens to top-30 by 20-day median dollar volume, then runs the existing
multi_model_fixture pipeline (raw AvL, 12-1 momentum, GKX-LGB, triple-barrier
meta-labeled AvL) with crypto-tuned costs and dev/holdout split.

Costs:
- commission: 4 bps one-way (Coinbase Prime / Binance institutional)
- spread:     5 bps one-way for top pairs
- cost-stress 2x

Universe: top-30 by ADV among the curated candidate pool with full
2018-2026 history. Coins with insufficient history are skipped.

Same dev/holdout discipline as equity runs:
- dev:     2018-01-01 → 2022-12-31  (~5 years)
- holdout: 2023-01-01 → 2026-05-26  (~3.4 years)

Usage:
    PYTHONPATH=src uv run python scripts/run_crypto_multi_model.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.backtests.multi_model_fixture import (
    FixtureSpec,
    cross_strategy_metrics,
    render_comparison_report,
    run_all_models_on_fixture,
)
from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
)

console = Console()

# Curated top-50 crypto candidates by approximate end-2024 market cap, excluding
# stablecoins and wrapped tokens. yfinance -USD pairs.
_CRYPTO_CANDIDATES: tuple[str, ...] = (
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "SOL-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
    "AVAX-USD", "ATOM-USD", "LINK-USD", "XLM-USD", "NEAR-USD", "FIL-USD",
    "UNI7083-USD", "UNI-USD", "ETC-USD", "HBAR-USD", "ICP-USD", "ALGO-USD",
    "AAVE-USD", "VET-USD", "EOS-USD", "THETA-USD", "MKR-USD", "EGLD-USD",
    "XTZ-USD", "AXS-USD", "SAND-USD", "MANA-USD", "FTM-USD", "ENJ-USD",
    "CHZ-USD", "GRT-USD", "ZEC-USD", "DASH-USD", "BAT-USD", "KSM-USD",
    "NEO-USD", "COMP-USD", "SUSHI-USD", "YFI-USD", "SNX-USD", "CRV-USD",
    "OMG-USD", "1INCH-USD",
)


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


def _coins_with_full_history(
    panel: pl.DataFrame, *, start: dt.date, min_days: int
) -> list[str]:
    counts = (
        panel.filter(pl.col("date") >= start)
        .group_by("symbol")
        .agg(pl.len().alias("n_days"))
        .filter(pl.col("n_days") >= min_days)
    )
    return counts["symbol"].to_list()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2026-05-26")
    p.add_argument("--dev-end", default="2022-12-31")
    p.add_argument("--holdout-start", default="2023-01-01")
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--min-history-days", type=int, default=1500)
    p.add_argument("--cache-root", default="data/processed/crypto/bars")
    p.add_argument("--out", default="reports/signal_research/crypto_multi_model")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    dev_end = dt.date.fromisoformat(args.dev_end)
    holdout_start = dt.date.fromisoformat(args.holdout_start)
    cache_root = Path(args.cache_root)

    console.print(f"[cyan]Pool[/cyan] of {len(_CRYPTO_CANDIDATES)} candidates")
    frames: list[pl.DataFrame] = []
    fetched = 0
    for tkr in _CRYPTO_CANDIDATES:
        df = _load_or_fetch(ticker=tkr, start=start, end=end, cache_root=cache_root)
        if df is None:
            continue
        try:
            frames.append(_normalize_one(df, tkr))
            fetched += 1
        except Exception as exc:
            console.print(f"[yellow]normalize-fail[/yellow] {tkr}: {exc}")
    console.print(f"[green]Fetched[/green] {fetched} tickers")

    panel_all = pl.concat(frames, how="diagonal_relaxed").drop_nulls(
        subset=["open", "high", "low", "close", "volume"]
    ).filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
        & (pl.col("close") > 0) & (pl.col("volume") > 0)
    )

    coins_with_hist = _coins_with_full_history(
        panel_all, start=start, min_days=args.min_history_days,
    )
    console.print(
        f"[cyan]Coins with ≥{args.min_history_days} days of history[/cyan]: "
        f"{len(coins_with_hist)}"
    )
    panel_all = panel_all.filter(pl.col("symbol").is_in(coins_with_hist))

    selected = _top_n_by_dollar_volume(panel_all, n=args.top_n)
    console.print(
        f"[cyan]Selected top {len(selected)}[/cyan] by median dollar volume: "
        f"{', '.join(selected[:15])}..."
    )
    bars = panel_all.filter(pl.col("symbol").is_in(selected))
    console.print(
        f"[green]Panel[/green] {bars.height} rows × "
        f"{bars['symbol'].n_unique()} symbols × {bars['date'].n_unique()} dates"
    )

    spec = FixtureSpec(
        universe_tickers=selected,
        start=start, end=end, dev_end=dev_end, holdout_start=holdout_start,
        pca_window=252,
        n_pca_components=3,  # lower than equity default; crypto has fewer factors
        z_entry=1.5,
        gkx_label_horizon=5,
        gkx_n_estimators=300,
        gkx_walk_forward_folds=5,
        gkx_walk_forward_embargo=10,
        commission_bps_one_way=4.0,  # crypto institutional
        spread_bps_one_way=5.0,
        cost_stress_multiplier=2.0,
        equity=1_000_000.0,
        q_quantile=0.25,
        cohort="focused_basket",
        data_quality_label="public_snapshot_not_pit",
        constituent_survivorship_applicable=False,  # directly-traded instruments
    )

    console.print("[cyan]Running 4 model families on crypto fixture...[/cyan]")
    results = run_all_models_on_fixture(bars=bars, spec=spec)
    for name, r in results.items():
        console.print(
            f"  [green]{name:30s}[/green] "
            f"dev={r.dev_metrics['sharpe']:+6.3f}  "
            f"hd={r.holdout_metrics['sharpe']:+6.3f}  "
            f"cs2x={r.cost_stress_metrics['sharpe']:+6.3f}  "
            f"pass={'YES' if r.research_pass else 'no'}"
        )

    cross = cross_strategy_metrics(results)
    console.print(
        f"[cyan]PBO[/cyan] raw_global={cross.pbo_raw_global:.3f}  "
        f"DSR for `{cross.best_name}`={cross.best_dsr:.3f}  "
        f"PSR_zero={cross.best_psr_zero:.3f}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = render_comparison_report(
        results, spec=spec, output_path=out_dir / "report.md"
    )

    # Append cross-strategy metrics section to the report (the existing renderer
    # doesn't include them by default — backwards-compat for the equity callers).
    extra = "\n".join([
        "",
        "## Cross-strategy multiple-testing controls",
        "",
        f"- **PBO raw_global**: {cross.pbo_raw_global:.3f}  (gate: ≤ 0.25)",
        f"- **Best strategy**: `{cross.best_name}`",
        f"- **DSR for best**: {cross.best_dsr:.3f}  (gate: ≥ 0.50)",
        f"- **PSR_zero for best**: {cross.best_psr_zero:.3f}",
        f"- **n_strategies in DSR deflation**: {cross.n_strategies}",
        "",
    ])
    with report_path.open("a") as fp:
        fp.write(extra)
    (out_dir / "crypto_pbo.json").write_text(json.dumps({
        "pbo_raw_global": cross.pbo_raw_global,
        "pbo_per_profile": cross.pbo_per_profile,
        "pbo_per_family": cross.pbo_per_family,
        "best_name": cross.best_name,
        "best_dsr": cross.best_dsr,
        "best_psr_zero": cross.best_psr_zero,
        "n_strategies": cross.n_strategies,
        "selected_tickers": selected,
    }, indent=2))

    any_pass = any(r.research_pass for r in results.values())
    if not any_pass:
        (out_dir / "failure_classification.md").write_text(
            "# Crypto multi-model — Failure classification\n\n"
            "No model variant passes all promotion gates on crypto top-30, "
            f"2018-2026 with hedge-fund-grade costs ({spec.commission_bps_one_way} "
            f"bps commission + {spec.spread_bps_one_way * 10:.1f} bps spread).\n\n"
            f"Best strategy: `{cross.best_name}` "
            f"with dev Sharpe={results[cross.best_name].dev_metrics['sharpe']:+.3f}, "
            f"holdout Sharpe={results[cross.best_name].holdout_metrics['sharpe']:+.3f}.\n"
        )

    console.print(f"[green]Report[/green] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
