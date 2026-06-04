"""CLI runner for the ~100k-strategy backtest-overfitting demonstration.

Demonstrates how selecting the "best" strategy from a large search grid
produces inflated in-sample Sharpe ratios that do not survive OOS.

Usage
-----
    PYTHONPATH=src uv run python scripts/run_strategy_zoo_overfitting.py \\
        --max-strategies 10000 \\
        --out reports/signal_research/strategy_zoo_overfitting_v1/run_10k

Smoke test (fast):
    PYTHONPATH=src uv run python scripts/run_strategy_zoo_overfitting.py \\
        --max-strategies 1000 --perm-max-strategies 500 --perm-n 3 \\
        --out reports/signal_research/strategy_zoo_overfitting_v1/smoke_1k
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return dt.date.today().isoformat()


def _load_or_fetch_panel(
    *,
    start: dt.date,
    end: dt.date,
    cache_dir: Path,
) -> dict[str, pl.DataFrame]:
    """Load cached parquet bars or download via yfinance.

    Returns bars per universe name (not per ticker).
    Missing tickers are fetched; already-cached ones are read from disk.
    """
    from quant_research_stack.strategy_benchmark.data import (
        UNIVERSES,
        build_universe_returns,
        fetch_daily_bars,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique tickers
    tickers = sorted({t for u in UNIVERSES for t in u.tickers})

    bars_by_ticker: dict[str, pl.DataFrame] = {}
    for ticker in tickers:
        safe = ticker.replace("=", "_").replace("^", "")
        cached = cache_dir / f"{safe}.parquet"
        if cached.exists():
            df = pl.read_parquet(cached)
            # Check that the cached range covers what we need
            if df.height > 0:
                cached_start = df["date"].min()
                cached_end = df["date"].max()
                if cached_start <= start and cached_end >= end - dt.timedelta(days=5):
                    bars_by_ticker[ticker] = df
                    print(f"  cache hit: {ticker} ({df.height} rows)")
                    continue
        print(f"  fetching:  {ticker} ...")
        bars = fetch_daily_bars(ticker=ticker, start=start, end=end)
        bars.write_parquet(cached)
        bars_by_ticker[ticker] = bars

    # Build per-universe series
    universes_out: dict[str, pl.DataFrame] = {}
    for u in UNIVERSES:
        universes_out[u.name] = build_universe_returns(
            universe=u, bars_by_ticker=bars_by_ticker
        )

    return universes_out


def _dedup_tiers(tiers_raw: tuple[int, ...], n_actual: int) -> tuple[int, ...]:
    """Remove duplicates and clip to n_actual; always include n_actual."""
    seen: set[int] = set()
    result: list[int] = []
    for t in sorted(tiers_raw):
        v = min(t, n_actual)
        if v not in seen:
            seen.add(v)
            result.append(v)
    if n_actual not in seen:
        result.append(n_actual)
    return tuple(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    today = _today()
    p = argparse.ArgumentParser(
        description="Strategy-zoo overfitting demonstration: IS vs OOS Sharpe."
    )
    p.add_argument("--start", default="2015-01-01", help="Bar start date (YYYY-MM-DD)")
    p.add_argument("--end", default=today, help="Bar end date (YYYY-MM-DD)")
    p.add_argument(
        "--max-strategies", type=int, default=10_000,
        help="Number of strategies to sample from the full grid (headline tier).",
    )
    p.add_argument("--oos-fraction", type=float, default=0.3, help="OOS fraction (0–1)")
    p.add_argument("--embargo-days", type=int, default=10, help="Purge+embargo gap (days)")
    p.add_argument(
        "--perm-max-strategies", type=int, default=2_000,
        help="Max strategies for permutation grid (kept smaller for speed).",
    )
    p.add_argument("--perm-n", type=int, default=5, help="Number of permutations")
    p.add_argument(
        "--out", default="reports/signal_research/strategy_zoo_overfitting_v1",
        help="Output directory for parquet + JSON artefacts.",
    )
    return p.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    cache_dir = Path("data/processed/strategy_zoo_overfitting_v1")

    print("=" * 70)
    print("Strategy Zoo Overfitting Demonstration")
    print(f"  start={args.start}  end={args.end}")
    print(f"  max_strategies={args.max_strategies}")
    print(f"  oos_fraction={args.oos_fraction}  embargo_days={args.embargo_days}")
    print(f"  perm_max_strategies={args.perm_max_strategies}  perm_n={args.perm_n}")
    print(f"  out_dir={out_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1 — Fetch / cache bars for all 11 universes
    # ------------------------------------------------------------------
    print("\n[1/5] Loading bars for all 11 universes ...")
    try:
        universes = _load_or_fetch_panel(start=start, end=end, cache_dir=cache_dir)
    except Exception as exc:
        print(f"ERROR: data fetch failed: {exc}", file=sys.stderr)
        print("DEFERRED — network unavailable or yfinance error. Run aborted.", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(universes)} universes.")
    for name, df in sorted(universes.items()):
        print(f"    {name}: {df.height} rows, {df['date'].min()} .. {df['date'].max()}")

    # ------------------------------------------------------------------
    # Step 2 — Main zoo run
    # ------------------------------------------------------------------
    from quant_research_stack.strategy_benchmark.zoo.grid import GridConfig
    from quant_research_stack.strategy_benchmark.zoo.runner import run_zoo

    print(f"\n[2/5] Running zoo (max_strategies={args.max_strategies}) ...")
    grid = GridConfig(max_strategies=args.max_strategies)
    res = run_zoo(
        universes=universes,
        grid=grid,
        oos_fraction=args.oos_fraction,
        embargo_days=args.embargo_days,
    )
    n_actual = len(res.specs)
    print(f"  Ran {n_actual} strategies in {res.wall_clock_sec:.1f}s")

    # ------------------------------------------------------------------
    # Step 3 — Analysis
    # ------------------------------------------------------------------
    from quant_research_stack.strategy_benchmark.zoo.analysis import (
        deflate_best,
        expected_vs_empirical,
    )

    print("\n[3/5] Computing expected-vs-empirical and DSR ...")
    raw_tiers = (1_000, 10_000, args.max_strategies)
    tiers = _dedup_tiers(raw_tiers, n_actual)
    sharpe_arr = res.metrics["is_sharpe"].to_numpy()
    tier_rows = expected_vs_empirical(sharpe_estimates=sharpe_arr, tiers=tiers)
    deflated = deflate_best(is_returns=res.is_returns.astype(float))

    # ------------------------------------------------------------------
    # Step 4 — Permutation control (smaller grid)
    # ------------------------------------------------------------------
    from quant_research_stack.strategy_benchmark.zoo.permutation import permutation_control

    print(
        f"\n[4/5] Permutation control "
        f"(max_strategies={args.perm_max_strategies}, n_perm={args.perm_n}) ..."
    )
    pgrid = GridConfig(max_strategies=args.perm_max_strategies)
    perm = permutation_control(
        universes=universes, grid=pgrid, n_permutations=args.perm_n, seed=42
    )
    print(f"  real best IS Sharpe: {perm['real_best_sharpe']:.4f}")
    print(f"  permuted mean:       {perm['permuted_best_sharpe_mean']:.4f}")
    print(f"  p-value:             {perm['p_value']:.4f}")

    # ------------------------------------------------------------------
    # Step 5 — Write artefacts
    # ------------------------------------------------------------------
    print(f"\n[5/5] Writing artefacts to {out_dir} ...")

    # metrics.parquet — full zoo results
    res.metrics.write_parquet(out_dir / "metrics.parquet")

    # tiers.json
    (out_dir / "tiers.json").write_text(
        json.dumps(tier_rows, indent=2), encoding="utf-8"
    )

    # deflated_best.json
    (out_dir / "deflated_best.json").write_text(
        json.dumps(deflated, indent=2), encoding="utf-8"
    )

    # permutation_control.json
    (out_dir / "permutation_control.json").write_text(
        json.dumps(perm, indent=2), encoding="utf-8"
    )

    # oos_decay.parquet — top-200 by IS Sharpe
    top200 = (
        res.metrics
        .sort("is_sharpe", descending=True)
        .head(200)
        .select(["strategy_id", "is_sharpe", "oos_sharpe"])
    )
    top200.write_parquet(out_dir / "oos_decay.parquet")

    # DSR pass count: strategies whose IS Sharpe > theoretical_max at the full-N tier
    # (i.e. strategies that would individually "pass" a deflated bar)
    theo_max_full_n = next(
        (row["theoretical_max"] for row in tier_rows if row["n_trials"] == n_actual),
        None,
    )
    if theo_max_full_n is None:
        # fall back to the last tier row
        theo_max_full_n = tier_rows[-1]["theoretical_max"]

    dsr_pass_count = int(np.sum(sharpe_arr > theo_max_full_n))

    # summary.json
    summary: dict = {
        "n_run": n_actual,
        "wall_clock_sec": round(res.wall_clock_sec, 2),
        "pbo": res.pbo,
        "best_is_sharpe": float(np.max(sharpe_arr)),
        "median_is_sharpe": float(np.median(sharpe_arr)),
        "best_oos_sharpe": float(res.metrics["oos_sharpe"].max()),
        "median_oos_sharpe": float(res.metrics["oos_sharpe"].median()),
        "dsr": deflated,
        "tiers": tier_rows,
        "permutation": perm,
        "dsr_pass_count": dsr_pass_count,
        "dsr_pass_threshold": theo_max_full_n,
        "args": {
            "start": args.start,
            "end": args.end,
            "max_strategies": args.max_strategies,
            "oos_fraction": args.oos_fraction,
            "embargo_days": args.embargo_days,
            "perm_max_strategies": args.perm_max_strategies,
            "perm_n": args.perm_n,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Strategies run:          {n_actual}")
    print(f"  Wall-clock:              {res.wall_clock_sec:.1f}s")
    print(f"  PBO probability:         {res.pbo['pbo_probability']:.4f}")
    print(f"  Best IS Sharpe:          {np.max(sharpe_arr):.4f}")
    print(f"  Best OOS Sharpe:         {float(res.metrics['oos_sharpe'].max()):.4f}")
    print(f"  DSR (best strat):        {deflated['dsr']:.4f}")
    print(f"  DSR pass count:          {dsr_pass_count} / {n_actual}")
    print(f"  Perm p-value:            {perm['p_value']:.4f}")
    print("\nExpected vs Empirical Sharpe by tier:")
    for row in tier_rows:
        print(
            f"    N={row['n_trials']:>6}: empirical={row['empirical_max']:.4f}  "
            f"theoretical={row['theoretical_max']:.4f}"
        )
    print(f"\nArtefacts written to: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
