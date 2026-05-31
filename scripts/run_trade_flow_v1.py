"""Trade-flow v1 — Mode B runner (L1-quote microstructure on Binance bookTicker).

Pre-registration: docs/research/intake/2026-05-29-trade-flow-v1.md (Mode B
addendum: source = Binance USDT-M futures bookTicker, InformationSource.
microstructure_book; free, streamed capped per day). research_only; hard-capped
by free_data_research_only.

Usage:
    PYTHONPATH=src uv run python scripts/run_trade_flow_v1.py \\
        --dates 2024-03-24,2024-03-25,2024-03-26 --max-rows-per-day 150000
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.crypto_research.perps.binance_bookticker import load_bookticker_day
from quant_research_stack.crypto_research.perps.features import build_l1_features
from quant_research_stack.crypto_research.perps.run_support import classification_metrics, write_report
from quant_research_stack.crypto_research.perps.training import (
    PerpWalkForwardConfig,
    train_perp_walk_forward,
)
from quant_research_stack.crypto_research.perps.validation import classify_perp_candidate

console = Console()
_EXEC_COLUMNS = ["best_bid", "best_ask", "relative_spread", "best_bid_size", "best_ask_size"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_window(symbol: str, dates: list[str], max_rows_per_day: int) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for date in dates:
        norm = load_bookticker_day(symbol, date, max_rows=max_rows_per_day)  # streams only what's needed
        console.print(f"  [green]{date}[/green] rows={norm.height:,}")
        frames.append(norm)
    frame = pl.concat(frames, how="vertical")
    return frame.unique(subset=["symbol", "event_time"], keep="last").sort(["symbol", "event_time"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade-flow v1 Mode B (L1 quotes on Binance bookTicker)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--dates", required=True, help="comma-separated YYYY-MM-DD bookTicker days (chronological)")
    p.add_argument("--max-rows-per-day", type=int, default=150_000)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    p.add_argument("--min-train-rows", type=int, default=100_000)
    p.add_argument("--test-rows", type=int, default=30_000)
    p.add_argument("--max-folds", type=int, default=4)
    p.add_argument("--embargo-rows", type=int, default=50)
    p.add_argument("--out-dir", default="reports/signal_research/microstructure/trade_flow_v1", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        console.print("[red]no dates provided[/red]")
        return 1
    console.print(f"[bold]Loading {len(dates)} day(s)[/bold] for {args.symbol} (L1 bookTicker)")
    frame = _load_window(args.symbol, dates, args.max_rows_per_day)
    horizons = tuple(sorted({1, 5, 15, 60, args.horizon}))
    features = build_l1_features(frame, horizons=horizons, rolling_windows=(10, 50, 200))
    target = f"future_mid_return_{args.horizon}"
    console.print(f"[bold]Walk-forward[/bold] rows={features.height:,} target={target}")
    wf = train_perp_walk_forward(
        features,
        config=PerpWalkForwardConfig(
            target_column=target, min_train_rows=args.min_train_rows, test_rows=args.test_rows,
            max_folds=args.max_folds, embargo_rows=args.embargo_rows,
        ),
    )
    if wf.predictions.is_empty():
        console.print("[red]walk-forward produced no OOS predictions (window too small).[/red]")
        return 1

    exec_cols = features.unique(subset=["symbol", "event_time"], keep="last").select(
        ["symbol", "event_time", f"future_best_bid_{args.horizon}", f"future_best_ask_{args.horizon}", *_EXEC_COLUMNS]
    )
    joined = wf.predictions.select(
        ["symbol", "event_time", "pred_ridge", "pred_hist_gradient", "pred_ensemble_mean"]
    ).join(exec_cols, on=["symbol", "event_time"], how="left")

    strategy_id = f"trade_flow_v1_bookticker_{args.symbol.lower()}_h{args.horizon}"
    metrics, diagnostics = classification_metrics(
        joined, horizon=args.horizon, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps, strategy_id=strategy_id
    )
    classification = classify_perp_candidate(metrics)

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    report_path = out_dir / f"trade_flow_v1_{run_id}.md"
    write_report(
        report_path,
        title="Trade-Flow v1 (Mode B — L1 quotes on Binance bookTicker)",
        run_id=run_id,
        header_lines=[
            "**Intake:** `docs/research/intake/2026-05-29-trade-flow-v1.md` (Mode B addendum)",
            f"**Git SHA:** `{_git_sha()}`",
            "**Information source:** `microstructure_book` (Binance USDT-M futures bookTicker)",
            "**Promotion intent:** `research_only` — hard-capped by `free_data_research_only`.",
        ],
        config_lines=[
            f"- Symbol: `{args.symbol}`", f"- Dates: `{args.dates}`",
            f"- Max rows/day: `{args.max_rows_per_day:,}`  Horizon (events): `{args.horizon}`",
            f"- Taker cost: fee `{args.fee_bps}` bps/side + slippage `{args.slippage_bps}` bps/side (+ spread via entry/exit)",
            f"- Walk-forward: min_train={args.min_train_rows:,} test={args.test_rows:,} folds={args.max_folds} embargo={args.embargo_rows}",
            f"- Feature rows: `{features.height:,}`",
        ],
        wf_model_metrics=wf.model_metrics, metrics=metrics, diagnostics=diagnostics, classification=classification,
        limitations=[
            "Free Binance public data; research_only ceiling enforced in code.",
            "bookTicker is L1 top-of-book; no full-depth reconstruction, no queue position.",
            "Rows capped per day (first N per day); not a full-day or multi-month run.",
            "Taker execution model; maker/passive fills not modeled. `not_investment_advice: true`",
        ],
    )
    summary = {
        "run_id": run_id, "git_sha": _git_sha(), "config": vars(args) | {"out_dir": str(args.out_dir)},
        "feature_rows": features.height, "model_metrics": wf.model_metrics,
        "metrics": metrics, "diagnostics": diagnostics, "classification": classification,
    }
    (out_dir / f"trade_flow_v1_{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    console.print(f"[bold]Classification:[/bold] {classification}")
    console.print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
