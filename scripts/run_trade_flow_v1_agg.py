"""Trade-flow v1 — Mode A runner (aggressor-signed flow on Binance spot aggTrades).

Pre-registration: docs/research/intake/2026-05-29-trade-flow-v1.md
(InformationSource.microstructure_tick). Midprice = trade price; spread is a
MODELED constant half-spread (no quotes in aggTrades). research_only; hard-capped
by free_data_research_only.

Usage:
    PYTHONPATH=src uv run python scripts/run_trade_flow_v1_agg.py \\
        --dates 2024-04-01,2024-04-02,2024-04-03 --max-rows-per-day 400000
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.crypto_research.perps.run_support import classification_metrics, write_report
from quant_research_stack.crypto_research.perps.trade_flow import (
    build_trade_flow_features,
    load_aggtrades_day,
    trade_flow_feature_columns,
)
from quant_research_stack.crypto_research.perps.training import (
    PerpWalkForwardConfig,
    train_perp_walk_forward,
)
from quant_research_stack.crypto_research.perps.validation import classify_perp_candidate

console = Console()
_WINDOWS = (10, 50, 200, 1000)
_EXEC_COLUMNS = ["best_bid", "best_ask", "relative_spread", "best_bid_size", "best_ask_size"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_window(symbol: str, dates: list[str], max_rows_per_day: int) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for date in dates:
        norm = load_aggtrades_day(symbol, date, max_rows=max_rows_per_day)  # streams only what's needed
        console.print(f"  [green]{date}[/green] trades={norm.height:,}")
        frames.append(norm)
    return pl.concat(frames, how="vertical").sort(["symbol", "event_time"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade-flow v1 Mode A (aggressor-signed flow on spot aggTrades)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--dates", required=True, help="comma-separated YYYY-MM-DD aggTrades days (chronological)")
    p.add_argument("--max-rows-per-day", type=int, default=400_000)
    p.add_argument("--horizon", type=int, default=20, help="trade-events ahead for the markout label")
    p.add_argument("--half-spread-bps", type=float, default=1.0, help="modeled half-spread (no quotes in aggTrades)")
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    p.add_argument("--min-train-rows", type=int, default=150_000)
    p.add_argument("--max-train-rows-per-fold", type=int, default=300_000)
    p.add_argument("--test-rows", type=int, default=50_000)
    p.add_argument("--max-folds", type=int, default=4)
    p.add_argument("--embargo-rows", type=int, default=200)
    p.add_argument("--out-dir", default="reports/signal_research/microstructure/trade_flow_v1_agg", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        console.print("[red]no dates provided[/red]")
        return 1
    console.print(f"[bold]Loading {len(dates)} day(s)[/bold] for {args.symbol} (spot aggTrades)")
    trades = _load_window(args.symbol, dates, args.max_rows_per_day)
    horizons = tuple(sorted({1, 5, args.horizon}))
    features = build_trade_flow_features(
        trades, horizons=horizons, windows=_WINDOWS, half_spread_bps=args.half_spread_bps
    )
    target = f"future_mid_return_{args.horizon}"
    feat_cols = trade_flow_feature_columns(_WINDOWS)
    train_frame = features.select(["symbol", "event_time", target, *feat_cols])
    console.print(f"[bold]Walk-forward[/bold] rows={train_frame.height:,} target={target} features={len(feat_cols)}")
    wf = train_perp_walk_forward(
        train_frame,
        config=PerpWalkForwardConfig(
            target_column=target, min_train_rows=args.min_train_rows, test_rows=args.test_rows,
            max_folds=args.max_folds, embargo_rows=args.embargo_rows,
            max_train_rows_per_fold=args.max_train_rows_per_fold,
        ),
    )
    if wf.predictions.is_empty():
        console.print("[red]walk-forward produced no OOS predictions (window too small).[/red]")
        return 1

    exec_cols = features.select(
        ["symbol", "event_time", f"future_best_bid_{args.horizon}", f"future_best_ask_{args.horizon}", *_EXEC_COLUMNS]
    )
    joined = wf.predictions.select(
        ["symbol", "event_time", "pred_ridge", "pred_hist_gradient", "pred_ensemble_mean"]
    ).join(exec_cols, on=["symbol", "event_time"], how="left")

    strategy_id = f"trade_flow_v1_aggtrades_{args.symbol.lower()}_h{args.horizon}"
    metrics, diagnostics = classification_metrics(
        joined, horizon=args.horizon, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps, strategy_id=strategy_id
    )
    classification = classify_perp_candidate(metrics)

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    report_path = out_dir / f"trade_flow_v1_agg_{run_id}.md"
    write_report(
        report_path,
        title="Trade-Flow v1 (Mode A — aggressor-signed flow on Binance spot aggTrades)",
        run_id=run_id,
        header_lines=[
            "**Intake:** `docs/research/intake/2026-05-29-trade-flow-v1.md`",
            f"**Git SHA:** `{_git_sha()}`",
            "**Information source:** `microstructure_tick` (Binance spot aggTrades, aggressor-signed)",
            "**Promotion intent:** `research_only` — hard-capped by `free_data_research_only`.",
        ],
        config_lines=[
            f"- Symbol: `{args.symbol}`  Dates: `{args.dates}`",
            f"- Max trades/day: `{args.max_rows_per_day:,}`  Markout horizon (events): `{args.horizon}`",
            f"- MODELED half-spread: `{args.half_spread_bps}` bps (no quotes in aggTrades; spread is synthesized)",
            f"- Taker cost: fee `{args.fee_bps}` bps/side + slippage `{args.slippage_bps}` bps/side (+ modeled spread)",
            f"- Walk-forward: min_train={args.min_train_rows:,} test={args.test_rows:,} folds={args.max_folds} embargo={args.embargo_rows}",
            f"- Feature rows: `{features.height:,}`  Features: `{len(feat_cols)}` (OFI/momentum/realized-vol/signed-count)",
        ],
        wf_model_metrics=wf.model_metrics, metrics=metrics, diagnostics=diagnostics, classification=classification,
        limitations=[
            "Free Binance public data; research_only ceiling enforced in code.",
            "**Spread is MODELED (constant half-spread), not observed** — aggTrades carry no quotes.",
            "Midprice reconstructed from trade price; no resting-book or queue model.",
            "Trades capped per day (first N per day); not a full-day or multi-month run.",
            "Taker execution model; maker/passive fills not modeled. `not_investment_advice: true`",
        ],
    )
    summary = {
        "run_id": run_id, "git_sha": _git_sha(), "config": vars(args) | {"out_dir": str(args.out_dir)},
        "feature_rows": features.height, "model_metrics": wf.model_metrics,
        "metrics": metrics, "diagnostics": diagnostics, "classification": classification,
    }
    (out_dir / f"trade_flow_v1_agg_{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    console.print(f"[bold]Classification:[/bold] {classification}")
    console.print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
