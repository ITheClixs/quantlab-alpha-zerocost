"""Trade-flow v1 — Mode B runner (L1-quote microstructure on Binance bookTicker).

Pre-registration: docs/research/intake/2026-05-29-trade-flow-v1.md (Mode B
addendum: source = Binance USDT-M futures bookTicker, InformationSource.
microstructure_book; free, downloadable).

Pipeline: download bounded daily bookTicker -> normalize (last quote per ms) ->
build_l1_features -> perps walk-forward (purged OOS folds) -> cost-aware event
backtest (taker baseline + 2x cost stress + 1-event delay stress) ->
validation (PBO/bootstrap/DSR/concentration) -> classify_perp_candidate ->
report. research_only; hard-capped by free_data_research_only.

Usage:
    PYTHONPATH=src uv run python scripts/run_trade_flow_v1.py \\
        --dates 2024-04-01,2024-04-02,2024-04-03,2024-04-04 \\
        --max-rows-per-day 150000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console

from quant_research_stack.crypto_research.perps.backtest import PerpBacktestConfig, run_event_backtest
from quant_research_stack.crypto_research.perps.binance_bookticker import load_bookticker_day
from quant_research_stack.crypto_research.perps.features import build_l1_features
from quant_research_stack.crypto_research.perps.training import (
    PerpWalkForwardConfig,
    train_perp_walk_forward,
)
from quant_research_stack.crypto_research.perps.validation import (
    bootstrap_sharpe_payload,
    classify_perp_candidate,
    concentration_payload,
    deflated_sharpe_payload,
    estimate_registry_pbo,
)

console = Console()
_MODELS = ("ridge", "hist_gradient", "ensemble_mean")
_EXEC_COLUMNS = ["best_bid", "best_ask", "relative_spread", "best_bid_size", "best_ask_size"]
# Only trade when |predicted edge| >= k * round-trip cost. None = trade every event
# (no threshold) — kept as a baseline to show why trading every tick is untradable.
_K_SWEEP: tuple[float | None, ...] = (None, 1.0, 1.5, 2.0, 3.0, 4.0)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _daily_sharpe(trades: pl.DataFrame) -> float:
    if trades.is_empty():
        return 0.0
    daily = (
        trades.with_columns(pl.col("event_time").dt.date().alias("d"))
        .group_by("d")
        .agg(pl.col("net_return").sum().alias("r"))
        .sort("d")
    )
    r = daily["r"].to_numpy().astype(np.float64)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * sqrt(365.0)) if sd > 0.0 else 0.0


def _load_window(symbol: str, dates: list[str], max_rows_per_day: int) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for date in dates:
        # Streams only the bytes needed for max_rows from Binance (no full-file download).
        norm = load_bookticker_day(symbol, date, max_rows=max_rows_per_day)
        console.print(f"  [green]{date}[/green] rows={norm.height:,}")
        frames.append(norm)
    frame = pl.concat(frames, how="vertical")
    # Last quote per millisecond -> unique (symbol, event_time) key for clean join-back.
    return frame.unique(subset=["symbol", "event_time"], keep="last").sort(["symbol", "event_time"])


def _backtest(joined: pl.DataFrame, model: str, *, horizon: int, fee_bps: float, slippage_bps: float,
              cost_multiplier: float = 1.0, latency_events: int = 0) -> Any:
    return run_event_backtest(
        joined,
        config=PerpBacktestConfig(
            prediction_column=f"pred_{model}",
            horizon=horizon,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            cost_multiplier=cost_multiplier,
            latency_events=latency_events,
        ),
    )


def _pbo_frame(joined: pl.DataFrame, *, horizon: int, fee_bps: float, slippage_bps: float) -> pl.DataFrame:
    merged: pl.DataFrame | None = None
    for model in _MODELS:
        trades = _backtest(joined, model, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps).trades
        if trades.is_empty():
            continue
        col = trades.select(["event_time", pl.col("net_return").alias(model)])
        merged = col if merged is None else merged.join(col, on="event_time", how="full", coalesce=True)
    if merged is None:
        return pl.DataFrame()
    return merged.sort("event_time").fill_null(0.0).with_row_index("event_index")


def _variant(joined: pl.DataFrame, *, horizon: int, fee_bps: float, slippage_bps: float,
             k: float | None, cost_multiplier: float = 1.0, latency_events: int = 0) -> Any:
    return run_event_backtest(
        joined,
        config=PerpBacktestConfig(
            prediction_column="pred_ensemble_mean",
            horizon=horizon,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_edge_to_cost_ratio=k,
            cost_multiplier=cost_multiplier,
            latency_events=latency_events,
        ),
    )


def _classification_metrics(
    joined: pl.DataFrame, *, horizon: int, fee_bps: float, slippage_bps: float, strategy_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Sweep the edge-over-cost threshold so a real strategy (trade only when the
    # predicted edge clears cost) gets a fair test, not "trade every tick".
    sweep: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for k in _K_SWEEP:
        res = _variant(joined, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps, k=k)
        key = "none" if k is None else f"{k}"
        results[key] = res
        sweep.append({"k": k, "trade_count": int(res.trades.height),
                      "net_total_return": res.metrics.get("net_total_return"),
                      "gross_total_return": res.metrics.get("gross_total_return"),
                      "trade_sharpe": res.metrics.get("trade_sharpe"),
                      "net_hit_rate": res.metrics.get("net_hit_rate")})
    traded = [row for row in sweep if row["trade_count"] > 0]
    pool = traded or sweep
    best_row = max(pool, key=lambda r: (float(r["net_total_return"] or -1e9), float(r["trade_sharpe"] or -1e9)))
    best_k = best_row["k"]
    best = results["none" if best_k is None else f"{best_k}"]
    best_trades = best.trades
    cost_2x = _variant(joined, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps, k=best_k, cost_multiplier=2.0)
    delay_1 = _variant(joined, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps, k=best_k, latency_events=1)
    best_net = best_trades["net_return"] if not best_trades.is_empty() else pl.Series("net_return", [], dtype=pl.Float64)

    pbo_frame = _pbo_frame(joined, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps)
    present = [m for m in _MODELS if m in pbo_frame.columns]
    pbo = (
        estimate_registry_pbo(pbo_frame, strategy_columns=present)
        if pbo_frame.height > 0 and len(present) >= 2
        else {"status": "not_estimated", "pbo_probability": None}
    )
    boot = bootstrap_sharpe_payload(best_net)
    dsr = deflated_sharpe_payload(best_net, trials=len(_K_SWEEP))
    conc = (
        concentration_payload(
            best_trades.with_columns(pl.col("event_time").dt.date().alias("event_date")),
            event_time_column="event_date",
        )
        if not best_trades.is_empty()
        else {"concentration_blocker": False}
    )

    metrics = {
        "strategy_id": strategy_id,
        "name": strategy_id,
        "pbo_probability": pbo.get("pbo_probability"),
        "bootstrap_ci_lower_95": boot.get("ci_lower_95"),
        "net_daily_sharpe": _daily_sharpe(best_trades),
        "net_total_return": best.metrics.get("net_total_return"),
        "cost_2x_net_total_return": cost_2x.metrics.get("net_total_return"),
        "delay_1_event_net_total_return": delay_1.metrics.get("net_total_return"),
        "deflated_sharpe_probability": dsr.get("probability"),
        "concentration_blocker": conc.get("concentration_blocker", False),
    }
    diagnostics = {
        "best_k": best_k,
        "threshold_sweep": sweep,
        "base_metrics": best.metrics,
        "cost_2x_metrics": cost_2x.metrics,
        "delay_1_metrics": delay_1.metrics,
        "pbo": pbo,
        "bootstrap": boot,
        "deflated_sharpe": dsr,
        "concentration": conc,
        "trade_count": int(best_trades.height),
    }
    return metrics, diagnostics


def _write_report(path: Path, *, run_id: str, args: argparse.Namespace, git_sha: str,
                  feature_rows: int, wf_model_metrics: dict, metrics: dict, diagnostics: dict,
                  classification: dict) -> None:
    def pct(value: Any) -> str:
        try:
            return f"{float(value) * 100.0:.4f}%"
        except Exception:
            return str(value)

    lines = [
        f"# Trade-Flow v1 (Mode B — L1 quotes on Binance bookTicker) `{run_id}`",
        "",
        "**Intake:** `docs/research/intake/2026-05-29-trade-flow-v1.md` (Mode B addendum)",
        f"**Git SHA:** `{git_sha}`",
        "**Information source:** `microstructure_book` (Binance USDT-M futures bookTicker)",
        "**Promotion intent:** `research_only` — hard-capped by `free_data_research_only`.",
        "",
        "## Configuration",
        f"- Symbol: `{args.symbol}`",
        f"- Dates: `{args.dates}`",
        f"- Max rows/day: `{args.max_rows_per_day:,}`",
        f"- Horizon (events): `{args.horizon}`",
        f"- Taker cost: fee `{args.fee_bps}` bps/side + slippage `{args.slippage_bps}` bps/side (+ spread via entry/exit)",
        f"- Walk-forward: min_train_rows=`{args.min_train_rows:,}` test_rows=`{args.test_rows:,}` "
        f"max_folds=`{args.max_folds}` embargo_rows=`{args.embargo_rows}`",
        f"- Feature rows: `{feature_rows:,}`",
        "",
        "## Walk-forward OOS model accuracy",
        "",
        "| model | folds | rows | mean IC | mean zero-mean R2 | mean directional acc. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, m in wf_model_metrics.items():
        lines.append(
            f"| `{model}` | {m.get('folds', 0)} | {int(m.get('rows', 0)):,} | "
            f"{m.get('mean_ic', 0.0):.5f} | {m.get('mean_zero_mean_r2', 0.0):.5f} | "
            f"{pct(m.get('mean_directional_accuracy', 0.0))} |"
        )
    lines += [
        "",
        "## Edge-over-cost threshold sweep (ensemble_mean)",
        "",
        "Trade only when `|predicted edge| >= k * round-trip cost`. `k=none` trades every event.",
        "",
        "| k | trades | gross total | net total | trade Sharpe | net hit rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics.get("threshold_sweep", []):
        lines.append(
            f"| `{row['k']}` | {int(row['trade_count']):,} | {pct(row['gross_total_return'])} | "
            f"{pct(row['net_total_return'])} | {row['trade_sharpe']} | {pct(row['net_hit_rate'])} |"
        )
    base_m = diagnostics["base_metrics"]
    lines += [
        "",
        f"## Best cost-aware variant (k=`{diagnostics.get('best_k')}`)",
        "",
        f"- Trades: `{base_m.get('trade_count', 0):,}`",
        f"- Gross total return: `{pct(base_m.get('gross_total_return'))}`",
        f"- **Net total return (taker): `{pct(base_m.get('net_total_return'))}`**",
        f"- Net total return @ 2x cost: `{pct(diagnostics['cost_2x_metrics'].get('net_total_return'))}`",
        f"- Net total return @ 1-event delay: `{pct(diagnostics['delay_1_metrics'].get('net_total_return'))}`",
        f"- Gross hit rate: `{pct(base_m.get('gross_hit_rate'))}`  Net hit rate: `{pct(base_m.get('net_hit_rate'))}`",
        f"- Net daily Sharpe: `{metrics['net_daily_sharpe']:.4f}`",
        f"- Max drawdown: `{pct(base_m.get('max_drawdown'))}`",
        "",
        "## Validation gate",
        "",
        f"- PBO probability: `{metrics['pbo_probability']}`",
        f"- Bootstrap Sharpe CI lower (95%): `{metrics['bootstrap_ci_lower_95']}`",
        f"- Deflated-Sharpe probability: `{metrics['deflated_sharpe_probability']}`",
        f"- Concentration blocker: `{metrics['concentration_blocker']}`",
        "",
        "## Classification",
        "",
        f"- strategy_id: `{classification['strategy_id']}`",
        f"- research_candidate: `{classification['research_candidate']}`",
        f"- promotion_eligible: `{classification['promotion_eligible']}` (hard-capped)",
        f"- production_candidate: `{classification['production_candidate']}`",
        f"- blockers: `{', '.join(classification['blockers'])}`",
        "",
        "## Limitations",
        "- Free Binance public data; research_only ceiling enforced in code.",
        "- bookTicker is L1 top-of-book; no full-depth book reconstruction, no queue position.",
        "- Rows capped per day (first N per day); not a full-day or multi-month run.",
        "- Taker execution model; maker/passive fills not modeled. `not_investment_advice: true`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade-flow v1 Mode B (L1 quotes on Binance bookTicker)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--dates", required=True, help="comma-separated YYYY-MM-DD bookTicker days (chronological)")
    p.add_argument("--max-rows-per-day", type=int, default=150_000)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    p.add_argument("--min-train-rows", type=int, default=60_000)
    p.add_argument("--test-rows", type=int, default=20_000)
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
    free_gb = shutil.disk_usage(".").free / 1e9
    console.print(f"[bold]Disk free:[/bold] {free_gb:.1f} GB  |  loading {len(dates)} day(s) for {args.symbol}")
    if free_gb < 5.0:
        console.print("[red]Less than 5 GB free — aborting download (CLAUDE.md §14).[/red]")
        return 1

    frame = _load_window(args.symbol, dates, args.max_rows_per_day)
    horizons = tuple(sorted({1, 5, 15, 60, args.horizon}))
    features = build_l1_features(frame, horizons=horizons, rolling_windows=(10, 50, 200))
    target = f"future_mid_return_{args.horizon}"
    console.print(f"[bold]Walk-forward[/bold] rows={features.height:,} target={target}")
    wf = train_perp_walk_forward(
        features,
        config=PerpWalkForwardConfig(
            target_column=target,
            min_train_rows=args.min_train_rows,
            test_rows=args.test_rows,
            max_folds=args.max_folds,
            embargo_rows=args.embargo_rows,
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
    metrics, diagnostics = _classification_metrics(
        joined, horizon=args.horizon, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps, strategy_id=strategy_id
    )
    classification = classify_perp_candidate(metrics)

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    report_path = out_dir / f"trade_flow_v1_{run_id}.md"
    _write_report(
        report_path, run_id=run_id, args=args, git_sha=_git_sha(),
        feature_rows=features.height, wf_model_metrics=wf.model_metrics,
        metrics=metrics, diagnostics=diagnostics, classification=classification,
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
