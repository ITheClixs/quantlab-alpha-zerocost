"""Shared run-support for trade-flow validation runners (Mode A and Mode B).

Both modes end up with a ``joined`` frame keyed by (symbol, event_time) carrying
walk-forward OOS predictions (``pred_ridge``/``pred_hist_gradient``/
``pred_ensemble_mean``) plus the execution columns the event backtest needs
(``best_bid``/``best_ask``/``future_best_bid_{h}``/``future_best_ask_{h}``/
``relative_spread``/sizes). Everything here operates on that contract, so the
two data loaders (L1 bookTicker quotes vs aggressor-signed trade flow) can share
the cost-aware threshold sweep, validation gate, and report shape.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_research_stack.crypto_research.perps.backtest import PerpBacktestConfig, run_event_backtest
from quant_research_stack.crypto_research.perps.validation import (
    bootstrap_sharpe_payload,
    concentration_payload,
    deflated_sharpe_payload,
    estimate_registry_pbo,
)

MODELS = ("ridge", "hist_gradient", "ensemble_mean")
# Only trade when |predicted edge| >= k * round-trip cost. None = trade every event.
K_SWEEP: tuple[float | None, ...] = (None, 1.0, 1.5, 2.0, 3.0, 4.0)


def daily_sharpe(trades: pl.DataFrame) -> float:
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


def variant_backtest(
    joined: pl.DataFrame, *, prediction_column: str, horizon: int, fee_bps: float,
    slippage_bps: float, k: float | None = None, cost_multiplier: float = 1.0, latency_events: int = 0,
) -> Any:
    return run_event_backtest(
        joined,
        config=PerpBacktestConfig(
            prediction_column=prediction_column,
            horizon=horizon,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_edge_to_cost_ratio=k,
            cost_multiplier=cost_multiplier,
            latency_events=latency_events,
        ),
    )


def pbo_frame(joined: pl.DataFrame, *, horizon: int, fee_bps: float, slippage_bps: float) -> pl.DataFrame:
    merged: pl.DataFrame | None = None
    for model in MODELS:
        col_name = f"pred_{model}"
        if col_name not in joined.columns:
            continue
        trades = variant_backtest(
            joined, prediction_column=col_name, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps
        ).trades
        if trades.is_empty():
            continue
        col = trades.select(["event_time", pl.col("net_return").alias(model)])
        merged = col if merged is None else merged.join(col, on="event_time", how="full", coalesce=True)
    if merged is None:
        return pl.DataFrame()
    return merged.sort("event_time").fill_null(0.0).with_row_index("event_index")


def classification_metrics(
    joined: pl.DataFrame, *, horizon: int, fee_bps: float, slippage_bps: float, strategy_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Edge-over-cost threshold sweep on the ensemble, then gate the best variant."""
    sweep: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for k in K_SWEEP:
        res = variant_backtest(joined, prediction_column="pred_ensemble_mean", horizon=horizon,
                               fee_bps=fee_bps, slippage_bps=slippage_bps, k=k)
        results["none" if k is None else f"{k}"] = res
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
    cost_2x = variant_backtest(joined, prediction_column="pred_ensemble_mean", horizon=horizon,
                               fee_bps=fee_bps, slippage_bps=slippage_bps, k=best_k, cost_multiplier=2.0)
    delay_1 = variant_backtest(joined, prediction_column="pred_ensemble_mean", horizon=horizon,
                               fee_bps=fee_bps, slippage_bps=slippage_bps, k=best_k, latency_events=1)
    best_net = best_trades["net_return"] if not best_trades.is_empty() else pl.Series("net_return", [], dtype=pl.Float64)

    frame = pbo_frame(joined, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps)
    present = [m for m in MODELS if m in frame.columns]
    pbo = (
        estimate_registry_pbo(frame, strategy_columns=present)
        if frame.height > 0 and len(present) >= 2
        else {"status": "not_estimated", "pbo_probability": None}
    )
    boot = bootstrap_sharpe_payload(best_net)
    dsr = deflated_sharpe_payload(best_net, trials=len(K_SWEEP))
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
        "net_daily_sharpe": daily_sharpe(best_trades),
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


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.4f}%"
    except Exception:
        return str(value)


def write_report(
    path: Path, *, title: str, run_id: str, header_lines: list[str], config_lines: list[str],
    wf_model_metrics: dict, metrics: dict, diagnostics: dict, classification: dict, limitations: list[str],
) -> None:
    lines = [f"# {title} `{run_id}`", "", *header_lines, "", "## Configuration", *config_lines, "",
             "## Walk-forward OOS model accuracy", "",
             "| model | folds | rows | mean IC | mean zero-mean R2 | mean directional acc. |",
             "|---|---:|---:|---:|---:|---:|"]
    for model, m in wf_model_metrics.items():
        lines.append(
            f"| `{model}` | {m.get('folds', 0)} | {int(m.get('rows', 0)):,} | {m.get('mean_ic', 0.0):.5f} | "
            f"{m.get('mean_zero_mean_r2', 0.0):.5f} | {_pct(m.get('mean_directional_accuracy', 0.0))} |"
        )
    lines += ["", "## Edge-over-cost threshold sweep (ensemble_mean)", "",
              "Trade only when `|predicted edge| >= k * round-trip cost`. `k=none` trades every event.", "",
              "| k | trades | gross total | net total | trade Sharpe | net hit rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in diagnostics.get("threshold_sweep", []):
        lines.append(
            f"| `{row['k']}` | {int(row['trade_count']):,} | {_pct(row['gross_total_return'])} | "
            f"{_pct(row['net_total_return'])} | {row['trade_sharpe']} | {_pct(row['net_hit_rate'])} |"
        )
    base_m = diagnostics["base_metrics"]
    lines += ["", f"## Best cost-aware variant (k=`{diagnostics.get('best_k')}`)", "",
              f"- Trades: `{base_m.get('trade_count', 0):,}`",
              f"- Gross total return: `{_pct(base_m.get('gross_total_return'))}`",
              f"- **Net total return (taker): `{_pct(base_m.get('net_total_return'))}`**",
              f"- Net total return @ 2x cost: `{_pct(diagnostics['cost_2x_metrics'].get('net_total_return'))}`",
              f"- Net total return @ 1-event delay: `{_pct(diagnostics['delay_1_metrics'].get('net_total_return'))}`",
              f"- Gross hit rate: `{_pct(base_m.get('gross_hit_rate'))}`  Net hit rate: `{_pct(base_m.get('net_hit_rate'))}`",
              f"- Net daily Sharpe: `{metrics['net_daily_sharpe']:.4f}`",
              f"- Max drawdown: `{_pct(base_m.get('max_drawdown'))}`",
              "", "## Validation gate", "",
              f"- PBO probability: `{metrics['pbo_probability']}`",
              f"- Bootstrap Sharpe CI lower (95%): `{metrics['bootstrap_ci_lower_95']}`",
              f"- Deflated-Sharpe probability: `{metrics['deflated_sharpe_probability']}`",
              f"- Concentration blocker: `{metrics['concentration_blocker']}`",
              "", "## Classification", "",
              f"- strategy_id: `{classification['strategy_id']}`",
              f"- research_candidate: `{classification['research_candidate']}`",
              f"- promotion_eligible: `{classification['promotion_eligible']}` (hard-capped)",
              f"- production_candidate: `{classification['production_candidate']}`",
              f"- blockers: `{', '.join(classification['blockers'])}`",
              "", "## Limitations", *[f"- {item}" for item in limitations]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
