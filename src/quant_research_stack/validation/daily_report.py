from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from quant_research_stack.validation.hit_rate import HitRateResult
from quant_research_stack.validation.reconcile import ReconcileSummary


@dataclass(frozen=True)
class PerSignalRow:
    signal_id: str
    symbol: str
    predicted_score: float
    confidence: float
    predicted_direction: int
    s2_decision: str
    fill_price: float | None
    horizon_minutes: int
    realized_return: float
    realized_direction: int
    hit: bool | None  # None when no trade was placed
    weight: float
    fill_ts_utc: datetime


@dataclass(frozen=True)
class DailyReportInputs:
    date_str: str
    stage: str
    broker_name: str
    rows: list[PerSignalRow]
    hit_rate: HitRateResult
    reconcile: ReconcileSummary
    daily_pnl_pct: float
    daily_dd_pct: float
    sharpe_rolling: float
    days_in_paper: int
    min_trading_days: int
    thresholds: dict[str, float] = field(default_factory=dict)


def _gate_mark(value: float, threshold: float, direction: str = "min") -> str:
    if direction == "min":
        return "✅" if value >= threshold else "❌"
    return "✅" if value <= threshold else "❌"


def render_markdown(inp: DailyReportInputs) -> str:
    n_pass = sum(1 for r in inp.rows if r.s2_decision == "pass")
    n_veto = sum(1 for r in inp.rows if r.s2_decision == "veto")
    n_ie = sum(1 for r in inp.rows if r.s2_decision == "insufficient_evidence")
    n_trades = sum(1 for r in inp.rows if r.fill_price is not None)

    lines = [
        f"# QuantLab paper validation — {inp.date_str}",
        "",
        f"Stage: {inp.stage} · Broker: {inp.broker_name} · TV chart account: "
        "Alpaca paper (operator-connected)",
        "",
        "## Headline",
        f"- n_signals: {len(inp.rows)}   (passed-S2: {n_pass} · vetoed: {n_veto} · "
        f"insufficient_evidence: {n_ie})",
        f"- n_trades: {n_trades}",
        f"- hit_rate (weighted): {inp.hit_rate.hit_rate:.3f}",
        f"- daily_pnl_pct: {inp.daily_pnl_pct:+.2f}",
        f"- daily_dd_pct: {inp.daily_dd_pct:.2f}",
        f"- Sharpe (rolling {inp.min_trading_days}d): {inp.sharpe_rolling:.2f}",
        f"- governor_block_rate: {inp.hit_rate.governor_block_rate:.2f}",
        "",
        "## Per-signal table",
        "| signal_id | symbol | predicted_score | confidence | s2_decision | "
        "fill_price | horizon_min | realized_return | hit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in inp.rows:
        fp = "—" if r.fill_price is None else f"{r.fill_price:.4f}"
        rr = "—" if math.isnan(r.realized_return) else f"{r.realized_return:+.4f}"
        hit_mark = "—" if r.hit is None else ("✅" if r.hit else "❌")
        lines.append(
            f"| {r.signal_id} | {r.symbol} | {r.predicted_score:+.4f} | "
            f"{r.confidence:.2f} | {r.s2_decision} | {fp} | {r.horizon_minutes} | "
            f"{rr} | {hit_mark} |"
        )

    flag = "⚠" if inp.reconcile.flagged else ""
    lines += [
        "",
        "## Position-book reconciliation",
        f"QuantLab book equity:    {inp.reconcile.book_equity}",
        f"Alpaca paper equity:     {inp.reconcile.broker_equity}",
        f"Diff bps:                {inp.reconcile.diff_bps:.2f} {flag}".rstrip(),
        "",
        "## TV chart cross-check (operator-filled)",
        "- [ ] I reviewed today's trades on the TV chart with Alpaca connected.",
        "- [ ] Any signal looked obviously wrong on the chart (please annotate):",
        "- Operator initials + date:",
        "",
        "## Promotion gate status (informational)",
    ]

    if "hit_rate_min" in inp.thresholds:
        t = inp.thresholds["hit_rate_min"]
        lines.append(
            f"- hit_rate_min ({t}):                 "
            f"{_gate_mark(inp.hit_rate.hit_rate, t, 'min')} {inp.hit_rate.hit_rate:.3f}"
        )
    if "sharpe_min" in inp.thresholds:
        t = inp.thresholds["sharpe_min"]
        lines.append(
            f"- sharpe_min ({t} rolling):            "
            f"{_gate_mark(inp.sharpe_rolling, t, 'min')} {inp.sharpe_rolling:.2f}"
        )
    if "max_daily_dd_pct" in inp.thresholds:
        t = inp.thresholds["max_daily_dd_pct"]
        lines.append(
            f"- max_daily_dd ({t}):                 "
            f"{_gate_mark(inp.daily_dd_pct, t, 'max')} {inp.daily_dd_pct:.2f}"
        )
    if "governor_block_rate_max" in inp.thresholds:
        t = inp.thresholds["governor_block_rate_max"]
        lines.append(
            f"- governor_block_rate_max ({t}):      "
            f"{_gate_mark(inp.hit_rate.governor_block_rate, t, 'max')} "
            f"{inp.hit_rate.governor_block_rate:.2f}"
        )
    if inp.days_in_paper >= inp.min_trading_days:
        days_mark = "✅"
    elif inp.days_in_paper > 0:
        days_mark = "🟡"
    else:
        days_mark = "❌"
    lines.append(
        f"- min_trading_days ({inp.min_trading_days}):               "
        f"{days_mark} {inp.days_in_paper} of {inp.min_trading_days}"
    )

    return "\n".join(lines) + "\n"


def build_per_signal_table(rows: list[PerSignalRow]) -> pl.DataFrame:
    return pl.DataFrame({
        "signal_id": [r.signal_id for r in rows],
        "symbol": [r.symbol for r in rows],
        "predicted_score": [float(r.predicted_score) for r in rows],
        "confidence": [float(r.confidence) for r in rows],
        "predicted_dir": [int(r.predicted_direction) for r in rows],
        "s2_decision": [r.s2_decision for r in rows],
        "fill_price": [r.fill_price for r in rows],
        "horizon_minutes": [int(r.horizon_minutes) for r in rows],
        "realized_return": [float(r.realized_return) for r in rows],
        "realized_dir": [int(r.realized_direction) for r in rows],
        "hit": [r.hit for r in rows],
        "weight": [float(r.weight) for r in rows],
        "fill_ts_utc": [r.fill_ts_utc for r in rows],
    })
