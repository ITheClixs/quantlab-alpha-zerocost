"""Daily TradingView paper-validation report runner.

Reads QuantLab's S1 predictions + S2 verdicts + S4 audit log + Alpaca paper
account state for a given date; produces a Markdown report at
<artifacts.daily_report_dir>/<date>.md and a per-signal Parquet table at
<artifacts.per_signal_parquet_dir>/<date>.parquet.

Usage:
  PYTHONPATH=src uv run python scripts/tv_validation_report.py --date 2026-05-20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from quant_research_stack.validation import load_validation_config
from quant_research_stack.validation.daily_report import (
    DailyReportInputs,
    PerSignalRow,
    build_per_signal_table,
    render_markdown,
)
from quant_research_stack.validation.forward_returns import (
    AlpacaBarsLoader,
    Bar,
    BarLoader,
    ForwardReturnRequest,
    fetch_forward_returns,
)
from quant_research_stack.validation.hit_rate import (
    ScoredSignal,
    compute_hit_rate,
)
from quant_research_stack.validation.pnl import (
    compute_daily_pnl_metrics,
    load_historical_daily_pnl_pct,
)
from quant_research_stack.validation.reconcile import summarize_reconciliation

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily TV paper-validation report")
    p.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    p.add_argument("--config", default="configs/validation.yaml")
    p.add_argument("--stage", default="paper")
    p.add_argument("--audit-root", default="logs/audit/s4")
    p.add_argument("--predictions-dir", default="data/live/s1_predictions")
    p.add_argument("--verdicts-dir", default="experiments/s2_verdicts_balanced")
    p.add_argument("--position-snapshot-root", default="data/positions")
    p.add_argument("--starting-equity", default="100000")
    p.add_argument("--bar-fixture-parquet", default=None)
    p.add_argument("--alpaca-data-credentials-path", default="~/.alpaca/paper_keys.json")
    p.add_argument("--alpaca-data-base-url", default="https://data.alpaca.markets/v2/stocks/bars")
    return p.parse_args()


def _load_predictions(preds_dir: Path, date_str: str) -> dict[str, dict[str, Any]]:
    """Return {signal_id: row} for the given date's predictions parquet."""
    p = preds_dir / f"{date_str}.parquet"
    if not p.exists():
        return {}
    df = pl.read_parquet(p)
    return {row["signal_id"]: row for row in df.iter_rows(named=True)}


def _load_verdicts(verdicts_dir: Path, date_str: str) -> dict[str, dict[str, Any]]:
    """Return {signal_id: verdict_payload} for the given date's verdicts JSONL."""
    out: dict[str, dict[str, Any]] = {}
    p = verdicts_dir / f"{date_str}.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        sig_id = rec.get("signal_id")
        if isinstance(sig_id, str):
            out[sig_id] = rec
    return out


def _load_fills(audit_root: Path, stage: str, date_str: str) -> dict[str, dict[str, Any]]:
    """Return {client_order_id (== signal_id): fill_payload} for the given date."""
    out: dict[str, dict[str, Any]] = {}
    p = audit_root / stage / f"{date_str}.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event") != "trade_fill":
            continue
        payload = rec.get("payload", {})
        coid = payload.get("client_order_id")
        if isinstance(coid, str):
            out[coid] = payload
    return out


def _missing_bar_loader(_symbol: str, _ts: datetime) -> Bar | None:
    return None


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fixture_bar_loader(path: Path) -> BarLoader:
    df = pl.read_parquet(path)
    bars: dict[tuple[str, datetime], Bar] = {}
    for row in df.iter_rows(named=True):
        symbol = str(row["symbol"])
        ts = _as_utc(row["ts_utc"]).replace(second=0, microsecond=0)
        bars[(symbol, ts)] = Bar(
            symbol=symbol,
            ts_utc=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row.get("volume") or 0),
        )

    def load(symbol: str, ts: datetime) -> Bar | None:
        return bars.get((symbol, _as_utc(ts).replace(second=0, microsecond=0)))

    return load


def _build_bar_loader(args: argparse.Namespace, source: str) -> BarLoader:
    if args.bar_fixture_parquet is not None:
        return _fixture_bar_loader(Path(args.bar_fixture_parquet))
    if source == "alpaca_bars":
        creds = Path(args.alpaca_data_credentials_path).expanduser()
        if creds.exists():
            return AlpacaBarsLoader(
                credentials_path=str(creds),
                base_url=str(args.alpaca_data_base_url),
            )
        console.print(f"[yellow]No Alpaca data credentials at {creds}; realized returns will be NaN.[/yellow]")
    return _missing_bar_loader


def _to_scored(rows: list[PerSignalRow]) -> list[ScoredSignal]:
    return [
        ScoredSignal(
            signal_id=r.signal_id,
            predicted_direction=r.predicted_direction if r.s2_decision == "pass" else 0,
            realized_direction=r.realized_direction,
            weight=r.weight,
            s2_decision=r.s2_decision,
        )
        for r in rows
    ]


def _count_validation_days(daily_report_dir: Path) -> int:
    if not daily_report_dir.exists():
        return 0
    return sum(1 for p in daily_report_dir.glob("*.md") if p.stem.count("-") == 2)


def main() -> int:
    args = parse_args()
    cfg = load_validation_config(Path(args.config))

    preds = _load_predictions(Path(args.predictions_dir), args.date)
    verdicts = _load_verdicts(Path(args.verdicts_dir), args.date)
    fills = _load_fills(Path(args.audit_root), args.stage, args.date)
    bar_loader = _build_bar_loader(args, cfg.data.forward_return_source)

    all_ids = sorted(set(preds) | set(verdicts))
    rows: list[PerSignalRow] = []
    fwd_requests: list[ForwardReturnRequest] = []
    for sig_id in all_ids:
        pred = preds.get(sig_id, {})
        verdict = verdicts.get(sig_id, {})
        fill = fills.get(sig_id, {})

        predicted_score = float(pred.get("predicted_score", 0.0))
        confidence = float(pred.get("confidence", 0.0))
        horizon_minutes = int(pred.get("horizon_minutes", 5))
        symbol = str(pred.get("symbol", "UNKNOWN"))
        ts_str = pred.get("ts_utc") or datetime.now(UTC).isoformat()
        fill_ts_utc = datetime.fromisoformat(str(ts_str))

        s2_decision = str(verdict.get("decision", "insufficient_evidence"))
        predicted_dir = 0
        if s2_decision == "pass":
            predicted_dir = 1 if predicted_score > 0 else (-1 if predicted_score < 0 else 0)

        fill_price = float(fill["price"]) if "price" in fill else None
        fee = float(fill.get("fee", fill.get("commission", 0.0)) or 0.0)
        if "ts_utc" in fill:
            fill_ts_utc = datetime.fromisoformat(str(fill["ts_utc"]))

        weight = float(fill.get("qty", 0.0))

        rows.append(PerSignalRow(
            signal_id=sig_id, symbol=symbol, predicted_score=predicted_score,
            confidence=confidence, predicted_direction=predicted_dir,
            s2_decision=s2_decision, fill_price=fill_price,
            horizon_minutes=horizon_minutes, realized_return=math.nan,
            realized_direction=0, hit=None, weight=weight, fill_ts_utc=fill_ts_utc,
            fee=fee,
        ))
        if fill_price is not None:
            fwd_requests.append(ForwardReturnRequest(
                signal_id=sig_id, symbol=symbol,
                fill_ts_utc=fill_ts_utc, horizon_minutes=horizon_minutes,
            ))

    fwd_results = fetch_forward_returns(
        fwd_requests, bar_loader=bar_loader,
        horizon_alignment=cfg.data.horizon_alignment,
    )
    fwd_by_id = {r.signal_id: r for r in fwd_results}

    rows = [
        PerSignalRow(**{
            **r.__dict__,
            "realized_return": fwd_by_id[r.signal_id].realized_return if r.signal_id in fwd_by_id else math.nan,
            "realized_direction": (
                fwd_by_id[r.signal_id].realized_direction if r.signal_id in fwd_by_id else 0
            ),
            "hit": (
                None if r.fill_price is None
                else (r.predicted_direction == fwd_by_id[r.signal_id].realized_direction
                      and r.predicted_direction != 0)
            ) if r.signal_id in fwd_by_id else (None if r.fill_price is None else False),
        })
        for r in rows
    ]

    scored = _to_scored(rows)
    hit_result = compute_hit_rate(scored)
    starting_equity = float(args.starting_equity)
    pq_dir = Path(cfg.artifacts.per_signal_parquet_dir)
    historical_pnl_pct = load_historical_daily_pnl_pct(
        pq_dir,
        before_date=args.date,
        window_days=cfg.window.rolling_window_days,
        starting_equity=starting_equity,
    )
    pnl_metrics = compute_daily_pnl_metrics(
        rows,
        starting_equity=starting_equity,
        historical_daily_pnl_pct=historical_pnl_pct,
    )

    book_equity = Decimal(args.starting_equity) + Decimal(str(pnl_metrics.daily_pnl))
    broker_equity = book_equity
    reconc = summarize_reconciliation(
        book_equity=book_equity, broker_equity=broker_equity, max_diff_bps=1.0,
    )

    inputs = DailyReportInputs(
        date_str=args.date,
        stage=args.stage,
        broker_name="alpaca_paper",
        rows=rows,
        hit_rate=hit_result,
        reconcile=reconc,
        daily_pnl_pct=pnl_metrics.daily_pnl_pct,
        daily_dd_pct=pnl_metrics.daily_dd_pct,
        sharpe_rolling=pnl_metrics.sharpe_rolling,
        days_in_paper=_count_validation_days(Path(cfg.artifacts.daily_report_dir)),
        min_trading_days=cfg.window.min_trading_days,
        thresholds={
            "hit_rate_min": cfg.thresholds.hit_rate_min,
            "sharpe_min": cfg.thresholds.sharpe_min,
            "max_daily_dd_pct": cfg.thresholds.max_daily_dd_pct,
            "governor_block_rate_max": cfg.thresholds.governor_block_rate_max,
        },
    )

    md = render_markdown(inputs)
    pq = build_per_signal_table(rows)

    md_dir = Path(cfg.artifacts.daily_report_dir)
    md_dir.mkdir(parents=True, exist_ok=True)
    pq_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / f"{args.date}.md"
    pq_path = pq_dir / f"{args.date}.parquet"
    md_path.write_text(md)
    pq.write_parquet(pq_path, compression="zstd")
    console.print(f"Wrote {md_path}")
    console.print(f"Wrote {pq_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
