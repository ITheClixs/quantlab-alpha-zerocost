"""Free-tier Massive.com REST ingestion CLI.

Honest about what the free tier authorizes (verified 2026-05-29): live market
status and previous-day EOD bars only, at 5 calls/min. Historical range
aggregates, live snapshots, and S3 flat-file downloads return 403 and need a
paid plan — see reports/signal_research/microstructure/massive_data_feed_audit.md.

Usage:
    PYTHONPATH=src uv run python scripts/massive_ingest.py status
    PYTHONPATH=src uv run python scripts/massive_ingest.py prev-close \\
        --tickers SPY,QQQ,DIA --out data/processed/massive_prev_close.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.feeds.massive_rest import (
    MassiveREST,
    NotAuthorizedError,
    bars_to_dataframe,
    upsert_panel,
)

console = Console()


def cmd_status(rest: MassiveREST) -> int:
    status = rest.market_status()
    console.print("[bold]Massive.com market status[/bold]")
    console.print(
        f"  NYSE={'open' if status.nyse_open else 'closed'}  "
        f"NASDAQ={'open' if status.nasdaq_open else 'closed'}  "
        f"OTC={'open' if status.otc_open else 'closed'}  "
        f"after_hours={status.after_hours}  early_hours={status.early_hours}  "
        f"server_time={status.server_time}"
    )
    return 0


def cmd_prev_close(rest: MassiveREST, tickers: list[str], out: Path) -> int:
    console.print(f"[cyan]Fetching previous-close bars[/cyan] for {len(tickers)} tickers (5/min)")
    bars = []
    for ticker in tickers:
        try:
            bar = rest.previous_close(ticker)
            bars.append(bar)
            console.print(f"  [green]{ticker}[/green] close={bar.close} @ {bar.timestamp_utc.date()}")
        except NotAuthorizedError as exc:
            console.print(f"  [red]{ticker} NOT_AUTHORIZED[/red]: {exc}")
            return 2
    incoming = bars_to_dataframe(bars)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = pl.read_parquet(out) if out.exists() else pl.DataFrame()
    panel = upsert_panel(existing, incoming)
    panel.write_parquet(out)
    console.print(f"[bold green]Wrote[/bold green] {panel.height} rows -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Massive.com free-tier REST ingestion")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print live market status")
    pc = sub.add_parser("prev-close", help="accumulate previous-day EOD bars into a parquet panel")
    pc.add_argument("--tickers", required=True, help="comma-separated symbols, e.g. SPY,QQQ,DIA")
    pc.add_argument("--out", default="data/processed/massive_prev_close.parquet", type=Path)
    args = parser.parse_args(argv)

    try:
        rest = MassiveREST.from_env()
    except ValueError as exc:
        console.print(f"[red]config error[/red]: {exc}")
        return 1

    if args.command == "status":
        return cmd_status(rest)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    return cmd_prev_close(rest, tickers, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
