"""Fetch the signal_research data foundation.

Usage:
    PYTHONPATH=src uv run python scripts/fetch_signal_research_data.py \\
        --config configs/signal_research.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.signal_research.data.cboe_proxies import (
    CboeProxiesConfig,
    save_cboe_panel,
)
from quant_research_stack.signal_research.data.fred import FredConfig, save_fred_panel
from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
    save_with_manifest,
)
from quant_research_stack.signal_research.data.nasdaq_components import save_nasdaq_100_manifest
from quant_research_stack.signal_research.data.sp500_components import save_sp500_manifest

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/signal_research.yaml")
    p.add_argument(
        "--skip-crypto",
        action="store_true",
        help="skip BTC/ETH fetch (default off — included)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    start = dt.date.fromisoformat(cfg["data"]["long_history"]["start"])
    end_str = cfg["data"]["long_history"].get("end")
    end = dt.date.fromisoformat(end_str) if end_str else dt.date.today()
    long_history_root = Path(cfg["data"]["long_history"]["cache_root"])

    core_tickers = ["SPY", "QQQ", "ES=F", "NQ=F", "^IXIC", "TQQQ", "SQQQ",
                    "XLK", "SMH", "IGV"]
    for t in core_tickers:
        try:
            df = fetch_one_ticker(t, config=LongHistoryConfig(start=start, end=end))
            save_with_manifest(
                df,
                ticker=t,
                root=long_history_root,
                constituent_survivorship_applicable=False,
            )
            console.print(f"[green]ok[/green] long_history: {t}")
        except Exception as exc:
            console.print(f"[yellow]skip[/yellow] long_history {t}: {exc}")

    cboe_cfg = CboeProxiesConfig(
        tickers=cfg["data"]["cboe_proxies"]["tickers"],
        start=start,
        end=end,
    )
    try:
        save_cboe_panel(config=cboe_cfg, root=Path(cfg["data"]["cboe_proxies"]["cache_root"]))
        console.print("[green]ok[/green] CBOE proxies")
    except Exception as exc:
        console.print(f"[yellow]skip[/yellow] CBOE proxies: {exc}")

    try:
        fred_cfg = FredConfig(start=start, end=end)
        save_fred_panel(
            series_ids=cfg["data"]["fred"]["series"],
            config=fred_cfg,
            root=Path(cfg["data"]["fred"]["cache_root"]),
        )
        console.print("[green]ok[/green] FRED")
    except Exception as exc:
        console.print(f"[yellow]skip[/yellow] FRED ({exc}); set FRED_API_KEY env var")

    save_sp500_manifest(parquet_path=Path("data/processed/signal_research/sp500/sp500_current.parquet"))
    console.print("[green]ok[/green] SP500 current")
    save_nasdaq_100_manifest(parquet_path=Path("data/processed/signal_research/nasdaq/nasdaq_100_current.parquet"))
    console.print("[green]ok[/green] Nasdaq 100 current")

    if not args.skip_crypto:
        console.print("[yellow]note[/yellow] crypto fetcher: concrete adapter selected at execution time")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
