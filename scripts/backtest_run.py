from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.backtest.report import BacktestReport
from quant_research_stack.backtest.runner import BacktestConfig, BacktestRunner
from quant_research_stack.brokers.fill_model import FillModelConfig
from quant_research_stack.feeds.market_types import Venue
from quant_research_stack.feeds.replayer import Replayer, ReplayerConfig

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a backtest from a YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="experiments/backtests")
    return p.parse_args()


async def _collect_events(replayer_cfg: ReplayerConfig) -> list:
    rep = Replayer(replayer_cfg)
    return [ev async for ev in rep.iterate()]


async def _run(cfg: dict, output_root: Path) -> int:
    rep_cfg_dict = cfg["replayer"]
    rep_cfg = ReplayerConfig(
        root=Path(rep_cfg_dict["root"]),
        venue=Venue(rep_cfg_dict["venue"]),
        symbols=tuple(rep_cfg_dict["symbols"]),
        start_utc=datetime.fromisoformat(str(rep_cfg_dict["start_utc"]).replace("Z", "+00:00")),
        end_utc=datetime.fromisoformat(str(rep_cfg_dict["end_utc"]).replace("Z", "+00:00")),
        speed=float(rep_cfg_dict.get("speed", 0.0)),
    )
    events = await _collect_events(rep_cfg)
    if not events:
        console.print(f"[red]no events found under {rep_cfg.root} for the requested window[/red]")
        return 2
    fm_dict = cfg["fill_model"]
    fill_model = FillModelConfig(
        commission_bps=float(fm_dict.get("commission_bps", 1.0)),
        slippage_bps=float(fm_dict.get("slippage_bps", 2.0)),
        half_spread_bps=float(fm_dict.get("half_spread_bps", 1.0)),
        fill_latency_ms=int(fm_dict.get("fill_latency_ms", 50)),
        reject_if_notional_above_pct_adv=fm_dict.get("reject_if_notional_above_pct_adv"),
        partial_fill_max_pct_of_book=float(fm_dict.get("partial_fill_max_pct_of_book", 0.10)),
    )
    bt_cfg = BacktestConfig(
        events=events,
        fill_model=fill_model,
        starting_cash=float(cfg.get("starting_cash", 100_000.0)),
        strategy_name=str(cfg["strategy_name"]),
        strategy_params=dict(cfg.get("strategy_params", {})),
        metrics_horizon_minutes=int(cfg.get("metrics_horizon_minutes", 5)),
    )
    result = await BacktestRunner(bt_cfg).run()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report = BacktestReport(output_root / run_id)
    report.write(result, run_id=run_id, strategy_name=bt_cfg.strategy_name)
    console.print(f"backtest complete: {output_root / run_id}")
    return 0


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    return asyncio.run(_run(cfg, Path(args.output_root)))


if __name__ == "__main__":
    sys.exit(main())
