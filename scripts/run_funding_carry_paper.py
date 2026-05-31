"""Funding-carry paper-trading simulation CLI (observation-only, QUANTLAB_STAGE=paper).

Runs the delta-neutral carry through the CarryLoop on REAL public Binance data with
simulated fills. NOT validation/promotion (strategy is DO_NOT_ADVANCE). No live broker.

Run: QUANTLAB_STAGE=paper PYTHONPATH=src uv run python scripts/run_funding_carry_paper.py \
       --config configs/paper_sim.yaml --max-cycles 5
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from quant_research_stack.execution.paper_sim.config import load_paper_sim_config
from quant_research_stack.execution.paper_sim.market_data import MarketDataPoller, MarketSnapshot
from quant_research_stack.execution.paper_sim.runner import CarryLoop, SnapshotSource, ensure_paper_stage


def build_rest_source(poller: MarketDataPoller) -> SnapshotSource:
    async def source(symbol: str, now_ms: int) -> MarketSnapshot:
        return await poller.snapshot(symbol, now_ms=now_ms)
    return source


async def _run(args: argparse.Namespace) -> None:
    ensure_paper_stage()
    cfg = load_paper_sim_config(args.config)
    poller = MarketDataPoller()
    loop = CarryLoop(cfg, audit_root=Path("logs/audit/paper_sim"),
                     snapshot_root=Path("logs/paper_sim_book"),
                     snapshot_source=build_rest_source(poller))
    try:
        await loop.run(max_cycles=args.max_cycles)
    finally:
        await poller.close()
    print(f"done. funding_pnl={loop.funding_pnl():.2f} positions={loop.positions()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Funding-carry paper sim (observation-only)")
    ap.add_argument("--config", default="configs/paper_sim.yaml")
    ap.add_argument("--max-cycles", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
