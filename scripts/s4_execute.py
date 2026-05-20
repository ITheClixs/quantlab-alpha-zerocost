"""S4 execution daemon. Stage resolved from QUANTLAB_STAGE env var.

Usage:
  QUANTLAB_STAGE=paper PYTHONPATH=src uv run python scripts/s4_execute.py \
    --risk-config configs/risk.yaml --exec-config configs/exec.yaml \
    --brokers-config configs/brokers.yaml --asset-class crypto --starting-equity 100000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.configs import load_exec_config, load_risk_config
from quant_research_stack.execution.kill_switch import KillSwitchWatcher
from quant_research_stack.execution.loop import S4Loop
from quant_research_stack.execution.router import BrokerRouter
from quant_research_stack.feeds.heartbeat import RecordedFeedHeartbeat

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S4 execution daemon")
    parser.add_argument("--risk-config", default="configs/risk.yaml")
    parser.add_argument("--exec-config", default="configs/exec.yaml")
    parser.add_argument("--brokers-config", default="configs/brokers.yaml")
    parser.add_argument("--asset-class", choices=["equity", "crypto"], required=True)
    parser.add_argument("--starting-equity", type=Decimal, required=True)
    parser.add_argument("--feed-recording-root", default="data/live/parquet")
    parser.add_argument("--max-tickets", type=int, default=None, help="Stop after N tickets (testing)")
    return parser.parse_args()


def _is_crypto_fn(sym: str) -> bool:
    return sym.endswith(("USDT", "BTC", "ETH", "BUSD"))


def _mid_lookup_stub(_sym: str) -> Decimal:
    return Decimal("50000")


async def main_async() -> int:
    args = parse_args()
    stage = os.environ.get("QUANTLAB_STAGE")
    if stage not in {"paper", "live_shadow", "live"}:
        console.print(f"[red]QUANTLAB_STAGE must be paper|live_shadow|live; got {stage!r}[/red]")
        return 2

    risk_cfg = load_risk_config(Path(args.risk_config))
    exec_cfg = load_exec_config(Path(args.exec_config))
    brokers_cfg = yaml.safe_load(Path(args.brokers_config).read_text())

    if args.max_tickets == 0:
        console.print("[green]S4 startup config validation OK[/green]")
        return 0

    audit = AuditLog(
        root=Path(exec_cfg.audit.root) / stage,
        chmod_after_close=exec_cfg.audit.chmod_after_close,
    )
    router = BrokerRouter(brokers_cfg)
    broker = router.resolve(stage, asset_class=args.asset_class)
    heartbeat = RecordedFeedHeartbeat(Path(args.feed_recording_root))

    loop = S4Loop(
        stage=stage,
        risk_cfg=risk_cfg,
        exec_cfg=exec_cfg,
        broker=broker,
        audit=audit,
        starting_equity=args.starting_equity,
        mid_price_lookup=_mid_lookup_stub,
        is_crypto=_is_crypto_fn,
        feed_heartbeat_lookup=heartbeat.last_tick_ts,
    )

    flag_path = Path(exec_cfg.kill_switch.repo_root_marker)
    if not flag_path.is_absolute():
        flag_path = Path.cwd() / flag_path
    killer_fired = asyncio.Event()

    async def on_kill(reason: str) -> None:
        console.print(f"[bold red]kill_trigger:[/bold red] {reason}")
        killer_fired.set()

    watcher = KillSwitchWatcher(
        flag_path=flag_path,
        poll_interval_s=exec_cfg.kill_switch.poll_interval_seconds,
        audit=audit,
        on_kill=on_kill,
    )
    watcher.install_signal_handlers()

    watch_task = asyncio.create_task(watcher.run())
    loop_task = asyncio.create_task(loop.run(max_tickets=args.max_tickets))
    killer_task = asyncio.create_task(killer_fired.wait())

    _done, pending = await asyncio.wait(
        {watch_task, loop_task, killer_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    watcher.stop()
    try:
        await broker.close()
    except Exception as exc:
        audit.append("broker_close_error", {"error": repr(exc)})
    exit_code = 137 if killer_fired.is_set() else 0
    audit.append("exit", {"reason": "kill_or_done", "exit_code": exit_code})
    audit.close_current()
    return exit_code


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
