from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.kill_switch import KillSwitchWatcher


@pytest.mark.asyncio
async def test_watcher_fires_when_flag_appears(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_TRADING"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)
    fired: list[str] = []

    async def on_kill(reason: str) -> None:
        fired.append(reason)

    watcher = KillSwitchWatcher(flag_path=flag, poll_interval_s=0.05, audit=audit, on_kill=on_kill)
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0.1)
    flag.touch()
    await asyncio.wait_for(asyncio.sleep(0.5), timeout=2.0)
    watcher.stop()
    await task
    assert "file_flag" in fired


@pytest.mark.asyncio
async def test_watcher_stops_cleanly(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_TRADING_NEVER"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)

    async def on_kill(_: str) -> None:
        pass

    watcher = KillSwitchWatcher(flag_path=flag, poll_interval_s=0.05, audit=audit, on_kill=on_kill)
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0.15)
    watcher.stop()
    await asyncio.wait_for(task, timeout=1.0)
