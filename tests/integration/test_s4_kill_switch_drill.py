from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.s4_integration


@pytest.mark.asyncio
async def test_kill_flag_fires_on_kill_callback(tmp_path: Path) -> None:
    from quant_research_stack.execution.audit import AuditLog
    from quant_research_stack.execution.kill_switch import KillSwitchWatcher

    flag = tmp_path / "KILL_TRADING"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)
    fired: list[str] = []

    async def on_kill(reason: str) -> None:
        fired.append(reason)

    watcher = KillSwitchWatcher(flag_path=flag, poll_interval_s=0.05, audit=audit, on_kill=on_kill)
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0.1)
    flag.touch()
    await asyncio.wait_for(asyncio.sleep(0.3), timeout=2.0)
    watcher.stop()
    await task
    audit.close_current()

    events = []
    for p in (tmp_path / "audit").iterdir():
        for line in p.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line)["event"])
    assert "kill_trigger" in events
    assert "file_flag" in fired
