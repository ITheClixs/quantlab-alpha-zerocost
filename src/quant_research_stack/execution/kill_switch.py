from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path

from quant_research_stack.execution.audit import AuditLog


class KillSwitchWatcher:
    """Watches a repo-root flag file and can install SIGTERM/SIGINT handlers."""

    def __init__(
        self,
        flag_path: Path,
        poll_interval_s: float,
        audit: AuditLog,
        on_kill: Callable[[str], Awaitable[None]],
    ) -> None:
        self._flag = Path(flag_path)
        self._poll = float(poll_interval_s)
        self._audit = audit
        self._on_kill = on_kill
        self._stop = False
        self._fired = False

    def stop(self) -> None:
        self._stop = True

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig_name in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig_name, self._schedule_signal_trigger, sig_name.name)
            except NotImplementedError:
                pass

    def _schedule_signal_trigger(self, reason: str) -> None:
        asyncio.create_task(self._trigger(reason))

    async def run(self) -> None:
        while not self._stop:
            if self._flag.exists() and not self._fired:
                await self._trigger("file_flag")
                return
            await asyncio.sleep(self._poll)

    async def _trigger(self, reason: str) -> None:
        if self._fired:
            return
        self._fired = True
        self._audit.append("kill_trigger", {"reason": reason})
        await self._on_kill(reason)
