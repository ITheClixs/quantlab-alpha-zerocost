from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """Append-only JSONL audit log with date rotation and chmod-a-w on close."""

    def __init__(self, root: Path | str, rotation: str = "daily", chmod_after_close: bool = True) -> None:
        if rotation != "daily":
            raise ValueError(f"only 'daily' rotation supported; got {rotation}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chmod_after_close = chmod_after_close
        self._current_day: str | None = None
        self._current_path: Path | None = None

    def _current_file(self) -> Path:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_day:
            if self._current_path is not None:
                self._maybe_chmod(self._current_path)
            self._current_day = today
            self._current_path = self.root / f"{today}.jsonl"
        return self._current_path

    def append(self, event: str, payload: dict[str, Any]) -> None:
        path = self._current_file()
        record = {
            "event": event,
            "not_investment_advice": True,
            "payload": payload,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        with path.open("a") as h:
            h.write(json.dumps(record) + "\n")

    def close_current(self) -> None:
        if self._current_path is not None:
            self._maybe_chmod(self._current_path)
            self._current_path = None
            self._current_day = None

    def _maybe_chmod(self, path: Path) -> None:
        if not self.chmod_after_close or not path.exists():
            return
        current = path.stat().st_mode
        os.chmod(path, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
