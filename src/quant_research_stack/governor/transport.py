from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from quant_research_stack.governor.signal_schema import GovernorVerdict


@dataclass
class VerdictWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, verdict: GovernorVerdict) -> None:
        line = verdict.model_dump_json()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def close_and_lock(self) -> None:
        if self.path.exists():
            mode = self.path.stat().st_mode & 0o7777
            self.path.chmod(mode & ~0o222)


def tail_verdicts(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
