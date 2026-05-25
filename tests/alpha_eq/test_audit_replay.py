"""audit_replay_check.py equity-backtest subcommand smoke."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _subprocess_env() -> dict[str, str]:
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
    }


def test_equity_backtest_audit_replay(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    rows = [{"event": "fill", "i": i} for i in range(5)]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = subprocess.run(
        [
            "uv", "run", "python", "scripts/audit_replay_check.py",
            "equity-backtest", "--audit-log", str(log),
        ],
        check=True, capture_output=True, text=True,
        env=_subprocess_env(),
    )
    assert "rows=5" in res.stdout
