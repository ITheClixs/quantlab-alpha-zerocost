"""Smoke for the JS-overlay comparison script."""

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
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", ".uv-cache"),
    }


def test_overlay_compare_smoke(tmp_path: Path) -> None:
    s1_eq_run = tmp_path / "s1_eq_run"
    s1_eq_run.mkdir()
    (s1_eq_run / "metrics.json").write_text(json.dumps({"holdout_sharpe": 0.85}))
    js_run = tmp_path / "js_run"
    js_run.mkdir()
    (js_run / "metrics.json").write_text(json.dumps({"holdout_sharpe": 0.40}))
    out = tmp_path / "compare.json"
    subprocess.run(
        [
            "uv", "run", "python", "scripts/s1_eq_overlay_compare.py",
            "--s1-eq-run", str(s1_eq_run),
            "--js-overlay-run", str(js_run),
            "--out", str(out),
        ],
        check=True, capture_output=True, text=True,
        env=_subprocess_env(),
    )
    payload = json.loads(out.read_text())
    assert payload["s1_eq_sharpe"] == 0.85
    assert payload["js_overlay_sharpe"] == 0.40
    assert payload["s1_eq_beats_js"] is True
