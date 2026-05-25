"""End-to-end smoke (spec §6.2): prepare → train fast_v1 → backtest standard."""

from __future__ import annotations

import os
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _subprocess_env() -> dict[str, str]:
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
    }


def test_fast_v1_standard_backtest_e2e(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(25):
            rows.append({
                "date": d, "symbol": f"S{s}",
                "open": 100.0 + float(rng.standard_normal()),
                "high": 101.0 + float(rng.standard_normal()),
                "low": 99.0 + float(rng.standard_normal()),
                "close": 100.0 + float(rng.standard_normal()),
                "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 100_000),
            })
    pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).write_parquet(
        raw / "panel.parquet"
    )

    eq = tmp_path / "eq"
    eq.mkdir()
    subprocess.run([
        "uv", "run", "python", "scripts/prepare_equity_data.py",
        "--panel", str(raw / "panel.parquet"),
        "--equity-root", str(eq),
        "--membership-source", "absent_prototype_only",
    ], check=True, env=_subprocess_env())

    runs = tmp_path / "runs"
    runs.mkdir()
    subprocess.run([
        "uv", "run", "python", "scripts/train_s1_eq.py",
        "--config", "configs/alpha_eq.yaml", "--mode", "fast_v1",
        "--equity-root", str(eq), "--experiments-root", str(runs),
    ], check=True, env=_subprocess_env())

    run = next(runs.iterdir())
    bt = tmp_path / "bt"
    bt.mkdir()
    subprocess.run([
        "uv", "run", "python", "scripts/backtest_s1_eq.py",
        "--config", "configs/backtest_eq.yaml", "--mode", "standard",
        "--equity-root", str(eq), "--run-dir", str(run),
        "--out-dir", str(bt),
    ], check=True, env=_subprocess_env())

    report = bt / "report.md"
    assert report.exists()
    text = report.read_text()
    assert "prototype-only" in text.lower()
    assert "Configuration" in text
