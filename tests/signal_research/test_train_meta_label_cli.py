"""CLI smoke test for supervised meta-label walk-forward training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl


def test_train_meta_label_cli_writes_artifacts(
    tmp_path: Path,
    synthetic_daily_bars: pl.DataFrame,
    subprocess_env: dict[str, str],
) -> None:
    data_root = tmp_path / "bars"
    data_root.mkdir()
    synthetic_daily_bars.write_parquet(data_root / "synthetic.parquet")
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/train_signal_research_meta_label.py",
            "--data-root",
            str(data_root),
            "--out",
            str(out_dir),
            "--lookback-days",
            "10",
            "--train-window-days",
            "90",
            "--test-window-days",
            "30",
            "--step-days",
            "30",
            "--purge-days",
            "5",
            "--min-train-events",
            "30",
            "--random-forest-estimators",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=subprocess_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "predictions.parquet").exists()
    assert (out_dir / "fold_metrics.parquet").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.md").exists()
