"""Reproducibility contract (spec §4.9):
- byte-identical: splits, configs, feature_cols, manifest hashes
- within tolerance: predictions, metrics
"""

from __future__ import annotations

import os
import subprocess
from datetime import date, timedelta
from pathlib import Path

import joblib
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


def _seed_minimal_root(root: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(80):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(8):
            rows.append({
                "date": d, "symbol": f"S{s}",
                "open": 100.0 + float(rng.standard_normal()),
                "high": 101.0 + float(rng.standard_normal()),
                "low": 99.0 + float(rng.standard_normal()),
                "close": 100.0 + float(rng.standard_normal()),
                "volume": 1_000_000,
            })
    pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).write_parquet(
        root / "panel.parquet"
    )


def test_two_runs_produce_identical_feature_cols_and_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _seed_minimal_root(raw)

    eq = tmp_path / "equities"
    eq.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "run", "python", "scripts/prepare_equity_data.py",
         "--panel", str(raw / "panel.parquet"),
         "--equity-root", str(eq),
         "--membership-source", "absent_prototype_only"],
        check=True, env=_subprocess_env(),
    )
    out_a = tmp_path / "run_a"
    out_a.mkdir()
    out_b = tmp_path / "run_b"
    out_b.mkdir()
    for out in (out_a, out_b):
        subprocess.run(
            ["uv", "run", "python", "scripts/train_s1_eq.py",
             "--config", "configs/alpha_eq.yaml", "--mode", "fast_v1",
             "--equity-root", str(eq), "--experiments-root", str(out)],
            check=True, env=_subprocess_env(),
        )
    run_a = next(out_a.iterdir())
    run_b = next(out_b.iterdir())
    # byte-identical: feature_cols.json, holdout_dates.json
    assert (run_a / "feature_cols.json").read_bytes() == (run_b / "feature_cols.json").read_bytes()
    assert (run_a / "holdout_dates.json").read_bytes() == (run_b / "holdout_dates.json").read_bytes()
    # Stacker weights within tolerance
    sa = joblib.load(run_a / "models" / "stacker.joblib")["estimator"].coef_
    sb = joblib.load(run_b / "models" / "stacker.joblib")["estimator"].coef_
    np.testing.assert_allclose(sa, sb, atol=1e-6)
