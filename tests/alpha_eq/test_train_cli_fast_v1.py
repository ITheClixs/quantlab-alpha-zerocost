"""CLI smoke for fast_v1 training."""

from __future__ import annotations

import os
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)


def _subprocess_env() -> dict[str, str]:
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", ".uv-cache"),
    }


def _seed_root(root: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(8):
            rows.append({
                "date": d,
                "symbol": f"S{s}",
                "open": 100.0 + float(rng.standard_normal()),
                "high": 101.0 + float(rng.standard_normal()),
                "low": 99.0 + float(rng.standard_normal()),
                "close": 100.0 + float(rng.standard_normal()),
                "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 100_000),
            })
    df = pl.DataFrame(rows)
    df.write_parquet(root / "sp500_tradable_prices.parquet")
    df.rename({c: f"{c}_tr" for c in ("open", "high", "low", "close")}).write_parquet(
        root / "sp500_total_return_prices.parquet"
    )
    df.write_parquet(root / "sp500_split_adjusted_prices.parquet")
    pl.DataFrame(
        schema={"ex_date": pl.Date, "symbol": pl.Utf8, "dividend_per_share": pl.Float64}
    ).write_parquet(root / "sp500_dividends.parquet")
    pl.DataFrame(
        {"date": df["date"], "symbol": df["symbol"], "adv_20d_dollar_lag1": [1e7] * df.height}
    ).write_parquet(root / "sp500_adv.parquet")
    pl.DataFrame(
        {
            "symbol": [f"S{s}" for s in range(8)],
            "borrow_tier": ["general"] * 8,
            "annual_bps": [100] * 8,
        }
    ).write_parquet(root / "sp500_borrow_proxy.parquet")
    pl.DataFrame(
        schema={
            "symbol": pl.Utf8, "exit_date": pl.Date, "exit_reason": pl.Utf8,
            "terminal_return_captured": pl.Boolean, "terminal_return_value": pl.Float64,
            "classification_source": pl.Utf8, "classification": pl.Utf8,
        }
    ).write_parquet(root / "sp500_delisting_audit.parquet")

    arts = {}
    for key in (
        "sp500_tradable_prices", "sp500_total_return_prices", "sp500_split_adjusted_prices",
        "sp500_dividends", "sp500_adv", "sp500_borrow_proxy", "sp500_delisting_audit",
    ):
        p = root / f"{key}.parquet"
        loaded = pl.read_parquet(p)
        arts[key] = ManifestArtifact(
            path=p.name,
            sha256=sha256_of_file(p),
            row_count=loaded.height,
            symbol_count=int(loaded["symbol"].n_unique()) if "symbol" in loaded.columns else 0,
            date_range_start=str(loaded["date"].min()) if "date" in loaded.columns else "",
            date_range_end=str(loaded["date"].max()) if "date" in loaded.columns else "",
            schema_fingerprint="cols:" + ",".join(loaded.columns),
        )
    m = EquityManifest(
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        artifacts=arts,
        data_quality_label=DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY,
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1",
        pit_membership_source="absent_prototype_only",
        delisting_audit_quality="audit_absent",
        delisting_audit_counters=DelistingAuditCounters(),
        build_command_line="x",
        python_version="3.11.0",
        package_versions={},
        warnings=[],
    )
    write_manifest(root / "_manifest.json", m)


def test_train_cli_fast_v1_smoke(tmp_path: Path) -> None:
    root = tmp_path / "equities"
    root.mkdir(parents=True, exist_ok=True)
    _seed_root(root)
    out = tmp_path / "runs"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", "scripts/train_s1_eq.py",
            "--config", "configs/alpha_eq.yaml",
            "--mode", "fast_v1",
            "--equity-root", str(root),
            "--experiments-root", str(out),
        ],
        check=True,
        capture_output=True, text=True,
        env=_subprocess_env(),
    )
    run_dirs = list(out.iterdir())
    assert run_dirs
    rd = run_dirs[0]
    assert (rd / "feature_cols.json").exists()
    assert (rd / "models" / "stacker.joblib").exists()
    assert (rd / "holdout_dates.json").exists()
    assert (rd / "metadata.json").exists()
