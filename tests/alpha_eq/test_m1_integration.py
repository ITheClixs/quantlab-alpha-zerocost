"""M1 integration: prepare → load round trip, with prototype-only label verification."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.data.loaders import EquityRootLoader


def _subprocess_env() -> dict[str, str]:
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
    }


def test_prepare_then_load_then_label_visible(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": [f"2020-01-{d:02d}" for d in range(2, 12)],
            "symbol": ["A"] * 10,
            "open": list(range(100, 110)),
            "high": list(range(101, 111)),
            "low": list(range(99, 109)),
            "close": [float(x) + 0.5 for x in range(100, 110)],
            "volume": [1_000_000] * 10,
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")).write_parquet(
        raw / "panel.parquet"
    )

    out_root = tmp_path / "processed" / "equities"
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", "scripts/prepare_equity_data.py",
            "--panel", str(raw / "panel.parquet"),
            "--equity-root", str(out_root),
            "--membership-source", "absent_prototype_only",
        ],
        check=True,
        env=_subprocess_env(),
    )

    loader = EquityRootLoader(root=out_root)
    df = loader.load_tradable_prices()
    assert df.height == 10
    manifest = json.loads((out_root / "_manifest.json").read_text())
    assert manifest["data_quality_label"] == "survivorship_prototype_only"
    assert "delisting_audit_counters" in manifest
