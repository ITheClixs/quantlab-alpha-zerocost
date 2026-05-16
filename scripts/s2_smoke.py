from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import polars as pl
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-signal end-to-end governor smoke.")
    p.add_argument("--config", default="configs/governor.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fixture = pl.DataFrame({
        "signal_id": [f"sig-smoke-{i}" for i in range(5)],
        "symbol": ["BTCUSDT"] * 5,
        "direction": [1, -1, 1, 0, 1],
        "confidence": [0.7, 0.4, 0.85, 0.5, 0.9],
        "horizon_minutes": [5, 15, 1, 60, 30],
        "regime_hint": ["trending", "mean_reverting", "trending", "unknown", "high_vol"],
        "recent_vol_label": ["med", "low", "high", "med", "high"],
        "trade_size_pct": [0.3, 0.6, 1.5, 0.1, 2.5],
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.parquet"
        fixture.write_parquet(path)
        from subprocess import run

        rc = run([
            sys.executable, "-m", "scripts.s2_govern",
            "--config", args.config,
            "--predictions", str(path),
            "--once",
        ], check=False).returncode
        console.print(f"smoke completed with rc={rc}")
        return rc


if __name__ == "__main__":
    sys.exit(main())
