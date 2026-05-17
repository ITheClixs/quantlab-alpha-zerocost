from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import polars as pl
import yaml
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
        tmp_path = Path(tmp)
        path = tmp_path / "smoke.parquet"
        fixture.write_parquet(path)
        cfg = yaml.safe_load(Path(args.config).read_text())
        cfg["transport"]["primary_verdicts_dir"] = str(tmp_path / "s2_verdicts")
        cfg["transport"]["tier3_verdicts_dir"] = str(tmp_path / "s2_verdicts_tier3")
        cfg["transport"]["audit_log_dir"] = str(tmp_path / "audit")
        smoke_config = tmp_path / "governor_smoke.yaml"
        smoke_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        from subprocess import run

        rc = run([
            sys.executable, "-m", "scripts.s2_govern",
            "--config", str(smoke_config),
            "--predictions", str(path),
            "--once",
        ], check=False).returncode
        console.print(f"smoke completed with rc={rc}")
        return rc


if __name__ == "__main__":
    sys.exit(main())
