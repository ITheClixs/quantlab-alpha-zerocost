from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the S1 success gate.")
    p.add_argument("--metrics-json", required=True, help="experiments/alpha_s1/<run_id>/metrics.json")
    p.add_argument("--min-holdout-r2", type=float, default=0.012)
    p.add_argument("--max-fold-std", type=float, default=0.002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics = json.loads(Path(args.metrics_json).read_text())
    holdout_r2 = float(metrics["holdout_weighted_zero_mean_r2"])
    fold_metrics = metrics["fold_metrics"]
    lgb_per_fold = np.array([fm["lgb_r2"] for fm in fold_metrics], dtype=np.float64)
    fold_std = float(np.std(lgb_per_fold))
    failed = []
    if holdout_r2 < args.min_holdout_r2:
        failed.append(f"holdout R² {holdout_r2:.6f} < {args.min_holdout_r2}")
    if fold_std > args.max_fold_std:
        failed.append(f"fold std {fold_std:.6f} > {args.max_fold_std}")
    if failed:
        console.print(f"[red]S1 success gate FAILED[/red]: {failed}")
        return 1
    console.print(f"[green]S1 success gate PASSED[/green]: holdout R²={holdout_r2:.6f}, fold std={fold_std:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
