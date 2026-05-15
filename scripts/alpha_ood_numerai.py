from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console

from quant_research_stack.alpha.metrics import weighted_zero_mean_r2

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOD sign-correctness check on Numerai signals data.")
    p.add_argument("--numerai-csv", default="data/raw/kaggle/datasets/code1110__yfinance-stock-price-data-for-numerai-signals")
    p.add_argument("--predictions-parquet", required=True, help="S1 predictions on the Numerai universe.")
    p.add_argument("--out-json", default="reports/alpha_ood_numerai.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preds = pl.read_parquet(args.predictions_parquet)
    if "pred" not in preds.columns or "true" not in preds.columns or "weight" not in preds.columns:
        raise SystemExit("predictions parquet must have 'pred', 'true', 'weight' columns")
    y = preds["true"].to_numpy().astype(np.float64)
    yhat = preds["pred"].to_numpy().astype(np.float64)
    w = preds["weight"].to_numpy().astype(np.float64)
    r2 = weighted_zero_mean_r2(y, yhat, w)
    sign_corr = float(np.mean(np.sign(y) == np.sign(yhat)))
    out = {"weighted_zero_mean_r2": r2, "sign_correctness": sign_corr}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    console.print(f"OOD R²={r2:.6f}, sign_correctness={sign_corr:.4f}")
    console.print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
