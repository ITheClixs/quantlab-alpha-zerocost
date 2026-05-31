"""Train a sentiment-surrogate LightGBM model on the benstaf/nasdaq_2013_2023 dataset.

Predicts the LLM-generated `llm_sentiment` score (1-5 ordinal) from OHLCV + technical
indicators + VIX + turbulence. The trained model becomes a fast sentiment signal that
the S2 governor and S1 feature pipeline can call without invoking a full LLM.

Train: 2013-2018 (NASDAQ-100 + DeepSeek labels, ~47k labeled rows).
Holdout: 2019-2023 (NASDAQ-100 + DeepSeek labels, ~46k labeled rows). Time-disjoint.

Saves model + metrics + predictions under experiments/sentiment_surrogate/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from rich.console import Console
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error

console = Console()


FEATURE_COLUMNS = (
    "close", "high", "low", "open", "volume",
    "day", "macd", "boll_ub", "boll_lb",
    "rsi_30", "cci_30", "dx_30",
    "close_30_sma", "close_60_sma",
    "vix", "turbulence",
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sentiment surrogate from benstaf/nasdaq.")
    p.add_argument(
        "--train-csv",
        default="data/raw/huggingface/benstaf__nasdaq_2013_2023/train_data_deepseek_sentiment_2013_2018.csv",
    )
    p.add_argument(
        "--holdout-csv",
        default="data/raw/huggingface/benstaf__nasdaq_2013_2023/trade_data_deepseek_sentiment_2019_2023.csv",
    )
    p.add_argument("--target-column", default="llm_sentiment")
    p.add_argument("--experiments-root", default="experiments/sentiment_surrogate")
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--early-stopping", type=int, default=50)
    return p.parse_args()


def _load_labeled(csv_path: str, target_col: str) -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_csv(csv_path, infer_schema_length=10000, null_values=["", "NaN", "nan"])
    df = df.filter(pl.col(target_col).is_not_null())
    feat_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(feat_present)
    if missing:
        raise RuntimeError(f"missing expected feature columns: {sorted(missing)}")
    return df, feat_present


def _to_arrays(df: pl.DataFrame, feat_cols: list[str], target_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df.select(feat_cols).to_numpy().astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[target_col].to_numpy().astype(np.float32)
    return x, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pred_int = np.clip(np.round(y_pred), 0, 5).astype(int)
    truth_int = y_true.astype(int)
    # Class-wise recall to detect mode-collapse.
    recalls_by_class = {}
    for cls in sorted(set(truth_int.tolist())):
        mask = truth_int == cls
        if mask.sum() > 0:
            recalls_by_class[f"recall_class_{cls}"] = float((pred_int[mask] == cls).mean())
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "accuracy_nearest_int": float((pred_int == truth_int).mean()),
        "within_1_class": float((np.abs(pred_int - truth_int) <= 1).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(truth_int, pred_int, weights="quadratic")),
        "macro_recall": float(np.mean(list(recalls_by_class.values()))) if recalls_by_class else 0.0,
        "per_class_recall": recalls_by_class,
        "n": int(len(y_true)),
    }


def main() -> int:
    args = parse_args()
    console.print(f"[bold]Loading train[/bold] from {args.train_csv}")
    t0 = time.time()
    train_df, feat_cols = _load_labeled(args.train_csv, args.target_column)
    console.print(f"  train labeled rows: {train_df.height:,} (in {time.time() - t0:.1f}s)")
    console.print(f"  features: {len(feat_cols)} ({', '.join(feat_cols[:8])}...)")

    console.print(f"[bold]Loading holdout[/bold] from {args.holdout_csv}")
    holdout_df, _ = _load_labeled(args.holdout_csv, args.target_column)
    console.print(f"  holdout labeled rows: {holdout_df.height:,}")

    # Time-ordered eval split on the train portion (last 10% for early stopping).
    train_df = train_df.sort("date")
    es_cut = int(train_df.height * 0.90)
    eval_df = train_df[es_cut:]
    fit_df = train_df[:es_cut]

    x_fit, y_fit = _to_arrays(fit_df, feat_cols, args.target_column)
    x_eval, y_eval = _to_arrays(eval_df, feat_cols, args.target_column)
    x_holdout, y_holdout = _to_arrays(holdout_df, feat_cols, args.target_column)

    console.print(
        f"  fit={x_fit.shape[0]:,}  internal_eval={x_eval.shape[0]:,}  holdout={x_holdout.shape[0]:,}"
    )
    fit_class_dist = np.bincount(y_fit.astype(int), minlength=6).tolist()
    console.print(f"  target class distribution (fit): {fit_class_dist}")

    # The mode (class 3 = neutral) dominates ~74%. A plain regressor collapses to
    # mode-prediction (acc≈74%, kappa≈0). Train a class-balanced multi-class classifier
    # so minority classes (bullish 4-5, bearish 0-1) get learned signal. Predicted class
    # becomes the ordinal score.
    console.print("[bold]Training LightGBM multi-class classifier (balanced)[/bold]")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=6,
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(
        x_fit, y_fit.astype(int),
        eval_set=[(x_eval, y_eval.astype(int))],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(args.early_stopping)],
    )
    fit_secs = time.time() - t0
    console.print(f"  trained in {fit_secs:.1f}s  best_iter={model.best_iteration_}")

    eval_pred = model.predict(x_eval).astype(np.float32)
    holdout_pred = model.predict(x_holdout).astype(np.float32)

    eval_metrics = _metrics(y_eval, eval_pred)
    holdout_metrics = _metrics(y_holdout, holdout_pred)
    console.print(f"  internal eval:  {eval_metrics}")
    console.print(f"  [bold green]holdout:[/bold green]      {holdout_metrics}")

    # Persist
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.experiments_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "feature_columns": feat_cols, "target": args.target_column},
                run_dir / "model.joblib")

    pl.DataFrame({
        "split": ["holdout"] * len(y_holdout),
        "date": holdout_df["date"].to_list(),
        "tic": holdout_df["tic"].to_list(),
        "sentiment_actual": y_holdout.astype(np.float32),
        "sentiment_predicted": holdout_pred.astype(np.float32),
    }).write_parquet(run_dir / "holdout_predictions.parquet", compression="zstd")

    importance_df = pl.DataFrame({
        "feature": feat_cols,
        "lgb_importance": model.booster_.feature_importance(importance_type="gain").tolist(),
    }).sort("lgb_importance", descending=True)
    importance_df.write_parquet(run_dir / "feature_importance.parquet", compression="zstd")

    metrics_payload = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "train_csv": args.train_csv,
        "holdout_csv": args.holdout_csv,
        "target_column": args.target_column,
        "feature_columns": feat_cols,
        "train_rows": int(train_df.height),
        "fit_rows": int(x_fit.shape[0]),
        "internal_eval_rows": int(x_eval.shape[0]),
        "holdout_rows": int(x_holdout.shape[0]),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "fit_seconds": round(fit_secs, 2),
        "internal_eval_metrics": eval_metrics,
        "holdout_metrics": holdout_metrics,
        "limitations": [
            "Predicts LLM-generated sentiment labels (DeepSeek-distilled). Not human-labeled.",
            "Surrogate model for fast inference; the upstream LLM remains the canonical signal.",
            "Trained on NASDAQ-100 universe; transfer to other universes not validated.",
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    latest = Path(args.experiments_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_id)

    console.print(f"[bold]Saved[/bold] to {run_dir}/")
    console.print("Files: model.joblib, holdout_predictions.parquet, feature_importance.parquet, metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
