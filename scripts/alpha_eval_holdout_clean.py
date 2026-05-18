"""Honest holdout eval after the responder-leak fix.

The streaming + legacy training scripts had a feature-set bug that leaked the other
responder_* columns into the feature matrix, producing artificially-high R². Both
scripts have been patched. This script provides an apples-to-apples holdout number
using the CLEAN feature set and a simple equal-weight ensemble of the three tree
models (LGB + XGB + CatBoost) — the linear stacker design has a separate bug worth
addressing later. Ridge is excluded because the sklearn float64 upcast OOMs on
24 GB RAM at this scale.

Pipeline:
  scan_jane_street -> select_tail_by_row_budget -> permanent_holdout_split
  -> build_training_features (clean) -> adversarial_drop_features
  -> fit 3 trees on full train (float32) -> mean-ensemble predict on holdout
  -> weighted_zero_mean_r2
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha.adversarial import adversarial_drop_features
from quant_research_stack.alpha.features import FeatureConfig, build_training_features
from quant_research_stack.alpha.io import (
    LoadConfig,
    permanent_holdout_split,
    scan_jane_street,
    select_tail_by_row_budget,
)
from quant_research_stack.alpha.metrics import weighted_zero_mean_r2
from quant_research_stack.alpha.models.catboost_model import CatBoostAlphaModel, CatBoostConfig
from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig
from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel, XGBoostConfig
from quant_research_stack.alpha.registry import RunMetadata, RunRegistry

console = Console()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-feature holdout eval (no responder leakage).")
    p.add_argument("--config", default="configs/alpha_5m.yaml")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Row budget. Overrides data.max_rows in config.")
    p.add_argument("--experiments-root", default="experiments/alpha_s1_clean")
    return p.parse_args()


def _materialize_f32(
    df: pl.DataFrame, feat_cols: list[str], target: str, weight: str, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sliced = df[indices.tolist()]
    x = np.nan_to_num(sliced.select(feat_cols).to_numpy().astype(np.float32, copy=False), nan=0.0)
    y = sliced[target].to_numpy().astype(np.float32, copy=False)
    w = sliced[weight].to_numpy().astype(np.float32, copy=False)
    return x, y, w


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    target_col = cfg["data"]["target_column"]
    weight_col = cfg["data"]["weight_column"]
    group_col = cfg["data"]["group_column"]
    max_rows = args.max_rows if args.max_rows is not None else cfg["data"].get("max_rows", 0)

    load_cfg = LoadConfig(
        target_column=target_col,
        weight_column=weight_col,
        group_column=group_col,
        holdout_fraction=cfg["data"]["permanent_holdout_fraction"],
    )

    console.print(f"[bold]Scanning JS lazily[/bold] from {cfg['data']['jane_street_root']}")
    lf = scan_jane_street(cfg["data"]["jane_street_root"], load_cfg)

    t0 = time.time()
    if max_rows and max_rows > 0:
        console.print(f"Selecting tail by row budget: max_rows={max_rows:,}")
        df = select_tail_by_row_budget(lf, group_col, max_rows=max_rows)
    else:
        df = lf.collect().sort(group_col)
    console.print(f"Materialized {df.height:,} rows in {time.time() - t0:.1f}s")

    train_df, holdout_df = permanent_holdout_split(df, load_cfg)
    console.print(f"Train rows={train_df.height:,}, holdout rows={holdout_df.height:,}")
    del df
    gc.collect()

    fcfg = FeatureConfig(
        lag_windows=cfg["features"]["lag_windows"],
        rolling_windows=cfg["features"]["rolling_windows"],
        include_noise_feature=cfg["features"]["include_noise_feature"],
        cross_sectional_ranks=cfg["features"]["cross_sectional_ranks"],
        noise_seed=42,
    )
    console.print("Building train features (clean, no responder leak)…")
    t0 = time.time()
    train_feats, feat_cols = build_training_features(train_df, fcfg)
    console.print(f"Train features: {len(feat_cols)} cols in {time.time() - t0:.1f}s")
    leaked = [c for c in feat_cols if c.startswith("responder_")]
    if leaked:
        raise RuntimeError(f"FAIL: leak guard broke — responder_* in feature set: {leaked}")
    del train_df
    gc.collect()

    console.print("Building holdout features…")
    holdout_feats, _ = build_training_features(holdout_df, fcfg)
    del holdout_df
    gc.collect()

    kept = adversarial_drop_features(train_feats, holdout_feats, feat_cols, auc_threshold=0.6)
    console.print(f"Adversarial filter: kept {len(kept)} / {len(feat_cols)} features")
    feat_cols = kept

    n = train_feats.height
    n_h = holdout_feats.height

    # Train: materialize once at float32. ~4M × ~600 × 4 = ~10 GB peak. Tight, but works.
    console.print("Materializing full train at float32…")
    t0 = time.time()
    x_tr, y_tr, w_tr = _materialize_f32(train_feats, feat_cols, target_col, weight_col, np.arange(n))
    console.print(f"  train: {x_tr.nbytes / 1e9:.2f}GB in {time.time() - t0:.1f}s")

    # Use last ~5% of train as the early-stopping eval set (NEVER touches holdout).
    es_cut = int(n * 0.95)
    x_val = x_tr[es_cut:]
    y_val = y_tr[es_cut:]
    w_val = w_tr[es_cut:]
    console.print(f"Early-stopping eval set: {x_val.shape[0]:,} rows (last 5% of train)")

    model_metrics: list[dict[str, Any]] = []

    console.print("[bold]Fitting LightGBM on full train…[/bold]")
    t0 = time.time()
    lcfg = cfg["models"]["lightgbm"]
    lmod = LightGBMAlphaModel(LightGBMConfig(**{
        k: lcfg[k] for k in ("num_leaves", "max_depth", "learning_rate", "n_estimators",
                             "early_stopping_rounds", "feature_fraction", "bagging_fraction")
    }))
    lmod.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)
    console.print(f"  LGB fit in {time.time() - t0:.1f}s")

    # Materialize holdout once, predict all 3 models against it.
    console.print("Materializing holdout at float32…")
    x_h, y_h, w_h = _materialize_f32(holdout_feats, feat_cols, target_col, weight_col, np.arange(n_h))
    console.print(f"  holdout: {x_h.nbytes / 1e9:.2f}GB")

    pred_lgb = lmod.predict(x_h)
    lgb_r2 = weighted_zero_mean_r2(y_h, pred_lgb, w_h)
    console.print(f"  LGB holdout R²: {lgb_r2:.6f}")
    model_metrics.append({"model": "lightgbm", "holdout_r2": float(lgb_r2)})
    del lmod
    gc.collect()

    console.print("[bold]Fitting XGBoost on full train…[/bold]")
    t0 = time.time()
    xcfg = cfg["models"]["xgboost"]
    xmod = XGBoostAlphaModel(XGBoostConfig(**{
        k: xcfg[k] for k in ("max_depth", "learning_rate", "n_estimators",
                             "early_stopping_rounds", "tree_method")
    }))
    xmod.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)
    console.print(f"  XGB fit in {time.time() - t0:.1f}s")
    pred_xgb = xmod.predict(x_h)
    xgb_r2 = weighted_zero_mean_r2(y_h, pred_xgb, w_h)
    console.print(f"  XGB holdout R²: {xgb_r2:.6f}")
    model_metrics.append({"model": "xgboost", "holdout_r2": float(xgb_r2)})
    del xmod
    gc.collect()

    console.print("[bold]Fitting CatBoost on full train…[/bold]")
    t0 = time.time()
    ccfg = cfg["models"]["catboost"]
    cmod = CatBoostAlphaModel(CatBoostConfig(**{
        k: ccfg[k] for k in ("depth", "learning_rate", "n_estimators", "early_stopping_rounds")
    }))
    cmod.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)
    console.print(f"  CatBoost fit in {time.time() - t0:.1f}s")
    pred_cat = cmod.predict(x_h)
    cat_r2 = weighted_zero_mean_r2(y_h, pred_cat, w_h)
    console.print(f"  CatBoost holdout R²: {cat_r2:.6f}")
    model_metrics.append({"model": "catboost", "holdout_r2": float(cat_r2)})
    del cmod, x_tr, y_tr, w_tr, x_val, y_val, w_val
    gc.collect()

    # Equal-weight ensemble
    pred_ensemble = (pred_lgb + pred_xgb + pred_cat) / 3.0
    ensemble_r2 = weighted_zero_mean_r2(y_h, pred_ensemble, w_h)
    console.print(f"[bold green]Ensemble (mean of 3 trees) holdout weighted zero-mean R²: {ensemble_r2:.6f}[/bold green]")

    # Persist
    reg = RunRegistry(root=Path(args.experiments_root))
    meta = RunMetadata(
        version="0.1.0",
        git_sha=_git_sha(),
        data_hashes={
            "jane_street_root": cfg["data"]["jane_street_root"],
            "max_rows_used": str(max_rows),
        },
        hyperparams=cfg,
        fold_definition={"strategy": "single-train+holdout (no folds)"},
    )
    run_id = reg.create_run(meta)
    reg.save_artifact(run_id, "metrics.json", json.dumps({
        "ensemble_holdout_weighted_zero_mean_r2": float(ensemble_r2),
        "per_model_holdout_r2": model_metrics,
        "n_features_after_adversarial": len(feat_cols),
        "training_rows": int(n),
        "holdout_rows": int(n_h),
        "max_rows_budget": int(max_rows),
        "profile": "clean_holdout_eval",
        "leak_guard": "asserted_no_responder_in_features",
    }, indent=2).encode())
    console.print(f"Run id: {run_id}")
    console.print(f"Artifacts under: {args.experiments_root}/{run_id}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
