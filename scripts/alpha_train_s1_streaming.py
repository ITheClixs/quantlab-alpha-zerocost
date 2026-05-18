"""S1 streaming trainer.

Memory-efficient counterpart to scripts/alpha_train_s1.py:
- Lazy scan of JS parquet (no whole-frame load).
- Optional row budget via --max-rows (or data.max_rows in config). Picks the most-recent
  contiguous date_ids whose cumulative row count fits the budget.
- All numpy arrays held at float32 (halves memory vs float64).
- Per-fold materialization: only the current fold's slice lives in numpy at any time.

Use this script when training on >1M rows. The old script remains for parity.
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

from quant_research_stack.alpha.adversarial import adversarial_drop_features, drop_below_noise_floor
from quant_research_stack.alpha.cv import PurgedKFold
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
from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig
from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig
from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel, XGBoostConfig
from quant_research_stack.alpha.registry import RunMetadata, RunRegistry
from quant_research_stack.alpha.stacking import LinearStacker

console = Console()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S1 streaming retrain (memory-efficient).")
    p.add_argument("--config", default="configs/alpha_5m.yaml")
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Row budget. Overrides data.max_rows in config. 0 = no cap (full dataset).",
    )
    p.add_argument("--experiments-root", default="experiments/alpha_s1")
    return p.parse_args()


def _build_features(df: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, list[str]]:
    fcfg = FeatureConfig(
        lag_windows=cfg["features"]["lag_windows"],
        rolling_windows=cfg["features"]["rolling_windows"],
        include_noise_feature=cfg["features"]["include_noise_feature"],
        cross_sectional_ranks=cfg["features"]["cross_sectional_ranks"],
        noise_seed=42,
    )
    return build_training_features(df, fcfg, date_col="date_id", symbol_col="symbol_id")


def _materialize_slice_f32(
    df: pl.DataFrame, feat_cols: list[str], target: str, weight: str, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, w) numpy arrays at float32 for the given row indices."""
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
        console.print("No row cap — collecting full lazy frame (may OOM on large parquet)")
        df = lf.collect().sort(group_col)
    console.print(f"Materialized {df.height:,} rows in {time.time() - t0:.1f}s")

    train_df, holdout_df = permanent_holdout_split(df, load_cfg)
    console.print(f"Train rows={train_df.height:,}, holdout rows={holdout_df.height:,}")
    del df
    gc.collect()

    console.print("Building train features…")
    t0 = time.time()
    train_feats, feat_cols = _build_features(train_df, cfg)
    console.print(f"Train features: {len(feat_cols)} cols built in {time.time() - t0:.1f}s")
    del train_df
    gc.collect()

    console.print("Building holdout features…")
    holdout_feats, _ = _build_features(holdout_df, cfg)
    del holdout_df
    gc.collect()

    # Adversarial drop (uses internal row caps to keep memory bounded)
    kept = adversarial_drop_features(train_feats, holdout_feats, feat_cols, auc_threshold=0.6)
    console.print(f"Adversarial filter: kept {len(kept)} / {len(feat_cols)} features")
    feat_cols = kept

    splitter = PurgedKFold(
        n_folds=cfg["cv"]["n_folds"],
        group_column="date_id",
        purge=cfg["cv"]["purge_days"],
        embargo=cfg["cv"]["embargo_days"],
    )

    # OOF arrays at float32 only
    n = train_feats.height
    oof_ridge = np.zeros(n, dtype=np.float32)
    oof_lgb = np.zeros(n, dtype=np.float32)
    oof_xgb = np.zeros(n, dtype=np.float32)
    oof_cat = np.zeros(n, dtype=np.float32)
    oof_mlp = np.zeros(n, dtype=np.float32)
    y_all = train_feats[target_col].to_numpy().astype(np.float32, copy=False)
    w_all = train_feats[weight_col].to_numpy().astype(np.float32, copy=False)

    fold_metrics: list[dict[str, Any]] = []
    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(train_feats)):
        console.print(
            f"[bold]Fold {fold_i + 1}/{cfg['cv']['n_folds']}[/bold]: "
            f"train={tr_idx.size:,}, test={te_idx.size:,}"
        )

        fold_t0 = time.time()
        x_tr, y_tr, w_tr = _materialize_slice_f32(
            train_feats, feat_cols, target_col, weight_col, tr_idx
        )
        x_te, y_te, w_te = _materialize_slice_f32(
            train_feats, feat_cols, target_col, weight_col, te_idx
        )
        console.print(
            f"  fold {fold_i + 1}: materialized {x_tr.nbytes / 1e9:.2f}GB train + "
            f"{x_te.nbytes / 1e9:.2f}GB test in {time.time() - fold_t0:.1f}s"
        )

        # Ridge: sklearn upcasts our float32 → float64 internally during fit, which OOMs
        # macOS jetsam at 2.5M+ rows. Subsample fit-set to 100k. Predict on full test still.
        sub_n = min(100_000, x_tr.shape[0])
        sub_idx = np.random.default_rng(42 + fold_i).choice(x_tr.shape[0], size=sub_n, replace=False)
        console.print(f"  fold {fold_i + 1}: ridge (fit on {sub_n:,}-row subsample)")
        rmod = RidgeAlphaModel(RidgeConfig(alpha=1.0))
        rmod.fit(x_tr[sub_idx], y_tr[sub_idx], w_tr[sub_idx])
        oof_ridge[te_idx] = rmod.predict(x_te)
        del rmod, sub_idx
        gc.collect()

        console.print(f"  fold {fold_i + 1}: lightgbm")
        lcfg = cfg["models"]["lightgbm"]
        lmod = LightGBMAlphaModel(LightGBMConfig(**{
            k: lcfg[k]
            for k in ("num_leaves", "max_depth", "learning_rate", "n_estimators",
                      "early_stopping_rounds", "feature_fraction", "bagging_fraction")
        }))
        lmod.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
        oof_lgb[te_idx] = lmod.predict(x_te)
        del lmod
        gc.collect()

        console.print(f"  fold {fold_i + 1}: xgboost")
        xcfg = cfg["models"]["xgboost"]
        xmod = XGBoostAlphaModel(XGBoostConfig(**{
            k: xcfg[k]
            for k in ("max_depth", "learning_rate", "n_estimators",
                      "early_stopping_rounds", "tree_method")
        }))
        xmod.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
        oof_xgb[te_idx] = xmod.predict(x_te)
        del xmod
        gc.collect()

        console.print(f"  fold {fold_i + 1}: catboost")
        ccfg = cfg["models"]["catboost"]
        cmod = CatBoostAlphaModel(CatBoostConfig(**{
            k: ccfg[k]
            for k in ("depth", "learning_rate", "n_estimators", "early_stopping_rounds")
        }))
        cmod.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
        oof_cat[te_idx] = cmod.predict(x_te)
        del cmod
        gc.collect()

        console.print(f"  fold {fold_i + 1}: mlp")
        mcfg = cfg["models"]["mlp"]
        mmod = MLPAlphaModel(MLPConfig(
            hidden_dims=mcfg["hidden_dims"],
            dropout=mcfg["dropout"],
            learning_rate=mcfg["learning_rate"],
            batch_size=mcfg["batch_size"],
            max_epochs=mcfg["max_epochs"],
            patience=mcfg["patience"],
            mixed_precision=mcfg["mixed_precision"],
        ))
        mmod.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
        oof_mlp[te_idx] = mmod.predict(x_te)
        del mmod
        gc.collect()

        fold_metrics.append({
            "fold": fold_i,
            "ridge_r2": weighted_zero_mean_r2(y_te, oof_ridge[te_idx], w_te),
            "lgb_r2": weighted_zero_mean_r2(y_te, oof_lgb[te_idx], w_te),
            "xgb_r2": weighted_zero_mean_r2(y_te, oof_xgb[te_idx], w_te),
            "cat_r2": weighted_zero_mean_r2(y_te, oof_cat[te_idx], w_te),
            "mlp_r2": weighted_zero_mean_r2(y_te, oof_mlp[te_idx], w_te),
            "wall_seconds": round(time.time() - fold_t0, 1),
        })
        console.print(f"  fold {fold_i + 1}: complete ({time.time() - fold_t0:.0f}s)")
        del x_tr, y_tr, w_tr, x_te, y_te, w_te
        gc.collect()

    # Stacking on OOF
    stack_x = np.column_stack([oof_ridge, oof_lgb, oof_xgb, oof_cat, oof_mlp]).astype(np.float32)
    stacker = LinearStacker(alpha=1e-3)
    stacker.fit(stack_x, y_all, w_all)

    # Final LGB on full train for noise-floor importance + holdout predict
    console.print("Final LightGBM fit on full train for noise-floor importance")
    x_full, _, _ = _materialize_slice_f32(
        train_feats, feat_cols, target_col, weight_col, np.arange(n)
    )
    final_lgb = LightGBMAlphaModel(LightGBMConfig(**{
        k: cfg["models"]["lightgbm"][k]
        for k in ("num_leaves", "max_depth", "learning_rate", "n_estimators",
                  "early_stopping_rounds", "feature_fraction", "bagging_fraction")
    }))
    final_lgb.fit(x_full, y_all, w_all, x_full[-1000:], y_all[-1000:], w_all[-1000:])
    importance = final_lgb.feature_importance()
    kept_after_noise = drop_below_noise_floor(feat_cols, importance, noise_feature="noise_seed42")
    console.print(f"Noise-floor filter: kept {len(kept_after_noise)} / {len(feat_cols)} features")

    # Holdout — materialize once at float32, use stacked predictions
    n_h = holdout_feats.height
    x_h, y_h, w_h = _materialize_slice_f32(
        holdout_feats, feat_cols, target_col, weight_col, np.arange(n_h)
    )

    # Same Ridge subsample trick as per-fold to avoid sklearn float64 upcast OOM
    h_sub_n = min(100_000, x_full.shape[0])
    h_sub = np.random.default_rng(7).choice(x_full.shape[0], size=h_sub_n, replace=False)
    console.print(f"Holdout ridge: fitting on {h_sub_n:,}-row subsample")
    h_ridge = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    h_ridge.fit(x_full[h_sub], y_all[h_sub], w_all[h_sub])
    h_pred_ridge = h_ridge.predict(x_h)
    del h_sub
    h_pred_lgb = final_lgb.predict(x_h)
    holdout_stack = np.column_stack([
        h_pred_ridge,
        h_pred_lgb,
        np.zeros(n_h, dtype=np.float32),
        np.zeros(n_h, dtype=np.float32),
        np.zeros(n_h, dtype=np.float32),
    ]).astype(np.float32)
    holdout_pred = stacker.predict(holdout_stack)
    holdout_r2 = weighted_zero_mean_r2(y_h, holdout_pred, w_h)
    console.print(f"[bold green]Holdout weighted zero-mean R²:[/bold green] {holdout_r2:.6f}")

    # Persist artifacts
    reg = RunRegistry(root=Path(args.experiments_root))
    meta = RunMetadata(
        version="0.1.0",
        git_sha=_git_sha(),
        data_hashes={
            "jane_street_root": cfg["data"]["jane_street_root"],
            "max_rows_used": str(max_rows),
        },
        hyperparams=cfg,
        fold_definition={
            "n_folds": cfg["cv"]["n_folds"],
            "purge": cfg["cv"]["purge_days"],
            "embargo": cfg["cv"]["embargo_days"],
        },
    )
    run_id = reg.create_run(meta)
    reg.save_artifact(run_id, "metrics.json", json.dumps({
        "fold_metrics": fold_metrics,
        "holdout_weighted_zero_mean_r2": float(holdout_r2),
        "n_features_after_adversarial": len(feat_cols),
        "n_features_after_noise_floor": len(kept_after_noise),
        "training_rows": int(n),
        "holdout_rows": int(n_h),
        "max_rows_budget": int(max_rows),
        "profile": "streaming",
    }, indent=2).encode())
    console.print(f"Run id: {run_id}")
    console.print(f"Artifacts under: experiments/alpha_s1/{run_id}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
