from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha.adversarial import adversarial_drop_features, drop_below_noise_floor
from quant_research_stack.alpha.cv import PurgedKFold
from quant_research_stack.alpha.features import FeatureConfig, build_feature_frame
from quant_research_stack.alpha.io import LoadConfig, load_jane_street, permanent_holdout_split
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
    p = argparse.ArgumentParser(description="S1 full retrain end-to-end.")
    p.add_argument("--config", default="configs/alpha.yaml")
    p.add_argument("--max-rows", type=int, default=None, help="Cap rows for smoke runs.")
    p.add_argument("--experiments-root", default="experiments/alpha_s1")
    return p.parse_args()


def _build_features(df: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, list[str]]:
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    fcfg = FeatureConfig(
        lag_windows=cfg["features"]["lag_windows"],
        rolling_windows=cfg["features"]["rolling_windows"],
        include_noise_feature=cfg["features"]["include_noise_feature"],
        cross_sectional_ranks=cfg["features"]["cross_sectional_ranks"],
        noise_seed=42,
    )
    built = build_feature_frame(
        df, fcfg, base_features=feature_cols, date_col="date_id", symbol_col="symbol_id"
    )
    feature_cols_all = [c for c in built.columns if c not in {"date_id", "symbol_id", "weight", cfg["data"]["target_column"]}]
    return built, feature_cols_all


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    load_cfg = LoadConfig(
        target_column=cfg["data"]["target_column"],
        weight_column=cfg["data"]["weight_column"],
        group_column=cfg["data"]["group_column"],
        holdout_fraction=cfg["data"]["permanent_holdout_fraction"],
    )
    console.print(f"Loading JS from {cfg['data']['jane_street_root']}")
    df = load_jane_street(cfg["data"]["jane_street_root"], load_cfg)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    train_df, holdout_df = permanent_holdout_split(df, load_cfg)
    console.print(f"Train rows={train_df.height}, holdout rows={holdout_df.height}")

    train_feats, feat_cols = _build_features(train_df, cfg)
    holdout_feats, _ = _build_features(holdout_df, cfg)

    # Adversarial drop
    kept = adversarial_drop_features(train_feats, holdout_feats, feat_cols, auc_threshold=0.6)
    console.print(f"Adversarial filter: kept {len(kept)} / {len(feat_cols)} features")
    feat_cols = kept

    # PurgedKFold
    splitter = PurgedKFold(
        n_folds=cfg["cv"]["n_folds"],
        group_column="date_id",
        purge=cfg["cv"]["purge_days"],
        embargo=cfg["cv"]["embargo_days"],
    )

    y = train_feats[cfg["data"]["target_column"]].to_numpy().astype(np.float64)
    w = train_feats[cfg["data"]["weight_column"]].to_numpy().astype(np.float64)
    x = train_feats.select(feat_cols).to_numpy().astype(np.float64)
    x = np.nan_to_num(x, nan=0.0)

    n = x.shape[0]
    oof_ridge = np.zeros(n)
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_cat = np.zeros(n)
    oof_mlp = np.zeros(n)

    fold_metrics: list[dict[str, Any]] = []
    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(train_feats)):
        console.print(f"Fold {fold_i + 1}/{cfg['cv']['n_folds']}: train={tr_idx.size}, test={te_idx.size}")

        rmod = RidgeAlphaModel(RidgeConfig(alpha=1.0))
        rmod.fit(x[tr_idx], y[tr_idx], w[tr_idx])
        oof_ridge[te_idx] = rmod.predict(x[te_idx])

        lcfg = cfg["models"]["lightgbm"]
        lmod = LightGBMAlphaModel(LightGBMConfig(**{k: lcfg[k] for k in ("num_leaves", "max_depth", "learning_rate", "n_estimators", "early_stopping_rounds", "feature_fraction", "bagging_fraction")}))
        lmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_lgb[te_idx] = lmod.predict(x[te_idx])

        xcfg = cfg["models"]["xgboost"]
        xmod = XGBoostAlphaModel(XGBoostConfig(**{k: xcfg[k] for k in ("max_depth", "learning_rate", "n_estimators", "early_stopping_rounds", "tree_method")}))
        xmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_xgb[te_idx] = xmod.predict(x[te_idx])

        ccfg = cfg["models"]["catboost"]
        cmod = CatBoostAlphaModel(CatBoostConfig(**{k: ccfg[k] for k in ("depth", "learning_rate", "n_estimators", "early_stopping_rounds")}))
        cmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_cat[te_idx] = cmod.predict(x[te_idx])

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
        mmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_mlp[te_idx] = mmod.predict(x[te_idx])

        fold_metrics.append({
            "fold": fold_i,
            "ridge_r2": weighted_zero_mean_r2(y[te_idx], oof_ridge[te_idx], w[te_idx]),
            "lgb_r2": weighted_zero_mean_r2(y[te_idx], oof_lgb[te_idx], w[te_idx]),
            "xgb_r2": weighted_zero_mean_r2(y[te_idx], oof_xgb[te_idx], w[te_idx]),
            "cat_r2": weighted_zero_mean_r2(y[te_idx], oof_cat[te_idx], w[te_idx]),
            "mlp_r2": weighted_zero_mean_r2(y[te_idx], oof_mlp[te_idx], w[te_idx]),
        })

    # Stacking on OOF
    stack_x = np.column_stack([oof_ridge, oof_lgb, oof_xgb, oof_cat, oof_mlp])
    stacker = LinearStacker(alpha=1e-3)
    stacker.fit(stack_x, y, w)

    # Noise-floor filter using LightGBM importance (from a final LGB fit on full train)
    final_lgb = LightGBMAlphaModel(LightGBMConfig(**{k: cfg["models"]["lightgbm"][k] for k in ("num_leaves", "max_depth", "learning_rate", "n_estimators", "early_stopping_rounds", "feature_fraction", "bagging_fraction")}))
    final_lgb.fit(x, y, w, x[-1000:], y[-1000:], w[-1000:])
    importance = final_lgb.feature_importance()
    kept_after_noise = drop_below_noise_floor(feat_cols, importance, noise_feature="noise_seed42")
    console.print(f"Noise-floor filter: kept {len(kept_after_noise)} / {len(feat_cols)} features")

    # Holdout eval
    y_h = holdout_feats[cfg["data"]["target_column"]].to_numpy().astype(np.float64)
    w_h = holdout_feats[cfg["data"]["weight_column"]].to_numpy().astype(np.float64)
    x_h = holdout_feats.select(feat_cols).to_numpy().astype(np.float64)
    x_h = np.nan_to_num(x_h, nan=0.0)

    h_ridge_pred = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    h_ridge_pred.fit(x, y, w)
    h_pred_ridge = h_ridge_pred.predict(x_h)
    h_pred_lgb = final_lgb.predict(x_h)
    holdout_stack = np.column_stack([
        h_pred_ridge, h_pred_lgb,
        np.zeros(x_h.shape[0]), np.zeros(x_h.shape[0]), np.zeros(x_h.shape[0]),
    ])
    holdout_pred = stacker.predict(holdout_stack)
    holdout_r2 = weighted_zero_mean_r2(y_h, holdout_pred, w_h)
    console.print(f"Holdout weighted zero-mean R²: {holdout_r2:.6f}")

    # Persist artifacts
    reg = RunRegistry(root=Path(args.experiments_root))
    meta = RunMetadata(
        version="0.1.0",
        git_sha=_git_sha(),
        data_hashes={"jane_street_root": cfg["data"]["jane_street_root"]},
        hyperparams=cfg,
        fold_definition={"n_folds": cfg["cv"]["n_folds"], "purge": cfg["cv"]["purge_days"], "embargo": cfg["cv"]["embargo_days"]},
    )
    run_id = reg.create_run(meta)
    reg.save_artifact(run_id, "metrics.json", json.dumps({
        "fold_metrics": fold_metrics,
        "holdout_weighted_zero_mean_r2": holdout_r2,
        "n_features_after_adversarial": len(feat_cols),
        "n_features_after_noise_floor": len(kept_after_noise),
    }, indent=2).encode())
    console.print(f"Run id: {run_id}")
    console.print(f"Artifacts under: experiments/alpha_s1/{run_id}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
