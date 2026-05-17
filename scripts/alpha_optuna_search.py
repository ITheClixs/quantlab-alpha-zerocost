from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import yaml
from rich.console import Console

from quant_research_stack.alpha.cv import PurgedKFold
from quant_research_stack.alpha.features import FeatureConfig, build_feature_frame
from quant_research_stack.alpha.io import LoadConfig, load_jane_street, permanent_holdout_split
from quant_research_stack.alpha.metrics import weighted_zero_mean_r2
from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for LightGBM on JS.")
    p.add_argument("--config", default="configs/alpha.yaml")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--study-name", default="alpha_lgb")
    p.add_argument("--out-json", default="reports/alpha_optuna_lgb.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    load_cfg = LoadConfig(
        target_column=cfg["data"]["target_column"],
        weight_column=cfg["data"]["weight_column"],
        group_column=cfg["data"]["group_column"],
        holdout_fraction=cfg["data"]["permanent_holdout_fraction"],
    )
    df = load_jane_street(cfg["data"]["jane_street_root"], load_cfg)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    train_df, _ = permanent_holdout_split(df, load_cfg)
    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    fcfg = FeatureConfig(
        lag_windows=cfg["features"]["lag_windows"],
        rolling_windows=cfg["features"]["rolling_windows"],
        include_noise_feature=cfg["features"]["include_noise_feature"],
        cross_sectional_ranks=cfg["features"]["cross_sectional_ranks"],
        noise_seed=42,
    )
    built = build_feature_frame(train_df, fcfg, base_features=feature_cols, date_col="date_id", symbol_col="symbol_id")
    fc = [c for c in built.columns if c not in {"date_id", "symbol_id", "weight", cfg["data"]["target_column"]}]
    y = built[cfg["data"]["target_column"]].to_numpy().astype(np.float64)
    w = built[cfg["data"]["weight_column"]].to_numpy().astype(np.float64)
    x = built.select(fc).to_numpy().astype(np.float64)
    x = np.nan_to_num(x, nan=0.0)

    splitter = PurgedKFold(
        n_folds=cfg["cv"]["n_folds"], group_column="date_id",
        purge=cfg["cv"]["purge_days"], embargo=cfg["cv"]["embargo_days"],
    )
    folds = list(splitter.split(built))

    def objective(trial: optuna.Trial) -> float:
        params = LightGBMConfig(
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            n_estimators=int(cfg["models"]["lightgbm"]["n_estimators"]),
            early_stopping_rounds=int(cfg["models"]["lightgbm"]["early_stopping_rounds"]),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        )
        scores: list[float] = []
        for tr, te in folds:
            mdl = LightGBMAlphaModel(params)
            mdl.fit(x[tr], y[tr], w[tr], x[te], y[te], w[te])
            scores.append(weighted_zero_mean_r2(y[te], mdl.predict(x[te]), w[te]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name=args.study_name, sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials)
    best = {"value": study.best_value, "params": study.best_params, "n_trials": args.n_trials}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(best, indent=2))
    console.print(f"Best CV R² = {best['value']:.6f}")
    console.print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
