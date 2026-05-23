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
from quant_research_stack.alpha.features import FeatureConfig, build_training_features
from quant_research_stack.alpha.io import (
    LoadConfig,
    permanent_holdout_split,
    scan_jane_street,
    select_tail_by_row_budget,
)
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
    # Lazy + tail-budget loading + leak-fixed feature selection. Mirrors the streaming
    # trainer so Optuna sees exactly the dataset and feature set the production run uses.
    lf = scan_jane_street(cfg["data"]["jane_street_root"], load_cfg)
    max_rows = args.max_rows if args.max_rows is not None else int(cfg["data"].get("max_rows", 0))
    if max_rows and max_rows > 0:
        df = select_tail_by_row_budget(lf, load_cfg.group_column, max_rows=max_rows)
    else:
        df = lf.collect().sort(load_cfg.group_column)
    train_df, _ = permanent_holdout_split(df, load_cfg)
    fcfg = FeatureConfig(
        lag_windows=cfg["features"]["lag_windows"],
        rolling_windows=cfg["features"]["rolling_windows"],
        include_noise_feature=cfg["features"]["include_noise_feature"],
        cross_sectional_ranks=cfg["features"]["cross_sectional_ranks"],
        noise_seed=42,
    )
    built, fc = build_training_features(train_df, fcfg, date_col="date_id", symbol_col="symbol_id")
    leaked = [c for c in fc if c.startswith("responder_")]
    if leaked:
        raise RuntimeError(f"leak guard tripped: responder_* in features: {leaked}")
    y = built[cfg["data"]["target_column"]].to_numpy().astype(np.float32)
    w = built[cfg["data"]["weight_column"]].to_numpy().astype(np.float32)
    x = built.select(fc).to_numpy().astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)

    splitter = PurgedKFold(
        n_folds=cfg["cv"]["n_folds"], group_column="date_id",
        purge=cfg["cv"]["purge_days"], embargo=cfg["cv"]["embargo_days"],
    )
    raw_folds = list(splitter.split(built))
    folds = [(tr, te) for tr, te in raw_folds if tr.size > 0 and te.size > 0]
    dropped = len(raw_folds) - len(folds)
    if dropped:
        console.print(f"[yellow]Dropped {dropped} empty purged folds from Optuna CV.[/yellow]")
    if not folds:
        raise RuntimeError(
            "PurgedKFold produced no non-empty train/test folds; increase --max-rows "
            "or reduce purge/embargo in the config."
        )

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
