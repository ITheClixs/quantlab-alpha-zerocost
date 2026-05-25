"""Refit-on-full + persistence of all required S1-EQ artifacts (spec §4.8, §4.11)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig
from quant_research_stack.alpha_eq.features.builder import write_feature_cols_json
from quant_research_stack.alpha_eq.models.lightgbm_model import (
    LightGBMEqConfig,
    LightGBMEqModel,
)
from quant_research_stack.alpha_eq.models.ridge import RidgeEqConfig, RidgeEqModel
from quant_research_stack.alpha_eq.models.xgboost_model import (
    XGBoostEqConfig,
    XGBoostEqModel,
)
from quant_research_stack.alpha_eq.stacking import LinearStackerEq

REQUIRED_FAST_V1_ARTIFACTS: tuple[str, ...] = (
    "feature_cols.json",
    "models/ridge.joblib",
    "models/lightgbm.txt",
    "models/lightgbm.config.json",
    "models/xgboost.json",
    "models/xgboost.config.json",
    "models/stacker.joblib",
    "metadata.json",
    "_artifact_sha256.json",
)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def persist_fast_v1_run(
    *,
    run_dir: Path,
    config: AlphaEqConfig,
    feature_cols: Sequence[str],
    dev_panel: pl.DataFrame,
    target: str,
) -> None:
    run_dir = Path(run_dir)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    write_feature_cols_json(run_dir / "feature_cols.json", list(feature_cols))

    panel = dev_panel.drop_nulls(subset=[*list(feature_cols), target])
    x = panel.select(list(feature_cols)).to_numpy().astype(np.float64)
    y = panel[target].to_numpy().astype(np.float64)

    r = RidgeEqModel(RidgeEqConfig(alpha=1.0))
    r.fit(x=x, y=y)
    r.save(run_dir / "models" / "ridge.joblib")

    lgb_cfg = LightGBMEqConfig(seed=config.reproducibility.lightgbm_seed, n_estimators=200)
    m_lgb = LightGBMEqModel(lgb_cfg)
    m_lgb.fit(x=x, y=y)
    m_lgb.save(
        run_dir / "models" / "lightgbm.txt",
        config_path=run_dir / "models" / "lightgbm.config.json",
    )

    xgb_cfg = XGBoostEqConfig(seed=config.reproducibility.xgboost_seed, n_estimators=200)
    m_xgb = XGBoostEqModel(xgb_cfg)
    m_xgb.fit(x=x, y=y)
    m_xgb.save(
        run_dir / "models" / "xgboost.json",
        config_path=run_dir / "models" / "xgboost.config.json",
    )

    # Stacker on training-set predictions (refit-on-full smoke; CV-OOF fit happens in train CLI)
    stacker = LinearStackerEq(
        alpha=config.stacker.alpha,
        prefer_non_negative=config.stacker.prefer_non_negative,
        feature_order=("ridge", "lightgbm", "xgboost"),
    )
    oof_smoke = np.column_stack([r.predict(x), m_lgb.predict(x), m_xgb.predict(x)])
    stacker.fit(oof_predictions=oof_smoke, y=y)
    stacker.save(run_dir / "models" / "stacker.joblib")

    metadata = {
        "git_sha": "filled-by-train_s1_eq.py",
        "data_manifest_sha256": "filled-by-train_s1_eq.py",
        "hyperparams": {
            "ridge": {"alpha": 1.0},
            "lightgbm": {},
            "xgboost": {},
            "stacker": {"alpha": config.stacker.alpha},
        },
        "mode": config.mode.value,
        "seeds": {
            "numpy": config.reproducibility.numpy_seed,
            "lightgbm": config.reproducibility.lightgbm_seed,
            "xgboost": config.reproducibility.xgboost_seed,
        },
        "research_dof": {
            "optuna_trials": {},
            "model_classes_searched": list(config.active_models()),
            "feature_sets_evaluated": 1,
            "threshold_sweeps": 0,
            "post_hoc_decisions": [],
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2))

    sha = {
        art: _sha256_file(run_dir / art)
        for art in REQUIRED_FAST_V1_ARTIFACTS
        if art != "_artifact_sha256.json"
    }
    (run_dir / "_artifact_sha256.json").write_text(json.dumps(sha, sort_keys=True, indent=2))
