"""Per-fold base-learner training loop (spec §4.3, §4.4)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig
from quant_research_stack.alpha_eq.models.lightgbm_model import (
    LightGBMEqConfig,
    LightGBMEqModel,
)
from quant_research_stack.alpha_eq.models.ridge import RidgeEqConfig, RidgeEqModel
from quant_research_stack.alpha_eq.models.xgboost_model import (
    XGBoostEqConfig,
    XGBoostEqModel,
)
from quant_research_stack.alpha_eq.training.cv import Fold
from quant_research_stack.alpha_eq.training.scalers import FoldScaler


def _split(
    panel: pl.DataFrame, fold: Fold, target: str, feature_cols: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = panel.filter(pl.col("date").is_in(list(fold.train_dates))).drop_nulls(
        subset=[target, *list(feature_cols)]
    )
    valid = panel.filter(pl.col("date").is_in(list(fold.validation_dates))).drop_nulls(
        subset=list(feature_cols)
    )
    x_tr = train.select(list(feature_cols)).to_numpy().astype(np.float64)
    y_tr = train[target].to_numpy().astype(np.float64)
    x_va = valid.select(list(feature_cols)).to_numpy().astype(np.float64)
    y_va = (
        valid[target].to_numpy().astype(np.float64)
        if target in valid.columns
        else np.array([])
    )
    return train, valid, x_tr, y_tr, x_va, y_va


def run_fold_loop(
    *,
    panel: pl.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    fold: Fold,
    config: AlphaEqConfig,
) -> pl.DataFrame:
    _train, valid, x_tr, y_tr, x_va, y_va = _split(panel, fold, target, feature_cols)

    scaler = FoldScaler(fold_id=fold.fold_id)
    scaler.fit(x=x_tr, train_dates=list(fold.train_dates))
    scaler.assert_transform_dates_outside_fit(transform_dates=list(fold.validation_dates))
    x_tr_s = scaler.transform(x_tr)
    x_va_s = scaler.transform(x_va)

    preds: dict[str, np.ndarray] = {}

    if "ridge" in config.active_models():
        m = RidgeEqModel(RidgeEqConfig(alpha=1.0))
        m.fit(x=x_tr_s, y=y_tr)
        preds["ridge"] = m.predict(x_va_s)

    if "lightgbm" in config.active_models():
        lgb_cfg = LightGBMEqConfig(seed=config.reproducibility.lightgbm_seed, n_estimators=200)
        m_lgb = LightGBMEqModel(lgb_cfg)
        m_lgb.fit(x=x_tr_s, y=y_tr, x_val=x_va_s, y_val=y_va if y_va.size else None)
        preds["lightgbm"] = m_lgb.predict(x_va_s)

    if "xgboost" in config.active_models():
        xgb_cfg = XGBoostEqConfig(seed=config.reproducibility.xgboost_seed, n_estimators=200)
        m_xgb = XGBoostEqModel(xgb_cfg)
        m_xgb.fit(x=x_tr_s, y=y_tr, x_val=x_va_s, y_val=y_va if y_va.size else None)
        preds["xgboost"] = m_xgb.predict(x_va_s)

    out = valid
    for name, p in preds.items():
        out = out.with_columns(pl.Series(f"pred_{name}", p))
    return out
