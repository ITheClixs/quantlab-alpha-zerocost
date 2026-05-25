"""S1-EQ inference loader (mirrors alpha/inference.py).

Also exposes evaluate_holdout() for the one-shot M5/M6 holdout evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.alpha_eq.models.lightgbm_model import LightGBMEqModel
from quant_research_stack.alpha_eq.models.ridge import RidgeEqModel
from quant_research_stack.alpha_eq.models.xgboost_model import XGBoostEqModel
from quant_research_stack.alpha_eq.stacking import LinearStackerEq


class FeatureSchemaMismatchError(RuntimeError):
    pass


class HoldoutAlreadyEvaluatedError(RuntimeError):
    pass


@dataclass(frozen=True)
class BoundEqPredictor:
    feature_columns: tuple[str, ...]
    ridge: RidgeEqModel
    lightgbm: LightGBMEqModel
    xgboost: XGBoostEqModel
    stacker: LinearStackerEq

    def predict_batch(self, df: pl.DataFrame) -> NDArray[np.float64]:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise FeatureSchemaMismatchError(f"missing feature columns: {missing}")
        x = df.select(list(self.feature_columns)).to_numpy().astype(np.float64)
        oof = np.column_stack(
            [self.ridge.predict(x), self.lightgbm.predict(x), self.xgboost.predict(x)]
        )
        return self.stacker.predict(oof)


def load_predictor_from_run(run_dir: Path) -> BoundEqPredictor:
    rd = Path(run_dir)
    cols = json.loads((rd / "feature_cols.json").read_text())["feature_columns"]
    return BoundEqPredictor(
        feature_columns=tuple(cols),
        ridge=RidgeEqModel.load(rd / "models" / "ridge.joblib"),
        lightgbm=LightGBMEqModel.load(
            rd / "models" / "lightgbm.txt",
            config_path=rd / "models" / "lightgbm.config.json",
        ),
        xgboost=XGBoostEqModel.load(
            rd / "models" / "xgboost.json",
            config_path=rd / "models" / "xgboost.config.json",
        ),
        stacker=LinearStackerEq.load(rd / "models" / "stacker.joblib"),
    )


def evaluate_holdout(
    *, run_dir: Path, holdout_features: pl.DataFrame, target: str
) -> None:
    """One-shot holdout evaluator (spec §4.10).

    Refuses to re-emit holdout_metrics.json once it exists for a given run_id.
    """
    rd = Path(run_dir)
    metrics_path = rd / "holdout_metrics.json"
    if metrics_path.exists():
        raise HoldoutAlreadyEvaluatedError(
            f"holdout_metrics.json already exists at {metrics_path}; "
            "create a new run_id for a fresh holdout evaluation"
        )
    predictor = load_predictor_from_run(rd)
    preds = predictor.predict_batch(holdout_features)
    out = holdout_features.with_columns(pl.Series("y_pred", preds))
    out.write_parquet(rd / "holdout_predictions.parquet")
    metrics: dict[str, object]
    if target in holdout_features.columns:
        y = holdout_features[target].to_numpy().astype(np.float64)
        valid_mask = ~np.isnan(y) & ~np.isnan(preds)
        if int(valid_mask.sum()) > 1:
            # Spearman rank correlation (manual, no scipy dependency)
            ranks_p = _ranks(preds[valid_mask])
            ranks_y = _ranks(y[valid_mask])
            ic = float(np.corrcoef(ranks_p, ranks_y)[0, 1])
        else:
            ic = 0.0
        metrics = {"holdout_rows": int(len(y)), "rank_ic": ic}
    else:
        metrics = {"holdout_rows": int(len(preds))}
    metrics_path.write_text(json.dumps(metrics, sort_keys=True, indent=2))


def _ranks(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    return ranks
